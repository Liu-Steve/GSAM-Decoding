import argparse
import itertools
import os
import sqlite3
import multiprocessing as mp
from collections import deque
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Iterable, Sequence


_FAST_PRAGMAS = """
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous  = OFF;
    PRAGMA temp_store   = MEMORY;
    PRAGMA cache_size   = -524288;          -- 512 MiB page cache (negative -> KiB)
"""

_SCHEMA_P1 = """
CREATE TABLE IF NOT EXISTS kgram_counts (
    kgram TEXT PRIMARY KEY,
    cnt   INTEGER NOT NULL
);
"""

_SCHEMA_P2 = """
CREATE TABLE IF NOT EXISTS follow_counts (
    prefix TEXT NOT NULL,
    suffix TEXT NOT NULL,
    cnt    INTEGER NOT NULL,
    PRIMARY KEY(prefix, suffix)
);
"""

def _open_db(path: str, schema_sql: str) -> sqlite3.Connection:
    """Open (and create if necessary) an SQLite DB with a given schema."""
    first = not os.path.exists(path)
    conn = sqlite3.connect(path, isolation_level=None)
    cur = conn.cursor()
    cur.executescript(_FAST_PRAGMAS)
    if first:
        cur.executescript(schema_sql)
    return conn


def _tokens_to_str(t: Sequence[int]) -> str:
    return ",".join(map(str, t))


def _str_to_tokens(s: str) -> tuple[int, ...]:
    return tuple(map(int, s.split(",")))


def _count_kgrams(int_iter: Iterable[int],
                  k: int,
                  conn: sqlite3.Connection,
                  batch: int = 100_000) -> None:
    """Populate table `kgram_counts` from a stream of ints."""
    cur = conn.cursor()
    q = ("INSERT INTO kgram_counts(kgram, cnt) VALUES(?,1) "
         "ON CONFLICT(kgram) DO UPDATE SET cnt = cnt + 1")

    win = deque(maxlen=k)
    todo: list[tuple[str]] = []

    for x in int_iter:
        win.append(x)
        if len(win) == k:
            todo.append((_tokens_to_str(win),))
            if len(todo) >= batch:
                cur.executemany(q, todo)
                todo.clear()
    if todo:
        cur.executemany(q, todo)


def top_kgrams(N: int,
               db_path: str
    ) -> list[tuple[tuple[int, ...], int]]:
    with _open_db(db_path, _SCHEMA_P1) as conn:
        cur = conn.cursor()
        cur.execute("SELECT kgram, cnt FROM kgram_counts ORDER BY cnt DESC LIMIT ?",
                    (N,))
        rows = [( _str_to_tokens(s), c ) for s, c in cur.fetchall()]
    return rows



def _count_followups(int_iter: Iterable[int],
                     k: int, v: int,
                     top_prefixes: Sequence[str],
                     conn: sqlite3.Connection,
                     batch: int = 100_000
    ) -> None:
    cur = conn.cursor()
    q = ("INSERT INTO follow_counts(prefix, suffix, cnt) VALUES(?,?,1) "
         "ON CONFLICT(prefix, suffix) DO UPDATE SET cnt = cnt + 1")

    top_set = set(top_prefixes)
    win = deque(maxlen=k + v)
    todo: list[tuple[str, str]] = []

    for x in int_iter:
        win.append(x)
        if len(win) == k + v:
            prefix = _tokens_to_str(itertools.islice(win, 0, k))
            if prefix in top_set:
                suffix = _tokens_to_str(itertools.islice(win, k, k + v))
                todo.append((prefix, suffix))
                if len(todo) >= batch:
                    cur.executemany(q, todo)
                    todo.clear()
    if todo:
        cur.executemany(q, todo)


def top_followups(int_iter: Iterable[int],
                  k: int,
                  v: int,
                  prefixes: Sequence[tuple[int, ...]],
                  M: int,
                  db_path: str
                  ) -> dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]]:
    conn = _open_db(db_path, _SCHEMA_P2)

    # Build counts
    _count_followups(
        int_iter,
        k, v,
        [_tokens_to_str(p) for p in prefixes],
        conn
    )

    # Fetch results
    cur = conn.cursor()
    out: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
    for p in prefixes:
        ps = _tokens_to_str(p)
        cur.execute(
            "SELECT suffix, cnt FROM follow_counts "
            "WHERE prefix = ? ORDER BY cnt DESC LIMIT ?", (ps, M))
        out[p] = [(_str_to_tokens(s), c) for s, c in cur.fetchall()]

    conn.close()
    return out

def _merge_tables(target_db: str,
                  worker_dbs: Sequence[str],
                  schema_sql: str,
                  table: str,
                  pk_cols: str
    ) -> None:
    """
    Merge *identical* tables from worker_dbs into target_db,
    summing counts on primary-key conflict.
    `pk_cols` is the comma-separated primary key column list (for ON CONFLICT).
    """
    with _open_db(target_db, schema_sql) as conn:
        cur = conn.cursor()

        for wdb in tqdm(worker_dbs):
            # Attach worker database
            cur.execute(f"ATTACH DATABASE '{wdb}' AS worker")
            
            # Create a temporary view of merged data
            cur.execute(f"""
            CREATE TEMPORARY TABLE merged AS
            SELECT {pk_cols}, SUM(cnt) as total_cnt
            FROM (
                SELECT {pk_cols}, cnt FROM {table}
                UNION ALL
                SELECT {pk_cols}, cnt FROM worker.{table}
            )
            GROUP BY {pk_cols}
            """)
            
            # Replace main table with merged data
            cur.execute(f"DELETE FROM {table}")
            cur.execute(f"INSERT INTO {table}({pk_cols}, cnt) SELECT {pk_cols}, total_cnt FROM merged")
            
            # Clean up
            cur.execute("DROP TABLE merged")
            cur.execute("DETACH DATABASE worker")


def batch_iter(stream, batch_size):
    it = iter(stream)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


def construct_task(bit, num_workers):
    tasks = []
    for wid in range(num_workers):
        try:
            batch = next(bit)
        except StopIteration:
            break
        tasks.append((wid, batch))
    return tasks

def process_batch(worker_id, batch, model_path, key_len, db_dir, dataset):
    db_path = os.path.join(db_dir, f"{dataset}_cnt_ngram_worker{worker_id}.sqlite")
    with _open_db(db_path, _SCHEMA_P1) as conn:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, use_cache=False, model_max_length=65536, legacy=True)
        for example in batch:
            tokens = tokenizer(example["text"])["input_ids"]
            _count_kgrams(tokens, key_len, conn)
        return len(batch)

def build_ngram_counts(
        model_path: str,
        dataset: str,
        db_dir: str,
        num_workers: int,
        key_len: int,
        thread_batch: int
    ):
    # Create the database directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)

    data_stream = load_dataset(dataset, split="train", streaming=True, trust_remote_code=True)
    total_examples = data_stream.info.splits["train"].num_examples
    
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_examples) as pbar:
            bit = batch_iter(data_stream, thread_batch)
            tasks = construct_task(bit, num_workers)
            while True:
                if not tasks:
                    break

                results = [
                    pool.apply_async(process_batch, args=(wid, batch, model_path, key_len, db_dir, dataset))
                    for wid, batch in tasks
                ]

                tasks = construct_task(bit, num_workers)

                for r in results:
                    count = r.get()
                    pbar.update(count)


def merge_ngram_counts(db_dir: str, dataset: str, num_workers: int):
    merged_db = os.path.join(db_dir, f"{dataset}_cnt_ngram_merged.sqlite")
    worker_dbs = [
        os.path.join(db_dir, f"{dataset}_cnt_ngram_worker{worker_id}.sqlite")
        for worker_id in range(num_workers)
    ]
    _merge_tables(merged_db, worker_dbs, _SCHEMA_P1,
                  table="kgram_counts", pk_cols="kgram")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build a cache for Shotgun.")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["count-ngram", "merge-ngram", "count-followup", "merge-followup"],
        help="The stage of cache building procedure.",
    )
    parser.add_argument(
        "--thread-batch",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--key-len",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.stage == "count-ngram":
        build_ngram_counts(
            args.model_path,
            args.dataset,
            args.db_dir,
            args.num_workers,
            args.key_len,
            args.thread_batch)
    elif args.stage == "count-followup":
        pass
    else:
        raise ValueError(f"Invalid stage: {args.stage}")
    