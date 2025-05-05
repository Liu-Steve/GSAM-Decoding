import argparse
import itertools
import os
import sqlite3
import multiprocessing as mp
from collections import deque
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Iterable, Sequence, List, Dict, Tuple, Any, Union


_FAST_PRAGMAS = """
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous  = OFF;
    PRAGMA temp_store   = MEMORY;
    PRAGMA cache_size   = -1048576;          -- 1 GiB page cache (negative -> KiB)
"""

def _generate_schema_cnt_ngram(key_len: int) -> str:
    """Generate kgram_counts schema SQL with dynamic key length."""
    cols = [f"k{i+1} INTEGER NOT NULL" for i in range(key_len)]
    pk_cols = ", ".join(f"k{i+1}" for i in range(key_len))
    return f"""
CREATE TABLE IF NOT EXISTS kgram_counts (
    {', '.join(cols)},
    cnt INTEGER NOT NULL,
    PRIMARY KEY({pk_cols})
);
"""

def _generate_schema_cnt_followup(key_len: int, val_len: int) -> str:
    """Generate follow_counts schema SQL with dynamic key and value length."""
    prefix_cols = [f"p{i+1} INTEGER NOT NULL" for i in range(key_len)]
    suffix_cols = [f"s{i+1} INTEGER NOT NULL" for i in range(val_len)]
    pk_cols = ", ".join([f"p{i+1}" for i in range(key_len)] + [f"s{i+1}" for i in range(val_len)])
    return f"""
CREATE TABLE IF NOT EXISTS follow_counts (
    {', '.join(prefix_cols)},
    {', '.join(suffix_cols)},
    cnt INTEGER NOT NULL,
    PRIMARY KEY({pk_cols})
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


def _count_kgrams(int_iter: Iterable[int],
                  k: int,
                  conn: sqlite3.Connection,
                  batch: int = 100_000) -> None:
    """Populate table `kgram_counts` from a stream of ints."""
    cur = conn.cursor()
    
    # Generate dynamic column names and placeholders
    cols = [f"k{i+1}" for i in range(k)]
    placeholders = ", ".join(["?"] * k)
    col_str = ", ".join(cols)
    pk_str = col_str
    
    q = (f"INSERT INTO kgram_counts({col_str}, cnt) VALUES({placeholders},1) "
         f"ON CONFLICT({pk_str}) DO UPDATE SET cnt = cnt + 1")

    win = deque(maxlen=k)
    todo: List[Tuple[int, ...]] = []

    for x in int_iter:
        win.append(x)
        if len(win) == k:
            todo.append(tuple(win))
            if len(todo) >= batch:
                cur.executemany(q, todo)
                todo.clear()
    if todo:
        cur.executemany(q, todo)


def top_kgrams(N: int,
               db_path: str,
               key_len: int
    ) -> List[Tuple[Tuple[int, ...], int]]:
    schema_sql = _generate_schema_cnt_ngram(key_len)
    with _open_db(db_path, schema_sql) as conn:
        cur = conn.cursor()
        
        cols = [f"k{i+1}" for i in range(key_len)]
        col_str = ", ".join(cols)
        
        cur.execute(f"SELECT {col_str}, cnt FROM kgram_counts ORDER BY cnt DESC LIMIT ?",
                    (N,))
        
        rows = []
        for row in cur.fetchall():
            kgram = tuple(row[i] for i in range(key_len))
            count = row[key_len]
            rows.append((kgram, count))
            
    return rows


def _count_followups(int_iter: Iterable[int],
                     k: int, v: int,
                     top_prefixes: set[Tuple[int, ...]],
                     conn: sqlite3.Connection,
                     batch: int = 100_000
    ) -> None:
    # Generate dynamic column names and placeholders
    p_cols = [f"p{i+1}" for i in range(k)]
    s_cols = [f"s{i+1}" for i in range(v)]
    placeholders = ", ".join(["?"] * (k + v))
    col_str = ", ".join(p_cols + s_cols)
    pk_str = col_str
    
    cur = conn.cursor()
    q = (f"INSERT INTO follow_counts({col_str}, cnt) VALUES({placeholders},1) "
         f"ON CONFLICT({pk_str}) DO UPDATE SET cnt = cnt + 1")

    win = deque(maxlen=k + v)
    todo: List[Tuple[int, ...]] = []

    for x in int_iter:
        win.append(x)
        if len(win) == k + v:
            prefix = tuple(itertools.islice(win, 0, k))
            if prefix in top_prefixes:
                suffix = tuple(itertools.islice(win, k, k + v))
                todo.append((*prefix, *suffix))
                if len(todo) >= batch:
                    cur.executemany(q, todo)
                    todo.clear()
    if todo:
        cur.executemany(q, todo)


def top_followups(int_iter: Iterable[int],
                  k: int,
                  v: int,
                  prefixes: Sequence[Tuple[int, ...]],
                  M: int,
                  db_path: str
                  ) -> Dict[Tuple[int, ...], List[Tuple[Tuple[int, ...], int]]]:
    schema_sql = _generate_schema_cnt_followup(k, v)
    conn = _open_db(db_path, schema_sql)

    # Build counts
    _count_followups(
        int_iter,
        k, v,
        prefixes,
        conn
    )

    # Fetch results
    cur = conn.cursor()
    out: Dict[Tuple[int, ...], List[Tuple[Tuple[int, ...], int]]] = {}
    
    p_cols = [f"p{i+1}" for i in range(k)]
    s_cols = [f"s{i+1}" for i in range(v)]
    p_where = " AND ".join(f"{col} = ?" for col in p_cols)
    s_col_str = ", ".join(s_cols)
    
    for p in prefixes:
        cur.execute(
            f"SELECT {s_col_str}, cnt FROM follow_counts "
            f"WHERE {p_where} ORDER BY cnt DESC LIMIT ?", 
            (*p, M))
        
        suffix_results = []
        for row in cur.fetchall():
            suffix = tuple(row[i] for i in range(v))
            count = row[v]
            suffix_results.append((suffix, count))
            
        out[p] = suffix_results

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


def _batch_iter(stream, batch_size):
    it = iter(stream)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


def _construct_data_for_tasks(bit, num_workers):
    tasks = []
    for wid in range(num_workers):
        try:
            batch = next(bit)
        except StopIteration:
            break
        tasks.append((wid, batch))
    return tasks

def _count_ngram_worker(worker_id, batch, model_path, key_len, db_dir, dataset):
    db_path = os.path.join(db_dir, f"{dataset}_cnt_ngram_worker{worker_id}.sqlite")
    schema_sql = _generate_schema_cnt_ngram(key_len)
    with _open_db(db_path, schema_sql) as conn:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, use_cache=False, model_max_length=2**20, legacy=True)
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
            bit = _batch_iter(data_stream, thread_batch)
            data_for_tasks = _construct_data_for_tasks(bit, num_workers)
            while True:
                if not data_for_tasks:
                    break

                results = [
                    pool.apply_async(_count_ngram_worker, args=(wid, batch, model_path, key_len, db_dir, dataset))
                    for wid, batch in data_for_tasks
                ]

                data_for_tasks = _construct_data_for_tasks(bit, num_workers)

                for r in results:
                    count = r.get()
                    pbar.update(count)


def merge_ngram_counts(db_dir: str, dataset: str, num_workers: int, key_len: int):
    merged_db = os.path.join(db_dir, f"{dataset}_cnt_ngram_merged.sqlite")
    worker_dbs = [
        os.path.join(db_dir, f"{dataset}_cnt_ngram_worker{worker_id}.sqlite")
        for worker_id in range(num_workers)
    ]
    
    # Generate pk_cols string for the merge operation
    pk_cols = ", ".join(f"k{i+1}" for i in range(key_len))
    
    schema_sql = _generate_schema_cnt_ngram(key_len)
    _merge_tables(merged_db, worker_dbs, schema_sql,
                  table="kgram_counts", pk_cols=pk_cols)

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
    parser.add_argument(
        "--val-len",
        type=int,
        default=None,
        help="Length of the value n-gram for follow-up counting. Required for count-followup stage."
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
    elif args.stage == "merge-ngram":
        merge_ngram_counts(
            args.db_dir, 
            args.dataset, 
            args.num_workers, 
            args.key_len)
    elif args.stage == "count-followup":
        if args.val_len is None:
            parser.error("--val-len is required for count-followup stage")
        # Implement count-followup logic here
        pass
    elif args.stage == "merge-followup":
        # Implement merge-followup logic here
        pass
    else:
        raise ValueError(f"Invalid stage: {args.stage}")
    