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
import math


_FAST_PRAGMAS = """
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous  = OFF;
    PRAGMA temp_store   = MEMORY;
    PRAGMA cache_size   = -1048576;          -- 1 GiB page cache (negative -> KiB)
"""


def _generate_schema_kgram_counts(prefix_len: int) -> str:
    """Generate `kgram_counts` schema SQL given prefix length (i.e., the `k` in `k-gram`).

    This function creates the SQL schema for the k-gram counting table.
    The schema includes `prefix_len` columns for token IDs (k1, k2, ...) and a count column.
    Each k-gram forms a primary key in the database.

    Example:
        When prefix_len=3, the generated schema will be:
        ```
        CREATE TABLE IF NOT EXISTS kgram_counts (
            k1 INTEGER NOT NULL,
            k2 INTEGER NOT NULL,
            k3 INTEGER NOT NULL,
            cnt INTEGER NOT NULL,
            PRIMARY KEY(k1, k2, k3)
        );
        ```

    Args:
    - `prefix_len`: Length of the k-gram (number of tokens in sequence)

    Returns:
    - SQL statement string for creating the table
    """

    prefix_cols = [f"k{i+1} INTEGER NOT NULL" for i in range(prefix_len)]
    pk_cols = ", ".join(f"k{i+1}" for i in range(prefix_len))

    return f"""
CREATE TABLE IF NOT EXISTS kgram_counts (
    {', '.join(prefix_cols)},
    cnt INTEGER NOT NULL,
    PRIMARY KEY({pk_cols})
);
"""


def _generate_schema_followup_counts(prefix_len: int, followup_len: int) -> str:
    """Generate `followup_counts` schema SQL given prefix and followup lengths.

    This function creates the SQL schema for the follow-up counting table.
    The schema includes `prefix_len` columns for prefix token IDs (p1, p2, ...),
    `followup_len` columns for followup token IDs (s1, s2, ...), and a count column.
    The combination of prefix and followup tokens forms a primary key in the database.

    Example:
        When prefix_len=2 and followup_len=3, the generated schema will be:
        ```
        CREATE TABLE IF NOT EXISTS followup_counts (
            p1 INTEGER NOT NULL,
            p2 INTEGER NOT NULL,
            s1 INTEGER NOT NULL,
            s2 INTEGER NOT NULL,
            s3 INTEGER NOT NULL,
            cnt INTEGER NOT NULL,
            PRIMARY KEY(p1, p2, s1, s2, s3)
        );
        ```

    Args:
    - `prefix_len`: Length of the prefix sequence (number of tokens in prefix)
    - `followup_len`: Length of the followup sequence (number of tokens in followup)

    Returns:
    - SQL statement string for creating the table
    """

    prefix_cols = [f"p{i+1} INTEGER NOT NULL" for i in range(prefix_len)]
    followup_cols = [f"s{i+1} INTEGER NOT NULL" for i in range(followup_len)]
    pk_cols = ", ".join([f"p{i+1}" for i in range(prefix_len)] + [f"s{i+1}" for i in range(followup_len)])

    return f"""
CREATE TABLE IF NOT EXISTS followup_counts (
    {', '.join(prefix_cols)},
    {', '.join(followup_cols)},
    cnt INTEGER NOT NULL,
    PRIMARY KEY({pk_cols})
);
"""


def _into_batch_iter(stream: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Convert a stream into a batch iterator.

    Args:
    - `stream`: Iterable of data.
    - `batch_size`: Number of items per batch.

    Returns:
    - An iterator of batches of data.
    """

    it = iter(stream)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


def _prepare_data_for_workers(
        batch_iter: Iterable[List[Any]],
        num_workers: int
    ) -> List[Tuple[int, List[Any]]]:
    """Prepare data for each worker task.

    Args:
    - `batch_iter`: Iterable of batches of data.
    - `num_workers`: Number of workers.

    Returns:
    - A list of (worker_id, batch) pairs.
    """

    tasks = []
    for wid in range(num_workers):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        tasks.append((wid, batch))
    return tasks


def _open_db(db_path: str, schema: str) -> sqlite3.Connection:
    """Open (and create if necessary) a SQLite DB with a given schema.

    Args:
    - `db_path`: Path to the database file.
    - `schema`: SQL schema for creating the database.

    Returns:
    - A connection to the database.
    """

    first = not os.path.exists(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None)
    cur = conn.cursor()
    cur.executescript(_FAST_PRAGMAS)
    if first:
        cur.executescript(schema)
    return conn


def _update_kgram_counts_db(
        it: Iterable[int],
        prefix_len: int,
        conn: sqlite3.Connection,
        query_batch_size: int = 100_000
    ) -> None:
    """Populate table `kgram_counts` from a stream of ints.
    
    The table counts the number of times each k-gram appears in the stream.
    `prefix_len` is the `k` in k-gram.

    Args:
    - `it`: Iterable of ints.
    - `prefix_len`: Length of the k-gram.
    - `conn`: Connection to the database.
    - `query_batch_size`: Number of SQLqueries to execute at a time.
    """

    cur = conn.cursor()
    
    # Generate SQL query string.
    prefix_cols = [f"k{i+1}" for i in range(prefix_len)]
    placeholders = ", ".join(["?"] * prefix_len)
    prefix_col_str = ", ".join(prefix_cols)
    pk_str = prefix_col_str
    query_str = (f"INSERT INTO kgram_counts({prefix_col_str}, cnt) VALUES({placeholders},1) "
                 f"ON CONFLICT({pk_str}) DO UPDATE SET cnt = cnt + 1")

    # Use a sliding window to collect k-grams.
    window = deque(maxlen=prefix_len)

    # Collect k-grams in a list and execute a single SQL query when
    # the list reaches the size of `query_batch_size`.
    todo = []
    for x in it:
        window.append(x)
        if len(window) == prefix_len:
            todo.append(tuple(window))
            if len(todo) >= query_batch_size:
                cur.executemany(query_str, todo)
                todo.clear()
    if todo:
        cur.executemany(query_str, todo)


def _worker_kgram_counts(
        worker_id: int,
        batch: Iterable[Dict[str, Any]],
        model_path: str,
        prefix_len: int,
        db_dir: str,
        dataset: str,
    ) -> int:
    """Update the k-gram counts database with a batch of data.

    Args:
    - `worker_id`: ID of the worker.
    - `batch`: Batch of data to update the database with.
    - `model_path`: Path to the model.
    - `prefix_len`: Length of the k-gram.
    - `db_dir`: Directory to load and save the database.
    - `dataset`: Name of the dataset.

    Returns:
    - Number of examples processed.
    """

    db_path = os.path.join(db_dir, f"{dataset}_cnt_ngram_worker{worker_id}.sqlite")
    schema = _generate_schema_kgram_counts(prefix_len)

    # Tokenize the batched examples and update the database.
    with _open_db(db_path, schema) as conn:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, use_cache=False,
            model_max_length=2**20, legacy=True
        )
        for example in batch:
            tokens = tokenizer(example["text"])["input_ids"]
            _update_kgram_counts_db(tokens, prefix_len, conn)

    return len(batch)


def build_kgram_counts(
        model_path: str,
        dataset: str,
        db_dir: str,
        num_workers: int,
        prefix_len: int,
        worker_batch_size: int
    ) -> None:
    """Build the k-gram counts database by tokenizing the training set from
    the dataset and updating the database in parallel. Each worker updates a
    separate database. These worker databases can be merged into a single
    database using the `merge_kgram_counts` function.

    Args:
    - `model_path`: Path to the model.
    - `dataset`: Name of the dataset.
    - `db_dir`: Directory to load and save the database.
    - `num_workers`: Number of workers.
    - `prefix_len`: Length of the k-gram.
    - `worker_batch_size`: Number of examples per worker batch.
    """

    # Create the database directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)

    # Load the dataset in streaming mode
    data_stream = load_dataset(dataset, split="train", streaming=True, trust_remote_code=True)
    num_examples = data_stream.info.splits["train"].num_examples

    # Create a batch iterator.
    batch_iter = _into_batch_iter(data_stream, worker_batch_size)

    # Create a pool of workers to update the database.
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=num_examples) as pbar:

            # Prepare data for each worker for the first iteration.
            data_for_workers = _prepare_data_for_workers(batch_iter, num_workers)

            # Run until the batch iterator is exhausted.
            while True:
                if not data_for_workers:
                    break

                # Run the workers in parallel.
                results = [
                    pool.apply_async(_worker_kgram_counts, args=(wid, batch, model_path, prefix_len, db_dir, dataset))
                    for wid, batch in data_for_workers
                ]

                # Prepare data for each worker for the next iteration while
                # the workers are running.
                data_for_workers = _prepare_data_for_workers(batch_iter, num_workers)

                # Wait for the workers to finish and update the progress bar.
                for r in results:
                    count = r.get()
                    pbar.update(count)


def get_top_kgrams(
        top_n: int,
        db_path: str,
        prefix_len: int
    ) -> List[Tuple[Tuple[int, ...], int]]:
    """Get the top N k-grams from the database.
    
    Args:
    - `top_n`: Number of top k-grams to return.
    - `db_path`: Path to the database containing k-gram counts.
    - `prefix_len`: Length of the k-gram (number of tokens in sequence).

    Returns:
    - A list of (k-gram, count) pairs sorted by count in descending order.
    """

    schema = _generate_schema_kgram_counts(prefix_len)
    with _open_db(db_path, schema) as conn:
        cur = conn.cursor()
        
        # Generate SQL query string
        prefix_cols = [f"k{i+1}" for i in range(prefix_len)]
        prefix_col_str = ", ".join(prefix_cols)
        query_str = f"SELECT {prefix_col_str}, cnt FROM kgram_counts ORDER BY cnt DESC LIMIT ?"

        cur.execute(query_str, (top_n,))

        # A list of (k-gram, count) pairs.
        rows = [
            (tuple(row[i] for i in range(prefix_len)), row[prefix_len])
            for row in cur.fetchall()
        ]

    return rows


def _merge_two_dbs(
        output_db_path: str,
        input_db_path1: str, 
        input_db_path2: str,
        schema: str,
        table: str,
        pk_cols: str
    ) -> str:
    """
    Merge two databases into a new output database.
    
    Args:
    - `output_db_path`: Path to the output merged database.
    - `input_db_path1`: Path to the first input database.
    - `input_db_path2`: Path to the second input database.
    - `schema`: SQL schema for creating the database.
    - `table`: Name of the table to merge.
    - `pk_cols`: Comma-separated primary key column list.
    
    Returns:
    - Path to the output database
    """
    # Skip if output already exists
    if os.path.exists(output_db_path):
        return output_db_path
        
    # Skip if either input doesn't exist
    if not os.path.exists(input_db_path1) or not os.path.exists(input_db_path2):
        if os.path.exists(input_db_path1):
            # Just copy the first database if the second doesn't exist
            with open(input_db_path1, 'rb') as src, open(output_db_path, 'wb') as dst:
                dst.write(src.read())
        elif os.path.exists(input_db_path2):
            # Just copy the second database if the first doesn't exist
            with open(input_db_path2, 'rb') as src, open(output_db_path, 'wb') as dst:
                dst.write(src.read())
        return output_db_path
    
    conn = None    
    try:
        conn = _open_db(output_db_path, schema)
        cur = conn.cursor()
        
        # Attach both input databases
        cur.execute(f"ATTACH DATABASE '{input_db_path1}' AS db1")
        cur.execute(f"ATTACH DATABASE '{input_db_path2}' AS db2")
        
        # Directly insert the aggregated data into the output table
        cur.execute(f"""
        INSERT INTO {table}({pk_cols}, cnt)
        SELECT {pk_cols}, SUM(cnt) as total_cnt
        FROM (
            SELECT {pk_cols}, cnt FROM db1.{table}
            UNION ALL
            SELECT {pk_cols}, cnt FROM db2.{table}
        )
        GROUP BY {pk_cols}
        """)
        
        # Clean up
        cur.execute("DETACH DATABASE db1")
        cur.execute("DETACH DATABASE db2")
    finally:
        # Make sure to close the connection even if an error occurs
        if conn:
            conn.close()
            
    return output_db_path

def _parallel_merge_worker(args):
    """Worker function for parallel merge tasks"""
    output_path, input_path1, input_path2, schema, table, pk_cols = args
    return _merge_two_dbs(output_path, input_path1, input_path2, schema, table, pk_cols)

def _parallel_hierarchical_merge(
        db_dir: str,
        dataset: str,
        num_workers: int,
        num_prev_workers: int,
        schema: str,
        table: str,
        pk_cols: str,
        db_type: str
    ) -> str:
    """
    Merge databases hierarchically in parallel using a tournament-like approach.
    
    Args:
    - `db_dir`: Directory containing the databases.
    - `dataset`: Name of the dataset.
    - `num_workers`: Number of workers to run the merge.
    - `num_prev_workers`: Number of worker databases to merge.
    - `schema`: SQL schema for creating the database.
    - `table`: Name of the table to merge.
    - `pk_cols`: Comma-separated primary key column list.
    - `db_type`: Type of database ("ngram" or "followup").
    
    Returns:
    - Path to the final merged database.
    """
    # Generate paths to the worker databases and track their index ranges
    current_files = [
        (os.path.join(db_dir, f"{dataset}_cnt_{db_type}_worker{worker_id}.sqlite"), worker_id, worker_id)
        for worker_id in range(num_prev_workers)
    ]
    
    final_db_path = os.path.join(db_dir, f"{dataset}_cnt_{db_type}_merged.sqlite")
    
    # If only one worker, just rename the file
    if num_prev_workers == 1:
        if os.path.exists(current_files[0][0]):
            with open(current_files[0][0], 'rb') as src, open(final_db_path, 'wb') as dst:
                dst.write(src.read())
        return final_db_path

    # Limit number of processes to avoid system resource exhaustion
    actual_workers = min(num_workers, mp.cpu_count())

    with mp.Pool(processes=actual_workers) as pool:
        iteration = 0

        # Continue merging until we have only one database
        while len(current_files) > 1:
            next_files = []
            merge_tasks = []

            # Prepare merge tasks for this iteration
            for i in range(0, len(current_files), 2):
                if i + 1 < len(current_files):
                    # We have a pair to merge
                    file1_path, start_idx1, end_idx1 = current_files[i]
                    file2_path, start_idx2, end_idx2 = current_files[i+1]

                    # Use the start index of the first file and end index of the second file
                    start_idx = start_idx1
                    end_idx = end_idx2

                    output_path = os.path.join(db_dir, f"{dataset}_cnt_{db_type}_merged_{start_idx}_{end_idx}.sqlite")
                    merge_tasks.append((output_path, file1_path, file2_path, schema, table, pk_cols))
                    next_files.append((output_path, start_idx, end_idx))
                else:
                    # Odd number of databases, pass this one to the next iteration
                    next_files.append(current_files[i])

            # Run merge tasks in parallel
            with tqdm(total=len(merge_tasks), desc=f"Merge iteration {iteration+1}") as pbar:
                # Run at most num_workers tasks at once using apply_async
                results = []
                for task in merge_tasks:
                    if len(results) >= actual_workers:
                        # Wait for one task to complete before adding more
                        results[0].get()
                        pbar.update(1)
                        results.pop(0)
                    # Add new task
                    results.append(pool.apply_async(_parallel_merge_worker, args=(task,)))

                # Wait for remaining tasks to complete
                for result in results:
                    result.get()
                    pbar.update(1)

            current_files = next_files
            iteration += 1
            
            # Force garbage collection after each iteration
            import gc
            gc.collect()

        # Rename the final merged database
        if current_files and os.path.exists(current_files[0][0]):
            with open(current_files[0][0], 'rb') as src, open(final_db_path, 'wb') as dst:
                dst.write(src.read())
    
    return final_db_path

def _merge_tables(merged_db_path: str,
                  worker_db_paths: Sequence[str],
                  schema: str,
                  table: str,
                  pk_cols: str
    ) -> None:
    """
    Merge *identical* tables from worker_dbs into target_db,
    summing counts on primary-key conflict.
    `pk_cols` is the comma-separated primary key column list (for ON CONFLICT).
    
    Note: This function is kept for backward compatibility but new code should use
    _parallel_hierarchical_merge for better performance.
    """
    with _open_db(merged_db_path, schema) as conn:
        cur = conn.cursor()

        for wdb in tqdm(worker_db_paths):
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

def merge_followup_counts(
        db_dir: str,
        dataset: str,
        num_workers: int,
        num_prev_workers: int,
        prefix_len: int,
        followup_len: int,
    ) -> None:
    """Merge the worker followup counts databases into a single database using parallel hierarchical merging.
    
    Args:
    - `db_dir`: Directory containing the databases.
    - `dataset`: Name of the dataset.
    - `num_workers`: Number of workers to run the merge.
    - `num_prev_workers`: Number of worker databases to merge.
    - `prefix_len`: Length of the prefix sequence.
    - `followup_len`: Length of the followup sequence.
    """
    # Generate pk_cols string for the merge operation
    prefix_cols = [f"p{i+1}" for i in range(prefix_len)]
    followup_cols = [f"s{i+1}" for i in range(followup_len)]
    pk_cols = ", ".join(prefix_cols + followup_cols)

    schema = _generate_schema_followup_counts(prefix_len, followup_len)
    
    # Use the parallel hierarchical merge
    _parallel_hierarchical_merge(
        db_dir, 
        dataset, 
        num_workers, 
        num_prev_workers, 
        schema, 
        "followup_counts", 
        pk_cols, 
        "followup"
    )


def merge_kgram_counts(
        db_dir: str,
        dataset: str,
        num_workers: int,
        num_prev_workers: int,
        prefix_len: int
    ) -> None:
    """Merge the worker k-gram counts databases into a single database using parallel hierarchical merging.
    
    Args:
    - `db_dir`: Directory containing the databases.
    - `dataset`: Name of the dataset.
    - `num_workers`: Number of workers to run the merge.
    - `num_prev_workers`: Number of worker databases to merge.
    - `prefix_len`: Length of the k-gram.
    """
    # Generate pk_cols string for the merge operation
    pk_cols = ", ".join(f"k{i+1}" for i in range(prefix_len))

    schema = _generate_schema_kgram_counts(prefix_len)

    # Use the parallel hierarchical merge
    _parallel_hierarchical_merge(
        db_dir, 
        dataset, 
        num_workers, 
        num_prev_workers, 
        schema, 
        "kgram_counts", 
        pk_cols, 
        "ngram"
    )


def _update_followup_counts_db(
        it: Iterable[int],
        prefix_len: int,
        followup_len: int,
        top_prefixes: set[Tuple[int, ...]],
        conn: sqlite3.Connection,
        query_batch_size: int = 100_000
    ) -> None:
    """Update the followup counts database with a stream of ints.

    The table counts the number of times each v-gram in the stream appears immediately
    after a top k-gram. `prefix_len` is the `k` in k-gram. `followup_len` is the `v`
    in v-gram.

    Args:
    - `it`: Iterable of ints.
    - `prefix_len`: Length of the prefix.
    - `followup_len`: Length of the followup.
    - `top_prefixes`: Set of top prefixes to update the database with.
    - `conn`: Connection to the database.
    - `query_batch_size`: Number of SQL queries to execute at a time.
    """

    cur = conn.cursor()

    # Generate SQL query string.
    prefix_cols = [f"p{i+1}" for i in range(prefix_len)]
    followup_cols = [f"s{i+1}" for i in range(followup_len)]
    placeholders = ", ".join(["?"] * (prefix_len + followup_len))
    col_str = ", ".join(prefix_cols + followup_cols)
    pk_str = col_str
    q = (f"INSERT INTO followup_counts({col_str}, cnt) VALUES({placeholders},1) "
         f"ON CONFLICT({pk_str}) DO UPDATE SET cnt = cnt + 1")

    # Use a sliding window to collect prefix and followup tokens.
    window = deque(maxlen=prefix_len+followup_len)

    # Collect prefix and followup tokens in a list and execute a single SQL query when
    # the list reaches the size of `query_batch_size`.
    todo = []
    for x in it:
        window.append(x)
        if len(window) == prefix_len + followup_len:
            prefix = tuple(itertools.islice(window, 0, prefix_len))
            if prefix in top_prefixes:
                suffix = tuple(itertools.islice(window, prefix_len, prefix_len + followup_len))
                todo.append((*prefix, *suffix))
                if len(todo) >= query_batch_size:
                    cur.executemany(q, todo)
                    todo.clear()
    if todo:
        cur.executemany(q, todo)


def get_top_followups(
        top_n: int,
        prefix_len: int,
        followup_len: int,
        top_prefixes: Sequence[Tuple[int, ...]],
        db_path: str
    ) -> Dict[Tuple[int, ...], List[Tuple[Tuple[int, ...], int]]]:
    """
    Get the top M followups for each prefix from a pre-built database.

    Args:
    - `prefix_len`: Length of prefix.
    - `followup_len`: Length of followup.
    - `prefixes`: Sequence of prefixes to get followups for.
    - `M`: Number of top followups to return per prefix.
    - `db_path`: Path to the database containing followup counts.

    Returns:
    - Dictionary mapping prefixes to lists of (followup, count) pairs.
    """
    schema = _generate_schema_followup_counts(prefix_len, followup_len)
    with _open_db(db_path, schema) as conn:

        # Fetch results
        cur = conn.cursor()
        out: Dict[Tuple[int, ...], List[Tuple[Tuple[int, ...], int]]] = {}
        
        prefix_cols = [f"p{i+1}" for i in range(prefix_len)]
        followup_cols = [f"s{i+1}" for i in range(followup_len)]
        prefix_where = " AND ".join(f"{col} = ?" for col in prefix_cols)
        followup_col_str = ", ".join(followup_cols)
        
        for p in top_prefixes:
            cur.execute(
                f"SELECT {followup_col_str}, cnt FROM followup_counts "
                f"WHERE {prefix_where} ORDER BY cnt DESC LIMIT ?", 
                (*p, top_n))
            
            followup_results = []
            for row in cur.fetchall():
                followup = tuple(row[i] for i in range(followup_len))
                count = row[followup_len]
                followup_results.append((followup, count))
                
            out[p] = followup_results

    return out


def _worker_followup_counts(
        worker_id: int,
        batch: Iterable[Dict[str, Any]],
        model_path: str,
        prefix_len: int,
        followup_len: int,
        top_prefixes: set[Tuple[int, ...]],
        db_dir: str,
        dataset: str,
    ) -> int:
    """Update the followup counts database with a batch of data.

    Args:
    - `worker_id`: ID of the worker.
    - `batch`: Batch of data to update the database with.
    - `model_path`: Path to the model.
    - `prefix_len`: Length of the prefix.
    - `followup_len`: Length of the followup.
    - `top_prefixes`: Set of top prefixes to update the database with.
    - `db_dir`: Directory to load and save the database.
    - `dataset`: Name of the dataset.

    Returns:
    - Number of examples processed.
    """
    db_path = os.path.join(db_dir, f"{dataset}_cnt_followup_worker{worker_id}.sqlite")
    schema = _generate_schema_followup_counts(prefix_len, followup_len)
    with _open_db(db_path, schema) as conn:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, use_cache=False, model_max_length=2**20, legacy=True)
        for example in batch:
            tokens = tokenizer(example["text"])["input_ids"]
            _update_followup_counts_db(tokens, prefix_len, followup_len, top_prefixes, conn)
        return len(batch)


def build_followup_counts(
        model_path: str,
        dataset: str,
        db_dir: str,
        num_workers: int,
        prefix_len: int,
        followup_len: int,
        top_prefixes: set[Tuple[int, ...]],
        thread_batch: int
    ) -> None:
    """Build the followup counts database by tokenizing the training set from
    the dataset and updating the database in parallel. Each worker updates a
    separate database. These worker databases can be merged into a single
    database using the `merge_followup_counts` function.

    Args:
    - `model_path`: Path to the model.
    - `dataset`: Name of the dataset.
    - `db_dir`: Directory to load and save the database.
    - `num_workers`: Number of workers.
    - `prefix_len`: Length of the prefix.
    - `followup_len`: Length of the followup.
    - `top_prefixes`: Set of top prefixes to update the database with.
    - `thread_batch`: Number of examples per worker batch.
    """

    # Create the database directory if it doesn't exist.
    os.makedirs(db_dir, exist_ok=True)

    # Load the dataset in streaming mode.
    data_stream = load_dataset(dataset, split="train", streaming=True, trust_remote_code=True)
    total_examples = data_stream.info.splits["train"].num_examples
    
    # Create a pool of workers to update the database.
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_examples) as pbar:

            # Create a batch iterator.
            batch_iter = _into_batch_iter(data_stream, thread_batch)

            # Prepare data for each worker for the first iteration.
            data_for_tasks = _prepare_data_for_workers(batch_iter, num_workers)

            # Run until the batch iterator is exhausted.
            while True:
                if not data_for_tasks:
                    break

                # Run the workers in parallel.
                results = [
                    pool.apply_async(_worker_followup_counts, args=(wid, batch, model_path, prefix_len, followup_len, top_prefixes, db_dir, dataset))
                    for wid, batch in data_for_tasks
                ]

                # Prepare data for each worker for the next iteration while
                # the workers are running.
                data_for_tasks = _prepare_data_for_workers(batch_iter, num_workers)

                # Wait for the workers to finish and update the progress bar.
                for r in results:
                    count = r.get()
                    pbar.update(count)

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
    parser.add_argument(
        "--top-ngrams",
        type=int,
        default=None,
        help="Number of top n-grams to use as prefixes for follow-up counting."
    )
    parser.add_argument(
        "--num-prev-workers",
        type=int,
        default=None,
        help="Number of previous worker databases to merge. Required for merge-followup stage."
    )
    args = parser.parse_args()

    if args.stage == "count-ngram":
        build_kgram_counts(
            args.model_path,
            args.dataset,
            args.db_dir,
            args.num_workers,
            args.key_len,
            args.thread_batch)
    elif args.stage == "merge-ngram":
        merge_kgram_counts(
            args.db_dir, 
            args.dataset, 
            args.num_workers, 
            args.num_prev_workers,
            args.key_len)
    elif args.stage == "count-followup":
        if args.val_len is None:
            parser.error("--val-len is required for count-followup stage")
        if args.top_ngrams is None:
            parser.error("--top-ngrams is required for count-followup stage")

        # Load top n-grams to use as prefixes
        merged_db = os.path.join(args.db_dir, f"{args.dataset}_cnt_ngram_merged.sqlite")
        top_ngrams = get_top_kgrams(args.top_ngrams, merged_db, args.key_len)  # Get top 1000 n-grams
        top_prefixes = set(kgram for kgram, _ in top_ngrams)

        build_followup_counts(
            args.model_path,
            args.dataset,
            args.db_dir,
            args.num_workers,
            args.key_len,
            args.val_len,
            top_prefixes,
            args.thread_batch)
    elif args.stage == "merge-followup":
        if args.val_len is None:
            parser.error("--val-len is required for merge-followup stage")

        merge_followup_counts(
            args.db_dir, 
            args.dataset, 
            args.num_workers, 
            args.num_prev_workers,
            args.key_len,
            args.val_len)
    else:
        raise ValueError(f"Invalid stage: {args.stage}")
    