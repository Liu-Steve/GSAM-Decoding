import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


SUBTASKS = [
    "mt_bench",
    "translation",
    "summarization",
    "qa",
    "math_reasoning",
    "rag",
    "overall",
]

MT_BENCH_CATEGORIES = {
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
}

MODEL_SIZE = "7b"
DEFAULT_DIR = Path(f"./data/spec_bench_{MODEL_SIZE}/model_answer")
DEFAULT_CSV = Path(f"./data/spec_bench_{MODEL_SIZE}/result.csv")
DEFAULT_TOKENIZER = f"/data/llm/vicuna-{MODEL_SIZE}-v1.3"
MODEL_PREFIX = f"vicuna-{MODEL_SIZE}-v1.3-"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def filter_records(records: list[dict], task: str) -> list[dict]:
    if task == "overall":
        return records
    if task == "mt_bench":
        return [record for record in records if record["category"] in MT_BENCH_CATEGORIES]
    return [record for record in records if record["category"] == task]


def model_name_from_file(path: Path) -> str:
    name = path.stem
    if name.startswith(MODEL_PREFIX):
        return name[len(MODEL_PREFIX):]
    return name


def generated_tokens_per_second(records: list[dict]) -> list[float]:
    speeds = []
    for record in records:
        choice = record["choices"][0]
        tokens = sum(choice["new_tokens"])
        seconds = sum(choice["wall_time"])
        if seconds > 0:
            speeds.append(tokens / seconds)
    return speeds


def baseline_tokens_per_second(records: list[dict], tokenizer) -> list[float]:
    speeds = []
    for record in records:
        choice = record["choices"][0]
        tokens = sum(len(tokenizer(turn).input_ids) - 1 for turn in choice["turns"])
        seconds = sum(choice["wall_time"])
        if seconds > 0:
            speeds.append(tokens / seconds)
    return speeds


def accept_lengths(records: list[dict]) -> list[int]:
    lengths = []
    for record in records:
        lengths.extend(record["choices"][0].get("accept_lengths", []))
    return lengths


def max_memory(path: Path) -> int:
    max_mem = 0
    for record in load_jsonl(path):
        max_mem = max(max_mem, int(record.get("memory_usage", 0)))
    return max_mem


def compute_task_metrics(
    spec_records: list[dict],
    baseline_records: list[dict],
    tokenizer,
    task: str,
) -> tuple[float, float]:
    spec_task_records = filter_records(spec_records, task)
    baseline_task_records = filter_records(baseline_records, task)
    spec_speeds = generated_tokens_per_second(spec_task_records)
    baseline_speeds = baseline_tokens_per_second(baseline_task_records, tokenizer)
    if not spec_speeds or not baseline_speeds:
        return float("nan"), float("nan")
    speedup = float(np.mean(spec_speeds) / np.mean(baseline_speeds))
    lengths = accept_lengths(spec_task_records)
    accept_mean = float(np.mean(lengths)) if lengths else float("nan")
    return speedup, accept_mean


def collect_result_dir(answer_dir: Path, tokenizer_path: str = DEFAULT_TOKENIZER) -> pd.DataFrame:
    answer_dir = Path(answer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    files = sorted(answer_dir.glob("*.jsonl"))
    baseline_files = [path for path in files if "vanilla" in path.name]
    if not baseline_files:
        raise FileNotFoundError(f"no vanilla baseline jsonl found in {answer_dir}")
    baseline_path = baseline_files[0]
    baseline_records = load_jsonl(baseline_path)

    rows = []
    for spec_path in files:
        if spec_path == baseline_path:
            continue
        spec_records = load_jsonl(spec_path)
        row = {
            "model": model_name_from_file(spec_path),
            "memory": max_memory(spec_path),
        }
        for task in SUBTASKS:
            speedup, accept_mean = compute_task_metrics(
                spec_records,
                baseline_records,
                tokenizer,
                task,
            )
            row[task] = speedup
            row[f"accept_{task}"] = accept_mean
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build speedup, memory, and accept-length tables from Spec-Bench JSONL outputs."
    )
    parser.add_argument("--dir-path", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_result_dir(args.dir_path, args.tokenizer_path)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_path, index=False)
    pd.set_option("display.max_rows", None)
    print(df.sort_values(by="overall"))


if __name__ == "__main__":
    main()
