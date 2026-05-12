import json
import os

import pandas as pd

from evaluation.speed import speed


subtask = [
    "mt_bench",
    "translation",
    "summarization",
    "qa",
    "math_reasoning",
    "rag",
    "overall",
]
dir_path = "./data/spec_bench/model_answer"
tokenizer = "/data/llm/vicuna-7b-v1.3"


def bytes_to_gib(value):
    return value / 1024 ** 3


def get_memory_metric(record):
    rss_memory = record.get("memory_usage")
    device_memory = record.get("device_memory_usage") or {}
    device_type = device_memory.get("device_type")

    if device_type == "cuda":
        memory_candidates = [
            ("cuda:max_memory_reserved", device_memory.get("max_memory_reserved")),
            ("cuda:max_memory_allocated", device_memory.get("max_memory_allocated")),
            ("cuda:memory_reserved", device_memory.get("memory_reserved")),
            ("cuda:memory_allocated", device_memory.get("memory_allocated")),
        ]
    elif device_type == "mps":
        memory_candidates = [
            ("mps:driver_allocated_memory", device_memory.get("driver_allocated_memory")),
            ("mps:current_allocated_memory", device_memory.get("current_allocated_memory")),
        ]
    else:
        memory_candidates = []

    memory_candidates = [(name, value) for name, value in memory_candidates if value is not None]
    if memory_candidates:
        source, value = max(memory_candidates, key=lambda item: item[1])
        return value, source, rss_memory

    return rss_memory, "rss", rss_memory


if __name__ == "__main__":
    file_paths = os.listdir(dir_path)
    baseline_path = [f for f in file_paths if "vanilla" in f][0]
    spec_path = [f for f in file_paths if f != baseline_path]

    ratio = {
        "model": [],
        "memory": [],
        "memory_gib": [],
        "memory_source": [],
        "rss_memory_gib": [],
    }
    ratio.update({k: [] for k in subtask})
    for spec in spec_path:
        ratio["model"].append(spec[15:-6])
        for task in subtask:
            (
                tokens_per_second,
                tokens_per_second_baseline,
                speedup_ratio,
                accept_lengths_list,
            ) = speed(
                os.path.join(dir_path, spec),
                os.path.join(dir_path, baseline_path),
                tokenizer,
                task,
                report=False,
            )
            ratio[task].append(speedup_ratio)
        with open(os.path.join(dir_path, spec), "r") as f:
            max_mem = 0
            max_rss = 0
            memory_source = "rss"
            for line in f:
                memory, source, rss_memory = get_memory_metric(json.loads(line))
                if memory is not None and memory >= max_mem:
                    max_mem = memory
                    memory_source = source
                if rss_memory is not None:
                    max_rss = max(max_rss, rss_memory)
        ratio["memory"].append(max_mem)
        ratio["memory_gib"].append(bytes_to_gib(max_mem) if max_mem else 0.0)
        ratio["memory_source"].append(memory_source)
        ratio["rss_memory_gib"].append(bytes_to_gib(max_rss) if max_rss else 0.0)

    df = pd.DataFrame(ratio)
    pd.set_option("display.max_rows", None)
    print(df.sort_values(by="overall"))
