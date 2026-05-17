import os
import json
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

if __name__ == "__main__":
    file_paths = os.listdir(dir_path)
    baseline_path = [f for f in file_paths if "vanilla" in f][0]
    spec_path = [f for f in file_paths if f != baseline_path]
    # spec_path = [
    #     "vicuna-7b-v1.3-samd_bias_10_bias_10_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_11_bias_11_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_12_bias_12_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_13_bias_13_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_14_bias_14_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_15_bias_15_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_16_bias_16_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_17_bias_17_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_18_bias_18_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_19_bias_19_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_bias_20_bias_20_threshold_5.jsonl",
    #     "vicuna-7b-v1.3-samd_dynamic_only_bias_5_threshold_5.jsonl",
    # ]

    ratio = {"model": [], "memory": []}
    ratio.update({k: [] for k in subtask})
    for spec in spec_path:
        ratio["model"].append(spec[15:-6])
        # ratio["model"].append(spec.split("-")[3].split(".")[0])
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
            for line in f:
                mem = json.loads(line)["memory_usage"]
                max_mem = max(mem, max_mem)
        ratio["memory"].append(max_mem)

    df = pd.DataFrame(ratio)
    pd.set_option("display.max_rows", None)
    print(df.sort_values(by="overall"))
