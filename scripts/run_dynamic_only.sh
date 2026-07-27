#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_ID="${MODEL_ID:-vicuna-7b-v1.3-csamd-dynamic-only}"
BASE_MODEL_ID="${BASE_MODEL_ID:-vicuna-7b-v1.3}"
GPU_IDS_STRING="${GPU_IDS:-0 1 2}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/eval_dynamic_only}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH must point to the Vicuna-7B-v1.3 model directory." >&2
    exit 2
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 2
fi

read -r -a GPU_IDS_ARRAY <<<"${GPU_IDS_STRING}"
if [[ "${#GPU_IDS_ARRAY[@]}" -ne 3 ]]; then
    echo "GPU_IDS must contain exactly three space-separated GPU ids." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

pids=()
for run_id in 1 2 3; do
    gpu_id="${GPU_IDS_ARRAY[$((run_id - 1))]}"
    bench_name="spec_bench_7b_percent_${run_id}"
    bench_dir="${REPO_ROOT}/data/${bench_name}"
    baseline="${bench_dir}/model_answer/${BASE_MODEL_ID}-vanilla-float16-temp-0.0.jsonl"
    answer_file="${bench_dir}/model_answer/${MODEL_ID}.jsonl"
    log_file="${LOG_DIR}/run_${run_id}.log"

    if [[ ! -f "${bench_dir}/question.jsonl" ]]; then
        echo "Missing benchmark questions: ${bench_dir}/question.jsonl" >&2
        exit 2
    fi
    if [[ ! -f "${baseline}" ]]; then
        echo "Missing reusable autoregressive baseline: ${baseline}" >&2
        exit 2
    fi
    if [[ -e "${answer_file}" && "${OVERWRITE:-0}" != "1" ]]; then
        echo "Refusing to overwrite ${answer_file}; set OVERWRITE=1 to rerun." >&2
        exit 2
    fi

    echo "Starting run ${run_id} on GPU ${gpu_id}; log: ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONPATH="${REPO_ROOT}" \
        "${PYTHON_BIN}" -m evaluation.inference_gsamd \
        --model-path "${MODEL_PATH}" \
        --model-id "${MODEL_ID}" \
        --bench-name "${bench_name}" \
        --answer-file "${answer_file}" \
        --temperature 0.0 \
        --dtype float16 \
        --samd_n_predicts 40 \
        --samd_len_threshold 5 \
        --samd_len_bias 5 \
        --attn_implementation sdpa \
        --samd_map_type lazy_int32 \
        --samd_lazy_threshold 1 \
        >"${log_file}" 2>&1 &
    pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "Completed run $((index + 1))."
    else
        echo "Run $((index + 1)) failed; inspect ${LOG_DIR}/run_$((index + 1)).log." >&2
        status=1
    fi
done

exit "${status}"
