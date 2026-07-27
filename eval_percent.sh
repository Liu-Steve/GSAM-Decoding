#!/usr/bin/env bash

set -u

MODEL_PATH=/data/llm
SPEC_BENCH_PATH=/home/user/program/GSAM-Decoding

MODEL_SIZE=7
VICUNA_PATH=${MODEL_PATH}/vicuna-${MODEL_SIZE}b-v1.3
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
BENCH_NAME=spec_bench_7b_percent
TEMP=0.0
TORCH_DTYPE=float16

GSAM_DIR=${SPEC_BENCH_PATH}/local_cache/gsamd
LOG_DIR=${SPEC_BENCH_PATH}/logs/eval_percent

PERCENTAGES=(1 5 10 50 100)
GPU_DEVICES=(1 2 3 4 5)
GSAM_FILES=(
    static_data_gsam_1.pb
    static_data_gsam_5.pb
    static_data_gsam_10.pb
    static_data_gsam_50.pb
    static_data_gsam.pb
)

mkdir -p "${LOG_DIR}"
cd "${SPEC_BENCH_PATH}" || exit 1

pids=()
for i in "${!PERCENTAGES[@]}"; do
    percentage=${PERCENTAGES[$i]}
    gpu=${GPU_DEVICES[$i]}
    static_sam_path=${GSAM_DIR}/${GSAM_FILES[$i]}
    log_path=${LOG_DIR}/gsamd-${percentage}pct.log

    if [[ ! -f "${static_sam_path}" ]]; then
        echo "Missing GSAM file: ${static_sam_path}" >&2
        exit 1
    fi

    echo "Starting ${percentage}% corpus experiment on GPU ${gpu}; log: ${log_path}"
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=${SPEC_BENCH_PATH} \
        python -m evaluation.inference_gsamd \
        --model-path "${VICUNA_PATH}" \
        --model-id "${MODEL_NAME}-gsamd-corpus-${percentage}pct" \
        --bench-name "${BENCH_NAME}" \
        --temperature "${TEMP}" \
        --dtype "${TORCH_DTYPE}" \
        --samd_n_predicts 40 \
        --samd_len_threshold 5 \
        --samd_len_bias 5 \
        --attn_implementation sdpa \
        --static_sam_path "${static_sam_path}" \
        --samd_map_type lazy_int32 \
        --samd_lazy_threshold 1 \
        >"${log_path}" 2>&1 &
    pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Completed ${PERCENTAGES[$i]}% corpus experiment."
    else
        echo "Failed ${PERCENTAGES[$i]}% corpus experiment; see ${LOG_DIR}/gsamd-${PERCENTAGES[$i]}pct.log" >&2
        status=1
    fi
done

exit "${status}"
