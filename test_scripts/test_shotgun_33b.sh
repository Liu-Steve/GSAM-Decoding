#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=33
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m evaluation.inference_shotgun \
    --model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-shotgun-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype \
    --max-query-len 96 \
    --chaining-reserve-len 16 \
    --cache-config 1048576 128 1 3 \
    --cache-config 1048576 128 1 3 "openwebtext_sample100/openwebtext_prefix1_followup3_lru_cache.pkl" true
    # --question-begin 0 \
    # --question-end 10
