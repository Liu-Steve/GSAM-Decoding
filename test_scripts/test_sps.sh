#!/bin/bash

CUDA_VISIBLE_DEVICES=2

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
Drafter_PATH=$MODEL_PATH/vicuna-68m
bench_NAME=spec_bench
torch_dtype=float16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m evaluation.inference_sps \
    --model-path $Vicuna_PATH \
    --drafter-path $Drafter_PATH \
    --model-id ${MODEL_NAME}-sps-68m-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype
    # --question-begin 0 \
    # --question-end 10
