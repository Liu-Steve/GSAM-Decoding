#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=13
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m evaluation.inference_baseline \
    --model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-baseline-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype
    # --question-begin 0 \
    # --question-end 10
