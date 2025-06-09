#!/bin/bash

CUDA_VISIBLE_DEVICES=2

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
Eagle_PATH=$MODEL_PATH/EAGLE-Vicuna-7B-v1.3
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m evaluation.inference_eagle \
    --ea-model-path $Eagle_PATH \
    --base-model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-eagle-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype
    # --question-begin 0 \
    # --question-end 10
