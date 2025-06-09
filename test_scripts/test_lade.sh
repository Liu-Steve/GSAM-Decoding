#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=33
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead \
    --model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} \
    --level 5 \
    --window 7 \
    --guess 7 \
    --bench-name $bench_NAME \
    --dtype $torch_dtype \
    --num-gpus-per-model 4 \
    --num-gpus-total 4
    # --question-begin 0 \
    # --question-end 10
