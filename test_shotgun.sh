#!/bin/bash

CUDA_VISIBLE_DEVICES=3

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m evaluation.inference_shotgun \
    --model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-shotgun-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype
