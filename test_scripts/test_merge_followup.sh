#!/bin/bash

CUDA_VISIBLE_DEVICES=3

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16
DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/openwebtext_key2_val2

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m model.shotgun.cache_builder \
    --stage merge-followup \
    --model-path $Vicuna_PATH \
    --dataset openwebtext \
    --db-dir $DB_DIR \
    --num-workers 12 \
    --num-prev-workers 24 \
    --prefix-len 2 \
    --followup-len 2 \
    --thread-batch 512
