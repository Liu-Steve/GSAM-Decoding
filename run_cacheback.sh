#!/bin/bash

CUDA_VISIBLE_DEVICES=0

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16

LEADER_LEN=1
FOLLOWER_LEN=3
LEADER_CAPACITY=1048576
FOLLOWUP_CAPACITY=128
FROZEN_TABLE_PATH=alpaca/tatsu-lab_alpaca_leader1_follower3_lru_cache.pkl


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m evaluation.inference_cacheback \
    --model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-cacheback-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype \
    --max-query-len 96 \
    --chaining-reserve-len 16 \
    --cache-config $LEADER_CAPACITY $FOLLOWUP_CAPACITY $LEADER_LEN $FOLLOWER_LEN \
    --cache-config $LEADER_CAPACITY $FOLLOWUP_CAPACITY $LEADER_LEN $FOLLOWER_LEN $FROZEN_TABLE_PATH true
