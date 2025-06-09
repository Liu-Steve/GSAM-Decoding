#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=13
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16
datastore_PATH=./model/rest/datastore/datastore_chat_large.idx

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest \
    --model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-rest-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype \
    --datastore-path $datastore_PATH
    # --question-begin 0 \
    # --question-end 10
