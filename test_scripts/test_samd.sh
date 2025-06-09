#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=13
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
Eagle_PATH=$MODEL_PATH/EAGLE-Vicuna-7B-v1.3
bench_NAME=spec_bench
torch_dtype=float16

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m evaluation.inference_samd \
    --model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-samd-${torch_dtype} \
    --bench-name $bench_NAME \
    --dtype $torch_dtype \
    --samd_n_predicts 40 \
    --samd_len_threshold 5 \
    --samd_len_bias 5 \
    --attn_implementation sdpa \
    --static_sam_path static_sam.pkl
    # --tree_model_path $Eagle_PATH
    # --tree_method eagle2 \
    # --question-begin 0 \
    # --question-end 10
