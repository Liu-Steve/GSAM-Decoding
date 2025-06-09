#!/bin/bash

set -e

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3

PREFIX_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/openwebtext_sample1000/prefix2
FOLLOWUP_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/openwebtext_sample1000/prefix2_followup2

mkdir -p $PREFIX_DB_DIR
mkdir -p $FOLLOWUP_DB_DIR

echo "Start building prefix k-gram DB"

python -m model.shotgun.cache_builder \
    --stage count-kgram \
    --model-path $Vicuna_PATH \
    --dataset openwebtext \
    --db-dir $PREFIX_DB_DIR \
    --num-workers 40 \
    --prefix-len 2 \
    --thread-batch 512 \
    --sample-rate 1000

echo "Finish building prefix k-gram DB"
echo "Start merging prefix k-gram DB"

python -m model.shotgun.cache_builder \
    --stage merge-kgram \
    --db-dir $PREFIX_DB_DIR \
    --dataset openwebtext \
    --num-workers 20 \
    --num-prev-workers 40 \
    --prefix-len 2

echo "Finish merging prefix k-gram DB"

cp $PREFIX_DB_DIR/openwebtext_cnt_ngram_merged.sqlite $FOLLOWUP_DB_DIR

echo "Start building followup DB"

python -m model.shotgun.cache_builder \
    --stage count-followup \
    --model-path $Vicuna_PATH \
    --dataset openwebtext \
    --db-dir $FOLLOWUP_DB_DIR \
    --num-workers 24 \
    --prefix-len 2 \
    --followup-len 2 \
    --thread-batch 2048 \
    --top-ngrams 10000000 \
    --sample-rate 1000

echo "Finish building followup DB"
echo "Start merging followup DB"

python -m model.shotgun.cache_builder \
    --stage merge-followup \
    --db-dir $FOLLOWUP_DB_DIR \
    --dataset openwebtext \
    --num-workers 12 \
    --num-prev-workers 24 \
    --prefix-len 2 \
    --followup-len 2

echo "Finish merging followup DB"
