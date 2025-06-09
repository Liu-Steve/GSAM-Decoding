#!/bin/bash

set -e

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3

echo "Start building prefix k-gram DBs for different prefix lengths"

for PREFIX_LEN in 1 2 3 4 5
do
    PREFIX_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}
    mkdir -p $PREFIX_DB_DIR
    
    echo "Building prefix k-gram DB with prefix length ${PREFIX_LEN}"
    
    python -m model.shotgun.cache_builder \
        --stage count-kgram \
        --model-path $Vicuna_PATH \
        --dataset alpaca \
        --db-dir $PREFIX_DB_DIR \
        --num-workers 40 \
        --prefix-len $PREFIX_LEN \
        --thread-batch 512 \
        --sample-rate 1
        
    echo "Finished building prefix k-gram DB with prefix length ${PREFIX_LEN}"
done

echo "Finished building all prefix k-gram DBs"

echo "Start merging prefix k-gram DBs for different prefix lengths"

for PREFIX_LEN in 1 2 3 4 5
do
    PREFIX_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}
    
    echo "Merging prefix k-gram DB with prefix length ${PREFIX_LEN}"
    
    python -m model.shotgun.cache_builder \
        --stage merge-kgram \
        --db-dir $PREFIX_DB_DIR \
        --dataset alpaca \
        --num-workers 20 \
        --num-prev-workers 40 \
        --prefix-len $PREFIX_LEN
        
    echo "Finished merging prefix k-gram DB with prefix length ${PREFIX_LEN}"
done

echo "Finished merging all prefix k-gram DBs"

for PREFIX_LEN in 1 2 3 4 5
do
    PREFIX_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}
    for FOLLOWUP_LEN in 1 2 3 4 5
    do
        FOLLOWUP_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}_followup${FOLLOWUP_LEN}
        mkdir -p $FOLLOWUP_DB_DIR
        cp $PREFIX_DB_DIR/alpaca_cnt_ngram_merged.sqlite $FOLLOWUP_DB_DIR
    done
done

echo "Start building followup DBs for different prefix and followup lengths"

for PREFIX_LEN in 1 2 3 4 5
do
    for FOLLOWUP_LEN in 1 2 3 4 5
    do
        PREFIX_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}
        FOLLOWUP_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}_followup${FOLLOWUP_LEN}
        
        echo "Building followup DB with prefix length ${PREFIX_LEN} and followup length ${FOLLOWUP_LEN}"
        
        python -m model.shotgun.cache_builder \
            --stage count-followup \
            --model-path $Vicuna_PATH \
            --dataset alpaca \
            --db-dir $FOLLOWUP_DB_DIR \
            --num-workers 24 \
            --prefix-len $PREFIX_LEN \
            --followup-len $FOLLOWUP_LEN \
            --thread-batch 2048 \
            --top-ngrams 10000000 \
            --sample-rate 1
            
        echo "Finished building followup DB with prefix length ${PREFIX_LEN} and followup length ${FOLLOWUP_LEN}"
    done
done

echo "Finished building all followup DBs"


echo "Start merging followup DBs for different prefix and followup lengths"

for PREFIX_LEN in 1 2 3 4 5
do
    for FOLLOWUP_LEN in 1 2 3 4 5
    do
        PREFIX_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}
        FOLLOWUP_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}_followup${FOLLOWUP_LEN}
        
        echo "Merging followup DB with prefix length ${PREFIX_LEN} and followup length ${FOLLOWUP_LEN}"
        
        python -m model.shotgun.cache_builder \
            --stage merge-followup \
            --dataset alpaca \
            --db-dir $FOLLOWUP_DB_DIR \
            --num-workers 12 \
            --num-prev-workers 24 \
            --prefix-len $PREFIX_LEN \
            --followup-len $FOLLOWUP_LEN
            
        echo "Finished merging followup DB with prefix length ${PREFIX_LEN} and followup length ${FOLLOWUP_LEN}"
    done
done

echo "Finished merging all followup DBs"


echo "Start building lru cache for different prefix and followup lengths"

for PREFIX_LEN in 1 2 3 4 5
do
    for FOLLOWUP_LEN in 1 2 3 4 5
    do
        PREFIX_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}
        FOLLOWUP_DB_DIR=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/prefix${PREFIX_LEN}_followup${FOLLOWUP_LEN}
        OUTPUT_PATH=/home/zhiyao/work/shotgun/Spec-Bench/alpaca/alpaca_prefix${PREFIX_LEN}_followup${FOLLOWUP_LEN}_lru_cache.pkl

        echo "Building lru cache with prefix length ${PREFIX_LEN} and followup length ${FOLLOWUP_LEN}"
        
        python -m model.shotgun.cache_builder \
            --stage build-lru-cache \
            --top-prefixes-n 10000000 \
            --top-followups-n 128 \
            --prefix-db-path $PREFIX_DB_DIR/alpaca_cnt_ngram_merged.sqlite \
            --followup-db-path $FOLLOWUP_DB_DIR/alpaca_cnt_followup_merged.sqlite \
            --prefix-len $PREFIX_LEN \
            --followup-len $FOLLOWUP_LEN \
            --output-path $OUTPUT_PATH

        echo "Finished building lru cache with prefix length ${PREFIX_LEN} and followup length ${FOLLOWUP_LEN}"
    done
done

echo "Finished building all lru caches"