#!/bin/bash
set -e

# -------------------------------------------------------------------------------
# build_frozen_table.sh
# -------------------------------------------------------------------------------
# Description:
#   This script builds a frozen table for Cacheback speculative decoding using a given
#   dataset. The frozen table stores common token sequences (leaders and followers)
#   that can be used to accelerate text generation through speculative decoding to
#   help reduce the cold start performance degradation.
#
# Usage:
#   Edit the configuration parameters in each section to match your environment
#   and run the script.
# -------------------------------------------------------------------------------


###########################################
### Path configurations                 ###
### EDIT THESE FOR YOUR OWN ENVIRONMENT ###
###########################################

# Spec-Bench paths
MODEL_PATH=/your_own_model_path/
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3

# A working directory for building the frozen table
WORKING_DIR=/your_own_working_dir/


###########################################
### Dataset configurations              ###
### EDIT THESE TO CHOOSE YOUR DATASET   ###
###########################################

# The dataset name in the Hugging Face Hub you want to use
FULL_DATASET_NAME=tatsu-lab/alpaca

# If the dataset name contains `/`, replace it with `_` to avoid invalid file names.
# No need to edit this.
LEGALIZED_DATASET_NAME=$(echo $FULL_DATASET_NAME | tr '/' '_')


############################################
### Table size configurations            ###
### EDIT THESE TO CHOOSE YOUR TABLE SIZE ###
############################################

# A frozen table will be generated for each combination of leader and follower lengths.
# The leader length will range from 1 to LEADER_LEN_MAX.
# The follower length will range from 1 to FOLLOWER_LEN_MAX.
LEADER_LEN_MAX=3
FOLLOWER_LEN_MAX=3

# The number of leaders to keep in the built frozen table.
LEADER_CAPACITY=10000000

# The number of followers to keep in the built frozen table for each leader.
FOLLOWER_CAPACITY=128


###################################################
### Parallelism configurations                  ###
### EDIT THESE TO SPEED UP THE BUILDING PROCESS ###
### AS LONG AS YOU HAVE ENOUGH MEMORY AND CPU   ###
###################################################

# Number of workers for counting leader k-grams.
NUM_LEADER_CNT_WORKERS=40

# Number of workers for merging leader k-grams.
NUM_LEADER_MERGE_WORKERS=20

# Number of workers for counting follower k-grams.
NUM_FOLLOWER_CNT_WORKERS=24

# Number of workers for merging follower k-grams.
NUM_FOLLOWER_MERGE_WORKERS=12

# The number of SQL queries to be executed in each thread in a batch
# when counting leader k-grams.
NUM_LEADER_KGRAM_CNT_THREAD_BATCH=512

# The number of SQL queries to be executed in each thread in a batch
# when counting follower k-grams.
NUM_FOLLOWER_CNT_THREAD_BATCH=2048


###################################################
### Sampling configurations                     ###
### EDIT THESE TO SPEED UP THE BUILDING PROCESS ###
###################################################

# The building process will sample the dataset with the rate
# 1/SAMPLE_RATE. Set a larger SAMPLE_RATE to speed up the building process.
SAMPLE_RATE=1


#####################################################################
### Code starts here                                              ###
### DO NOT EDIT THE CODE BELOW UNLESS YOU KNOW WHAT YOU ARE DOING ###
#####################################################################

# Step 1: Count the number of occurrences of leader k-grams in parallel.
#         Each worker will save the results in a SQLite database.
echo "Start building leader k-gram DBs for different leader lengths"

for LEADER_LEN in $(seq 1 $LEADER_LEN_MAX)
do
    LEADER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}
    mkdir -p $LEADER_DB_DIR
    
    echo "Building leader k-gram DB with leader length ${LEADER_LEN}"
    
    python -m model.cacheback.cache_builder \
        --stage count-kgram \
        --model-path $Vicuna_PATH \
        --dataset $FULL_DATASET_NAME \
        --db-dir $LEADER_DB_DIR \
        --num-workers $NUM_LEADER_CNT_WORKERS \
        --leader-len $LEADER_LEN \
        --thread-batch $NUM_LEADER_KGRAM_CNT_THREAD_BATCH \
        --sample-rate $SAMPLE_RATE
        
    echo "Finished building leader k-gram DB with leader length ${LEADER_LEN}"
done

echo "Finished building all leader k-gram DBs"

# Step 2: Merge the leader k-grams counts in parallel into a single database.
echo "Start merging leader k-gram DBs for different leader lengths"

for LEADER_LEN in $(seq 1 $LEADER_LEN_MAX)
do
    LEADER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}
    
    echo "Merging leader k-gram DB with leader length ${LEADER_LEN}"
    
    python -m model.cacheback.cache_builder \
        --stage merge-kgram \
        --db-dir $LEADER_DB_DIR \
        --dataset $FULL_DATASET_NAME \
        --num-workers $NUM_LEADER_MERGE_WORKERS \
        --num-prev-workers $NUM_LEADER_CNT_WORKERS \
        --leader-len $LEADER_LEN
        
    echo "Finished merging leader k-gram DB with leader length ${LEADER_LEN}"
done

echo "Finished merging all leader k-gram DBs"

# Step 3: Count the number of occurrences of follower k-grams in parallel.
#         Each worker will save the results in a SQLite database.
for LEADER_LEN in $(seq 1 $LEADER_LEN_MAX)
do
    LEADER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}
    for FOLLOWER_LEN in $(seq 1 $FOLLOWER_LEN_MAX)
    do
        FOLLOWER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}_follower${FOLLOWER_LEN}
        mkdir -p $FOLLOWER_DB_DIR
        cp $LEADER_DB_DIR/${LEGALIZED_DATASET_NAME}_cnt_ngram_merged.sqlite $FOLLOWER_DB_DIR
    done
done

echo "Start building follower DBs for different leader and follower lengths"

for LEADER_LEN in $(seq 1 $LEADER_LEN_MAX)
do
    for FOLLOWER_LEN in $(seq 1 $FOLLOWER_LEN_MAX)
    do
        FOLLOWER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}_follower${FOLLOWER_LEN}
        
        echo "Building follower DB with leader length ${LEADER_LEN} and follower length ${FOLLOWER_LEN}"
        
        python -m model.cacheback.cache_builder \
            --stage count-follower \
            --model-path $Vicuna_PATH \
            --dataset $FULL_DATASET_NAME \
            --db-dir $FOLLOWER_DB_DIR \
            --num-workers $NUM_FOLLOWER_CNT_WORKERS \
            --leader-len $LEADER_LEN \
            --follower-len $FOLLOWER_LEN \
            --thread-batch $NUM_FOLLOWER_CNT_THREAD_BATCH \
            --top-ngrams $LEADER_CAPACITY \
            --sample-rate $SAMPLE_RATE
            
        echo "Finished building follower DB with leader length ${LEADER_LEN} and follower length ${FOLLOWER_LEN}"
    done
done

echo "Finished building all follower DBs"

# Step 4: Merge the follower k-grams counts in parallel into a single database.
echo "Start merging follower DBs for different leader and follower lengths"

for LEADER_LEN in $(seq 1 $LEADER_LEN_MAX)
do
    for FOLLOWER_LEN in $(seq 1 $FOLLOWER_LEN_MAX)
    do
        FOLLOWER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}_follower${FOLLOWER_LEN}
        
        echo "Merging follower DB with leader length ${LEADER_LEN} and follower length ${FOLLOWER_LEN}"
        
        python -m model.cacheback.cache_builder \
            --stage merge-follower \
            --dataset $FULL_DATASET_NAME \
            --db-dir $FOLLOWER_DB_DIR \
            --num-workers $NUM_FOLLOWER_MERGE_WORKERS \
            --num-prev-workers $NUM_FOLLOWER_CNT_WORKERS \
            --leader-len $LEADER_LEN \
            --follower-len $FOLLOWER_LEN
            
        echo "Finished merging follower DB with leader length ${LEADER_LEN} and follower length ${FOLLOWER_LEN}"
    done
done

echo "Finished merging all follower DBs"

# Step 5: Build the frozen cache table and save it as a pickle file.
echo "Start building lru cache for different leader and follower lengths"

for LEADER_LEN in $(seq 1 $LEADER_LEN_MAX)
do
    for FOLLOWER_LEN in $(seq 1 $FOLLOWER_LEN_MAX)
    do
        LEADER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}
        FOLLOWER_DB_DIR=$WORKING_DIR/leader${LEADER_LEN}_follower${FOLLOWER_LEN}
        OUTPUT_PATH=$WORKING_DIR/${LEGALIZED_DATASET_NAME}_leader${LEADER_LEN}_follower${FOLLOWER_LEN}_lru_cache.pkl

        echo "Building lru cache with leader length ${LEADER_LEN} and follower length ${FOLLOWER_LEN}"
        
        python -m model.cacheback.cache_builder \
            --stage build-lru-cache \
            --top-leaders-n $LEADER_CAPACITY \
            --top-followers-n $FOLLOWER_CAPACITY \
            --leader-db-path $LEADER_DB_DIR/${LEGALIZED_DATASET_NAME}_cnt_ngram_merged.sqlite \
            --follower-db-path $FOLLOWER_DB_DIR/${LEGALIZED_DATASET_NAME}_cnt_follower_merged.sqlite \
            --leader-len $LEADER_LEN \
            --follower-len $FOLLOWER_LEN \
            --output-path $OUTPUT_PATH

        echo "Finished building lru cache with leader length ${LEADER_LEN} and follower length ${FOLLOWER_LEN}"
    done
done

echo "Finished building all lru caches"
