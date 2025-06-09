#!/bin/bash

MODEL_PATH=/home/zhiyao/work/Spec-Bench-Models
MODEL_SIZE=7
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
bench_NAME=spec_bench
torch_dtype=float16

for PREFIX in 1 2 3 4 5
do
    # Array to store background process IDs
    pids=()
    
    for FOLLOWUP in 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=${FOLLOWUP} python -m evaluation.inference_shotgun \
            --model-path $Vicuna_PATH \
            --model-id ${MODEL_NAME}-shotgun-${torch_dtype}-p${PREFIX}-f${FOLLOWUP} \
            --bench-name $bench_NAME \
            --dtype $torch_dtype \
            --answer-file data/shotgun_128_answer/${MODEL_NAME}-shotgun-${torch_dtype}-p${PREFIX}-f${FOLLOWUP}.jsonl \
            --cache-config 1048576 128 ${PREFIX} ${FOLLOWUP} \
            --cache-config 1048576 128 ${PREFIX} ${FOLLOWUP} "openwebtext_sample100/openwebtext_prefix${PREFIX}_followup${FOLLOWUP}_lru_cache.pkl" true &
        
        # Store process ID
        pids+=($!)
    done
    
    # Wait for all background processes to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo "All processes for prefix ${PREFIX} have completed."
done
