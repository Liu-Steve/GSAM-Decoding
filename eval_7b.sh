#Path and Parameters
MODEL_PATH=/data/llm/
SPEC_BENCH_PATH=/home/user/program/GSAM-Decoding/

cd $SPEC_BENCH_PATH

run_csam() {
    csam_name=$1
    csam_path=$2
    map_type=$3
    lazy_threshold=$4

    if [ "$map_type" = "lazy" ] || [ "$map_type" = "lazy_int32" ]; then
        map_suffix=${map_type}-t${lazy_threshold}
    else
        map_suffix=${map_type}
    fi

    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} PYTHONPATH=$SPEC_BENCH_PATH python -m evaluation.inference_gsamd \
        --model-path $Vicuna_PATH \
        --model-id ${MODEL_NAME}-csam-${csam_name}-${map_suffix} \
        --bench-name $bench_NAME \
        --temperature $TEMP \
        --dtype $torch_dtype \
        --samd_n_predicts 40 \
        --samd_len_threshold 5 \
        --samd_len_bias 5 \
        --attn_implementation sdpa \
        --static_sam_path $csam_path \
        --samd_map_type $map_type \
        --samd_lazy_threshold $lazy_threshold
}

run_csam_matrix() {
    csam_name=$1
    csam_path=$2

    run_csam $csam_name $csam_path unordered 1
    run_csam $csam_name $csam_path int32 1

    for threshold in 1 2 3 4 5; do
        run_csam $csam_name $csam_path lazy $threshold
    done

    for threshold in 1 2 3 4 5; do
        run_csam $csam_name $csam_path lazy_int32 $threshold
    done
}

# Run the evaluation
MODEL_SIZE=7
GPU_DEVICES=0
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
Eagle_PATH=$MODEL_PATH/EAGLE-Vicuna-${MODEL_SIZE}B-v1.3
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
SAM_PATH=$SPEC_BENCH_PATH/local_cache/static_sam_origin_dict.pkl
REST_PATH=$SPEC_BENCH_PATH/local_cache/datastore_chat_large.idx
CACHEBACK_PATH=$SPEC_BENCH_PATH/local_cache/openwebtext_leader1_follower3_lru_cache.pkl
CSAM_GSAM_PATH=$SPEC_BENCH_PATH/local_cache/gsamd/static_data_gsam.pb
CSAM_SAM_PATH=$SPEC_BENCH_PATH/local_cache/gsamd/static_data_sam.pb
TEMP=0.0

bench_NAME="spec_bench_7b"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --datastore-path $REST_PATH
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_cacheback --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-cacheback --bench-name $bench_NAME --dtype $torch_dtype --max-query-len 96 --chaining-reserve-len 16 --cache-config 1048576 128 1 3 $CACHEBACK_PATH False
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} PYTHONPATH=$SPEC_BENCH_PATH python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd-origin --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --static_sam_path $SAM_PATH
run_csam_matrix gsamd $CSAM_GSAM_PATH
run_csam_matrix samd $CSAM_SAM_PATH
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} PYTHONPATH=$SPEC_BENCH_PATH python -m evaluation.inference_gsamd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-gsamd-eagle2 --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --static_sam_path $CSAM_GSAM_PATH --tree_method eagle2 --tree_model_path $Eagle_PATH --samd_map_type lazy_int32 --samd_lazy_threshold 1
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} PYTHONPATH=$SPEC_BENCH_PATH python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd-eagle2 --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --attn_implementation sdpa --static_sam_path $SAM_PATH --tree_method eagle2 --tree_model_path $Eagle_PATH
