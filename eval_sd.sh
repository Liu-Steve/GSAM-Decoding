#Path and Parameters
MODEL_PATH=/data/llm/
SPEC_BENCH_PATH=/home/user/program/GSAM-Decoding/
GPU_DEVICES=3

cd $SPEC_BENCH_PATH

# Run the evaluation
MODEL_SIZE=7
Vicuna_PATH=$MODEL_PATH/vicuna-${MODEL_SIZE}b-v1.3
MODEL_NAME=vicuna-${MODEL_SIZE}b-v1.3
SAM_PATH=$SPEC_BENCH_PATH/local_cache/static_sam_origin_dict.pkl
GSAM_PATH=$SPEC_BENCH_PATH/local_cache/static_data_gsam_small_dict.pb
REST_PATH=$SPEC_BENCH_PATH/local_cache/datastore_chat_large.idx
CACHEBACK_PATH=$SPEC_BENCH_PATH/local_cache/openwebtext_leader1_follower3_lru_cache.pkl
TEMP=0.0
GPU_DEVICES=${GPU_DEVICES}

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --datastore-path $REST_PATH
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_cacheback --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-cacheback --bench-name $bench_NAME --dtype $torch_dtype --max-query-len 96 --chaining-reserve-len 16 --cache-config 1048576 128 1 3 $CACHEBACK_PATH False
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} PYTHONPATH=$SPEC_BENCH_PATH python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd-origin --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 20 --attn_implementation sdpa --static_sam_path $SAM_PATH
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} PYTHONPATH=$SPEC_BENCH_PATH python -m evaluation.inference_gsamd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-gsamd-small --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 20 --attn_implementation sdpa --static_sam_path $GSAM_PATH
