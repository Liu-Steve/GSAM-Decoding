"""Generate answers with GSAMD."""

import argparse

import torch
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from evaluation.eval import reorg_answer_file, run_eval
from model.gsamd import DraftModel, SamdConfig, SamdGenerationConfig, SamdModel, load_sam


def gsamd_forward(
    inputs,
    model: SamdModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
    temperature: float = 0.0,
    do_sample: bool = False,
):
    max_cache_len = model.lm.config.max_position_embeddings
    input_ids = inputs.input_ids
    outputs = model.generate(
        input_ids,
        generation_config=SamdGenerationConfig(
            max_new_tokens=max_new_tokens,
            max_cache_len=max_cache_len,
            greedy=not do_sample,
            temperature=temperature,
        ),
    )
    output_ids = outputs.output_ids
    new_token = outputs.decode_tokens
    step = outputs.decode_steps
    accept_length_list = outputs.accepet_length_per_step
    return output_ids, new_token, step, accept_length_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument("--samd_n_predicts", type=int, default=40)
    parser.add_argument("--static_sam_path", type=str, default=None)
    parser.add_argument("--samd_len_threshold", type=int, default=5)
    parser.add_argument("--samd_len_bias", type=int, default=5)
    parser.add_argument("--samd_tree_path", type=str, default=None)
    parser.add_argument("--tree_method", type=str, default=None, choices=["token_recycle", "eagle2"])
    parser.add_argument("--tree_model_path", type=str, default="path/to/EAGLE-Vicuna-7B-v1.3")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--use_gsam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_small_dict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--samd_map_type",
        type=str,
        default="",
        choices=["", "lazy", "unordered", "int32", "lazy_int32"],
        help=(
            "runtime transition map backend for static/dynamic SAM. Empty keeps "
            "the map metadata stored in the static SAM file."
        ),
    )
    parser.add_argument(
        "--samd_lazy_threshold",
        type=int,
        default=1,
        help="runtime inline transition threshold for lazy map backends",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    device_map = "cuda" if args.num_gpus_total == 1 else "auto"
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    device = next(model.lm_head.parameters()).device
    sam = (
        load_sam(
            args.static_sam_path,
            map_type=args.samd_map_type,
            lazy_threshold=args.samd_lazy_threshold,
        )
        if args.static_sam_path is not None
        else None
    )
    if sam is not None:
        sam.device = device
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
        len_threshold=args.samd_len_threshold,
        len_bias=args.samd_len_bias,
        tree_path=args.samd_tree_path,
        use_gsam=args.use_gsam,
        use_small_dict=args.use_small_dict,
        map_type=args.samd_map_type,
        lazy_threshold=args.samd_lazy_threshold,
    )
    draft = DraftModel(
        samd_config,
        sam_static=sam,
        lm=model,
        dtype=str_to_torch_dtype(args.dtype),
        device=device,
    )
    samd_model = SamdModel(
        samd_config,
        model,
        draft,
        tokenizer.eos_token_id,
        str_to_torch_dtype(args.dtype),
        device,
    )
    do_sample = args.temperature > 0

    run_eval(
        model=samd_model,
        tokenizer=tokenizer,
        forward_func=gsamd_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)
