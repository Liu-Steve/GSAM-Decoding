"""Generate answers with GSAMD."""

import argparse

import torch
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from evaluation.eval import reorg_answer_file, run_eval
from model.gsamd import DraftModel, SamdConfig, SamdGenerationConfig, SamdModel, load_sam


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str | None, device: torch.device) -> str:
    if dtype_name is not None:
        return dtype_name
    if device.type == "cuda":
        return "float16"
    return "float32"


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
        default=None,
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    requested_device = resolve_device(args.device)
    dtype_name = resolve_dtype(args.dtype, requested_device)
    dtype = str_to_torch_dtype(dtype_name)
    print(f"Loading model on {requested_device} with dtype {dtype_name}")

    model_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    if requested_device.type == "cuda" and args.num_gpus_total > 1:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            **model_kwargs,
        )
        model.to(requested_device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {"": str(requested_device)}

    device = next(model.lm_head.parameters()).device
    sam = load_sam(args.static_sam_path) if args.static_sam_path is not None else None
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
    )
    draft = DraftModel(
        samd_config,
        sam_static=sam,
        lm=model,
        dtype=dtype,
        device=device,
    )
    samd_model = SamdModel(
        samd_config,
        model,
        draft,
        tokenizer.eos_token_id,
        dtype,
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
        device=str(device),
    )

    reorg_answer_file(answer_file)
