"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

import torch
from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.eval import reorg_answer_file, run_eval


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


def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False, use_cache=True):
    input_ids = inputs.input_ids
    output_ids = model.generate(
        input_ids,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
    )
    new_token = len(output_ids[0][len(input_ids[0]):])
    step = new_token
    accept_length_list = [1] * new_token
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the dtype. Defaults to float16 on CUDA and float32 elsewhere.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    device = resolve_device(args.device)
    dtype_name = resolve_dtype(args.dtype, device)
    print(f"Loading model on {device} with dtype {dtype_name}")

    model_kwargs = dict(
        torch_dtype=str_to_torch_dtype(dtype_name),
        low_cpu_mem_usage=True,
    )
    if device.type == "cuda" and args.num_gpus_total > 1:
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
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    runtime_device = next(model.parameters()).device
    do_sample = args.temperature > 0
    use_cache = runtime_device.type != "mps"

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_forward,
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
        use_cache=use_cache,
        clear_cache_after_question=runtime_device.type == "mps",
        device=str(runtime_device),
    )

    reorg_answer_file(answer_file)
