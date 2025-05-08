import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.shotgun.shotgun import shotgun
from model.shotgun.lru_cache import ShotgunCache, ShotgunCacheConfig

class ParseCacheConfigAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        
        def parse_int(value, param_name):
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"{param_name} must be an integer, got '{value}'")
        
        def parse_bool(value, param_name):
            value_lower = value.lower()
            if value_lower not in ('true', 'false'):
                raise ValueError(f"{param_name} must be either 'true' or 'false', got '{value}'")
            return value_lower == 'true'

        # Initialize cache configs if not already present
        if not hasattr(namespace, 'cache_configs'):
            namespace.cache_configs = []
        
        # Check if we have at least the required arguments
        if len(values) < 4:
            raise ValueError("Cache config requires at least 4 arguments: prefix_capacity, followup_capacity, prefix_len, followup_len")
        
        # Parse the required arguments
        prefix_capacity = parse_int(values[0], "prefix_capacity")
        followup_capacity = parse_int(values[1], "followup_capacity")
        prefix_len = parse_int(values[2], "prefix_len")
        followup_len = parse_int(values[3], "followup_len")
        
        # Parse optional arguments
        file_path = None
        frozen = False
        
        if len(values) >= 5:
            file_path = values[4]
        
        if len(values) >= 6:
            frozen = parse_bool(values[5], "frozen")
        
        config = ShotgunCacheConfig(
            prefix_capacity=prefix_capacity,
            followup_capacity=followup_capacity,
            prefix_len=prefix_len,
            followup_len=followup_len,
            file_path=file_path,
            frozen=frozen
        )
        
        namespace.cache_configs.append(config)


def shotgun_forward(inputs, model, tokenizer, max_new_tokens, shotgun_cache):
    input_ids = inputs.input_ids

    output_ids, step, accept_length_list = shotgun(
        model,
        input_ids, 
        max_length=len(input_ids[0])+max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        shotgun_cache=shotgun_cache,
    )

    input_len = len(input_ids[0])
    new_tokens = output_ids[0, input_len:].tolist()
    num_new_tokens = len(new_tokens)

    if tokenizer.eos_token_id in new_tokens:
        for i, id in enumerate(new_tokens):
            if id == tokenizer.eos_token_id:
                eos_token_ids_index = i
        invalid_len = num_new_tokens - eos_token_ids_index - 1
        if invalid_len > 0:
            accept_length_list[-1] -= invalid_len
            num_new_tokens -= invalid_len

    return output_ids, num_new_tokens, step, accept_length_list


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
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    
    # Add shotgun cache configuration arguments
    parser.add_argument(
        "--cache-config",
        action=ParseCacheConfigAction,
        nargs='+',
        required=True,
        metavar="CONFIG_VALUE",
        help="Specify cache configuration(s). Format: "
             "prefix_capacity followup_capacity prefix_len followup_len [file_path] [frozen]. "
             "Optional values: file_path(default=None) frozen(default=false).",
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Initialize shotgun cache with user provided configurations
    shotgun_cache = ShotgunCache(args.cache_configs)

    forward_func = lambda inputs, model, tokenizer, max_new_tokens: \
        shotgun_forward(
            inputs, 
            model, 
            tokenizer, 
            max_new_tokens, 
            shotgun_cache
        )

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=forward_func,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
    )

    reorg_answer_file(answer_file)
