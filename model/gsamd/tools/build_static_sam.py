import argparse
import os

from datasets import load_from_disk
from transformers import AutoTokenizer

from model.gsamd import build_sam, dump_sam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path", type=str, default="/data/llm/vicuna-7b-v1.3"
    )
    parser.add_argument(
        "--sam_data_path",
        type=str,
        default="~/program/GSAM-Decoding/data/sam_data/sam_dialogues",
    )
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--n_predicts", type=int, default=40)
    parser.add_argument(
        "--sam_path", type=str, default="~/program/GSAM-Decoding/local_cache/gsamd_test"
    )
    parser.add_argument(
        "--use_gsam", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use_small_dict", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--map_type",
        choices=("lazy", "unordered", "int32", "lazy_int32"),
        default=None,
        help=(
            "transition map backend. Defaults to lazy when --use_small_dict is set "
            "and unordered when --no-use_small_dict is set."
        ),
    )
    parser.add_argument(
        "--lazy_threshold",
        type=int,
        default=1,
        help="number of inline transitions before the lazy map builds a hash table",
    )
    parser.add_argument(
        "--build_map_ablation",
        action="store_true",
        help="build unordered, lazy thresholds 1..5, and int32 suffix-map variants",
    )
    return parser.parse_args()


def resolve_map_type(args):
    return args.map_type or ("lazy" if args.use_small_dict else "unordered")


def map_label(map_type, lazy_threshold):
    if map_type == "lazy":
        return f"lazy_t{lazy_threshold}"
    if map_type == "int32":
        return "suffix_int32"
    if map_type == "lazy_int32":
        return f"lazy_suffix_int32_t{lazy_threshold}"
    return "unordered"


def main():
    args = parse_args()
    sam_data = load_from_disk(os.path.expanduser(args.sam_data_path))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def tokenize_fn(data_point, add_eos_token=False):
        text = data_point["prompt"] + data_point["response"]
        result = tokenizer(
            text,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    batch_tokens = sam_data.map(
        tokenize_fn,
        desc="Processing sam dialogue datasets",
    )["input_ids"]
    for i in range(len(tokenizer)):
        batch_tokens.append([i])

    sam_path = os.path.expanduser(args.sam_path)
    os.makedirs(os.path.dirname(sam_path), exist_ok=True)
    configs = (
        [("unordered", 1)]
        + [("lazy", threshold) for threshold in range(1, 6)]
        + [("int32", 1)]
        + [("lazy_int32", threshold) for threshold in range(1, 6)]
        if args.build_map_ablation
        else [(resolve_map_type(args), args.lazy_threshold)]
    )
    for map_type, lazy_threshold in configs:
        sam = build_sam(
            batch_tokens,
            tokenizer.eos_token_id,
            n_predicts=args.n_predicts,
            use_gsam=args.use_gsam,
            use_small_dict=(map_type == "lazy"),
            map_type=map_type,
            lazy_threshold=lazy_threshold,
        )
        if not args.build_map_ablation and args.map_type is None and lazy_threshold == 1:
            file_name = (
                f"static_data_{'gsam' if args.use_gsam else 'sam'}_"
                f"{'small' if args.use_small_dict else 'normal'}_dict.pb"
            )
        else:
            file_name = (
                f"static_data_{'gsam' if args.use_gsam else 'sam'}_"
                f"{map_label(map_type, lazy_threshold)}.pb"
            )
        output_path = os.path.join(sam_path, file_name)
        dump_sam(output_path, sam)
        print(
            "saved {} (map_type={}, lazy_threshold={}, transition bytes={})".format(
                output_path,
                sam.map_type,
                sam.lazy_threshold,
                sam.transition_memory_usage(),
            )
        )


if __name__ == "__main__":
    main()
