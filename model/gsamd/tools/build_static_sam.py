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
        "--sam_path", type=str, default="~/program/GSAM-Decoding/local_cache/"
    )
    parser.add_argument(
        "--use_gsam", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use_small_dict", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


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

    sam = build_sam(
        batch_tokens,
        tokenizer.eos_token_id,
        n_predicts=args.n_predicts,
        use_gsam=args.use_gsam,
        use_small_dict=args.use_small_dict,
    )
    sam_path = os.path.expanduser(args.sam_path)
    os.makedirs(os.path.dirname(sam_path), exist_ok=True)
    file_name = f"static_data_{'gsam' if args.use_gsam else 'sam'}_{'small' if args.use_small_dict else 'normal'}_dict.pb"
    sam_path = os.path.join(sam_path, file_name)
    dump_sam(sam_path, sam)


if __name__ == "__main__":
    main()
