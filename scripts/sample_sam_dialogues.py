#!/usr/bin/env python3
"""Create reproducible nested samples from the SAM dialogue corpus."""

import argparse
import math
import shutil
from pathlib import Path

from datasets import Dataset, load_from_disk


DEFAULT_PERCENTAGES = (1, 5, 10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample a Hugging Face Dataset saved on disk."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("~/program/GSAM-Decoding/data/sam_data/sam_dialogues"),
        help="source Dataset directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/program/GSAM-Decoding/data/sam_data"),
        help="parent directory for sam_dialogue_<percentage> outputs",
    )
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=DEFAULT_PERCENTAGES,
        help="integer percentages to sample (default: 1 5 10 50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output directories if they already exist",
    )
    return parser.parse_args()


def validate_percentages(percentages: list[int]) -> list[int]:
    unique = sorted(set(percentages))
    if not unique or any(value <= 0 or value > 100 for value in unique):
        raise ValueError("percentages must be integers in the range [1, 100]")
    return unique


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    percentages = validate_percentages(args.percentages)

    dataset = load_from_disk(str(input_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"expected a Dataset at {input_path}, got {type(dataset).__name__}"
        )
    if len(dataset) == 0:
        raise ValueError(f"source Dataset is empty: {input_path}")

    output_paths = {
        percentage: output_root / f"sam_dialogues_{percentage}"
        for percentage in percentages
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(map(str, existing))
        raise FileExistsError(f"output already exists: {joined}; use --overwrite")

    # A single permutation makes the samples nested and directly comparable.
    shuffled = dataset.shuffle(seed=args.seed, keep_in_memory=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"source={input_path} rows={len(dataset)} "
        f"seed={args.seed} columns={dataset.column_names}"
    )
    for percentage in percentages:
        output_path = output_paths[percentage]
        if output_path.exists():
            shutil.rmtree(output_path)

        sample_size = max(1, math.floor(len(dataset) * percentage / 100))
        sample = shuffled.select(range(sample_size))
        sample.save_to_disk(str(output_path))
        print(
            f"saved {percentage}%: rows={sample_size} path={output_path}"
        )


if __name__ == "__main__":
    main()
