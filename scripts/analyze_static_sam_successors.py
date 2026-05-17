#!/usr/bin/env python3
"""Analyze successor-count distribution in GSAMD static SAM protobuf files."""

from __future__ import annotations

import argparse
import math
import mmap
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT = Path("local_cache/static_data_sam_normal_dict.pb")
DEFAULT_OUTPUT_DIR = Path("assets")

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LENGTH_DELIMITED = 2
WIRE_FIXED32 = 5

FIELD_NAMES = {
    "edge": "all outgoing transitions",
    "topk_edge": "top-k outgoing transitions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the distribution of successor counts for states stored in "
            "local_cache/static_data_*.pb and plot a pie chart."
        )
    )
    parser.add_argument(
        "pb_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"protobuf file to analyze (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output chart path (default: assets/<input_stem>_successor_distribution.png)",
    )
    parser.add_argument(
        "--edge-field",
        choices=("edge", "topk_edge"),
        default="edge",
        help="which State repeated Edge field to count (default: edge)",
    )
    parser.add_argument(
        "--count-mode",
        choices=("transitions", "unique-states"),
        default="transitions",
        help=(
            "count outgoing transitions or unique destination state ids "
            "(default: transitions)"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="number of most frequent successor-count buckets shown separately",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for the generated chart",
    )
    return parser.parse_args()


def read_varint(data: mmap.mmap, pos: int, end: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while pos < end and shift <= 63:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
    raise ValueError("malformed protobuf varint")


def skip_field(data: mmap.mmap, pos: int, end: int, wire: int) -> int:
    if wire == WIRE_VARINT:
        _, pos = read_varint(data, pos, end)
        return pos
    if wire == WIRE_FIXED64:
        pos += 8
    elif wire == WIRE_LENGTH_DELIMITED:
        length, pos = read_varint(data, pos, end)
        pos += length
    elif wire == WIRE_FIXED32:
        pos += 4
    else:
        raise ValueError(f"unsupported protobuf wire type: {wire}")
    if pos > end:
        raise ValueError("malformed protobuf field length")
    return pos


def count_edge_dest_state(data: mmap.mmap, pos: int, end: int) -> int | None:
    while pos < end:
        tag, pos = read_varint(data, pos, end)
        field = tag >> 3
        wire = tag & 0x7
        if field == 2 and wire == WIRE_VARINT:
            value, _ = read_varint(data, pos, end)
            return value
        pos = skip_field(data, pos, end, wire)
    return None


def count_state_successors(
    data: mmap.mmap,
    pos: int,
    end: int,
    edge_field_number: int,
    count_mode: str,
) -> int:
    transition_count = 0
    destination_states: set[int] | None = set() if count_mode == "unique-states" else None

    while pos < end:
        tag, pos = read_varint(data, pos, end)
        field = tag >> 3
        wire = tag & 0x7
        if field == edge_field_number and wire == WIRE_LENGTH_DELIMITED:
            length, pos = read_varint(data, pos, end)
            edge_end = pos + length
            if edge_end > end:
                raise ValueError("malformed Edge message length")
            if destination_states is None:
                transition_count += 1
            else:
                dest = count_edge_dest_state(data, pos, edge_end)
                if dest is not None:
                    destination_states.add(dest)
            pos = edge_end
        else:
            pos = skip_field(data, pos, end, wire)

    return transition_count if destination_states is None else len(destination_states)


def iter_state_successor_counts(
    data: mmap.mmap,
    edge_field: str,
    count_mode: str,
) -> Iterable[int]:
    edge_field_number = 5 if edge_field == "edge" else 6
    pos = 0
    end = len(data)

    while pos < end:
        tag, pos = read_varint(data, pos, end)
        field = tag >> 3
        wire = tag & 0x7
        if field == 9 and wire == WIRE_LENGTH_DELIMITED:
            length, pos = read_varint(data, pos, end)
            state_end = pos + length
            if state_end > end:
                raise ValueError("malformed State message length")
            yield count_state_successors(
                data,
                pos,
                state_end,
                edge_field_number=edge_field_number,
                count_mode=count_mode,
            )
            pos = state_end
        else:
            pos = skip_field(data, pos, end, wire)


def analyze(path: Path, edge_field: str, count_mode: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    with path.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for successor_count in iter_state_successor_counts(data, edge_field, count_mode):
                counts[successor_count] += 1
    return counts


def format_bucket(successor_count: int) -> str:
    return f"{successor_count} successor" if successor_count == 1 else f"{successor_count} successors"


def top_buckets(counts: Counter[int], top_n: int) -> list[tuple[str, int]]:
    top = counts.most_common(max(top_n, 1))
    top_keys = {key for key, _ in top}
    buckets = [(format_bucket(key), value) for key, value in top]
    other = sum(value for key, value in counts.items() if key not in top_keys)
    if other:
        buckets.append(("Other", other))
    return buckets


def plot_pie(
    counts: Counter[int],
    output: Path,
    title: str,
    top_n: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = top_buckets(counts, top_n)
    labels = [label for label, _ in buckets]
    values = [value for _, value in buckets]

    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 24,
            "figure.titlesize": 26,
        }
    )

    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=dpi)
    colors = plt.get_cmap("tab20").colors[: len(values)]
    explode = [0.1] + [0] * top_n
    wedges, _ = ax.pie(
        values,
        labels=None,
        shadow=True,
        colors=colors,
        startangle=90,
        explode=explode,
        counterclock=False,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
    )

    total = sum(values)
    label_items = []
    for wedge, label, value in zip(wedges, labels, values, strict=True):
        angle = (wedge.theta1 + wedge.theta2) / 2.0
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))
        force_right = label in {"Other", "5 successors"}
        label_items.append(
            {
                "label": label,
                "side": 1 if force_right or x >= 0 else -1,
                "x": x,
                "y": y,
                "text": f"{label}\n{value / total:.1%} ({value:,})",
            }
        )

    preferred_label_y = {
        "Other": 1.02,
        "5 successors": 0.54,
        "4 successors": 1.02,
        "3 successors": 0.52,
        "2 successors": 0.02,
        "1 successor": -1.05,
    }
    for side in (-1, 1):
        side_items = sorted(
            [item for item in label_items if item["side"] == side],
            key=lambda item: item["y"],
        )
        if not side_items:
            continue
        min_gap = 0.42
        for item in side_items:
            item["label_y"] = preferred_label_y.get(
                item["label"],
                max(-1.05, min(1.05, item["y"] * 1.16)),
            )
        for index in range(1, len(side_items)):
            previous = side_items[index - 1]
            current = side_items[index]
            current["label_y"] = max(current["label_y"], previous["label_y"] + min_gap)
        overflow = side_items[-1]["label_y"] - 1.05
        if overflow > 0:
            for item in side_items:
                item["label_y"] -= overflow
        for index in range(len(side_items) - 2, -1, -1):
            following = side_items[index + 1]
            current = side_items[index]
            current["label_y"] = min(current["label_y"], following["label_y"] - min_gap)

    for item in label_items:
        side = item["side"]
        label_x = side * 1.72
        anchor_x = item["x"] * 0.92
        anchor_y = item["y"] * 0.92
        text_edge_x = label_x - side * 0.04
        diagonal_run = max(abs(item["label_y"] - anchor_y), 0.16)
        bend_x = anchor_x + side * diagonal_run
        ax.plot(
            [text_edge_x, bend_x, anchor_x],
            [item["label_y"], item["label_y"], anchor_y],
            color="#4B5563",
            linewidth=1.2,
            solid_capstyle="round",
            zorder=3,
            clip_on=False,
        )
        ax.text(
            label_x,
            item["label_y"],
            item["text"],
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=23,
            clip_on=False,
        )

    fig.suptitle(title, x=0.5, y=0.97, ha="center", fontsize=26)
    ax.axis("equal")
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-1.38, 1.38)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def print_summary(counts: Counter[int], path: Path, edge_field: str, count_mode: str) -> None:
    total_states = sum(counts.values())
    single_successor_states = counts.get(1, 0)
    single_ratio = single_successor_states / total_states if total_states else 0.0
    zero_successor_states = counts.get(0, 0)

    print(f"Input: {path}")
    print(f"Edge field: {edge_field} ({FIELD_NAMES[edge_field]})")
    print(f"Count mode: {count_mode}")
    print(f"Total states: {total_states:,}")
    print(
        "States with exactly 1 successor: "
        f"{single_successor_states:,} ({single_ratio:.2%})"
    )
    print(f"States with 0 successors: {zero_successor_states:,}")
    print("Top buckets:")
    for successor_count, state_count in counts.most_common(10):
        ratio = state_count / total_states if total_states else 0.0
        print(f"  {successor_count:>4} successors: {state_count:>12,} ({ratio:>7.2%})")


def main() -> None:
    args = parse_args()
    input_path = args.pb_path
    if not input_path.exists():
        raise FileNotFoundError(f"protobuf file not found: {input_path}")
    output_path = args.output or DEFAULT_OUTPUT_DIR / (
        f"{input_path.stem}_successor_distribution.pdf"
    )

    counts = analyze(input_path, args.edge_field, args.count_mode)
    if not counts:
        raise RuntimeError(f"no states found in {input_path}")

    title = "Frequency Distribution of Successor Counts\nin Suffix Automaton States"
    plot_pie(counts, output_path, title, args.top_n, args.dpi)
    print_summary(counts, input_path, args.edge_field, args.count_mode)
    print(f"Pie chart saved to: {output_path}")


if __name__ == "__main__":
    main()
