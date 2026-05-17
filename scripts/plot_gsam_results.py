#!/usr/bin/env python3
"""Plot GSAM Decoding experiment figures from GSAM_RESULT.md."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_RESULT = Path("GSAM_RESULT.md")
DEFAULT_CSV = Path("GSAM_RESULT.csv")
DEFAULT_OUTPUT_DIR = Path("assets")

METHOD_LABELS = {
    "lade": "LOOKAHEAD",
    "rest": "REST",
    "recycling": "Token Recycling",
    "pld": "PLD",
    "cacheback": "CacheBack",
    "samd-small": "SAM Decoding\nSmall Dict",
    "gsamd-small": "Compact SAM Decoding",
    "samd-origin": "SAM Decoding",
    "gsamd-normal": "GSAM On\nSmall Dict Off",
    "samd-normal": "GSAM Off\nSmall Dict Off",
}

MEMORY_SPEED_METHODS = [
    "lade",
    "rest",
    "recycling",
    "pld",
    "cacheback",
    "samd-origin",
    "gsamd-small",
]

ABLATED_METHODS = [
    ("samd-origin", "Original\nSAM Decoding"),
    ("gsamd-small", "GSAM On\nSmall Dict On"),
    ("samd-small", "GSAM Off\nSmall Dict On"),
    ("gsamd-normal", "GSAM On\nSmall Dict Off"),
    ("samd-normal", "GSAM Off\nSmall Dict Off"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GSAM results to CSV and plot experiment figures."
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RESULT,
        help=f"result markdown file (default: {DEFAULT_RESULT})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV extracted from the markdown Overall table (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for generated figures (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for generated figures",
    )
    return parser.parse_args()


def parse_overall_records(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_overall = False
    headers: list[str] | None = None
    records: list[dict[str, str]] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "## Overall":
            in_overall = True
            continue
        if in_overall and stripped.startswith("## "):
            break
        if not in_overall or not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if set(cells[0]) <= {"-"}:
            continue
        if headers is None:
            headers = cells
            continue

        if len(cells) != len(headers):
            raise ValueError(f"malformed Overall table row: {line}")
        records.append(dict(zip(headers, cells, strict=True)))

    if not records:
        raise ValueError(f"Overall table not found in {path}")
    return records


def extract_overall_csv(result_path: Path, csv_path: Path) -> None:
    records = parse_overall_records(result_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def read_overall_csv(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for record in reader:
            model = record.pop("model")
            rows[model] = {key: float(value) for key, value in record.items()}
    if not rows:
        raise ValueError(f"Overall CSV has no rows: {path}")
    return rows


def bytes_to_gb(value: float) -> float:
    return value / 1_000_000_000


def require_methods(rows: dict[str, dict[str, float]], methods: list[str]) -> None:
    missing = [method for method in methods if method not in rows]
    if missing:
        raise KeyError(f"missing Overall rows: {', '.join(missing)}")


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 25,
            "axes.labelsize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "figure.titlesize": 27,
            "savefig.bbox": "tight",
        }
    )


def plot_memory_speed(
    rows: dict[str, dict[str, float]],
    output: Path,
    dpi: int,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import numpy as np
    from PIL import Image

    methods = MEMORY_SPEED_METHODS
    require_methods(rows, methods)

    fig, ax = plt.subplots(figsize=(12.4, 7.6), dpi=dpi)
    colors = {
        "lade": "#4C78A8",
        "rest": "#9CA3AF",
        "recycling": "#59A14F",
        "pld": "#F58518",
        "cacheback": "#B279A2",
        "samd-origin": "#6B7280",
        "gsamd-small": "#D62728",
    }
    markers = {
        "gsamd-small": "*",
        "samd-origin": "D",
    }
    sizes = {
        "gsamd-small": 520,
        "samd-origin": 120,
    }

    label_offsets = {
        "lade": (10, 0),
        "rest": (-42, 14),
        "recycling": (12, -20),
        "pld": (12, 6),
        "cacheback": (12, 16),
        "samd-origin": (-86, 12),
        "gsamd-small": (12, -42),
    }

    for name in methods:
        memory = bytes_to_gb(rows[name]["memory"])
        speed = rows[name]["overall"]
        is_gsam = name == "gsamd-small"
        ax.scatter(
            [memory],
            [speed],
            marker=markers.get(name, "o"),
            s=sizes.get(name, 105),
            color=colors[name],
            edgecolor="black" if is_gsam else "white",
            linewidth=0.9 if is_gsam else 1.0,
            zorder=5 if is_gsam else 3,
        )
        ax.annotate(
            METHOD_LABELS.get(name, name),
            (memory, speed),
            xytext=label_offsets.get(name, (8, 8)),
            textcoords="offset points",
            fontsize=21 if is_gsam else 18,
            fontweight="bold" if is_gsam else "normal",
            color="#B22222" if is_gsam else "black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 1.5,
            }
            if is_gsam
            else None,
        )

    max_memory = max(bytes_to_gb(rows[name]["memory"]) for name in methods)
    min_memory = min(bytes_to_gb(rows[name]["memory"]) for name in methods)
    ax.set_ylabel("Overall Speedup")
    ax.set_xlabel("Memory Usage (GB)")
    ax.set_title("Memory Efficiency vs. Overall Speedup", pad=14)
    ax.set_xscale("log")
    ax.set_xlim(min_memory * 0.82, max_memory * 1.12)
    ax.set_ylim(1.18, 1.84)
    x_ticks = [1, 2, 5, 10, 20, 50]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(tick) for tick in x_ticks])
    arrow = Image.open(DEFAULT_OUTPUT_DIR / "arrow.png").convert("RGBA")
    arrow_data = np.array(arrow)
    arrow_data[np.all(arrow_data[:, :, :3] > 245, axis=-1), 3] = 0
    resample = getattr(Image, "Resampling", Image).BICUBIC
    arrow = Image.fromarray(arrow_data).rotate(45, resample=resample, expand=True)
    arrow_box = AnnotationBbox(
        OffsetImage(np.asarray(arrow) / 255.0, zoom=0.55),
        (0.075, 0.90),
        xycoords=ax.transAxes,
        frameon=False,
        pad=0,
        box_alignment=(0.5, 0.5),
    )
    arrow_box.set_zorder(4)
    ax.add_artist(arrow_box)
    ax.text(
        0.1,
        0.82,
        "Better:\nlower memory, \nhigher speedup",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#B22222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_memory_ablation(
    rows: dict[str, dict[str, float]],
    output: Path,
    dpi: int,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [method for method, _ in ABLATED_METHODS]
    require_methods(rows, methods)

    labels = [label for _, label in ABLATED_METHODS]
    values = [bytes_to_gb(rows[method]["memory"]) for method in methods]
    baseline = values[0]
    reductions = [(baseline - value) / baseline * 100.0 for value in values]

    colors = ["#6B7280", "#D62728", "#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=dpi)
    bars = ax.bar(x, values, color=colors, width=0.68, edgecolor="#222222", linewidth=0.7)

    for index, (bar, value, reduction) in enumerate(zip(bars, values, reductions, strict=True)):
        if index == 0:
            text = "Baseline"
        else:
            text = f"{reduction:.1f}% lower"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + baseline * 0.025,
            text,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 0.5,
            f"{value:.2f} GB",
            ha="center",
            va="center",
            rotation=0,
            fontsize=13,
            color="white",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Memory Usage (GB)")
    ax.set_title("Memory Usage Ablation")
    ax.set_ylim(0, baseline * 1.18)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    extract_overall_csv(args.result, args.csv)
    rows = read_overall_csv(args.csv)
    memory_speed_output = args.output_dir / "memory_speed.pdf"
    ablation_output = args.output_dir / "memory_ablation.pdf"

    plot_memory_speed(rows, memory_speed_output, args.dpi)
    plot_memory_ablation(rows, ablation_output, args.dpi)

    print(f"Overall CSV saved to: {args.csv}")
    print(f"Memory-speed figure saved to: {memory_speed_output}")
    print(f"Ablation figure saved to: {ablation_output}")


if __name__ == "__main__":
    main()
