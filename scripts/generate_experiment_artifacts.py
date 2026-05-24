from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIC_DIR = ROOT / "assets"
GENERATED_DIR = ROOT / "generated"

DEFAULT_7B_RESULTS = [
    ROOT / "data" / "spec_bench_7b_1" / "result.csv",
    ROOT / "data" / "spec_bench_7b_2" / "result.csv",
    ROOT / "data" / "spec_bench_7b_3" / "result.csv",
]
DEFAULT_13B_RESULT = ROOT / "data" / "spec_bench_13b" / "result.csv"

TASKS = [
    ("mt_bench", "MT"),
    ("translation", "Trans."),
    ("summarization", "Sum."),
    ("qa", "QA"),
    ("math_reasoning", "Math"),
    ("rag", "RAG"),
]

METHOD_LABELS = {
    "lade-level-5-win-7-guess-7-float16": "LOOKAHEAD",
    "rest-temperature-0.0-top_p-0": "REST",
    "recycling": "Recycling",
    "pld-float16": "PLD",
    "cacheback": "CacheBack",
    "samd-origin": "SAMD",
    "csam-gsamd-lazy_int32-t1": "C-SAMD",
    "samd-eagle2": "SAMD + EAGLE-2",
    "gsamd-eagle2": "C-SAMD + EAGLE-2",
}

ABLATION_LABELS = {
    "samd-origin": "SAMD",
    "csam-samd-unordered": "SAMD + STL",
    "csam-gsamd-unordered": "GSAMD + STL",
    "csam-samd-int32": "SAMD + OA",
    "csam-gsamd-int32": "GSAMD + OA",
    "csam-samd-lazy-t1": "SAMD lazy + STL",
    "csam-gsamd-lazy-t1": "GSAMD lazy + STL",
    "csam-samd-lazy_int32-t1": "SAMD lazy + OA",
    "csam-gsamd-lazy_int32-t1": "C-SAMD",
}

MAIN_METHODS = [
    "lade-level-5-win-7-guess-7-float16",
    "rest-temperature-0.0-top_p-0",
    "recycling",
    "pld-float16",
    "cacheback",
    "samd-origin",
    "csam-gsamd-lazy_int32-t1",
]

ABLATION_METHODS = [
    "samd-origin",
    "csam-samd-unordered",
    "csam-gsamd-unordered",
    "csam-samd-int32",
    "csam-gsamd-int32",
    "csam-samd-lazy-t1",
    "csam-gsamd-lazy-t1",
    "csam-samd-lazy_int32-t1",
    "csam-gsamd-lazy_int32-t1",
]


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def bytes_to_gb(value: float) -> float:
    return value / 1_000_000_000


def row_for(summary: pd.DataFrame, name: str) -> pd.Series:
    rows = summary.loc[summary["model"] == name]
    if rows.empty:
        raise KeyError(f"missing model in summary: {name}")
    return rows.iloc[0]


def speed_cell(row: pd.Series, key: str = "overall") -> str:
    return f"${row[f'{key}_mean']:.3f} \\pm {row[f'{key}_std']:.3f}$"


def plain_speed(row: pd.Series, key: str = "overall") -> str:
    return f"{row[f'{key}_mean']:.3f}"


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def bold_cell(value: str) -> str:
    return f"\\textbf{{{value}}}"


def bold_math_cell(value: str) -> str:
    return f"\\textbf{{{value}}}"


def maybe_bold(value: str, bold: bool) -> str:
    return bold_cell(value) if bold else value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_runs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one result CSV is required")
    frames = []
    for run_id, path in enumerate(paths, start=1):
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "model" not in frame.columns:
            raise ValueError(f"missing model column in {path}")
        frame = frame.copy()
        frame["run_id"] = run_id
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    numeric_cols = [col for col in combined.columns if col not in {"model", "run_id"}]
    grouped = combined.groupby("model", sort=False)[numeric_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def make_table(tabular: str, caption: str, label: str, starred: bool = False) -> str:
    env = "table*" if starred else "table"
    return (
        f"\\begin{{{env}}}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\end{{{env}}}\n"
    )


def generate_main_table(summary: pd.DataFrame) -> None:
    rows = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Memory (GB) & Overall Speedup & Accept Length \\\\",
        "\\midrule",
    ]
    for name in MAIN_METHODS:
        row = row_for(summary, name)
        label = METHOD_LABELS[name]
        if name == "csam-gsamd-lazy_int32-t1":
            label = f"\\textbf{{{label}}}"
            memory = f"\\textbf{{{bytes_to_gb(row['memory_mean']):.2f}}}"
            speed = f"\\textbf{{{speed_cell(row)}}}"
            accept = f"\\textbf{{{row['accept_overall_mean']:.3f}}}"
        else:
            memory = f"{bytes_to_gb(row['memory_mean']):.2f}"
            speed = speed_cell(row)
            accept = f"{row['accept_overall_mean']:.3f}"
        rows.append(f"{label} & {memory} & {speed} & {accept} \\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}", "}"])
    table = make_table(
        "\n".join(rows),
        "Memory, overall speedup, and accepted draft length of retrieval-based speculative decoding methods. Speedup is reported as the 3-run mean $\\pm$ standard deviation; accepted length is the mean number of accepted tokens per verification step.",
        "tab:main-results",
    )
    write_text(GENERATED_DIR / "main_results.tex", table)


def generate_task_table(summary: pd.DataFrame) -> None:
    rows = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Memory (GB) & Overall Speedup & Accept Length \\\\",
        "\\midrule",
    ]
    for name in MAIN_METHODS:
        row = row_for(summary, name)
        is_best = name == "csam-gsamd-lazy_int32-t1"
        label = maybe_bold(METHOD_LABELS[name], is_best)
        memory = maybe_bold(f"{bytes_to_gb(row['memory_mean']):.2f}", is_best)
        speed = bold_math_cell(speed_cell(row, "overall")) if is_best else speed_cell(row, "overall")
        accept = maybe_bold(f"{row['accept_overall_mean']:.3f}", is_best)
        rows.append(f"{label} & {memory} & {speed} & {accept} \\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}", "}"])
    table = make_table(
        "\n".join(rows),
        "Spec-Bench overall results for retrieval-based speculative decoding methods. Speedup is reported as the 3-run mean $\\pm$ standard deviation; accepted length is the overall mean number of accepted tokens per verification step.",
        "tab:task-results",
    )
    write_text(GENERATED_DIR / "task_results.tex", table)

def generate_ablation_table(summary: pd.DataFrame) -> None:
    metadata = {
        "samd-origin": ("SAM", "Python dict", "--"),
        "csam-samd-unordered": ("SAM", "STL", "--"),
        "csam-gsamd-unordered": ("GSAM", "STL", "--"),
        "csam-samd-int32": ("SAM", "OA", "--"),
        "csam-gsamd-int32": ("GSAM", "OA", "--"),
        "csam-samd-lazy-t1": ("SAM", "lazy + STL", "1"),
        "csam-gsamd-lazy-t1": ("GSAM", "lazy + STL", "1"),
        "csam-samd-lazy_int32-t1": ("SAM", "lazy + OA", "1"),
        "csam-gsamd-lazy_int32-t1": ("GSAM", "lazy + OA", "1"),
    }
    rows = [
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllcrrr}",
        "\\toprule",
        "Variant & Construction & Backing table & $\\tau$ & Memory (GB) & Overall Speedup & Accept Length \\\\",
        "\\midrule",
    ]
    for name in ABLATION_METHODS:
        row = row_for(summary, name)
        construction, table, threshold = metadata[name]
        is_best = name == "csam-gsamd-lazy_int32-t1"
        label = maybe_bold(tex_escape(ABLATION_LABELS[name]), is_best)
        construction = maybe_bold(construction, is_best)
        table = maybe_bold(table, is_best)
        threshold = maybe_bold(threshold, is_best)
        memory = maybe_bold(f"{bytes_to_gb(row['memory_mean']):.2f}", is_best)
        speed = bold_math_cell(speed_cell(row)) if is_best else speed_cell(row)
        accept = maybe_bold(f"{row['accept_overall_mean']:.3f}", is_best)
        rows.append(
            f"{label} & {construction} & {table} & {threshold} & "
            f"{memory} & {speed} & {accept} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", "}"])
    table = make_table(
        "\n".join(rows),
        "Ablation of construction and transition-container choices. STL denotes the C++ standard-library hash table, and OA denotes the Open Addressing hash table used by SuffixDecoding. Overall speedup is reported as the 3-run mean $\\pm$ standard deviation; accepted length is reported as the 3-run mean.",
        "tab:ablation",
        starred=True,
    )
    write_text(GENERATED_DIR / "ablation_results.tex", table)

def generate_eagle_table(eagle: pd.DataFrame) -> None:
    samd = row_for(eagle, "samd-eagle2")
    csamd = row_for(eagle, "gsamd-eagle2")
    rows = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & SAMD + EAGLE-2 & C-SAMD + EAGLE-2 \\\\",
        "\\midrule",
        f"Memory (GB) & {bytes_to_gb(samd['memory_mean']):.2f} & {bytes_to_gb(csamd['memory_mean']):.2f} \\\\",
        f"Overall Speedup & {speed_cell(samd)} & {speed_cell(csamd)} \\\\",
        f"Accept Length & {samd['accept_overall_mean']:.3f} & {csamd['accept_overall_mean']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "}",
    ]
    table = make_table(
        "\n".join(rows),
        "SAMD variants mixed with EAGLE-2. Speedup is the 3-run mean $\\pm$ standard deviation; accepted length is shown as a table value because its variance is negligible.",
        "tab:eagle2-mix",
    )
    write_text(GENERATED_DIR / "eagle2_mix_results.tex", table)


def generate_13b_table(summary_13b: pd.DataFrame) -> None:
    samd = row_for(summary_13b, "samd-origin")
    csamd = row_for(summary_13b, "csam-gsamd-lazy_int32-t1")
    rows = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & SAMD & C-SAMD \\\\",
        "\\midrule",
        f"Memory (GB) & {bytes_to_gb(samd['memory_mean']):.2f} & {bytes_to_gb(csamd['memory_mean']):.2f} \\\\",
        f"Overall Speedup & {plain_speed(samd)} & {plain_speed(csamd)} \\\\",
        f"Accept Length & {samd['accept_overall_mean']:.3f} & {csamd['accept_overall_mean']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "}",
    ]
    table = make_table(
        "\n".join(rows),
        "Vicuna-13B-v1.3 results for SAMD and C-SAMD. The experiment is run once; speedup and accepted length are reported without standard deviations.",
        "tab:vicuna13b-results",
    )
    write_text(GENERATED_DIR / "vicuna13b_results.tex", table)

def generate_lazy_threshold_accept_table(summary: pd.DataFrame) -> None:
    series = [
        ("csam-samd-lazy-t", "SAMD lazy + STL"),
        ("csam-gsamd-lazy-t", "GSAMD lazy + STL"),
        ("csam-samd-lazy_int32-t", "SAMD lazy + OA"),
        ("csam-gsamd-lazy_int32-t", "C-SAMD"),
    ]
    thresholds = [1, 2, 3, 4, 5]
    rows = [
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Variant & $\\tau=1$ & $\\tau=2$ & $\\tau=3$ & $\\tau=4$ & $\\tau=5$ \\\\",
        "\\midrule",
    ]
    for prefix, label in series:
        values = []
        for tau in thresholds:
            row = row_for(summary, f"{prefix}{tau}")
            value = f"{row['accept_overall_mean']:.3f}"
            if prefix == "csam-gsamd-lazy_int32-t" and tau == 1:
                value = bold_cell(value)
            values.append(value)
        label = bold_cell(label) if label == "C-SAMD" else label
        rows.append(f"{label} & " + " & ".join(values) + " \\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}", "}"])
    table = make_table(
        "\n".join(rows),
        "Accepted draft length for different lazy inline thresholds. Values are the overall 3-run mean; variances are negligible, so no error bars are shown.",
        "tab:lazy-threshold-accept",
        starred=False,
    ).replace("\\begin{table}[t]", "\\begin{table}[H]", 1)
    write_text(GENERATED_DIR / "lazy_threshold_accept.tex", table)

def generate_lazy_threshold_speedup_table(summary: pd.DataFrame) -> None:
    series = [
        ("csam-samd-lazy-t", "\\shortstack{SAMD lazy\\\\+ STL}"),
        ("csam-gsamd-lazy-t", "\\shortstack{GSAMD lazy\\\\+ STL}"),
        ("csam-samd-lazy_int32-t", "\\shortstack{SAMD lazy\\\\+ OA}"),
        ("csam-gsamd-lazy_int32-t", "\\shortstack{\\textbf{C-SAMD}}"),
    ]
    thresholds = [1, 2, 3, 4, 5]
    rows = [
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "$\\tau$ & " + " & ".join(label for _, label in series) + " \\\\",
        "\\midrule",
    ]
    for tau in thresholds:
        values = []
        for prefix, _ in series:
            row = row_for(summary, f"{prefix}{tau}")
            value = speed_cell(row)
            if prefix == "csam-gsamd-lazy_int32-t" and tau == 1:
                value = bold_math_cell(value)
            values.append(value)
        rows.append(f"{tau} & " + " & ".join(values) + " " + "\\\\")
    rows.extend(["\\bottomrule", "\\end{tabular}", "}"])
    table = make_table(
        "\n".join(rows),
        "Overall speedup for different lazy inline thresholds. Values are reported as 3-run mean $\\pm$ standard deviation.",
        "tab:lazy-threshold-speedup",
        starred=False,
    ).replace("\\begin{table}[t]", "\\begin{table}[H]", 1)
    write_text(GENERATED_DIR / "lazy_threshold_speedup.tex", table)

def plot_memory_speed(summary: pd.DataFrame, output: Path, dpi: int, title: str = "Memory-Speed Trade-off") -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    methods = [method for method in MAIN_METHODS if method in set(summary["model"])]
    plot_order = [method for method in methods if method != "samd-origin"] + [method for method in methods if method == "samd-origin"]
    colors = {
        "lade-level-5-win-7-guess-7-float16": "#4C78A8",
        "rest-temperature-0.0-top_p-0": "#4D32A9",
        "recycling": "#59A14F",
        "pld-float16": "#F58518",
        "cacheback": "#B279A2",
        "samd-origin": "#6B7280",
        "csam-gsamd-lazy_int32-t1": "#D62728",
    }
    markers = {
        "csam-gsamd-lazy_int32-t1": "*",
        "samd-origin": "D",
    }
    sizes = {
        "csam-gsamd-lazy_int32-t1": 520,
        "samd-origin": 120,
    }
    label_offsets = {
        "lade-level-5-win-7-guess-7-float16": (8, -2),
        "rest-temperature-0.0-top_p-0": (-45, 12),
        "recycling": (9, -14),
        "pld-float16": (9, 0),
        "cacheback": (9, 13),
        "samd-origin": (-45, 10),
        "csam-gsamd-lazy_int32-t1": (10, -26),
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=dpi)
    for name in plot_order:
        row = row_for(summary, name)
        memory = bytes_to_gb(row["memory_mean"])
        speed = row["overall_mean"]
        is_compact = name == "csam-gsamd-lazy_int32-t1"
        ax.scatter(
            [memory],
            [speed],
            marker=markers.get(name, "o"),
            s=sizes.get(name, 105),
            color=colors[name],
            edgecolor="black" if is_compact else "white",
            linewidth=0.9 if is_compact else 1.0,
            zorder=5 if is_compact else 3,
        )
        ax.annotate(
            METHOD_LABELS[name],
            (memory, speed),
            xytext=label_offsets.get(name, (8, 8)),
            textcoords="offset points",
            fontsize=16 if is_compact else 14,
            fontweight="bold" if is_compact else "normal",
            color="#B22222" if is_compact else "black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 1.5,
            }
            if is_compact
            else None,
        )

    memories = [bytes_to_gb(row_for(summary, name)["memory_mean"]) for name in methods]
    speeds = [row_for(summary, name)["overall_mean"] for name in methods]
    ax.set_ylabel("Overall Speedup", fontsize=16)
    ax.set_xlabel("CPU RSS (GB)", fontsize=16)
    ax.set_title(title, pad=10, fontsize=17)
    ax.set_xscale("log")
    ax.set_xlim(min(memories) * 0.82, max(memories) * 1.12)
    ax.set_ylim(min(1.15, min(speeds) * 0.96), max(speeds) * 1.06)
    x_ticks = [1, 2, 5, 10, 20, 50]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(tick) for tick in x_ticks], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    arrow = Image.open(PIC_DIR / "arrow.png").convert("RGBA")
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
        0.11,
        0.77,
        "Lower memory\nhigher speedup",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=14,
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


def plot_specbench_radar(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    methods = [method for method in MAIN_METHODS if method in set(summary["model"])]
    plot_order = [method for method in methods if method != "samd-origin"] + [method for method in methods if method == "samd-origin"]
    keys = [key for key, _ in TASKS]
    base_labels = [label for _, label in TASKS]
    max_by_key = {key: max(row_for(summary, name)[f"{key}_mean"] for name in methods) for key in keys}
    labels = [f"{label}\n{max_by_key[key]:.2f}x" for key, label in zip(keys, base_labels)]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    colors = {
        "lade-level-5-win-7-guess-7-float16": "#4C78A8",
        "rest-temperature-0.0-top_p-0": "#9CA3AF",
        "recycling": "#59A14F",
        "pld-float16": "#F58518",
        "cacheback": "#B279A2",
        "samd-origin": "#6B7280",
        "csam-gsamd-lazy_int32-t1": "#D62728",
    }
    linestyles = {"samd-origin": "--", "csam-gsamd-lazy_int32-t1": "-"}

    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=dpi, subplot_kw={"projection": "polar"})
    for name in plot_order:
        row = row_for(summary, name)
        values = [row[f"{key}_mean"] / max_by_key[key] for key in keys]
        values += values[:1]
        is_compact = name == "csam-gsamd-lazy_int32-t1"
        ax.plot(
            angles,
            values,
            color=colors[name],
            linewidth=2.8 if is_compact else (2.4 if name == "samd-origin" else 1.6),
            linestyle=linestyles.get(name, "-"),
            label=METHOD_LABELS[name],
            alpha=0.86 if is_compact else (0.98 if name == "samd-origin" else 0.68),
        )
        if is_compact:
            ax.fill(angles, values, color=colors[name], alpha=0.07)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", pad=12)
    ax.set_ylim(0.0, 1.18)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_rlabel_position(92)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_title("Task-Level Speedup on Spec-Bench", pad=12, fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True, fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_memory_ablation(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    methods = [
        "samd-origin",
        "csam-samd-unordered",
        "csam-gsamd-unordered",
        "csam-samd-int32",
        "csam-samd-lazy_int32-t1",
        "csam-gsamd-lazy_int32-t1",
    ]
    labels = [
        "Original\nSAMD",
        "SAMD\nSTL",
        "GSAMD\nSTL",
        "SAMD\nOA",
        "SAMD lazy\nOA t=1",
        "C-SAMD",
    ]
    values = [bytes_to_gb(row_for(summary, name)["memory_mean"]) for name in methods]
    colors = ["#9E9E9E", "#56B4E9", "#56B4E9", "#009E73", "#0072B2", "#D62728"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=dpi)
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="#333333", linewidth=0.7)
    base = values[0]
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.2f}", ha="center", va="bottom", fontsize=16)
        if i > 0:
            red = (1 - val / base) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val / 2,
                f"{red:.1f}%\nlower",
                ha="center",
                va="center",
                color="white",
                fontsize=13,
                fontweight="bold",
            )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("CPU RSS (GB)", fontsize=16)
    ax.set_title("Index Memory Ablation", pad=10, fontsize=20)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_lazy_threshold(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    series = [
        ("csam-samd-lazy-t", "SAMD lazy + STL", "#4C78A8", "o"),
        ("csam-gsamd-lazy-t", "GSAMD lazy + STL", "#59A14F", "s"),
        ("csam-samd-lazy_int32-t", "SAMD lazy + OA", "#F58518", "^"),
        ("csam-gsamd-lazy_int32-t", "C-SAMD", "#D62728", "*"),
    ]
    thresholds = np.array([1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=dpi)
    for prefix, label, color, marker in series:
        names = [f"{prefix}{tau}" for tau in thresholds]
        memories = np.array([bytes_to_gb(row_for(summary, name)["memory_mean"]) for name in names])
        memory_stds = np.array([bytes_to_gb(row_for(summary, name)["memory_std"]) for name in names])
        marker_size = 14 if marker == "*" else 8
        ax.plot(thresholds, memories, marker=marker, markersize=marker_size, linewidth=2.4, color=color, label=label)
        ax.fill_between(thresholds, memories - memory_stds, memories + memory_stds, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("Lazy Inline Threshold $\tau$", fontsize=16)
    ax.set_ylabel("CPU RSS (GB)", fontsize=16)
    ax.set_title("Memory by Lazy Inline Threshold", pad=10, fontsize=20)
    ax.set_xticks(thresholds)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_eagle2_mix(eagle: pd.DataFrame, output: Path, dpi: int) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    methods = [name for name in ["samd-eagle2", "gsamd-eagle2"] if name in set(eagle["model"])]
    plot_order = [name for name in methods if name != "samd-eagle2"] + [name for name in methods if name == "samd-eagle2"]
    keys = [key for key, _ in TASKS]
    base_labels = [label for _, label in TASKS]
    max_by_key = {key: max(row_for(eagle, name)[f"{key}_mean"] for name in methods) for key in keys}
    labels = [f"{label}\n{max_by_key[key]:.2f}x" for key, label in zip(keys, base_labels)]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    colors = {"samd-eagle2": "#6B7280", "gsamd-eagle2": "#D62728"}

    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=dpi, subplot_kw={"projection": "polar"})
    for name in plot_order:
        row = row_for(eagle, name)
        values = [row[f"{key}_mean"] / max_by_key[key] for key in keys]
        values += values[:1]
        is_compact = name == "gsamd-eagle2"
        ax.plot(
            angles,
            values,
            color=colors[name],
            linewidth=2.8 if is_compact else 2.5,
            linestyle="-" if is_compact else "--",
            label=METHOD_LABELS[name],
            alpha=0.96,
        )
        if is_compact:
            ax.fill(angles, values, color=colors[name], alpha=0.10)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0.0, 1.18)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_rlabel_position(92)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_title("Task-Level Speedup with EAGLE-2", pad=14, fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def generate_all(root: Path, result_7b: list[Path], result_13b: Path, dpi: int) -> None:
    summary = summarize_runs(result_7b)
    summary_13b = summarize_runs([result_13b])
    eagle = summary[summary["model"].isin(["samd-eagle2", "gsamd-eagle2"])].reset_index(drop=True)

    PIC_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    generate_main_table(summary)
    generate_task_table(summary)
    generate_ablation_table(summary)
    generate_eagle_table(eagle)
    generate_13b_table(summary_13b)
    generate_lazy_threshold_accept_table(summary)
    generate_lazy_threshold_speedup_table(summary)
    plot_memory_speed(summary, PIC_DIR / "retrieval_memory_speed_tradeoff.pdf", dpi, title="Memory-Speed Trade-off on Vicuna-7B")
    plot_memory_speed(summary_13b, PIC_DIR / "vicuna13b_memory_speed_tradeoff.pdf", dpi, title="Memory-Speed Trade-off on Vicuna-13B")
    plot_specbench_radar(summary, PIC_DIR / "specbench_task_speedup_radar.pdf", dpi)
    plot_memory_ablation(summary, PIC_DIR / "index_memory_ablation.pdf", dpi)
    plot_lazy_threshold(summary, PIC_DIR / "lazy_threshold_memory.pdf", dpi)
    plot_eagle2_mix(eagle, PIC_DIR / "eagle2_hybrid_task_speedup_radar.pdf", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures from raw experiment CSV files.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Paper root for generated tables, figures, and summary CSV outputs.")
    parser.add_argument("--result-7b", type=Path, nargs=3, default=DEFAULT_7B_RESULTS, metavar=("RUN1", "RUN2", "RUN3"), help="Three raw Vicuna-7B-v1.3 result CSV files.")
    parser.add_argument("--result-13b", type=Path, default=DEFAULT_13B_RESULT, help="Raw Vicuna-13B-v1.3 result CSV file.")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    root = args.root.resolve()
    result_7b = [path if path.is_absolute() else (root / path) for path in args.result_7b]
    result_13b = args.result_13b if args.result_13b.is_absolute() else (root / args.result_13b)
    generate_all(root, result_7b, result_13b, args.dpi)


if __name__ == "__main__":
    main()
