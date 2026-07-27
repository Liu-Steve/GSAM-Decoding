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
DEFAULT_DYNAMIC_ONLY_RESULTS = [
    ROOT / "data" / f"spec_bench_7b_percent_{run}" / "result.csv"
    for run in range(1, 4)
]

DEFAULT_DEGREE_STATS = ROOT / "data" / "index_degree_stats.csv"
DEFAULT_DEGREE_INDEXES = {
    "sam-100": ROOT / "local_cache" / "gsamd" / "static_data_sam.pb",
    "gsam-1": ROOT / "local_cache" / "gsamd" / "static_data_gsam_1.pb",
    "gsam-5": ROOT / "local_cache" / "gsamd" / "static_data_gsam_5.pb",
    "gsam-10": ROOT / "local_cache" / "gsamd" / "static_data_gsam_10.pb",
    "gsam-50": ROOT / "local_cache" / "gsamd" / "static_data_gsam_50.pb",
    "gsam-100": ROOT / "local_cache" / "gsamd" / "static_data_gsam.pb",
}
CORPUS_PERCENTAGES = (1, 5, 10, 50, 100)

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
    "cacheback": "Cacheback",
    "csamd-dynamic-only": "C-SAMD (dynamic only)",
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
    "csamd-dynamic-only",
    "samd-origin",
    "csam-gsamd-lazy_int32-t1",
]

PLOT_METHODS = [name for name in MAIN_METHODS if name != "csamd-dynamic-only"]

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


def make_table(
    tabular: str,
    caption: str,
    label: str,
    starred: bool = False,
    size: str = "\\small",
) -> str:
    env = "table*" if starred else "table"
    return (
        f"\\begin{{{env}}}[t]\n"
        "\\centering\n"
        f"{size}\n"
        f"{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\end{{{env}}}\n"
    )


def resolve_degree_indexes(overrides: list[str]) -> dict[str, Path]:
    paths = {name: path.resolve() for name, path in DEFAULT_DEGREE_INDEXES.items()}
    for value in overrides:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in paths or not raw_path:
            expected = ", ".join(paths)
            raise ValueError(
                f"invalid --degree-index {value!r}; expected NAME=PATH where NAME is one of: {expected}"
            )
        paths[name] = Path(raw_path).expanduser().resolve()
    return paths


def collect_degree_stats(index_paths: dict[str, Path], output: Path) -> pd.DataFrame:
    from analyze_static_sam_successors import analyze

    rows = []
    for index_id, path in index_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        construction, percentage_text = index_id.split("-", maxsplit=1)
        print(f"Scanning degree statistics: {index_id} <- {path}")
        counts = analyze(path, edge_field="edge", count_mode="transitions")
        total_states = sum(counts.values())
        if total_states == 0:
            raise ValueError(f"no states found in {path}")
        try:
            source_index = str(path.relative_to(ROOT))
        except ValueError:
            source_index = str(path)
        row = {
            "construction": construction,
            "corpus_percent": int(percentage_text),
            "source_index": source_index,
            "total_states": total_states,
        }
        for degree in range(1, 6):
            row[f"degree_{degree}_states"] = counts.get(degree, 0)
        row["degree_other_states"] = total_states - sum(
            counts.get(degree, 0) for degree in range(1, 6)
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote degree-stat cache: {output}")
    return frame


def load_degree_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"degree-stat cache not found: {path}; rerun with --refresh-degree-stats"
        )
    frame = pd.read_csv(path)
    bucket_columns = [f"degree_{degree}_states" for degree in range(1, 6)] + [
        "degree_other_states"
    ]
    required_columns = {
        "construction",
        "corpus_percent",
        "source_index",
        "total_states",
        *bucket_columns,
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing degree-stat columns in {path}: {missing}")

    numeric_columns = ["corpus_percent", "total_states", *bucket_columns]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    expected = {"sam-100", *(f"gsam-{value}" for value in CORPUS_PERCENTAGES)}
    actual = {
        f"{row.construction}-{int(row.corpus_percent)}"
        for row in frame.itertuples(index=False)
    }
    if actual != expected:
        raise ValueError(
            f"degree-stat rows in {path} are {sorted(actual)}; expected {sorted(expected)}"
        )
    for row in frame.itertuples(index=False):
        bucket_total = sum(getattr(row, column) for column in bucket_columns)
        if int(bucket_total) != int(row.total_states):
            raise ValueError(
                f"degree buckets do not sum to total_states for {row.construction}-{int(row.corpus_percent)}"
            )
    return frame


def degree_row(stats: pd.DataFrame, construction: str, corpus_percent: int) -> pd.Series:
    rows = stats.loc[
        (stats["construction"] == construction)
        & (stats["corpus_percent"] == corpus_percent)
    ]
    if len(rows) != 1:
        raise KeyError(
            f"expected one degree-stat row for {construction}-{corpus_percent}, found {len(rows)}"
        )
    return rows.iloc[0]


def compact_uncertainty_cell(mean: float, std: float) -> str:
    std_text = f"{std:.3f}"
    return rf"${mean:.3f}{{\pm}}{std_text}$"


def generate_degree_distribution_table(stats: pd.DataFrame) -> None:
    backslash = chr(92)
    row_end = backslash * 2
    rows = [
        f"{backslash}setlength{{{backslash}tabcolsep}}{{4.2pt}}",
        f"{backslash}begin{{tabular}}{{lrrrrrr}}",
        f"{backslash}toprule",
        f"Out-degree & 1 & 2 & 3 & 4 & 5 & Other {row_end}",
        f"{backslash}midrule",
    ]
    for construction, label in (("sam", "SAM"), ("gsam", "GSAM")):
        row = degree_row(stats, construction, 100)
        total_states = float(row["total_states"])
        bucket_columns = [f"degree_{degree}_states" for degree in range(1, 6)] + [
            "degree_other_states"
        ]
        values = [
            f"{100.0 * float(row[column]) / total_states:.1f}"
            for column in bucket_columns
        ]
        padded_label = f"{label:<5}"
        rows.append(padded_label + "& " + " & ".join(values) + f" {row_end}")
    rows.extend(
        [f"{backslash}bottomrule", f"{backslash}end{{tabular}}"]
    )
    table = make_table(
        "\n".join(rows),
        f"State out-degree distribution ({backslash}%). Low-degree states remain dominant after generalized construction.",
        "tab:degree-distribution",
    )
    write_text(GENERATED_DIR / "degree_distribution.tex", table)


def generate_corpus_scaling_table(
    scaling_summary: pd.DataFrame,
    degree_stats: pd.DataFrame,
) -> None:
    backslash = chr(92)
    row_end = backslash * 2
    rows = [
        f"{backslash}setlength{{{backslash}tabcolsep}}{{2.7pt}}",
        f"{backslash}begin{{tabular}}{{rrrrr}}",
        f"{backslash}toprule",
        f"Corpus & RSS (GB) & Speedup & Accepted & Degree 1 {row_end}",
        f"{backslash}midrule",
    ]
    for percentage in CORPUS_PERCENTAGES:
        result = row_for(scaling_summary, f"gsamd-corpus-{percentage}pct")
        degree = degree_row(degree_stats, "gsam", percentage)
        memory = compact_uncertainty_cell(
            bytes_to_gb(result["memory_mean"]),
            bytes_to_gb(result["memory_std"]),
        )
        speed = compact_uncertainty_cell(result["overall_mean"], result["overall_std"])
        accept_value = result["accept_overall_mean"]
        accept = f"{accept_value:.3f}"
        degree_one = 100.0 * degree["degree_1_states"] / degree["total_states"]
        corpus_label = f"{percentage}{backslash}%".ljust(6)
        rows.append(
            f"{corpus_label}& {memory} & {speed} & {accept} & "
            f"{degree_one:.1f}{backslash}% {row_end}"
        )
    rows.extend(
        [f"{backslash}bottomrule", f"{backslash}end{{tabular}}"]
    )
    table = make_table(
        "\n".join(rows),
        "C-SAMD corpus scaling (three runs). RSS is full-process host memory; Accepted is the accepted draft length; Degree 1 is the fraction of single-successor states.",
        "tab:corpus-scaling",
    )
    write_text(GENERATED_DIR / "corpus_scaling.tex", table)


def generate_main_table(summary: pd.DataFrame) -> None:
    rows = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Host RSS (GB) & Overall Speedup & Accepted Length \\\\",
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
        "Overall Spec-Bench results. RSS is full-process host memory; the dynamic-only row uses no external corpus. Speedup is the three-run mean $\\pm$ standard deviation, and accepted length is the overall mean per verification step.",
        "tab:main-results",
    )
    write_text(GENERATED_DIR / "main_results.tex", table)


def generate_task_table(summary: pd.DataFrame) -> None:
    rows = [
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Host RSS (GB) & Overall Speedup & Accepted Length \\\\",
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
        "Overall Spec-Bench results. RSS is full-process host memory; the dynamic-only row uses no external corpus. Speedup is the three-run mean $\\pm$ standard deviation, and accepted length is the overall mean per verification step.",
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
    backslash = chr(92)
    row_end = backslash * 2
    tau_label = f"${backslash}tau$"
    rows = [
        f"{backslash}begin{{tabular}}{{lllcrrr}}",
        f"{backslash}toprule",
        f"Variant & Construction & Transition container & {tau_label} & Host RSS (GB) & Overall speedup & Accepted length {row_end}",
        f"{backslash}midrule",
    ]
    for name in ABLATION_METHODS:
        row = row_for(summary, name)
        construction, container, threshold = metadata[name]
        is_best = name == "csam-gsamd-lazy_int32-t1"
        label = maybe_bold(ABLATION_LABELS[name], is_best)
        construction = maybe_bold(construction, is_best)
        container = maybe_bold(container, is_best)
        threshold = maybe_bold(threshold, is_best)
        memory_mean = row["memory_mean"]
        memory_value = f"{bytes_to_gb(memory_mean):.2f}"
        memory = maybe_bold(memory_value, is_best)
        speed = bold_math_cell(speed_cell(row)) if is_best else speed_cell(row)
        accept_mean = row["accept_overall_mean"]
        accept_value = f"{accept_mean:.3f}"
        accept = maybe_bold(accept_value, is_best)
        rows.append(
            f"{label} & {construction} & {container} & {threshold} & "
            f"{memory} & {speed} & {accept} {row_end}"
        )
    rows.extend([f"{backslash}bottomrule", f"{backslash}end{{tabular}}"])
    size = f"{backslash}small\n{backslash}setlength{{{backslash}tabcolsep}}{{4.2pt}}"
    table = make_table(
        "\n".join(rows),
        "Full ablation of construction and transition-container choices. STL is the standard-library hash table in the common C++ core; OA is adapted from ArcticInference. Speedup is the three-run mean $" + backslash + "pm$ standard deviation, and accepted length is the three-run mean.",
        "tab:ablation",
        starred=True,
        size=size,
    )
    write_text(GENERATED_DIR / "ablation_results.tex", table)


def generate_eagle_table(eagle: pd.DataFrame) -> None:
    samd = row_for(eagle, "samd-eagle2")
    csamd = row_for(eagle, "gsamd-eagle2")
    backslash = chr(92)
    row_end = backslash * 2
    rows = [
        f"{backslash}begin{{tabular}}{{lrr}}",
        f"{backslash}toprule",
        f"Task & SAMD + EAGLE-2 & C-SAMD + EAGLE-2 {row_end}",
        f"{backslash}midrule",
    ]
    task_labels = [
        ("mt_bench", "MT-Bench"),
        ("translation", "Translation"),
        ("summarization", "Summarization"),
        ("qa", "QA"),
        ("math_reasoning", "Math"),
        ("rag", "RAG"),
    ]
    for key, label in task_labels:
        samd_speed = compact_uncertainty_cell(
            samd[f"{key}_mean"], samd[f"{key}_std"]
        )
        csamd_speed = compact_uncertainty_cell(
            csamd[f"{key}_mean"], csamd[f"{key}_std"]
        )
        rows.append(f"{label} & {samd_speed} & {csamd_speed} {row_end}")
    rows.extend([f"{backslash}bottomrule", f"{backslash}end{{tabular}}"])
    size = "\n".join(
        [
            f"{backslash}scriptsize",
            f"{backslash}setlength{{{backslash}tabcolsep}}{{4pt}}",
        ]
    )
    table = make_table(
        "\n".join(rows),
        "Task-level speedups for the EAGLE-2 hybrids, reported as three-run means $\\pm$ standard deviations. Aggregate RSS, speedup, and accepted length appear in the text.",
        "tab:eagle2-mix",
        size=size,
    )
    write_text(GENERATED_DIR / "eagle2_mix_results.tex", table)

def generate_13b_table(summary_13b: pd.DataFrame) -> None:
    samd = row_for(summary_13b, "samd-origin")
    csamd = row_for(summary_13b, "csam-gsamd-lazy_int32-t1")
    rows = [
        "\\resizebox{0.8\\columnwidth}{!}{%",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & SAMD & C-SAMD \\\\",
        "\\midrule",
        f"Host RSS (GB) & {bytes_to_gb(samd['memory_mean']):.2f} & {bytes_to_gb(csamd['memory_mean']):.2f} \\\\",
        f"Overall Speedup & {plain_speed(samd)} & {plain_speed(csamd)} \\\\",
        f"Accepted Length & {samd['accept_overall_mean']:.3f} & {csamd['accept_overall_mean']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "}",
    ]
    table = make_table(
        "\n".join(rows),
        "Vicuna-13B-v1.3 results for SAMD and C-SAMD. The experiment is run once; speedup and accepted length are reported without standard deviations.",
        "tab:vicuna13b-results",
        size="\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{0.9}",
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
        "Overall speedup for different lazy inline thresholds. Values are reported as the three-run mean $\\pm$ standard deviation.",
        "tab:lazy-threshold-speedup",
        starred=False,
    )
    write_text(GENERATED_DIR / "lazy_threshold_speedup.tex", table)

def plot_memory_speed(summary: pd.DataFrame, output: Path, dpi: int, title: str = "Memory-Speed Trade-off") -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    methods = [method for method in PLOT_METHODS if method in set(summary["model"])]
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
    ax.set_xlabel("Host Memory (GB)", fontsize=16)
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

    methods = [method for method in PLOT_METHODS if method in set(summary["model"])]
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
    ax.set_ylabel("Host Memory (GB)", fontsize=16)
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

    ax.set_xlabel("Lazy Inline Threshold", fontsize=16)
    ax.set_ylabel("Host Memory (GB)", fontsize=16)
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


def generate_all(
    root: Path,
    result_7b: list[Path],
    dynamic_only_results: list[Path],
    result_13b: Path,
    degree_stats_path: Path,
    refresh_degree_stats: bool,
    degree_index_paths: dict[str, Path],
    dpi: int,
    generate_plots: bool,
) -> None:
    global PIC_DIR, GENERATED_DIR
    PIC_DIR = root / "assets"
    GENERATED_DIR = root / "generated"

    summary = summarize_runs(result_7b)
    scaling_summary = summarize_runs(dynamic_only_results)
    dynamic_row = scaling_summary.loc[
        scaling_summary["model"] == "csamd-dynamic-only"
    ]
    if dynamic_row.empty:
        raise KeyError("missing csamd-dynamic-only in dynamic-only result CSV files")
    summary = pd.concat([summary, dynamic_row], ignore_index=True)
    summary_13b = summarize_runs([result_13b])
    eagle = summary[
        summary["model"].isin(["samd-eagle2", "gsamd-eagle2"])
    ].reset_index(drop=True)
    degree_stats = (
        collect_degree_stats(degree_index_paths, degree_stats_path)
        if refresh_degree_stats
        else load_degree_stats(degree_stats_path)
    )

    PIC_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    generate_main_table(summary)
    generate_task_table(summary)
    generate_degree_distribution_table(degree_stats)
    generate_ablation_table(summary)
    generate_eagle_table(eagle)
    generate_13b_table(summary_13b)
    generate_lazy_threshold_accept_table(summary)
    generate_lazy_threshold_speedup_table(summary)
    generate_corpus_scaling_table(scaling_summary, degree_stats)

    if not generate_plots:
        return
    plot_memory_speed(
        summary,
        PIC_DIR / "retrieval_memory_speed_tradeoff.pdf",
        dpi,
        title="Memory-Speed Trade-off on Vicuna-7B",
    )
    plot_memory_speed(
        summary_13b,
        PIC_DIR / "vicuna13b_memory_speed_tradeoff.pdf",
        dpi,
        title="Memory-Speed Trade-off on Vicuna-13B",
    )
    plot_specbench_radar(summary, PIC_DIR / "specbench_task_speedup_radar.pdf", dpi)
    plot_memory_ablation(summary, PIC_DIR / "index_memory_ablation.pdf", dpi)
    plot_lazy_threshold(summary, PIC_DIR / "lazy_threshold_memory.pdf", dpi)
    plot_eagle2_mix(
        eagle,
        PIC_DIR / "eagle2_hybrid_task_speedup_radar.pdf",
        dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper tables and figures from experiment results."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="paper root for generated tables and figures",
    )
    parser.add_argument(
        "--result-7b",
        type=Path,
        nargs=3,
        default=DEFAULT_7B_RESULTS,
        metavar=("RUN1", "RUN2", "RUN3"),
        help="three Vicuna-7B-v1.3 result CSV files",
    )
    parser.add_argument(
        "--dynamic-only-results",
        type=Path,
        nargs=3,
        default=DEFAULT_DYNAMIC_ONLY_RESULTS,
        metavar=("RUN1", "RUN2", "RUN3"),
        help="three CSV files containing dynamic-only and corpus-scaling rows",
    )
    parser.add_argument(
        "--result-13b",
        type=Path,
        default=DEFAULT_13B_RESULT,
        help="Vicuna-13B-v1.3 result CSV file",
    )
    parser.add_argument(
        "--degree-stats",
        type=Path,
        default=DEFAULT_DEGREE_STATS,
        help="cached structural statistics used by degree and corpus tables",
    )
    parser.add_argument(
        "--refresh-degree-stats",
        action="store_true",
        help="rescan protobuf indexes and replace --degree-stats",
    )
    parser.add_argument(
        "--degree-index",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override a refresh input: sam-100 or gsam-{1,5,10,50,100}",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="write all generated TeX tables without regenerating figures",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    result_7b = [path.expanduser().resolve() for path in args.result_7b]
    dynamic_only_results = [
        path.expanduser().resolve() for path in args.dynamic_only_results
    ]
    result_13b = args.result_13b.expanduser().resolve()
    degree_stats_path = args.degree_stats.expanduser().resolve()
    degree_index_paths = resolve_degree_indexes(args.degree_index)
    generate_all(
        root=root,
        result_7b=result_7b,
        dynamic_only_results=dynamic_only_results,
        result_13b=result_13b,
        degree_stats_path=degree_stats_path,
        refresh_degree_stats=args.refresh_degree_stats,
        degree_index_paths=degree_index_paths,
        dpi=args.dpi,
        generate_plots=not args.tables_only,
    )


if __name__ == "__main__":
    main()
