from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


CSV_NAMES = {
    "joined": "kernel_exact_joined.csv",
    "same": "kernel_exact_same_distance_correlations.csv",
    "profile": "kernel_exact_profile_similarity.csv",
    "length": "kernel_exact_length_correlations.csv",
}

COLORS = {
    "attention": "#0072B2",
    "correlation": "#D55E00",
    "length": "#009E73",
    "gray": "#4D4D4D",
    "light_gray": "#D9D9D9",
}
N_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parents[1] / "attn_analysis_results" / "attn_analysis_4"
    parser = argparse.ArgumentParser(
        description=(
            "Make publication-style figures from compare_attention_kernel_to_exact_correlations.py CSV files."
        )
    )
    parser.add_argument("--input_dir", default=str(default_input), help="Directory containing kernel/exact CSV files.")
    parser.add_argument("--output_dir", default=None, help="Default: <input_dir>/paper_figures.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output formats, e.g. pdf png svg.")
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--same_n", type=int, default=None, help="Representative same-distance N. Defaults to top row.")
    parser.add_argument("--same_key", choices=["P", "b"], default=None)
    parser.add_argument("--same_layer", type=int, default=None)
    parser.add_argument("--same_head", type=int, default=None)
    parser.add_argument("--same_distance", type=int, default=None)

    parser.add_argument("--profile_n", type=int, default=14)
    parser.add_argument("--profile_g", type=float, default=0.7)
    parser.add_argument("--profile_key", choices=["P", "b"], default="b")

    parser.add_argument("--length_n", type=int, default=None, help="Representative length-correlation N.")
    parser.add_argument("--length_key", choices=["P", "b"], default=None)
    parser.add_argument("--length_layer", type=int, default=None)
    parser.add_argument("--length_head", type=int, default=None)

    parser.add_argument("--overview_width_mm", type=float, default=180.0)
    parser.add_argument("--overview_height_mm", type=float, default=135.0)
    parser.add_argument("--no_standalone", action="store_true", help="Only write the 2x2 overview figure.")
    return parser.parse_args()


def mm_to_in(value: float) -> float:
    return float(value) / 25.4


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_inputs(input_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    data: Dict[str, List[Dict[str, str]]] = {}
    for name, csv_name in CSV_NAMES.items():
        path = input_dir / csv_name
        if not path.exists():
            raise FileNotFoundError(f"Missing required CSV: {path}")
        data[name] = read_csv(path)
    return data


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_finite(value: Any) -> bool:
    return math.isfinite(fnum(value))


def parse_semicolon_floats(text: str) -> np.ndarray:
    values = [fnum(item) for item in str(text).split(";") if item != ""]
    return np.asarray(values, dtype=float)


def minmax_scale(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if int(np.sum(finite)) == 0:
        return np.full_like(array, np.nan, dtype=float)
    lo = float(np.min(array[finite]))
    hi = float(np.max(array[finite]))
    if hi - lo <= 1e-12:
        out = np.zeros_like(array, dtype=float)
        out[finite] = 0.5
        out[~finite] = np.nan
        return out
    return (array - lo) / (hi - lo)


def set_nature_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2)


def panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.18,
        1.1,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig: Any, output_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> List[Path]:
    paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def row_matches_same(row: Mapping[str, str], args: argparse.Namespace) -> bool:
    checks = [
        (args.same_n, "num_qubits"),
        (args.same_layer, "layer"),
        (args.same_head, "head"),
        (args.same_distance, "distance"),
    ]
    for expected, key in checks:
        if expected is not None and inum(row.get(key)) != int(expected):
            return False
    if args.same_key is not None and row.get("key_type") != args.same_key:
        return False
    return True


def choose_same_row(same_rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Mapping[str, str]:
    candidates = [row for row in same_rows if row_matches_same(row, args) and is_finite(row.get("pearson"))]
    if not candidates:
        candidates = [row for row in same_rows if is_finite(row.get("pearson"))]
    if not candidates:
        raise ValueError("No finite same-distance correlation rows found.")
    return max(candidates, key=lambda row: fnum(row.get("pearson")))


def same_series(
    joined_rows: Sequence[Mapping[str, str]],
    same_row: Mapping[str, str],
) -> List[Mapping[str, str]]:
    rows = [
        row
        for row in joined_rows
        if inum(row.get("num_qubits")) == inum(same_row.get("num_qubits"))
        and row.get("key_type") == same_row.get("key_type")
        and inum(row.get("layer")) == inum(same_row.get("layer"))
        and inum(row.get("head")) == inum(same_row.get("head"))
        and inum(row.get("distance")) == inum(same_row.get("distance"))
    ]
    return sorted(rows, key=lambda row: fnum(row.get("g")))


def plot_layer_head_heatmap(ax: Any, same_rows: Sequence[Mapping[str, str]]) -> Any:
    grouped: Dict[Tuple[int, int], Mapping[str, str]] = {}
    for row in same_rows:
        layer = inum(row.get("layer"))
        head = inum(row.get("head"))
        pearson = fnum(row.get("pearson"))
        if layer < 0 or head < 0 or not math.isfinite(pearson):
            continue
        key = (layer, head)
        if key not in grouped or abs(pearson) > abs(fnum(grouped[key].get("pearson"))):
            grouped[key] = row

    layers = sorted({key[0] for key in grouped})
    heads = sorted({key[1] for key in grouped})
    matrix = np.full((len(layers), len(heads)), np.nan, dtype=float)
    for li, layer in enumerate(layers):
        for hi, head in enumerate(heads):
            row = grouped.get((layer, head))
            if row is not None:
                matrix[li, hi] = fnum(row.get("pearson"))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([str(head) for head in heads])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Strongest fixed-distance correlation")
    ax.text(
        0.01,
        0.99,
        "signed max over N, key, r",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=5.8,
        color=COLORS["gray"],
    )
    return image


def plot_same_distance_curve(
    ax: Any,
    joined_rows: Sequence[Mapping[str, str]],
    same_rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
) -> Mapping[str, str]:
    row = choose_same_row(same_rows, args)
    rows = same_series(joined_rows, row)
    if len(rows) < 2:
        raise ValueError("Representative same-distance row has fewer than two g points.")

    g_values = np.asarray([fnum(item.get("g")) for item in rows], dtype=float)
    attention = np.asarray([fnum(item.get("attention_mass")) for item in rows], dtype=float)
    corr = np.asarray([abs(fnum(item.get("correlation"))) for item in rows], dtype=float)

    ax.plot(g_values, minmax_scale(attention), "o-", color=COLORS["attention"], label="attention")
    ax.plot(g_values, minmax_scale(corr), "s--", color=COLORS["correlation"], label="|ZZ|")
    ax.set_xlabel("g")
    ax.set_ylabel("min-max scaled value")
    ax.set_title("A fixed-distance channel")
    ax.legend(frameon=False, loc="best", handlelength=1.6)
    clean_axis(ax)
    ax.text(
        0.03,
        0.05,
        f"N={inum(row.get('num_qubits'))}, {row.get('key_type')}, "
        f"L{inum(row.get('layer'))}H{inum(row.get('head'))}, r={inum(row.get('distance'))}\n"
        f"Pearson={fnum(row.get('pearson')):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=COLORS["gray"],
    )
    return row


def choose_profile_row(profile_rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Mapping[str, str]:
    candidates = [
        row
        for row in profile_rows
        if inum(row.get("num_qubits")) == args.profile_n
        and row.get("key_type") == args.profile_key
        and abs(fnum(row.get("g")) - args.profile_g) <= 1e-9
        and is_finite(row.get("profile_cosine"))
    ]
    if not candidates:
        candidates = [
            row
            for row in profile_rows
            if row.get("key_type") == args.profile_key and is_finite(row.get("profile_cosine"))
        ]
    if not candidates:
        candidates = [row for row in profile_rows if is_finite(row.get("profile_cosine"))]
    if not candidates:
        raise ValueError("No finite profile similarity rows found.")
    return max(candidates, key=lambda row: fnum(row.get("profile_cosine")))


def plot_profile_overlay(ax: Any, profile_rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Mapping[str, str]:
    row = choose_profile_row(profile_rows, args)
    distances = parse_semicolon_floats(row.get("distances", ""))
    attention = parse_semicolon_floats(row.get("attention_profile", ""))
    corr = parse_semicolon_floats(row.get("correlation_profile", ""))

    ax.plot(distances, attention, "o-", color=COLORS["attention"], label="attention profile")
    ax.plot(distances, corr, "s--", color=COLORS["correlation"], label="|ZZ| profile")
    ax.set_xlabel("site distance r")
    ax.set_ylabel("normalized profile")
    ax.set_title("Spatial profile at fixed g")
    ax.legend(frameon=False, loc="best", handlelength=1.6)
    clean_axis(ax)
    ax.text(
        0.03,
        0.05,
        f"N={inum(row.get('num_qubits'))}, g={fnum(row.get('g')):g}, {row.get('key_type')}\n"
        f"L{inum(row.get('layer'))}H{inum(row.get('head'))}, cosine={fnum(row.get('profile_cosine')):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=COLORS["gray"],
    )
    return row


def row_matches_length(row: Mapping[str, str], args: argparse.Namespace) -> bool:
    checks = [
        (args.length_n, "num_qubits"),
        (args.length_layer, "layer"),
        (args.length_head, "head"),
    ]
    for expected, key in checks:
        if expected is not None and inum(row.get(key)) != int(expected):
            return False
    if args.length_key is not None and row.get("key_type") != args.length_key:
        return False
    return True


def choose_length_row(length_rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Mapping[str, str]:
    candidates = [row for row in length_rows if row_matches_length(row, args) and is_finite(row.get("pearson"))]
    if not candidates:
        candidates = [row for row in length_rows if is_finite(row.get("pearson"))]
    if not candidates:
        raise ValueError("No finite length correlation rows found.")
    return max(candidates, key=lambda row: abs(fnum(row.get("pearson"))))


def matching_profile_rows(
    profile_rows: Sequence[Mapping[str, str]],
    length_row: Mapping[str, str],
) -> List[Mapping[str, str]]:
    rows = [
        row
        for row in profile_rows
        if inum(row.get("num_qubits")) == inum(length_row.get("num_qubits"))
        and row.get("key_type") == length_row.get("key_type")
        and inum(row.get("layer")) == inum(length_row.get("layer"))
        and inum(row.get("head")) == inum(length_row.get("head"))
    ]
    return sorted(rows, key=lambda row: fnum(row.get("g")))


def plot_length_scatter(
    ax: Any,
    profile_rows: Sequence[Mapping[str, str]],
    length_rows: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
) -> Mapping[str, str]:
    row = choose_length_row(length_rows, args)
    rows = matching_profile_rows(profile_rows, row)
    if len(rows) < 2:
        raise ValueError("Representative length row has fewer than two g points.")

    x = np.asarray([fnum(item.get("attention_length")) for item in rows], dtype=float)
    y = np.asarray([fnum(item.get("correlation_length_proxy")) for item in rows], dtype=float)
    g_values = np.asarray([fnum(item.get("g")) for item in rows], dtype=float)
    scatter = ax.scatter(x, y, c=g_values, cmap="viridis", s=18, edgecolors="none")
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) >= 2 and len(np.unique(x[finite])) >= 2:
        coeff = np.polyfit(x[finite], y[finite], 1)
        x_fit = np.linspace(float(np.min(x[finite])), float(np.max(x[finite])), 100)
        ax.plot(x_fit, np.polyval(coeff, x_fit), color=COLORS["gray"], linewidth=0.9)
    ax.set_xlabel("attention length")
    ax.set_ylabel("ZZ length proxy")
    ax.set_title("Length-scale tracking")
    clean_axis(ax)
    ax.text(
        0.03,
        0.05,
        f"N={inum(row.get('num_qubits'))}, {row.get('key_type')}, "
        f"L{inum(row.get('layer'))}H{inum(row.get('head'))}\n"
        f"Pearson={fnum(row.get('pearson')):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=COLORS["gray"],
    )
    return row, scatter


def plot_overview(data: Mapping[str, Sequence[Mapping[str, str]]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    fig = plt.figure(
        figsize=(mm_to_in(args.overview_width_mm), mm_to_in(args.overview_height_mm)),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.35)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    image = plot_layer_head_heatmap(ax_a, data["same"])
    clean_axis(ax_a)
    cbar = fig.colorbar(image, ax=ax_a, fraction=0.048, pad=0.02)
    cbar.set_label("Pearson r", labelpad=2)
    cbar.outline.set_linewidth(0.5)
    panel_label(ax_a, "a")

    plot_same_distance_curve(ax_b, data["joined"], data["same"], args)
    panel_label(ax_b, "b")

    plot_profile_overlay(ax_c, data["profile"], args)
    panel_label(ax_c, "c")

    _, scatter = plot_length_scatter(ax_d, data["profile"], data["length"], args)
    cbar_d = fig.colorbar(scatter, ax=ax_d, fraction=0.048, pad=0.02)
    cbar_d.set_label("g", labelpad=2)
    cbar_d.outline.set_linewidth(0.5)
    panel_label(ax_d, "d")

    return save_figure(fig, output_dir, "paper_kernel_exact_overview", args.formats, args.dpi)


def plot_profile_similarity_by_g(
    profile_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(mm_to_in(120), mm_to_in(68)))
    by_n_g: Dict[Tuple[int, float], List[float]] = defaultdict(list)
    for row in profile_rows:
        value = fnum(row.get("profile_cosine"))
        if not math.isfinite(value):
            continue
        by_n_g[(inum(row.get("num_qubits")), fnum(row.get("g")))].append(value)

    num_qubits_values = sorted({key[0] for key in by_n_g})
    for idx, num_qubits in enumerate(num_qubits_values):
        g_values = sorted({key[1] for key in by_n_g if key[0] == num_qubits})
        medians: List[float] = []
        q25: List[float] = []
        q75: List[float] = []
        for g_value in g_values:
            values = np.asarray(by_n_g[(num_qubits, g_value)], dtype=float)
            medians.append(float(np.median(values)))
            q25.append(float(np.quantile(values, 0.25)))
            q75.append(float(np.quantile(values, 0.75)))
        color = N_COLORS[idx % len(N_COLORS)]
        ax.plot(g_values, medians, "o-", color=color, label=f"N={num_qubits}", linewidth=1.0, markersize=2.6)
        ax.fill_between(g_values, q25, q75, color=color, alpha=0.13, linewidth=0)

    ax.set_xlabel("g")
    ax.set_ylabel("profile cosine similarity")
    ax.set_ylim(0.0, 1.03)
    ax.legend(frameon=False, ncol=2, handlelength=1.4)
    clean_axis(ax)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "profile_similarity_by_g", args.formats, args.dpi)
    plt.close(fig)
    return paths


def plot_same_distance_by_layer(
    same_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in same_rows:
        layer = inum(row.get("layer"))
        pearson = fnum(row.get("pearson"))
        if layer < 0 or not math.isfinite(pearson):
            continue
        grouped[layer].append(abs(pearson))

    layers = sorted(grouped)
    values = [grouped[layer] for layer in layers]
    fig, ax = plt.subplots(figsize=(mm_to_in(90), mm_to_in(62)))
    box = ax.boxplot(
        values,
        positions=np.arange(len(layers)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        boxprops={"linewidth": 0.6},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#BFD7EA")
        patch.set_edgecolor("#3F6C8F")
    ax.axhline(0.99, color=COLORS["correlation"], linestyle="--", linewidth=0.8, alpha=0.85)
    ax.text(len(layers) - 0.15, 0.992, "|r|=0.99", ha="right", va="bottom", fontsize=5.8, color=COLORS["correlation"])
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_xlabel("layer")
    ax.set_ylabel("|Pearson r| over fixed-distance channels")
    ax.set_ylim(0.0, 1.02)
    clean_axis(ax)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "same_distance_correlation_by_layer", args.formats, args.dpi)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir / "paper_figures"
    data = load_inputs(input_dir)

    set_nature_style()
    written: List[Path] = []
    written.extend(plot_overview(data, output_dir, args))
    if not args.no_standalone:
        written.extend(plot_profile_similarity_by_g(data["profile"], output_dir, args))
        written.extend(plot_same_distance_by_layer(data["same"], output_dir, args))

    for path in written:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
