from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from matplotlib import ticker
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
    "sky": "#56B4E9",
    "gray": "#4D4D4D",
    "light_gray": "#D9D9D9",
}
N_COLORS = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["yellow"], COLORS["sky"]]
KEY_COLORS = {"g": COLORS["blue"], "P": COLORS["orange"], "b": COLORS["green"]}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "attn_analysis_results"
    parser = argparse.ArgumentParser(
        description=(
            "Make publication-style figures from the first three attention-analysis result folders."
        )
    )
    parser.add_argument("--distance_dir", default=str(root / "attn_analysis_1"))
    parser.add_argument("--kernel_dir", default=str(root / "attn_analysis_2"))
    parser.add_argument("--zz_dir", default=str(root / "attn_analysis_3"))
    parser.add_argument("--output_dir", default=str(root / "paper_figures_core_attention"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--query_group", default="b")
    parser.add_argument("--distance_metric", default="mean_site_distance_norm")
    parser.add_argument("--distance_n", type=int, default=None, help="Representative N for distance heatmaps. Defaults to largest N.")

    parser.add_argument("--kernel_n", type=int, default=None, help="Representative N for site-kernel panels. Defaults to largest N.")
    parser.add_argument("--kernel_profile_g", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])

    parser.add_argument("--zz_target", default="zz_infinite")
    parser.add_argument("--zz_threshold", type=float, default=0.99)
    parser.add_argument("--overview_width_mm", type=float, default=180.0)
    parser.add_argument("--overview_height_mm", type=float, default=135.0)
    return parser.parse_args()


def mm_to_in(value: float) -> float:
    return float(value) / 25.4


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def finite(value: Any) -> bool:
    return math.isfinite(fnum(value))


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
            "lines.linewidth": 1.15,
            "lines.markersize": 3.0,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def load_inputs(args: argparse.Namespace) -> Dict[str, List[Dict[str, str]]]:
    paths = {
        "distance": Path(args.distance_dir) / "attention_distance_metrics.csv",
        "kernel": Path(args.kernel_dir) / "attention_site_kernel.csv",
        "zz_corr": Path(args.zz_dir) / "attention_zz_correlations.csv",
        "zz_joined": Path(args.zz_dir) / "attention_zz_joined_long.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} CSV: {path}")
    return {name: read_csv_rows(path) for name, path in paths.items()}


def unique_ints(rows: Sequence[Mapping[str, str]], key: str) -> List[int]:
    return sorted({inum(row.get(key)) for row in rows})


def representative_n(rows: Sequence[Mapping[str, str]], key: str, requested: Optional[int]) -> int:
    values = unique_ints(rows, key)
    if not values:
        raise ValueError(f"No values found for {key}.")
    if requested is not None:
        return int(requested)
    return max(values)


def summary_distance_rows(rows: Sequence[Mapping[str, str]], query_group: str) -> List[Dict[str, str]]:
    selected = [
        dict(row)
        for row in rows
        if row.get("query_group") == query_group and inum(row.get("layer")) == -1 and inum(row.get("head")) == -1
    ]
    if selected:
        return selected

    grouped: Dict[Tuple[int, float], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("query_group") != query_group:
            continue
        if inum(row.get("layer")) < 0 or inum(row.get("head")) < 0:
            continue
        grouped[(inum(row.get("num_qubits")), fnum(row.get("g")))].append(row)

    out: List[Dict[str, str]] = []
    for (num_qubits, g_value), group in grouped.items():
        merged = dict(group[0])
        merged["num_qubits"] = str(num_qubits)
        merged["g"] = str(g_value)
        merged["layer"] = "-1"
        merged["head"] = "-1"
        for key in group[0].keys():
            values = [fnum(row.get(key)) for row in group if finite(row.get(key))]
            if values:
                merged[key] = f"{float(np.mean(values)):.12g}"
        out.append(merged)
    return sorted(out, key=lambda row: (inum(row.get("num_qubits")), fnum(row.get("g"))))


def plot_distance_metric(ax: Any, rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> None:
    summary = summary_distance_rows(rows, args.query_group)
    metric = args.distance_metric
    if metric not in summary[0]:
        raise KeyError(f"{metric!r} is not a column in attention_distance_metrics.csv.")

    for idx, num_qubits in enumerate(unique_ints(summary, "num_qubits")):
        series = sorted([row for row in summary if inum(row.get("num_qubits")) == num_qubits], key=lambda row: fnum(row.get("g")))
        ax.plot(
            [fnum(row.get("g")) for row in series],
            [fnum(row.get(metric)) for row in series],
            "o-",
            color=N_COLORS[idx % len(N_COLORS)],
            label=f"N={num_qubits}",
        )
    ax.set_xlabel("g")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Attention span")
    ax.legend(frameon=False, handlelength=1.5, ncol=2)
    clean_axis(ax)


def choose_mass_columns(row: Mapping[str, str]) -> Tuple[List[str], str]:
    enrichment = ["attention_to_g_enrichment", "attention_to_P_enrichment", "attention_to_b_enrichment"]
    if all(key in row for key in enrichment):
        return enrichment, "enrichment over uniform attention"
    raw = ["attention_to_g", "attention_to_P", "attention_to_b"]
    return raw, "attention mass"


def mean_by_g(rows: Sequence[Mapping[str, str]], metric: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    grouped: Dict[float, List[float]] = defaultdict(list)
    for row in rows:
        value = fnum(row.get(metric))
        if math.isfinite(value):
            grouped[fnum(row.get("g"))].append(value)
    g_values = sorted(grouped)
    means = [float(np.mean(grouped[g])) for g in g_values]
    lows = [float(np.min(grouped[g])) for g in g_values]
    highs = [float(np.max(grouped[g])) for g in g_values]
    return g_values, means, lows, highs


def plot_key_masses(ax: Any, rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> None:
    summary = summary_distance_rows(rows, args.query_group)
    columns, ylabel = choose_mass_columns(summary[0])
    labels = ["g", "P", "b"]
    for column, label in zip(columns, labels):
        g_values, means, lows, highs = mean_by_g(summary, column)
        color = KEY_COLORS[label]
        ax.plot(g_values, means, "o-", color=color, label=label)
        if any(abs(hi - lo) > 1e-12 for lo, hi in zip(lows, highs)):
            ax.fill_between(g_values, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("g")
    ax.set_ylabel(ylabel)
    ax.set_title("Key-type allocation")
    ax.legend(frameon=False, handlelength=1.5, ncol=3)
    clean_axis(ax)


def head_metric_span(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    num_qubits: int,
    query_group: str,
) -> Tuple[np.ndarray, List[int], List[int], Dict[Tuple[int, int], float]]:
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for row in rows:
        layer = inum(row.get("layer"))
        head = inum(row.get("head"))
        if layer < 0 or head < 0:
            continue
        if inum(row.get("num_qubits")) != num_qubits or row.get("query_group") != query_group:
            continue
        value = fnum(row.get(metric))
        if math.isfinite(value):
            grouped[(layer, head)].append(value)

    layers = sorted({key[0] for key in grouped})
    heads = sorted({key[1] for key in grouped})
    matrix = np.full((len(layers), len(heads)), np.nan, dtype=float)
    spans: Dict[Tuple[int, int], float] = {}
    for li, layer in enumerate(layers):
        for hi, head in enumerate(heads):
            values = grouped.get((layer, head), [])
            if values:
                span = float(np.max(values) - np.min(values))
                spans[(layer, head)] = span
                matrix[li, hi] = span
    return matrix, layers, heads, spans


def plot_distance_head_span(ax: Any, rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Tuple[int, Dict[Tuple[int, int], float]]:
    num_qubits = representative_n(rows, "num_qubits", args.distance_n)
    matrix, layers, heads, spans = head_metric_span(rows, args.distance_metric, num_qubits, args.query_group)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([str(head) for head in heads])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"Head sensitivity, N={num_qubits}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("range over g", labelpad=2)
    cbar.outline.set_linewidth(0.5)
    clean_axis(ax)
    return num_qubits, spans


def plot_top_head_curves(ax: Any, rows: Sequence[Mapping[str, str]], args: argparse.Namespace, num_qubits: int, spans: Mapping[Tuple[int, int], float]) -> None:
    top = sorted(spans, key=lambda key: spans[key], reverse=True)[:3]
    for idx, (layer, head) in enumerate(top):
        series = sorted(
            [
                row
                for row in rows
                if inum(row.get("num_qubits")) == num_qubits
                and inum(row.get("layer")) == layer
                and inum(row.get("head")) == head
                and row.get("query_group") == args.query_group
            ],
            key=lambda row: fnum(row.get("g")),
        )
        ax.plot(
            [fnum(row.get("g")) for row in series],
            [fnum(row.get(args.distance_metric)) for row in series],
            "o-",
            color=N_COLORS[idx % len(N_COLORS)],
            label=f"L{layer}H{head}",
        )
    ax.set_xlabel("g")
    ax.set_ylabel(args.distance_metric.replace("_", " "))
    ax.set_title("Most responsive heads")
    ax.legend(frameon=False, handlelength=1.5)
    clean_axis(ax)


def plot_attention_distance_figure(data: Mapping[str, Sequence[Mapping[str, str]]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    fig = plt.figure(figsize=(mm_to_in(args.overview_width_mm), mm_to_in(args.overview_height_mm)), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.35)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    plot_distance_metric(axes[0], data["distance"], args)
    panel_label(axes[0], "a")
    plot_key_masses(axes[1], data["distance"], args)
    panel_label(axes[1], "b")
    num_qubits, spans = plot_distance_head_span(axes[2], data["distance"], args)
    panel_label(axes[2], "c")
    plot_top_head_curves(axes[3], data["distance"], args, num_qubits, spans)
    panel_label(axes[3], "d")
    return save_figure(fig, output_dir, "paper_attention_distance_overview", args.formats, args.dpi)


def summary_kernel_rows(rows: Sequence[Mapping[str, str]], query_group: str, key_type: str, num_qubits: int) -> List[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row.get("query_group") == query_group
        and row.get("key_type") == key_type
        and inum(row.get("num_qubits")) == num_qubits
        and inum(row.get("layer")) == -1
        and inum(row.get("head")) == -1
        and row.get("delta_site") not in ("", "g")
    ]


def kernel_matrix(rows: Sequence[Mapping[str, str]]) -> Tuple[np.ndarray, List[int], List[float]]:
    deltas = sorted({inum(row.get("delta_site")) for row in rows})
    g_values = sorted({fnum(row.get("g")) for row in rows})
    lookup = {(inum(row.get("delta_site")), fnum(row.get("g"))): fnum(row.get("attention_mass")) for row in rows}
    matrix = np.full((len(deltas), len(g_values)), np.nan, dtype=float)
    for i, delta in enumerate(deltas):
        for j, g_value in enumerate(g_values):
            matrix[i, j] = lookup.get((delta, g_value), np.nan)
    return matrix, deltas, g_values


def plot_kernel_heatmap(ax: Any, rows: Sequence[Mapping[str, str]], key_type: str, title: str) -> Any:
    matrix, deltas, g_values = kernel_matrix(rows)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(g_values)))
    ax.set_xticklabels([f"{g:g}" for g in g_values])
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels([str(delta) for delta in deltas])
    ax.set_xlabel("g")
    ax.set_ylabel("site displacement r")
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("attention mass", labelpad=2)
    cbar.outline.set_linewidth(0.5)
    clean_axis(ax)
    return im


def aggregate_abs_kernel(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[float, int], float]:
    grouped: Dict[Tuple[float, int], float] = defaultdict(float)
    for row in rows:
        g_value = fnum(row.get("g"))
        distance = abs(inum(row.get("delta_site")))
        grouped[(g_value, distance)] += fnum(row.get("attention_mass"))
    return grouped


def closest_available(values: Sequence[float], requested: float) -> Optional[float]:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float(array[int(np.argmin(np.abs(array - float(requested))))])


def plot_kernel_abs_profiles(ax: Any, rows: Sequence[Mapping[str, str]], requested_g: Sequence[float],
                             r_min: float = 3.0, r_max: float = 6.0) -> None:
    grouped = aggregate_abs_kernel(rows)
    available_g = sorted({key[0] for key in grouped})
    chosen_g: List[float] = []
    for item in requested_g:
        nearest = closest_available(available_g, item)
        if nearest is not None and nearest not in chosen_g:
            chosen_g.append(nearest)
    distances = np.array(sorted({key[1] for key in grouped}))
    
    # 筛选范围
    mask = (distances >= r_min) & (distances <= r_max)
    distances_filtered = distances[mask]
    
    if len(distances_filtered) == 0:
        print(f"Warning: No data points in range [{r_min}, {r_max}]")
        return
    
    for idx, g_value in enumerate(chosen_g):
        values = np.asarray([grouped.get((g_value, d), np.nan) for d in distances], dtype=float)
        values_filtered = values[mask]
        
        # 过滤掉NaN值
        valid_mask = ~np.isnan(values_filtered)
        if valid_mask.any():
            ax.plot(distances_filtered[valid_mask], values_filtered[valid_mask], "o-", 
                    color=N_COLORS[idx % len(N_COLORS)], 
                    label=f"g={g_value:g}")
            
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax.set_xlabel("|r|")
    ax.set_ylabel("normalized mass")
    ax.set_title(f"Absolute-distance profile ({r_min} ≤ |r| ≤ {r_max})")
    ax.legend(frameon=False, handlelength=1.5)
    clean_axis(ax)


def plot_near_far_fraction(ax: Any, kernel_rows: Sequence[Mapping[str, str]], num_qubits: int, args: argparse.Namespace) -> None:
    for key_idx, key_type in enumerate(["P", "b"]):
        rows = summary_kernel_rows(kernel_rows, args.query_group, key_type, num_qubits)
        grouped: Dict[float, Dict[str, float]] = defaultdict(lambda: {"near": 0.0, "far": 0.0})
        for row in rows:
            distance = abs(inum(row.get("delta_site")))
            bucket = "near" if distance <= 1 else "far"
            grouped[fnum(row.get("g"))][bucket] += fnum(row.get("attention_mass"))
        g_values = sorted(grouped)
        fractions: List[float] = []
        for g_value in g_values:
            near = grouped[g_value]["near"]
            far = grouped[g_value]["far"]
            total = near + far
            fractions.append(far / total if total > 0 else float("nan"))
        ax.plot(g_values, fractions, "o-", color=[COLORS["orange"], COLORS["green"]][key_idx], label=f"to {key_type}")
    ax.set_xlabel("g")
    ax.set_ylabel("far-mass fraction, |r| ≥ 2")
    ax.set_title("Long-range allocation")
    ax.legend(frameon=False, handlelength=1.5)
    clean_axis(ax)


def plot_site_kernel_figure(data: Mapping[str, Sequence[Mapping[str, str]]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    num_qubits = representative_n(data["kernel"], "num_qubits", args.kernel_n)
    b_rows = summary_kernel_rows(data["kernel"], args.query_group, "b", num_qubits)
    p_rows = summary_kernel_rows(data["kernel"], args.query_group, "P", num_qubits)
    if not b_rows or not p_rows:
        raise ValueError("Could not find summary kernel rows for both P and b key types.")

    fig = plt.figure(figsize=(mm_to_in(args.overview_width_mm), mm_to_in(args.overview_height_mm)), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.35)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    plot_kernel_heatmap(axes[0], b_rows, "b", f"N={num_qubits}, b queries → b keys")
    panel_label(axes[0], "a")
    plot_kernel_heatmap(axes[1], p_rows, "P", f"N={num_qubits}, b queries → P keys")
    panel_label(axes[1], "b")
    plot_kernel_abs_profiles(axes[2], b_rows, args.kernel_profile_g)
    panel_label(axes[2], "c")
    plot_near_far_fraction(axes[3], data["kernel"], num_qubits, args)
    panel_label(axes[3], "d")
    return save_figure(fig, output_dir, "paper_site_kernel_overview", args.formats, args.dpi)


def best_corr_by_layer_head(rows: Sequence[Mapping[str, str]], target: str) -> Tuple[np.ndarray, List[int], List[int]]:
    grouped: Dict[Tuple[int, int], Mapping[str, str]] = {}
    for row in rows:
        if row.get("target") != target:
            continue
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
    return matrix, layers, heads


def plot_zz_corr_heatmap(ax: Any, rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> None:
    matrix, layers, heads = best_corr_by_layer_head(rows, args.zz_target)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([str(head) for head in heads])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"Best correlation with {args.zz_target}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Pearson r", labelpad=2)
    cbar.outline.set_linewidth(0.5)
    clean_axis(ax)


def choose_top_zz_row(rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Mapping[str, str]:
    candidates = [row for row in rows if row.get("target") == args.zz_target and finite(row.get("pearson"))]
    if not candidates:
        raise ValueError(f"No finite correlations found for target={args.zz_target}.")
    return max(candidates, key=lambda row: abs(fnum(row.get("pearson"))))


def matching_joined_metric_rows(joined_rows: Sequence[Mapping[str, str]], corr_row: Mapping[str, str]) -> List[Mapping[str, str]]:
    rows = [
        row
        for row in joined_rows
        if row.get("metric") == corr_row.get("metric")
        and row.get("attention_num_qubits") == corr_row.get("attention_num_qubits")
        and inum(row.get("layer")) == inum(corr_row.get("layer"))
        and inum(row.get("head")) == inum(corr_row.get("head"))
        and inum(row.get("distance")) == inum(corr_row.get("distance"))
    ]
    return sorted(rows, key=lambda row: fnum(row.get("g")))


def minmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.full_like(array, np.nan, dtype=float)
    lo = float(np.min(array[finite_mask]))
    hi = float(np.max(array[finite_mask]))
    if hi - lo <= 1e-12:
        return np.full_like(array, 0.5, dtype=float)
    return (array - lo) / (hi - lo)


def plot_top_zz_metric_curve(ax: Any, corr_rows: Sequence[Mapping[str, str]], joined_rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> Mapping[str, str]:
    row = choose_top_zz_row(corr_rows, args)
    series = matching_joined_metric_rows(joined_rows, row)
    if len(series) < 2:
        raise ValueError("Top ZZ correlation row has fewer than two joined points.")
    g_values = [fnum(item.get("g")) for item in series]
    metric = [fnum(item.get("metric_value")) for item in series]
    target = [fnum(item.get(args.zz_target)) for item in series]
    ax.plot(g_values, minmax(metric), "o-", color=COLORS["blue"], label=row.get("metric", "attention"))
    ax.plot(g_values, minmax(target), "s--", color=COLORS["orange"], label=args.zz_target)
    ax.set_xlabel("g")
    ax.set_ylabel("min-max scaled value")
    ax.set_title("Representative metric")
    ax.legend(frameon=False, handlelength=1.5)
    ax.text(
        0.03,
        0.05,
        f"d={inum(row.get('distance'))}, N={row.get('attention_num_qubits')}\n"
        f"L{inum(row.get('layer'))}H{inum(row.get('head'))}, r={fnum(row.get('pearson')):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=COLORS["gray"],
    )
    clean_axis(ax)
    return row


def plot_strong_count_by_metric(ax: Any, rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("target") != args.zz_target:
            continue
        pearson = fnum(row.get("pearson"))
        if math.isfinite(pearson) and abs(pearson) >= args.zz_threshold:
            counts[row.get("metric", "")] += 1
    top = counts.most_common(8)
    labels = [item[0].replace("attention_to_", "to_").replace("_", " ") for item in top]
    values = [item[1] for item in top]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=COLORS["sky"], edgecolor="#3F6C8F", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"count with |r| ≥ {args.zz_threshold:g}")
    ax.set_title("Repeated strong metrics")
    clean_axis(ax)


def plot_pearson_distribution(ax: Any, rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> None:
    targets = sorted({row.get("target", "") for row in rows})
    values: List[List[float]] = []
    labels: List[str] = []
    for target in targets:
        vals = [abs(fnum(row.get("pearson"))) for row in rows if row.get("target") == target and finite(row.get("pearson"))]
        if vals:
            labels.append(target.replace("_", " "))
            values.append(vals)
    box = ax.boxplot(
        values,
        positions=np.arange(len(values)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        boxprops={"linewidth": 0.6},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#E5E5E5")
        patch.set_edgecolor("#595959")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("|Pearson r|")
    ax.set_title("Correlation targets")
    ax.set_ylim(0.0, 1.02)
    clean_axis(ax)


def plot_attention_zz_figure(data: Mapping[str, Sequence[Mapping[str, str]]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    fig = plt.figure(figsize=(mm_to_in(args.overview_width_mm), mm_to_in(args.overview_height_mm)), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, wspace=0.33, hspace=0.38)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    plot_zz_corr_heatmap(axes[0], data["zz_corr"], args)
    panel_label(axes[0], "a")
    plot_top_zz_metric_curve(axes[1], data["zz_corr"], data["zz_joined"], args)
    panel_label(axes[1], "b")
    plot_strong_count_by_metric(axes[2], data["zz_corr"], args)
    panel_label(axes[2], "c")
    plot_pearson_distribution(axes[3], data["zz_corr"], args)
    panel_label(axes[3], "d")
    return save_figure(fig, output_dir, "paper_attention_zz_overview", args.formats, args.dpi)


def main() -> None:
    args = parse_args()
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    data = load_inputs(args)
    output_dir = Path(args.output_dir)
    set_nature_style()

    written: List[Path] = []
    written.extend(plot_attention_distance_figure(data, output_dir, args))
    written.extend(plot_site_kernel_figure(data, output_dir, args))
    written.extend(plot_attention_zz_figure(data, output_dir, args))
    for path in written:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
