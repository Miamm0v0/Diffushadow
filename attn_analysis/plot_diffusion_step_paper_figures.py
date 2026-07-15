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
KEY_COLORS = {"g": COLORS["blue"], "P": COLORS["orange"], "b": COLORS["green"]}
G_COLORS = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["yellow"], COLORS["sky"]]


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parents[1] / "attn_analysis_results" / "diff_analysis"
    parser = argparse.ArgumentParser(
        description="Make publication-style figures from diffusion-step attention analysis CSV files."
    )
    parser.add_argument("--input_dir", default=str(default_input))
    parser.add_argument("--output_dir", default=None, help="Default: <input_dir>/paper_figures.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--query_scope", default="all_b", choices=["all_b", "masked_b", "filled_b"])
    parser.add_argument("--kernel_key", default="b", choices=["b", "P"])
    parser.add_argument("--kernel_n", type=int, default=None, help="Representative N for kernel panels. Defaults to largest N.")
    parser.add_argument("--kernel_g", type=float, default=0.7)
    parser.add_argument("--selected_g", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--overview_width_mm", type=float, default=180.0)
    parser.add_argument("--overview_height_mm", type=float, default=135.0)
    parser.add_argument("--no_standalone", action="store_true")
    return parser.parse_args()


def mm_to_in(value: float) -> float:
    return float(value) / 25.4


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_inputs(input_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    paths = {
        "metrics": input_dir / "diffusion_step_attention_metrics.csv",
        "kernel": input_dir / "diffusion_step_site_kernel.csv",
        "profile": input_dir / "diffusion_step_kernel_exact_profile_similarity.csv",
    }
    for key in ("metrics", "kernel"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required CSV: {paths[key]}")
    data = {key: read_csv_rows(path) for key, path in paths.items() if path.exists()}
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


def closest_value(values: Sequence[float], target: float) -> Optional[float]:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float(array[int(np.argmin(np.abs(array - float(target))))])


def selected_summary_rows(metrics_rows: Sequence[Mapping[str, str]], query_scope: str) -> List[Mapping[str, str]]:
    return [
        row
        for row in metrics_rows
        if row.get("query_scope") == query_scope
        and inum(row.get("layer")) == -1
        and inum(row.get("head")) == -1
    ]


def median_iqr_by_step(rows: Sequence[Mapping[str, str]], metric: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        value = fnum(row.get(metric))
        if math.isfinite(value):
            grouped[inum(row.get("step"))].append(value)
    steps = sorted(grouped)
    med = [float(np.median(grouped[step])) for step in steps]
    q25 = [float(np.quantile(grouped[step], 0.25)) for step in steps]
    q75 = [float(np.quantile(grouped[step], 0.75)) for step in steps]
    return steps, med, q25, q75


def plot_routing_panel(ax: Any, metrics_rows: Sequence[Mapping[str, str]], query_scope: str) -> None:
    rows = selected_summary_rows(metrics_rows, query_scope)
    metrics = [("attention_to_g", "g"), ("attention_to_P", "P"), ("attention_to_b", "b")]
    for metric, label in metrics:
        steps, med, q25, q75 = median_iqr_by_step(rows, metric)
        color = KEY_COLORS[label]
        ax.plot(steps, med, "o-", color=color, label=f"to {label}")
        ax.fill_between(steps, q25, q75, color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("attention mass")
    ax.set_title(f"Information routing ({query_scope})")
    ax.legend(frameon=False, handlelength=1.4, ncol=3)
    clean_axis(ax)


def plot_distance_panel(
    ax: Any,
    metrics_rows: Sequence[Mapping[str, str]],
    query_scope: str,
    selected_g: Sequence[float],
) -> None:
    rows = selected_summary_rows(metrics_rows, query_scope)
    available_g = sorted({fnum(row.get("g")) for row in rows})
    chosen_g: List[float] = []
    for g_value in selected_g:
        nearest = closest_value(available_g, g_value)
        if nearest is not None and nearest not in chosen_g:
            chosen_g.append(nearest)
    for idx, g_value in enumerate(chosen_g):
        grouped: Dict[int, List[float]] = defaultdict(list)
        for row in rows:
            if abs(fnum(row.get("g")) - g_value) > 1e-12:
                continue
            value = fnum(row.get("mean_site_distance_norm"))
            if math.isfinite(value):
                grouped[inum(row.get("step"))].append(value)
        steps = sorted(grouped)
        med = [float(np.median(grouped[step])) for step in steps]
        q25 = [float(np.quantile(grouped[step], 0.25)) for step in steps]
        q75 = [float(np.quantile(grouped[step], 0.75)) for step in steps]
        color = G_COLORS[idx % len(G_COLORS)]
        ax.plot(steps, med, "o-", color=color, label=f"g={g_value:g}")
        ax.fill_between(steps, q25, q75, color=color, alpha=0.11, linewidth=0)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("normalized site distance")
    ax.set_title("Spatial range during denoising")
    ax.legend(frameon=False, handlelength=1.4, ncol=2)
    clean_axis(ax)


def representative_n(rows: Sequence[Mapping[str, str]], requested: Optional[int]) -> int:
    values = sorted({inum(row.get("num_qubits")) for row in rows})
    if not values:
        raise ValueError("No num_qubits values found.")
    return int(requested) if requested is not None else max(values)


def plot_kernel_step_heatmap(
    ax: Any,
    kernel_rows: Sequence[Mapping[str, str]],
    query_scope: str,
    key_type: str,
    num_qubits: int,
    requested_g: float,
) -> None:
    available_g = sorted({
        fnum(row.get("g"))
        for row in kernel_rows
        if inum(row.get("num_qubits")) == num_qubits
        and row.get("query_scope") == query_scope
        and row.get("key_type") == key_type
        and inum(row.get("layer")) == -1
        and inum(row.get("head")) == -1
    })
    g_value = closest_value(available_g, requested_g)
    if g_value is None:
        raise ValueError("No matching g value found for kernel heatmap.")
    selected = [
        row
        for row in kernel_rows
        if inum(row.get("num_qubits")) == num_qubits
        and row.get("query_scope") == query_scope
        and row.get("key_type") == key_type
        and inum(row.get("layer")) == -1
        and inum(row.get("head")) == -1
        and abs(fnum(row.get("g")) - g_value) <= 1e-12
        and row.get("delta_site") not in ("", "g")
    ]
    deltas = sorted({inum(row.get("delta_site")) for row in selected})
    steps = sorted({inum(row.get("step")) for row in selected})
    lookup = {(inum(row.get("delta_site")), inum(row.get("step"))): fnum(row.get("attention_mass")) for row in selected}
    matrix = np.full((len(deltas), len(steps)), np.nan, dtype=float)
    for i, delta in enumerate(deltas):
        for j, step in enumerate(steps):
            matrix[i, j] = lookup.get((delta, step), np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(step) for step in steps])
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels([str(delta) for delta in deltas])
    ax.set_xlabel("denoising step")
    ax.set_ylabel("site displacement r")
    ax.set_title(f"Kernel dynamics, N={num_qubits}, g={g_value:g}, to {key_type}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("attention mass", labelpad=2)
    cbar.outline.set_linewidth(0.5)
    clean_axis(ax)


def filtered_profile_rows(
    profile_rows: Sequence[Mapping[str, str]],
    query_scope: str,
    key_type: str,
    num_qubits: Optional[int] = None,
) -> List[Mapping[str, str]]:
    out = [
        row
        for row in profile_rows
        if row.get("query_scope") == query_scope
        and row.get("key_type") == key_type
        and finite(row.get("profile_cosine"))
    ]
    if num_qubits is not None:
        out = [row for row in out if inum(row.get("num_qubits")) == num_qubits]
    return out


def plot_similarity_panel(
    ax: Any,
    profile_rows: Sequence[Mapping[str, str]],
    query_scope: str,
    key_type: str,
    num_qubits: int,
    selected_g: Sequence[float],
) -> None:
    rows = filtered_profile_rows(profile_rows, query_scope, key_type, num_qubits)
    available_g = sorted({fnum(row.get("g")) for row in rows})
    chosen_g: List[float] = []
    for g_value in selected_g:
        nearest = closest_value(available_g, g_value)
        if nearest is not None and nearest not in chosen_g:
            chosen_g.append(nearest)
    for idx, g_value in enumerate(chosen_g):
        grouped: Dict[int, List[float]] = defaultdict(list)
        for row in rows:
            if abs(fnum(row.get("g")) - g_value) > 1e-12:
                continue
            grouped[inum(row.get("step"))].append(fnum(row.get("profile_cosine")))
        steps = sorted(grouped)
        med = [float(np.median(grouped[step])) for step in steps]
        q25 = [float(np.quantile(grouped[step], 0.25)) for step in steps]
        q75 = [float(np.quantile(grouped[step], 0.75)) for step in steps]
        color = G_COLORS[idx % len(G_COLORS)]
        ax.plot(steps, med, "o-", color=color, label=f"g={g_value:g}")
        ax.fill_between(steps, q25, q75, color=color, alpha=0.11, linewidth=0)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("profile cosine")
    ax.set_ylim(0.35, 1.03)
    ax.set_title(f"Kernel-exact alignment, N={num_qubits}, to {key_type}")
    ax.legend(frameon=False, handlelength=1.4, ncol=2)
    clean_axis(ax)


def plot_overview(data: Mapping[str, List[Dict[str, str]]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    if "profile" not in data:
        raise FileNotFoundError("diffusion_step_kernel_exact_profile_similarity.csv is required for the overview figure.")
    num_qubits = representative_n(data["kernel"], args.kernel_n)
    fig = plt.figure(
        figsize=(mm_to_in(args.overview_width_mm), mm_to_in(args.overview_height_mm)),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.35)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]

    plot_routing_panel(axes[0], data["metrics"], args.query_scope)
    panel_label(axes[0], "a")
    plot_distance_panel(axes[1], data["metrics"], args.query_scope, args.selected_g)
    panel_label(axes[1], "b")
    plot_kernel_step_heatmap(axes[2], data["kernel"], args.query_scope, args.kernel_key, num_qubits, args.kernel_g)
    panel_label(axes[2], "c")
    plot_similarity_panel(axes[3], data["profile"], args.query_scope, args.kernel_key, num_qubits, args.selected_g)
    panel_label(axes[3], "d")
    return save_figure(fig, output_dir, "paper_diffusion_step_overview", args.formats, args.dpi)


def plot_similarity_heatmaps(data: Mapping[str, List[Dict[str, str]]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    if "profile" not in data:
        return []
    rows = data["profile"]
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(180), mm_to_in(72)), constrained_layout=True)
    for ax, key_type in zip(axes, ["b", "P"]):
        selected = filtered_profile_rows(rows, args.query_scope, key_type)
        steps = sorted({inum(row.get("step")) for row in selected})
        g_values = sorted({fnum(row.get("g")) for row in selected})
        matrix = np.full((len(g_values), len(steps)), np.nan, dtype=float)
        for i, g_value in enumerate(g_values):
            for j, step in enumerate(steps):
                vals = [
                    fnum(row.get("profile_cosine"))
                    for row in selected
                    if abs(fnum(row.get("g")) - g_value) <= 1e-12 and inum(row.get("step")) == step
                ]
                if vals:
                    matrix[i, j] = float(np.median(vals))
        im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(step) for step in steps])
        ax.set_yticks(range(len(g_values)))
        ax.set_yticklabels([f"{g:g}" for g in g_values])
        ax.set_xlabel("denoising step")
        ax.set_ylabel("g")
        ax.set_title(f"{args.query_scope} -> {key_type}")
        clean_axis(ax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("median profile cosine", labelpad=2)
        cbar.outline.set_linewidth(0.5)
    return save_figure(fig, output_dir, "diffusion_step_similarity_heatmaps", args.formats, args.dpi)


def plot_scope_comparison(data: Mapping[str, List[Dict[str, str]]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    rows = [
        row
        for row in data["metrics"]
        if inum(row.get("layer")) == -1 and inum(row.get("head")) == -1
    ]
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(180), mm_to_in(62)), constrained_layout=True, sharex=True)
    metrics = [("attention_to_g", "to g"), ("attention_to_P", "to P"), ("attention_to_b", "to b")]
    for ax, (metric, title) in zip(axes, metrics):
        for scope, color in [("masked_b", COLORS["blue"]), ("all_b", COLORS["orange"])]:
            scope_rows = [row for row in rows if row.get("query_scope") == scope]
            steps, med, q25, q75 = median_iqr_by_step(scope_rows, metric)
            ax.plot(steps, med, "o-", color=color, label=scope)
            ax.fill_between(steps, q25, q75, color=color, alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("denoising step")
        clean_axis(ax)
    axes[0].set_ylabel("attention mass")
    axes[-1].legend(frameon=False, handlelength=1.4)
    return save_figure(fig, output_dir, "diffusion_step_query_scope_comparison", args.formats, args.dpi)


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
        written.extend(plot_similarity_heatmaps(data, output_dir, args))
        written.extend(plot_scope_comparison(data, output_dir, args))
    for path in written:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
