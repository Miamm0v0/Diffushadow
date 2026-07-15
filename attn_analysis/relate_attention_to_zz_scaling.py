from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_METRICS = [
    "mean_site_distance",
    "mean_site_distance_norm",
    "attention_to_g",
    "attention_to_P",
    "attention_to_b",
    "attention_to_near_site_1",
    "entropy_norm",
]
TARGETS = ["zz_infinite", "abs_slope", "r_squared"]


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1] / "attention_analysis" / "results" / "attention_zz_scaling"
    parser = argparse.ArgumentParser(
        description=(
            "Join attention metrics with ZZ finite-size scaling fits and rank which "
            "attention quantities track extrapolated physics."
        )
    )
    parser.add_argument(
        "--attention_csv",
        required=True,
        help="attention_distance_metrics.csv from analyze_attention_distance_oseq_rope.py.",
    )
    parser.add_argument(
        "--fit_csv",
        required=True,
        help=(
            "CSV from plot_zz_vs_inverse_qubit.py or plot_zz_extrapolated_vs_g.py. "
            "Requires distance, g_value, zz_infinite columns."
        ),
    )
    parser.add_argument(
        "--query_group",
        default="b",
        help="Attention query_group to use, usually b.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Attention metric columns to correlate with ZZ scaling values.",
    )
    parser.add_argument(
        "--attention_level",
        choices=["summary", "per_layer", "per_head"],
        default="per_head",
        help=(
            "summary uses layer=-1/head=-1; per_layer uses head=-1 for each layer; "
            "per_head scans individual heads."
        ),
    )
    parser.add_argument(
        "--aggregate_qubits",
        action="store_true",
        help="Average attention metrics across qubit sizes before correlating over g.",
    )
    parser.add_argument(
        "--num_qubits",
        nargs="+",
        type=int,
        default=None,
        help="Optional attention N values to keep before optional aggregation.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=None,
        help="Optional ZZ distances d to keep.",
    )
    parser.add_argument(
        "--g_match_tol",
        type=float,
        default=1e-6,
        help="Tolerance when matching attention g values to fit_csv g_value.",
    )
    parser.add_argument("--min_points", type=int, default=4, help="Minimum g points for a correlation.")
    parser.add_argument("--top_k_plots", type=int, default=8, help="Number of strongest correlations to plot.")
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def to_float(row: Mapping[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(row: Mapping[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return int(float(value))


def select_attention_rows(rows: Sequence[Mapping[str, str]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    keep_n = set(args.num_qubits) if args.num_qubits is not None else None
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("query_group") != args.query_group:
            continue
        layer = to_int(row, "layer")
        head = to_int(row, "head")
        if args.attention_level == "summary" and not (layer == -1 and head == -1):
            continue
        if args.attention_level == "per_layer" and not (layer >= 0 and head == -1):
            continue
        if args.attention_level == "per_head" and not (layer >= 0 and head >= 0):
            continue

        num_qubits = to_int(row, "num_qubits")
        if keep_n is not None and num_qubits not in keep_n:
            continue

        typed: Dict[str, Any] = {
            "num_qubits": num_qubits,
            "g": to_float(row, "g"),
            "layer": layer,
            "head": head,
            "query_group": row.get("query_group", ""),
        }
        for metric in args.metrics:
            typed[metric] = to_float(row, metric)
        selected.append(typed)
    if not selected:
        raise ValueError("No attention rows matched the requested filters.")
    return selected


def aggregate_attention_qubits(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["g"]), int(row["layer"]), int(row["head"]), str(row["query_group"]))].append(row)

    aggregated: List[Dict[str, Any]] = []
    for (g_value, layer, head, query_group), group_rows in sorted(groups.items()):
        out: Dict[str, Any] = {
            "num_qubits": "mean",
            "g": g_value,
            "layer": layer,
            "head": head,
            "query_group": query_group,
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=float)
            out[metric] = float(np.nanmean(values))
        aggregated.append(out)
    return aggregated


def load_fit_rows(path: Path, distances: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    raw_rows = read_csv_rows(path)
    required = {"distance", "g_value", "zz_infinite"}
    if raw_rows:
        missing = required.difference(raw_rows[0].keys())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    keep_distances = set(distances) if distances is not None else None
    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        distance = int(float(row["distance"]))
        if keep_distances is not None and distance not in keep_distances:
            continue
        slope = to_float(row, "slope")
        out = {
            "distance": distance,
            "g_value": to_float(row, "g_value"),
            "zz_infinite": to_float(row, "zz_infinite"),
            "slope": slope,
            "abs_slope": abs(slope) if math.isfinite(slope) else float("nan"),
            "r_squared": to_float(row, "r_squared"),
            "fit_degree": to_int(row, "fit_degree", default=0),
            "fit_num_points": to_int(row, "num_points", default=0),
            "fit_qubits": row.get("qubits", ""),
        }
        rows.append(out)
    if not rows:
        raise ValueError("No fit rows matched the requested filters.")
    return rows


def nearest_fit_by_g(fit_rows: Sequence[Mapping[str, Any]], g_value: float, tol: float) -> List[Mapping[str, Any]]:
    by_distance: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in fit_rows:
        by_distance[int(row["distance"])].append(row)

    matched: List[Mapping[str, Any]] = []
    for distance, rows in by_distance.items():
        diffs = np.asarray([abs(float(row["g_value"]) - g_value) for row in rows], dtype=float)
        index = int(np.argmin(diffs))
        if diffs[index] <= tol:
            matched.append(rows[index])
    return matched


def join_attention_and_fit(
    attention_rows: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    g_match_tol: float,
) -> List[Dict[str, Any]]:
    joined: List[Dict[str, Any]] = []
    for attn_row in attention_rows:
        g_value = float(attn_row["g"])
        matched_fit_rows = nearest_fit_by_g(fit_rows, g_value, g_match_tol)
        for fit_row in matched_fit_rows:
            for metric in metrics:
                metric_value = float(attn_row[metric])
                if not math.isfinite(metric_value):
                    continue
                out = {
                    "distance": int(fit_row["distance"]),
                    "g": g_value,
                    "attention_num_qubits": attn_row["num_qubits"],
                    "layer": int(attn_row["layer"]),
                    "head": int(attn_row["head"]),
                    "query_group": attn_row["query_group"],
                    "metric": metric,
                    "metric_value": metric_value,
                    "zz_infinite": float(fit_row["zz_infinite"]),
                    "slope": float(fit_row["slope"]),
                    "abs_slope": float(fit_row["abs_slope"]),
                    "r_squared": float(fit_row["r_squared"]),
                    "fit_degree": int(fit_row["fit_degree"]),
                    "fit_num_points": int(fit_row["fit_num_points"]),
                    "fit_qubits": fit_row["fit_qubits"],
                }
                joined.append(out)
    if not joined:
        raise ValueError("No rows could be joined. Check g grids and --g_match_tol.")
    return joined


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(rankdata(x), rankdata(y))


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size < 2 or len(np.unique(x)) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def compute_correlations(joined_rows: Sequence[Mapping[str, Any]], min_points: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, Any, int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        key = (
            int(row["distance"]),
            row["attention_num_qubits"],
            int(row["layer"]),
            int(row["head"]),
            str(row["metric"]),
        )
        groups[key].append(row)

    corr_rows: List[Dict[str, Any]] = []
    for (distance, num_qubits, layer, head, metric), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        rows = sorted(rows, key=lambda row: float(row["g"]))
        x_all = np.asarray([float(row["metric_value"]) for row in rows], dtype=float)
        g_values = ";".join(f"{float(row['g']):.12g}" for row in rows)
        for target in TARGETS:
            y_all = np.asarray([float(row[target]) for row in rows], dtype=float)
            finite = np.isfinite(x_all) & np.isfinite(y_all)
            x = x_all[finite]
            y = y_all[finite]
            if x.size < min_points:
                continue
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            slope, intercept = linear_fit(x, y)
            corr_rows.append(
                {
                    "distance": distance,
                    "attention_num_qubits": num_qubits,
                    "layer": layer,
                    "head": head,
                    "metric": metric,
                    "target": target,
                    "num_points": int(x.size),
                    "pearson": pearson,
                    "spearman": spearman,
                    "linear_slope": slope,
                    "linear_intercept": intercept,
                    "g_values": g_values,
                }
            )
    corr_rows.sort(key=lambda row: abs(float(row["pearson"])) if math.isfinite(float(row["pearson"])) else -1.0, reverse=True)
    return corr_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def group_joined_for_plot(
    joined_rows: Sequence[Mapping[str, Any]],
    corr_row: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    return [
        row
        for row in joined_rows
        if int(row["distance"]) == int(corr_row["distance"])
        and str(row["attention_num_qubits"]) == str(corr_row["attention_num_qubits"])
        and int(row["layer"]) == int(corr_row["layer"])
        and int(row["head"]) == int(corr_row["head"])
        and str(row["metric"]) == str(corr_row["metric"])
        and math.isfinite(float(row[str(corr_row["target"])]))
    ]


def safe_name(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_")


def plot_top_correlations(
    joined_rows: Sequence[Mapping[str, Any]],
    corr_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    top_k: int,
    dpi: int,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")

    paths: List[Path] = []
    for rank, corr_row in enumerate(corr_rows[: max(0, top_k)], start=1):
        rows = group_joined_for_plot(joined_rows, corr_row)
        if len(rows) < 2:
            continue
        metric = str(corr_row["metric"])
        target = str(corr_row["target"])
        x = np.asarray([float(row["metric_value"]) for row in rows], dtype=float)
        y = np.asarray([float(row[target]) for row in rows], dtype=float)
        g = [float(row["g"]) for row in rows]

        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.scatter(x, y, s=34, alpha=0.9)
        if len(np.unique(x)) >= 2:
            slope, intercept = linear_fit(x, y)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(x_fit, slope * x_fit + intercept, color="black", linewidth=1.3, alpha=0.75)
        for xi, yi, gi in zip(x, y, g):
            ax.annotate(f"{gi:g}", (xi, yi), textcoords="offset points", xytext=(3, 4), fontsize=8)

        ax.set_xlabel(metric)
        ax.set_ylabel(target)
        ax.set_title(
            f"rank {rank}: d={corr_row['distance']}, N={corr_row['attention_num_qubits']}, "
            f"L{corr_row['layer']}H{corr_row['head']}, r={float(corr_row['pearson']):.3f}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        filename = (
            f"top{rank:02d}_d{corr_row['distance']}_N{safe_name(corr_row['attention_num_qubits'])}_"
            f"L{corr_row['layer']}H{corr_row['head']}_{metric}_vs_{target}.png"
        )
        path = output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    attention_rows = select_attention_rows(read_csv_rows(Path(args.attention_csv)), args)
    if args.aggregate_qubits:
        attention_rows = aggregate_attention_qubits(attention_rows, args.metrics)
    fit_rows = load_fit_rows(Path(args.fit_csv), args.distances)
    joined_rows = join_attention_and_fit(attention_rows, fit_rows, args.metrics, args.g_match_tol)
    corr_rows = compute_correlations(joined_rows, args.min_points)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = output_dir / "attention_zz_joined_long.csv"
    corr_path = output_dir / "attention_zz_correlations.csv"
    write_csv(
        joined_path,
        joined_rows,
        [
            "distance",
            "g",
            "attention_num_qubits",
            "layer",
            "head",
            "query_group",
            "metric",
            "metric_value",
            "zz_infinite",
            "slope",
            "abs_slope",
            "r_squared",
            "fit_degree",
            "fit_num_points",
            "fit_qubits",
        ],
    )
    write_csv(
        corr_path,
        corr_rows,
        [
            "distance",
            "attention_num_qubits",
            "layer",
            "head",
            "metric",
            "target",
            "num_points",
            "pearson",
            "spearman",
            "linear_slope",
            "linear_intercept",
            "g_values",
        ],
    )
    print(f"Wrote joined rows: {joined_path}")
    print(f"Wrote correlations: {corr_path}")

    if corr_rows:
        print("Strongest attention/ZZ-scaling correlations:")
        for row in corr_rows[: min(10, len(corr_rows))]:
            print(
                f"  d={row['distance']}, N={row['attention_num_qubits']}, "
                f"L{row['layer']}H{row['head']}, {row['metric']} vs {row['target']}: "
                f"pearson={float(row['pearson']):.4g}, spearman={float(row['spearman']):.4g}"
            )
    else:
        print("No correlations met --min_points.")

    if not args.no_plots and corr_rows and args.top_k_plots > 0:
        for path in plot_top_correlations(joined_rows, corr_rows, output_dir, args.top_k_plots, args.dpi):
            print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
