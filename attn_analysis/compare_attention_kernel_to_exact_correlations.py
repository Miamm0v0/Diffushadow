from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


CORRELATION_KEY_CANDIDATES = {
    "ZZ": ["ZZ_curves", "exact_corr_ZZ", "corr_ZZ_mean", "zz_mean", "exact_zz_correlation"],
    "XX": ["exact_corr_XX", "corr_XX_mean"],
    "YY": ["exact_corr_YY", "corr_YY_mean"],
    "spin_dot": ["exact_corr_spin_dot", "corr_spin_dot_mean"],
    "Xstring": ["Xs_curves", "xs_mean"],
}
PARAM_KEY_CANDIDATES = [
    "h_values",
    "h_values_true",
    "hs_gpt",
    "J_values_dense",
    "Js",
    "Delta_values_dense",
    "delta_values",
    "Deltas",
]


@dataclass
class ExactCorrelationData:
    file_path: str
    num_qubits: int
    params: np.ndarray
    values: np.ndarray
    correlation_key: str
    param_key: str


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1] / "attention_analysis" / "results" / "kernel_exact_correlations"
    parser = argparse.ArgumentParser(
        description=(
            "Compare attention site kernels with exact quantum correlation profiles. "
            "This joins K_{attention}(r, g) from attention_site_kernel.csv with "
            "C(r, g) from exact/eval .npz files."
        )
    )
    parser.add_argument(
        "--kernel_csv",
        required=True,
        help="attention_site_kernel.csv from analyze_attention_site_kernel_oseq_rope.py.",
    )
    parser.add_argument(
        "--exact_files",
        nargs="+",
        required=True,
        help=(
            "Exact/eval .npz files, for example DshadowGPT/eval_exact_cache/exact_TFI_N12.npz "
            "or eval_new.py output files containing ZZ_curves."
        ),
    )
    parser.add_argument(
        "--num_qubits",
        nargs="+",
        type=int,
        default=None,
        help="Optional N values matching --exact_files when N cannot be inferred.",
    )
    parser.add_argument(
        "--correlation",
        default="ZZ",
        choices=sorted(CORRELATION_KEY_CANDIDATES),
        help="Which exact correlation family to compare against.",
    )
    parser.add_argument(
        "--correlation_key",
        default="auto",
        help="Override .npz key for exact correlation values.",
    )
    parser.add_argument(
        "--param_key",
        default="auto",
        help="Override .npz key for scanned parameter values.",
    )
    parser.add_argument(
        "--query_group",
        default="b",
        help="Kernel query_group to keep, usually b.",
    )
    parser.add_argument(
        "--key_types",
        nargs="+",
        default=["b"],
        choices=["b", "P", "non_g"],
        help="Attention key types to compare with the exact correlation.",
    )
    parser.add_argument(
        "--attention_level",
        choices=["summary", "per_layer", "per_head"],
        default="per_head",
        help=(
            "summary uses layer=-1/head=-1; per_layer uses head=-1; "
            "per_head scans individual heads."
        ),
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=None,
        help="Physical distances r/d to compare. Defaults to distances available in both kernel and exact data.",
    )
    parser.add_argument(
        "--min_distance",
        type=int,
        default=1,
        help="Minimum site distance to include. Default excludes r=0 because exact ZZ starts at d=1.",
    )
    parser.add_argument(
        "--use_abs_correlation",
        action="store_true",
        help="Compare attention profiles with abs(C(r,g)) instead of signed C(r,g).",
    )
    parser.add_argument(
        "--normalize_kernel_profile",
        dest="normalize_kernel_profile",
        action="store_true",
        default=True,
        help="Normalize attention mass over selected distances before profile/length comparisons.",
    )
    parser.add_argument(
        "--no_normalize_kernel_profile",
        dest="normalize_kernel_profile",
        action="store_false",
        help="Use raw attention masses for profile/length comparisons.",
    )
    parser.add_argument(
        "--normalize_correlation_profile",
        dest="normalize_correlation_profile",
        action="store_true",
        default=True,
        help="Normalize absolute exact-correlation mass over selected distances for profile/length comparisons.",
    )
    parser.add_argument(
        "--no_normalize_correlation_profile",
        dest="normalize_correlation_profile",
        action="store_false",
        help="Use raw exact-correlation values for profile/length comparisons.",
    )
    parser.add_argument(
        "--g_match_tol",
        type=float,
        default=1e-6,
        help=(
            "Tolerance for nearest g matching, and endpoint tolerance for interpolation. "
            "The attention CSV usually has a sparse g grid while exact .npz files are dense."
        ),
    )
    parser.add_argument(
        "--g_match_mode",
        choices=["interp", "nearest"],
        default="interp",
        help="How to evaluate exact correlations at attention CSV g values.",
    )
    parser.add_argument("--min_points", type=int, default=4, help="Minimum g points for a correlation.")
    parser.add_argument("--top_k_plots", type=int, default=10, help="Number of top comparison scatter plots.")
    parser.add_argument("--profile_plot_g", nargs="+", type=float, default=None, help="Optional g values for profile overlay plots.")
    parser.add_argument("--output_dir", default=str(default_output), help="Output directory.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def infer_num_qubits(file_path: str, npz_data: Mapping[str, Any], explicit_n: Optional[int] = None) -> int:
    if explicit_n is not None:
        return int(explicit_n)
    if "N" in npz_data:
        return int(np.asarray(npz_data["N"]).reshape(-1)[0])

    name = os.path.basename(file_path)
    patterns = [
        r"(?i)(?:^|[_\-])N(\d+)(?:[_\-.]|$)",
        r"(?i)(?:^|[_\-])(\d+)q(?:ubit)?s?(?:[_\-.]|$)",
        r"(?i)(?:^|[_\-])qubits?[_\-]?(\d+)(?:[_\-.]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot infer num_qubits from {file_path}. Pass --num_qubits.")


def infer_key(data: Mapping[str, Any], requested_key: str, candidates: Sequence[str], field_name: str) -> str:
    if requested_key != "auto":
        if requested_key not in data:
            raise KeyError(f"{field_name} key {requested_key!r} not found.")
        return requested_key
    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Could not infer {field_name} key. Tried: {', '.join(candidates)}")


def as_2d(values: Any, key: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{key} must have shape (num_distances, num_params), got {array.shape}.")
    return array


def load_exact_file(file_path: str, args: argparse.Namespace, explicit_n: Optional[int]) -> ExactCorrelationData:
    data = np.load(file_path, allow_pickle=True)
    param_key = infer_key(data, args.param_key, PARAM_KEY_CANDIDATES, "parameter")
    corr_candidates = CORRELATION_KEY_CANDIDATES[args.correlation]
    corr_key = infer_key(data, args.correlation_key, corr_candidates, f"{args.correlation} correlation")

    params = np.asarray(data[param_key], dtype=float).reshape(-1)
    values = as_2d(data[corr_key], corr_key)
    if values.shape[1] != params.shape[0]:
        raise ValueError(
            f"{file_path}: {corr_key}.shape[1] ({values.shape[1]}) does not match "
            f"{param_key} length ({params.shape[0]})."
        )
    num_qubits = infer_num_qubits(file_path, data, explicit_n)
    return ExactCorrelationData(
        file_path=file_path,
        num_qubits=num_qubits,
        params=params,
        values=values,
        correlation_key=corr_key,
        param_key=param_key,
    )


def load_exact_files(args: argparse.Namespace) -> Dict[int, ExactCorrelationData]:
    if args.num_qubits is not None and len(args.num_qubits) != len(args.exact_files):
        raise ValueError("--num_qubits must have the same length as --exact_files.")

    out: Dict[int, ExactCorrelationData] = {}
    for idx, file_path in enumerate(args.exact_files):
        explicit_n = args.num_qubits[idx] if args.num_qubits is not None else None
        item = load_exact_file(file_path, args, explicit_n)
        if item.num_qubits in out:
            raise ValueError(f"Duplicate exact file for N={item.num_qubits}: {out[item.num_qubits].file_path} and {file_path}")
        out[item.num_qubits] = item
    return out


def kernel_level_matches(layer: int, head: int, attention_level: str) -> bool:
    if attention_level == "summary":
        return layer == -1 and head == -1
    if attention_level == "per_layer":
        return layer >= 0 and head == -1
    if attention_level == "per_head":
        return layer >= 0 and head >= 0
    raise ValueError(f"Unknown attention_level: {attention_level}")


def load_kernel_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw_rows = read_csv_rows(Path(args.kernel_csv))
    selected: List[Dict[str, Any]] = []
    for row in raw_rows:
        if row.get("query_group") != args.query_group:
            continue
        if row.get("key_type") not in set(args.key_types):
            continue
        delta_text = row.get("delta_site", "")
        if delta_text == "g":
            continue
        distance = abs(to_int(delta_text))
        if distance < args.min_distance:
            continue
        if args.distances is not None and distance not in set(args.distances):
            continue

        layer = to_int(row.get("layer"))
        head = to_int(row.get("head"))
        if not kernel_level_matches(layer, head, args.attention_level):
            continue

        selected.append(
            {
                "num_qubits": to_int(row.get("num_qubits")),
                "g": to_float(row.get("g")),
                "layer": layer,
                "head": head,
                "query_group": row.get("query_group", ""),
                "key_type": row.get("key_type", ""),
                "distance": distance,
                "attention_mass": to_float(row.get("attention_mass")),
            }
        )
    if not selected:
        raise ValueError("No kernel rows matched the requested filters.")
    return selected


def aggregate_kernel_abs_distance(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, float, int, int, str, int], float] = defaultdict(float)
    for row in rows:
        key = (
            int(row["num_qubits"]),
            float(row["g"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["key_type"]),
            int(row["distance"]),
        )
        grouped[key] += float(row["attention_mass"])

    out: List[Dict[str, Any]] = []
    for (num_qubits, g_value, layer, head, key_type, distance), mass in sorted(grouped.items()):
        out.append(
            {
                "num_qubits": num_qubits,
                "g": g_value,
                "layer": layer,
                "head": head,
                "key_type": key_type,
                "distance": distance,
                "attention_mass": mass,
            }
        )
    return out


def nearest_param_index(params: np.ndarray, target: float, tol: float) -> Optional[int]:
    distances = np.abs(params - float(target))
    index = int(np.argmin(distances))
    if distances[index] > tol:
        return None
    return index


def interp_param_value(params: np.ndarray, values: np.ndarray, target: float, tol: float) -> Optional[float]:
    finite = np.isfinite(params) & np.isfinite(values)
    params = params[finite]
    values = values[finite]
    if params.size < 2:
        return None

    order = np.argsort(params)
    params = params[order]
    values = values[order]
    target = float(target)
    if target < float(params[0]) - tol or target > float(params[-1]) + tol:
        return None
    target = min(max(target, float(params[0])), float(params[-1]))
    return float(np.interp(target, params, values))


def exact_value_at(
    exact: ExactCorrelationData,
    distance: int,
    g_value: float,
    args: argparse.Namespace,
) -> Optional[Tuple[float, float, str]]:
    if distance < 1 or distance > exact.values.shape[0]:
        return None

    if args.g_match_mode == "nearest":
        index = nearest_param_index(exact.params, g_value, args.g_match_tol)
        if index is None:
            return None
        return float(exact.values[distance - 1, index]), float(exact.params[index]), "nearest"

    value = interp_param_value(exact.params, exact.values[distance - 1], g_value, args.g_match_tol)
    if value is None:
        return None
    return value, float(g_value), "interp"


def join_kernel_exact(
    kernel_rows: Sequence[Mapping[str, Any]],
    exact_by_n: Mapping[int, ExactCorrelationData],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    joined: List[Dict[str, Any]] = []
    for row in kernel_rows:
        num_qubits = int(row["num_qubits"])
        if num_qubits not in exact_by_n:
            continue
        distance = int(row["distance"])
        exact = exact_by_n[num_qubits]
        lookup = exact_value_at(exact, distance, float(row["g"]), args)
        if lookup is None:
            continue
        corr_value, exact_parameter, parameter_match_mode = lookup
        joined.append(
            {
                "num_qubits": num_qubits,
                "g": float(row["g"]),
                "exact_parameter": exact_parameter,
                "parameter_match_mode": parameter_match_mode,
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "key_type": row["key_type"],
                "distance": distance,
                "attention_mass": float(row["attention_mass"]),
                "correlation": corr_value,
                "abs_correlation": abs(corr_value),
                "correlation_family": args.correlation,
                "correlation_key": exact.correlation_key,
                "param_key": exact.param_key,
                "exact_file": exact.file_path,
            }
        )
    if not joined:
        raise ValueError("No kernel rows could be joined with exact data. Check N/g grids and --g_match_tol.")
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
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return float("nan")
    return pearsonr(rankdata(x[finite]), rankdata(y[finite]))


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return float("nan")
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0 or not math.isfinite(total):
        return np.full_like(values, np.nan, dtype=float)
    return values / total


def compute_same_distance_correlations(joined_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str, int, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        key = (
            int(row["num_qubits"]),
            str(row["key_type"]),
            int(row["layer"]),
            int(row["head"]),
            int(row["distance"]),
        )
        groups[key].append(row)

    target_key = "abs_correlation" if args.use_abs_correlation else "correlation"
    out: List[Dict[str, Any]] = []
    for (num_qubits, key_type, layer, head, distance), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        rows = sorted(rows, key=lambda row: float(row["g"]))
        if len(rows) < args.min_points:
            continue
        x = np.asarray([float(row["attention_mass"]) for row in rows], dtype=float)
        y = np.asarray([float(row[target_key]) for row in rows], dtype=float)
        out.append(
            {
                "comparison": "same_distance_over_g",
                "num_qubits": num_qubits,
                "key_type": key_type,
                "layer": layer,
                "head": head,
                "distance": distance,
                "num_points": len(rows),
                "pearson": pearsonr(x, y),
                "spearman": spearmanr(x, y),
                "g_values": ";".join(f"{float(row['g']):.12g}" for row in rows),
            }
        )
    out.sort(key=lambda row: abs(float(row["pearson"])) if math.isfinite(float(row["pearson"])) else -1.0, reverse=True)
    return out


def profile_groups(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, float, str, int, int], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[int, float, str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        key = (
            int(row["num_qubits"]),
            float(row["g"]),
            str(row["key_type"]),
            int(row["layer"]),
            int(row["head"]),
        )
        groups[key].append(row)
    return groups


def compute_profile_rows(joined_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    target_key = "abs_correlation" if args.use_abs_correlation else "correlation"
    out: List[Dict[str, Any]] = []

    for (num_qubits, g_value, key_type, layer, head), rows in sorted(profile_groups(joined_rows).items(), key=lambda item: str(item[0])):
        rows = sorted(rows, key=lambda row: int(row["distance"]))
        if len(rows) < 2:
            continue
        distances = np.asarray([int(row["distance"]) for row in rows], dtype=float)
        attn = np.asarray([float(row["attention_mass"]) for row in rows], dtype=float)
        corr_signed = np.asarray([float(row["correlation"]) for row in rows], dtype=float)
        corr_for_profile = np.abs(corr_signed) if args.use_abs_correlation else corr_signed
        corr_mass = np.abs(corr_signed)

        attn_profile = normalize_nonnegative(attn) if args.normalize_kernel_profile else attn
        corr_profile = normalize_nonnegative(corr_mass) if args.normalize_correlation_profile else corr_for_profile

        attention_length = float(np.sum(distances * attn_profile)) if np.all(np.isfinite(attn_profile)) else float("nan")
        correlation_length_proxy = float(np.sum(distances * normalize_nonnegative(corr_mass))) if float(np.sum(corr_mass)) > 0 else float("nan")

        out.append(
            {
                "num_qubits": num_qubits,
                "g": g_value,
                "key_type": key_type,
                "layer": layer,
                "head": head,
                "num_distances": len(rows),
                "distances": ";".join(str(int(d)) for d in distances),
                "profile_pearson": pearsonr(attn_profile, corr_profile),
                "profile_spearman": spearmanr(attn_profile, corr_profile),
                "profile_cosine": cosine_similarity(attn_profile, corr_profile),
                "attention_length": attention_length,
                "correlation_length_proxy": correlation_length_proxy,
                "attention_profile": ";".join(f"{v:.12g}" for v in attn_profile),
                "correlation_profile": ";".join(f"{v:.12g}" for v in corr_profile),
                "signed_correlation_profile": ";".join(f"{v:.12g}" for v in corr_signed),
            }
        )
    return out


def compute_length_correlations(profile_rows: Sequence[Mapping[str, Any]], min_points: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        groups[(int(row["num_qubits"]), str(row["key_type"]), int(row["layer"]), int(row["head"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (num_qubits, key_type, layer, head), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        rows = sorted(rows, key=lambda row: float(row["g"]))
        if len(rows) < min_points:
            continue
        x = np.asarray([float(row["attention_length"]) for row in rows], dtype=float)
        y = np.asarray([float(row["correlation_length_proxy"]) for row in rows], dtype=float)
        out.append(
            {
                "comparison": "length_over_g",
                "num_qubits": num_qubits,
                "key_type": key_type,
                "layer": layer,
                "head": head,
                "num_points": len(rows),
                "pearson": pearsonr(x, y),
                "spearman": spearmanr(x, y),
                "g_values": ";".join(f"{float(row['g']):.12g}" for row in rows),
            }
        )
    out.sort(key=lambda row: abs(float(row["pearson"])) if math.isfinite(float(row["pearson"])) else -1.0, reverse=True)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_name(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_")


def plot_top_same_distance(
    joined_rows: Sequence[Mapping[str, Any]],
    corr_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")

    target_key = "abs_correlation" if args.use_abs_correlation else "correlation"
    paths: List[Path] = []
    for rank, corr in enumerate(corr_rows[: max(args.top_k_plots, 0)], start=1):
        rows = [
            row
            for row in joined_rows
            if int(row["num_qubits"]) == int(corr["num_qubits"])
            and row["key_type"] == corr["key_type"]
            and int(row["layer"]) == int(corr["layer"])
            and int(row["head"]) == int(corr["head"])
            and int(row["distance"]) == int(corr["distance"])
        ]
        rows = sorted(rows, key=lambda row: float(row["g"]))
        if len(rows) < 2:
            continue
        x = np.asarray([float(row["attention_mass"]) for row in rows], dtype=float)
        y = np.asarray([float(row[target_key]) for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.scatter(x, y, s=34, alpha=0.9)
        if len(np.unique(x)) >= 2:
            coeff = np.polyfit(x, y, 1)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(x_fit, np.polyval(coeff, x_fit), color="black", linewidth=1.2, alpha=0.75)
        for row, xi, yi in zip(rows, x, y):
            ax.annotate(f"{float(row['g']):g}", (xi, yi), textcoords="offset points", xytext=(3, 4), fontsize=8)
        ax.set_xlabel(f"K_{corr['key_type']}(r={corr['distance']}) attention mass")
        ax.set_ylabel(("abs " if args.use_abs_correlation else "") + args.correlation)
        ax.set_title(
            f"rank {rank}: N={corr['num_qubits']}, L{corr['layer']}H{corr['head']}, "
            f"r={corr['distance']}, pearson={float(corr['pearson']):.3f}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / (
            f"top_same_distance_{rank:02d}_N{corr['num_qubits']}_{corr['key_type']}_"
            f"L{corr['layer']}H{corr['head']}_r{corr['distance']}.png"
        )
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_length_correlations(
    profile_rows: Sequence[Mapping[str, Any]],
    length_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")

    paths: List[Path] = []
    for rank, corr in enumerate(length_rows[: max(args.top_k_plots, 0)], start=1):
        rows = [
            row
            for row in profile_rows
            if int(row["num_qubits"]) == int(corr["num_qubits"])
            and row["key_type"] == corr["key_type"]
            and int(row["layer"]) == int(corr["layer"])
            and int(row["head"]) == int(corr["head"])
        ]
        rows = sorted(rows, key=lambda row: float(row["g"]))
        if len(rows) < 2:
            continue
        x = np.asarray([float(row["attention_length"]) for row in rows], dtype=float)
        y = np.asarray([float(row["correlation_length_proxy"]) for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        ax.scatter(x, y, s=34, alpha=0.9)
        if len(np.unique(x)) >= 2:
            coeff = np.polyfit(x, y, 1)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(x_fit, np.polyval(coeff, x_fit), color="black", linewidth=1.2, alpha=0.75)
        for row, xi, yi in zip(rows, x, y):
            ax.annotate(f"{float(row['g']):g}", (xi, yi), textcoords="offset points", xytext=(3, 4), fontsize=8)
        ax.set_xlabel("attention length")
        ax.set_ylabel(f"{args.correlation} length proxy")
        ax.set_title(
            f"length rank {rank}: N={corr['num_qubits']}, L{corr['layer']}H{corr['head']}, "
            f"pearson={float(corr['pearson']):.3f}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / (
            f"top_length_{rank:02d}_N{corr['num_qubits']}_{corr['key_type']}_"
            f"L{corr['layer']}H{corr['head']}.png"
        )
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def choose_profile_g_values(args: argparse.Namespace, profile_rows: Sequence[Mapping[str, Any]]) -> List[float]:
    available = sorted({float(row["g"]) for row in profile_rows})
    if args.profile_plot_g is None:
        if len(available) <= 3:
            return available
        return [available[0], available[len(available) // 2], available[-1]]
    chosen: List[float] = []
    for requested in args.profile_plot_g:
        diffs = np.abs(np.asarray(available, dtype=float) - float(requested))
        index = int(np.argmin(diffs))
        if diffs[index] <= args.g_match_tol:
            chosen.append(available[index])
    return sorted(set(chosen))


def profile_score(rows: Sequence[Mapping[str, Any]]) -> float:
    scores = [
        abs(float(row["profile_cosine"]))
        for row in rows
        if math.isfinite(float(row["profile_cosine"]))
    ]
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def choose_profile_overlay_rows(
    profile_rows: Sequence[Mapping[str, Any]],
    chosen_g: Sequence[float],
) -> List[Mapping[str, Any]]:
    chosen_g_set = set(chosen_g)

    summary_rows = [
        row
        for row in profile_rows
        if int(row["layer"]) == -1 and int(row["head"]) == -1 and float(row["g"]) in chosen_g_set
    ]
    if summary_rows:
        return summary_rows

    grouped: Dict[Tuple[int, str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        if float(row["g"]) not in chosen_g_set:
            continue
        key = (int(row["num_qubits"]), str(row["key_type"]), int(row["layer"]), int(row["head"]))
        grouped[key].append(row)

    best_by_qubit_key: Dict[Tuple[int, str], Tuple[int, float, Tuple[int, str, int, int]]] = {}
    for group_key, rows in grouped.items():
        num_qubits, key_type, _, _ = group_key
        coverage = len({float(row["g"]) for row in rows})
        score = profile_score(rows)
        selector = (coverage, score, group_key)
        short_key = (num_qubits, key_type)
        if short_key not in best_by_qubit_key or selector > best_by_qubit_key[short_key]:
            best_by_qubit_key[short_key] = selector

    selected: List[Mapping[str, Any]] = []
    for _, _, group_key in sorted(best_by_qubit_key.values(), key=lambda item: item[2]):
        selected.extend(grouped[group_key])
    return selected


def plot_profile_overlays(
    profile_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")

    chosen_g = choose_profile_g_values(args, profile_rows)
    if not chosen_g:
        return []

    candidates = choose_profile_overlay_rows(profile_rows, chosen_g)

    paths: List[Path] = []
    grouped: Dict[Tuple[int, str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(int(row["num_qubits"]), str(row["key_type"]), int(row["layer"]), int(row["head"]))].append(row)

    for (num_qubits, key_type, layer, head), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: float(row["g"]))
        fig, axes = plt.subplots(1, len(rows), figsize=(4.4 * len(rows), 3.6), sharey=True)
        if len(rows) == 1:
            axes = [axes]
        for ax, row in zip(axes, rows):
            distances = np.asarray([float(x) for x in str(row["distances"]).split(";") if x != ""], dtype=float)
            attn = np.asarray([float(x) for x in str(row["attention_profile"]).split(";") if x != ""], dtype=float)
            corr = np.asarray([float(x) for x in str(row["correlation_profile"]).split(";") if x != ""], dtype=float)
            ax.plot(distances, attn, "o-", label="attention")
            ax.plot(distances, corr, "s--", label=args.correlation)
            ax.set_xlabel("distance")
            ax.set_title(f"g={float(row['g']):g}")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("normalized profile")
        axes[-1].legend(frameon=False)
        fig.suptitle(f"N={num_qubits}, {key_type}, L{layer}H{head}", y=1.04)
        fig.tight_layout()
        path = output_dir / f"profile_overlay_N{num_qubits}_{key_type}_L{layer}H{head}.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    exact_by_n = load_exact_files(args)
    kernel_rows = aggregate_kernel_abs_distance(load_kernel_rows(args))
    joined_rows = join_kernel_exact(kernel_rows, exact_by_n, args)
    same_distance_rows = compute_same_distance_correlations(joined_rows, args)
    profile_rows = compute_profile_rows(joined_rows, args)
    length_rows = compute_length_correlations(profile_rows, args.min_points)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joined_path = output_dir / "kernel_exact_joined.csv"
    same_path = output_dir / "kernel_exact_same_distance_correlations.csv"
    profile_path = output_dir / "kernel_exact_profile_similarity.csv"
    length_path = output_dir / "kernel_exact_length_correlations.csv"

    write_csv(
        joined_path,
        joined_rows,
        [
            "num_qubits",
            "g",
            "exact_parameter",
            "parameter_match_mode",
            "layer",
            "head",
            "key_type",
            "distance",
            "attention_mass",
            "correlation",
            "abs_correlation",
            "correlation_family",
            "correlation_key",
            "param_key",
            "exact_file",
        ],
    )
    write_csv(
        same_path,
        same_distance_rows,
        [
            "comparison",
            "num_qubits",
            "key_type",
            "layer",
            "head",
            "distance",
            "num_points",
            "pearson",
            "spearman",
            "g_values",
        ],
    )
    write_csv(
        profile_path,
        profile_rows,
        [
            "num_qubits",
            "g",
            "key_type",
            "layer",
            "head",
            "num_distances",
            "distances",
            "profile_pearson",
            "profile_spearman",
            "profile_cosine",
            "attention_length",
            "correlation_length_proxy",
            "attention_profile",
            "correlation_profile",
            "signed_correlation_profile",
        ],
    )
    write_csv(
        length_path,
        length_rows,
        [
            "comparison",
            "num_qubits",
            "key_type",
            "layer",
            "head",
            "num_points",
            "pearson",
            "spearman",
            "g_values",
        ],
    )

    print(f"Wrote joined kernel/exact rows: {joined_path}")
    print(f"Wrote same-distance correlations: {same_path}")
    print(f"Wrote profile similarities: {profile_path}")
    print(f"Wrote length correlations: {length_path}")

    if same_distance_rows:
        print("Strongest same-distance K(r,g) vs C(r,g) correlations:")
        for row in same_distance_rows[: min(10, len(same_distance_rows))]:
            print(
                f"  N={row['num_qubits']}, {row['key_type']}, "
                f"L{row['layer']}H{row['head']}, r={row['distance']}: "
                f"pearson={float(row['pearson']):.4g}, spearman={float(row['spearman']):.4g}"
            )
    if length_rows:
        print("Strongest attention-length vs correlation-length correlations:")
        for row in length_rows[: min(10, len(length_rows))]:
            print(
                f"  N={row['num_qubits']}, {row['key_type']}, "
                f"L{row['layer']}H{row['head']}: "
                f"pearson={float(row['pearson']):.4g}, spearman={float(row['spearman']):.4g}"
            )

    if not args.no_plots:
        for path in plot_top_same_distance(joined_rows, same_distance_rows, output_dir, args):
            print(f"Wrote figure: {path}")
        for path in plot_length_correlations(profile_rows, length_rows, output_dir, args):
            print(f"Wrote figure: {path}")
        for path in plot_profile_overlays(profile_rows, output_dir, args):
            print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
