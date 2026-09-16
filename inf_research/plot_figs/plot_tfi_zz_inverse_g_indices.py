"""Plot ZZ correlations versus 1/N for selected g indices.

Use either eval_new.py NPZ files for the original mean-value fit, or bootstrap
NPZ files for finite-size confidence intervals.  The two inputs are mutually
exclusive.

Bootstrap example:
  python inf_research/plot_zz_inverse_qubit_g0.3_g0.9.py \
    --bootstrap_files bootstrap_N8.npz bootstrap_N10.npz \
                      bootstrap_N12.npz bootstrap_N16.npz \
    --bootstrap_num_qubits 8 10 12 16 \
    --g-index 3 9 \
    --distances 1 \
    --output_dir zz_invN_g03_g09_figs

Eval example:
  python inf_research/plot_zz_inverse_qubit_g0.3_g0.9.py \
    --files result_N8.npz result_N10.npz result_N12.npz result_N16.npz \
    --num_qubits 8 10 12 16 \
    --g-index 3 9 \
    --distances 1
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

from plot_zz_inverse_qubit import (
    FIT_LINE_STYLES,
    NATURE_PALETTE,
    infer_num_qubits,
    import_matplotlib,
    load_all_results,
    select_distances,
    style_axis,
)
from bootstrap_inf import fit_bootstrap_intercepts


plt = None
POINT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BOOTSTRAP_PARAM_KEYS = ("params", "hs_gpt", "hs", "Js", "Deltas", "J2s", "alphas")
INFINITE_ANNOTATION_OFFSETS = {
    4: (10, 36),
    10: (6, 78),
}
DEFAULT_INFINITE_ANNOTATION_OFFSET = (10, 9)


@dataclass
class BootstrapZZResult:
    file_path: str
    num_qubits: int
    params: np.ndarray
    d_values: np.ndarray
    values: np.ndarray


@dataclass
class BootstrapFitRecord:
    source_mode: str
    distance: int
    g_index: int
    g_value: float
    zz_infinite: float
    ci_low: float
    ci_high: float
    bootstrap_mean: float
    bootstrap_median: float
    bootstrap_std: float
    slope: float
    r_squared: float
    fit_degree: int
    bootstrap_repetitions: int
    confidence: float
    pairing: str
    num_points: int
    qubits: str
    inverse_qubits: str
    zz_values: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot selected g-index ZZ finite-size curves together in one figure."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--files",
        nargs="+",
        help="eval_new.py output .npz files. Each file should correspond to one qubit size.",
    )
    parser.add_argument(
        "--num_qubits",
        nargs="+",
        type=int,
        default=None,
        help="Optional qubit sizes matching --files.",
    )
    input_group.add_argument(
        "--bootstrap_files",
        nargs="+",
        help=(
            "Per-N NPZ files from bootstrap_snapshot_observables.py. These files "
            "provide both the finite-N points and the bootstrap finite-size fits."
        ),
    )
    parser.add_argument(
        "--bootstrap_num_qubits",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional qubit sizes matching --bootstrap_files. Normally N is read "
            "from each NPZ or inferred from its filename."
        ),
    )
    parser.add_argument(
        "--bootstrap_key",
        default="bootstrap_zz",
        help="Bootstrap observable key. Default: bootstrap_zz.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Percentile confidence level for finite-N and N->infinity intervals.",
    )
    parser.add_argument(
        "--pairing",
        choices=["independent", "index"],
        default="independent",
        help=(
            "How bootstrap replicates are paired across N. This has the same "
            "semantics as bootstrap_finite_size_ci.py."
        ),
    )
    parser.add_argument(
        "--bootstrap_repetitions",
        "--B",
        dest="bootstrap_repetitions",
        type=int,
        default=None,
        help=(
            "Number of stored bootstrap replicates to use. By default all files "
            "must contain the same B and all replicates are used."
        ),
    )
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument(
        "--g-index",
        "--g_index",
        "--g_indices",
        dest="g_indices",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Indices along the saved parameter dimension to plot, e.g. "
            "--g-index 3 9."
        ),
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=[1],
        help="Correlation distances d to plot. Defaults to d=1.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="zz_invN_g03_g09_figs",
        help="Directory where the figure and fit CSV are saved.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional full output path for the figure.",
    )
    parser.add_argument(
        "--param_key",
        type=str,
        default="auto",
        help="Parameter key in .npz. Use auto for hs_gpt/Js/Deltas.",
    )
    parser.add_argument(
        "--zz_key",
        type=str,
        default="auto",
        help="ZZ mean key in .npz. Use auto for zz_mean/corr_ZZ_mean.",
    )
    parser.add_argument(
        "--zz_std_key",
        type=str,
        default="auto",
        help="ZZ std key in .npz. Use auto for zz_std/corr_ZZ_std.",
    )
    parser.add_argument(
        "--param_label",
        type=str,
        default="g",
        help="Label used in titles and legends for the scanned parameter.",
    )
    parser.add_argument(
        "--match_tol",
        type=float,
        default=1e-6,
        help=(
            "Tolerance used to verify that a selected g index represents the same "
            "saved parameter value in every N file."
        ),
    )
    parser.add_argument(
        "--fit_degree",
        type=int,
        default=1,
        help="Polynomial degree for fitting ZZ as a function of 1/N.",
    )
    parser.add_argument(
        "--fit_csv",
        type=str,
        default="zz_g03_g09_infinite_fit_values.csv",
        help="CSV filename for fitted N -> infinity ZZ values.",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Figure file format.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--fig_width", type=float, default=7.2, help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=5.2, help="Figure height in inches.")
    parser.add_argument(
        "--no_errorbar",
        action="store_true",
        help="Do not draw finite-N or N->infinity bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--annotate_n",
        action="store_true",
        help="Annotate each point with its qubit size N.",
    )
    infinite_annotation_group = parser.add_mutually_exclusive_group()
    infinite_annotation_group.add_argument(
        "--annotate_infinite",
        "--annotate-infinite",
        dest="annotate_infinite",
        action="store_true",
        help="Show the extrapolated value and CI beside the x=0 marker.",
    )
    infinite_annotation_group.add_argument(
        "--no_annotate_infinite",
        "--no-annotate-infinite",
        dest="annotate_infinite",
        action="store_false",
        help="Hide the extrapolated-value text beside the x=0 marker.",
    )
    parser.set_defaults(annotate_infinite=True)
    parser.add_argument(
        "--no_y_break",
        action="store_true",
        help="Disable automatic broken y-axis even when g=0.3 and g=0.9 ranges are separated.",
    )
    parser.add_argument(
        "--y_break_min_gap_fraction",
        type=float,
        default=0.05,
        help="Minimum empty y-gap fraction of total y range required to use a broken y-axis.",
    )
    parser.add_argument(
        "--y_pad_fraction",
        type=float,
        default=0.12,
        help="Padding added around each retained y-axis segment.",
    )
    return parser.parse_args()


def default_output_path(args):
    g_part = "_".join(f"gidx{int(index)}" for index in args.g_indices)
    d_part = "_".join(f"d{int(d)}" for d in args.distances)
    return os.path.join(args.output_dir, f"zz_vs_invN_{g_part}_{d_part}.{args.save_format}")


def load_bootstrap_results(args):
    if args.bootstrap_files is None:
        return {}
    if (
        args.bootstrap_num_qubits is not None
        and len(args.bootstrap_num_qubits) != len(args.bootstrap_files)
    ):
        raise ValueError(
            "--bootstrap_num_qubits must have the same length as --bootstrap_files."
        )

    results = {}
    for file_index, file_path in enumerate(args.bootstrap_files):
        explicit_n = (
            args.bootstrap_num_qubits[file_index]
            if args.bootstrap_num_qubits is not None
            else None
        )
        with np.load(file_path, allow_pickle=True) as data:
            if args.bootstrap_key not in data:
                raise KeyError(
                    f"{file_path}: bootstrap key '{args.bootstrap_key}' was not found."
                )
            param_key = next((key for key in BOOTSTRAP_PARAM_KEYS if key in data), None)
            if param_key is None:
                raise KeyError(
                    f"{file_path}: could not find a parameter array; tried "
                    f"{', '.join(BOOTSTRAP_PARAM_KEYS)}."
                )
            params = np.asarray(data[param_key], dtype=float).reshape(-1)
            values = np.asarray(data[args.bootstrap_key], dtype=float)
            if values.ndim == 2:
                values = values[:, np.newaxis, :]
            elif values.ndim != 3:
                raise ValueError(
                    f"{file_path}:{args.bootstrap_key} must have shape [B,H] or "
                    f"[B,D,H], got {values.shape}."
                )
            if values.shape[-1] != params.size:
                raise ValueError(
                    f"{file_path}:{args.bootstrap_key} parameter dimension "
                    f"{values.shape[-1]} != {params.size}."
                )
            if "d_values" in data:
                d_values = np.asarray(data["d_values"], dtype=int).reshape(-1)
                if d_values.size != values.shape[1]:
                    raise ValueError(
                        f"{file_path}: d_values has {d_values.size} entries but "
                        f"{args.bootstrap_key} has {values.shape[1]} distance rows."
                    )
            else:
                d_values = np.arange(1, values.shape[1] + 1, dtype=int)
            num_qubits = infer_num_qubits(file_path, data, explicit_n)

        if num_qubits in results:
            raise ValueError(
                f"Duplicate bootstrap qubit size N={num_qubits}: "
                f"{results[num_qubits].file_path} and {file_path}."
            )
        results[num_qubits] = BootstrapZZResult(
            file_path=file_path,
            num_qubits=num_qubits,
            params=params,
            d_values=d_values,
            values=values,
        )
    return results


def format_array(values):
    return ";".join(f"{float(value):.12g}" for value in np.asarray(values).reshape(-1))


def select_bootstrap_distances(bootstrap_results, requested_distances):
    available = sorted(
        {
            int(distance)
            for result in bootstrap_results.values()
            for distance in result.d_values
        }
    )
    distances = []
    for distance in requested_distances:
        value = int(distance)
        if value < 1:
            raise ValueError("--distances values must be >= 1.")
        if value not in distances:
            distances.append(value)
    missing = [distance for distance in distances if distance not in available]
    if missing:
        raise ValueError(
            f"Requested distances {missing} are absent from all bootstrap files. "
            f"Available distances: {available}."
        )
    return distances


def collect_eval_fit_series(results, g_index, distance, args):
    xs = []
    ys = []
    yerrs = []
    ns = []
    missing_ns = []
    selected_g_value = None

    for result in results:
        distance_matches = np.flatnonzero(result.d_values == int(distance))
        if distance_matches.size == 0:
            missing_ns.append(int(result.num_qubits))
            continue
        if g_index < 0 or g_index >= result.params.size:
            raise IndexError(
                f"{result.file_path}: g index {g_index} is outside the saved "
                f"parameter range [0, {result.params.size - 1}]."
            )
        file_g_value = float(result.params[g_index])
        if selected_g_value is None:
            selected_g_value = file_g_value
        elif abs(file_g_value - selected_g_value) > args.match_tol:
            raise ValueError(
                f"g index {g_index} is inconsistent across eval files: expected "
                f"{selected_g_value:.12g}, but {result.file_path} stores "
                f"{file_g_value:.12g}."
            )

        distance_index = int(distance_matches[0])
        xs.append(1.0 / float(result.num_qubits))
        ys.append(float(result.zz_mean[distance_index, g_index]))
        if not args.no_errorbar and result.zz_std is not None:
            yerrs.append(float(result.zz_std[distance_index, g_index]))
        else:
            yerrs.append(np.nan)
        ns.append(int(result.num_qubits))

    if len(xs) < args.fit_degree + 1:
        raise ValueError(
            f"Degree-{args.fit_degree} eval fit for g index {g_index}, "
            f"d={int(distance)} needs at least {args.fit_degree + 1} usable N "
            f"values, got {len(xs)}."
        )

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    yerr = np.asarray(yerrs, dtype=float)
    ns_array = np.asarray(ns, dtype=int)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]
    ns_array = ns_array[order]
    if np.any(np.isnan(yerr)):
        yerr = None

    coefficients = np.polyfit(x, y, args.fit_degree)
    fitted = np.polyval(coefficients, x)
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else float("nan")

    return {
        "x": x,
        "y": y,
        "yerr": yerr,
        "ns": ns_array,
        "g_value": float(selected_g_value),
        "coefficients": coefficients,
        "zz_infinite": float(coefficients[-1]),
        "slope": float(coefficients[-2]) if args.fit_degree >= 1 else 0.0,
        "r_squared": r_squared,
        "missing_ns": missing_ns,
    }


def collect_bootstrap_fit_series(
    bootstrap_results,
    eval_ns,
    g_index,
    distance,
    args,
    rng,
):
    selected_by_n = []
    missing_ns = []
    selected_g_value = None
    for num_qubits in sorted({int(value) for value in eval_ns}):
        result = bootstrap_results.get(num_qubits)
        if result is None:
            missing_ns.append(num_qubits)
            continue
        distance_matches = np.flatnonzero(result.d_values == int(distance))
        if distance_matches.size == 0:
            missing_ns.append(num_qubits)
            continue
        if g_index < 0 or g_index >= result.params.size:
            raise IndexError(
                f"{result.file_path}: g index {g_index} is outside the saved "
                f"parameter range [0, {result.params.size - 1}]."
            )
        file_g_value = float(result.params[g_index])
        if selected_g_value is None:
            selected_g_value = file_g_value
        elif abs(file_g_value - selected_g_value) > args.match_tol:
            raise ValueError(
                f"g index {g_index} is inconsistent across bootstrap files: "
                f"expected {selected_g_value:.12g}, but {result.file_path} stores "
                f"{file_g_value:.12g}."
            )
        samples = result.values[:, int(distance_matches[0]), g_index]
        selected_by_n.append((num_qubits, samples))

    if len(selected_by_n) < args.fit_degree + 1:
        raise ValueError(
            f"Degree-{args.fit_degree} bootstrap fit for g index {g_index}, "
            f"d={int(distance)} needs at least "
            f"{args.fit_degree + 1} usable N values, got {len(selected_by_n)}."
        )

    counts = [samples.shape[0] for _, samples in selected_by_n]
    if args.bootstrap_repetitions is None:
        if len(set(counts)) != 1:
            raise ValueError(
                f"Stored bootstrap counts differ for {args.bootstrap_key}: {counts}. "
                "Pass --B no larger than the minimum to select a common count."
            )
        bootstrap_count = counts[0]
    else:
        bootstrap_count = args.bootstrap_repetitions
        if bootstrap_count <= 0:
            raise ValueError("--bootstrap_repetitions/--B must be positive.")
        if bootstrap_count > min(counts):
            raise ValueError(
                f"Requested B={bootstrap_count} exceeds stored counts {counts}."
            )

    aligned = []
    for _, samples in selected_by_n:
        samples = samples[:bootstrap_count]
        if args.pairing == "independent":
            samples = samples[rng.permutation(bootstrap_count)]
        aligned.append(samples)

    # Match bootstrap_finite_size_ci.py exactly: [N, B, component, parameter].
    finite = np.stack(aligned, axis=0)[:, :, np.newaxis, np.newaxis]
    ns = np.asarray([num_qubits for num_qubits, _ in selected_by_n], dtype=int)
    inverse_sizes = 1.0 / ns.astype(float)
    intercepts, fit_of_mean, slopes, r2_values = fit_bootstrap_intercepts(
        inverse_sizes, finite, args.fit_degree
    )
    finite_mean = np.mean(finite[:, :, 0, 0], axis=1)
    coefficients = np.polyfit(inverse_sizes, finite_mean, args.fit_degree)

    alpha = (1.0 - args.confidence) / 2.0
    infinite_samples = intercepts[:, 0, 0]
    infinite_ci_low, infinite_ci_high = np.nanquantile(
        infinite_samples, [alpha, 1.0 - alpha]
    )
    finite_ci_low = np.nanquantile(finite[:, :, 0, 0], alpha, axis=1)
    finite_ci_high = np.nanquantile(finite[:, :, 0, 0], 1.0 - alpha, axis=1)

    return {
        "x": inverse_sizes,
        "y": finite_mean,
        "ns": ns,
        "g_value": float(selected_g_value),
        "finite_ci_low": finite_ci_low,
        "finite_ci_high": finite_ci_high,
        "coefficients": coefficients,
        "zz_infinite": float(fit_of_mean[0, 0]),
        "infinite_ci_low": float(infinite_ci_low),
        "infinite_ci_high": float(infinite_ci_high),
        "bootstrap_mean": float(np.nanmean(infinite_samples)),
        "bootstrap_median": float(np.nanmedian(infinite_samples)),
        "bootstrap_std": float(np.nanstd(infinite_samples, ddof=1))
        if bootstrap_count > 1
        else 0.0,
        "slope": float(slopes[0, 0]),
        "r_squared": float(r2_values[0, 0]),
        "bootstrap_count": bootstrap_count,
        "missing_ns": missing_ns,
    }


def update_range(values_by_g, g_value, *arrays):
    values = []
    for array in arrays:
        if array is None:
            continue
        arr = np.asarray(array, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if values:
        values_by_g.setdefault(float(g_value), []).append(np.concatenate(values))


def build_plot_items(results, bootstrap_results, g_indices, distances, args, rng):
    items = []
    fit_records = []
    values_by_g = {}
    use_bootstrap = bool(bootstrap_results)
    bootstrap_ns = sorted(bootstrap_results)

    for curve_index, g_index in enumerate(g_indices):
        color = NATURE_PALETTE[curve_index % len(NATURE_PALETTE)]
        for d_index, distance in enumerate(distances):
            if use_bootstrap:
                series = collect_bootstrap_fit_series(
                    bootstrap_results,
                    bootstrap_ns,
                    int(g_index),
                    int(distance),
                    args,
                    rng,
                )
            else:
                series = collect_eval_fit_series(
                    results,
                    int(g_index),
                    int(distance),
                    args,
                )
            g_value = series["g_value"]
            if series["missing_ns"]:
                source_name = "bootstrap" if use_bootstrap else "eval"
                print(
                    f"Warning: excluded N={series['missing_ns']} from the {source_name} fit "
                    f"for g index {int(g_index)} ({args.param_label}={g_value:g}), "
                    f"d={int(distance)} "
                    "because the requested distance is unavailable."
                )

            x = series["x"]
            y = series["y"]
            ns = series["ns"]
            if use_bootstrap:
                yerr = None
                ci_low = None if args.no_errorbar else series["finite_ci_low"]
                ci_high = None if args.no_errorbar else series["finite_ci_high"]
                infinite_ci_low = (
                    None if args.no_errorbar else series["infinite_ci_low"]
                )
                infinite_ci_high = (
                    None if args.no_errorbar else series["infinite_ci_high"]
                )
                record_ci_low = series["infinite_ci_low"]
                record_ci_high = series["infinite_ci_high"]
                bootstrap_mean = series["bootstrap_mean"]
                bootstrap_median = series["bootstrap_median"]
                bootstrap_std = series["bootstrap_std"]
                bootstrap_count = series["bootstrap_count"]
                confidence = args.confidence
                pairing = args.pairing
            else:
                yerr = series["yerr"]
                ci_low = None
                ci_high = None
                infinite_ci_low = None
                infinite_ci_high = None
                record_ci_low = float("nan")
                record_ci_high = float("nan")
                bootstrap_mean = float("nan")
                bootstrap_median = float("nan")
                bootstrap_std = float("nan")
                bootstrap_count = 0
                confidence = float("nan")
                pairing = ""

            fit_record = BootstrapFitRecord(
                source_mode="bootstrap" if use_bootstrap else "eval",
                distance=int(distance),
                g_index=int(g_index),
                g_value=g_value,
                zz_infinite=series["zz_infinite"],
                ci_low=record_ci_low,
                ci_high=record_ci_high,
                bootstrap_mean=bootstrap_mean,
                bootstrap_median=bootstrap_median,
                bootstrap_std=bootstrap_std,
                slope=series["slope"],
                r_squared=series["r_squared"],
                fit_degree=args.fit_degree,
                bootstrap_repetitions=bootstrap_count,
                confidence=confidence,
                pairing=pairing,
                num_points=len(x),
                qubits=format_array(ns),
                inverse_qubits=format_array(x),
                zz_values=format_array(y),
            )
            fit_records.append(fit_record)

            x_fit = np.linspace(0.0, float(np.max(x)), 240)
            y_fit = np.polyval(series["coefficients"], x_fit)
            marker = POINT_MARKERS[d_index % len(POINT_MARKERS)]
            linestyle = FIT_LINE_STYLES[d_index % len(FIT_LINE_STYLES)]
            series_label = rf"$g={g_value:.2f}$"
            if use_bootstrap:
                infinite_annotation = (
                    f"extrapolated = {fit_record.zz_infinite:.6g}\n"
                    f"{100.0 * fit_record.confidence:g}% CI = "
                    f"[{fit_record.ci_low:.6g}, {fit_record.ci_high:.6g}]"
                )
            else:
                infinite_annotation = f"extrapolated = {fit_record.zz_infinite:.6g}"

            infinite_range = [fit_record.zz_infinite]
            if infinite_ci_low is not None and infinite_ci_high is not None:
                infinite_range.extend([infinite_ci_low, infinite_ci_high])
            update_range(
                values_by_g,
                g_value,
                y,
                ci_low,
                ci_high,
                y_fit,
                infinite_range,
            )

            items.append(
                {
                    "g_index": int(g_index),
                    "g_value": g_value,
                    "distance": int(distance),
                    "x": x,
                    "y": y,
                    "yerr": yerr,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "infinite_ci_low": infinite_ci_low,
                    "infinite_ci_high": infinite_ci_high,
                    "ns": ns,
                    "x_fit": x_fit,
                    "y_fit": y_fit,
                    "zz_infinite": fit_record.zz_infinite,
                    "infinite_annotation": infinite_annotation,
                    "color": color,
                    "marker": marker,
                    "linestyle": linestyle,
                    "label": series_label,
                }
            )

    return items, fit_records, values_by_g


def expand_range(low, high, pad_fraction):
    span = float(high - low)
    if span <= 0:
        span = max(abs(float(low)), 1.0) * 0.02
    pad = span * float(pad_fraction)
    return float(low - pad), float(high + pad)


def choose_y_segments(values_by_g, args):
    if args.no_y_break or len(values_by_g) != 2:
        return None

    ranges = []
    for g_value, chunks in values_by_g.items():
        values = np.concatenate(chunks)
        ranges.append((float(np.min(values)), float(np.max(values)), float(g_value)))
    ranges.sort(key=lambda item: item[0])

    lower_min, lower_max, lower_g = ranges[0]
    upper_min, upper_max, upper_g = ranges[1]
    gap = upper_min - lower_max
    total_span = max(upper_max - lower_min, 1e-12)
    if gap <= args.y_break_min_gap_fraction * total_span:
        return None

    lower_ylim = expand_range(lower_min, lower_max, args.y_pad_fraction)
    upper_ylim = expand_range(upper_min, upper_max, args.y_pad_fraction)
    height_low = max(lower_ylim[1] - lower_ylim[0], 1e-12)
    height_high = max(upper_ylim[1] - upper_ylim[0], 1e-12)

    return {
        "lower_g": lower_g,
        "upper_g": upper_g,
        "lower_ylim": lower_ylim,
        "upper_ylim": upper_ylim,
        "height_ratios": [height_high, height_low],
    }


def draw_axis_break_marks(ax_top, ax_bottom):
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bottom.xaxis.tick_bottom()

    d = 0.012
    kwargs = dict(color="black", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)


def draw_item(
    ax,
    item,
    label=None,
    annotate_n=False,
    annotate_infinite=True,
):
    ci_low = item.get("ci_low")
    ci_high = item.get("ci_high")
    if ci_low is not None:
        finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
        if np.any(finite_ci):
            ci_mid = 0.5 * (ci_low[finite_ci] + ci_high[finite_ci])
            ax.errorbar(
                item["x"][finite_ci],
                ci_mid,
                yerr=np.vstack(
                    [ci_mid - ci_low[finite_ci], ci_high[finite_ci] - ci_mid]
                ),
                fmt="none",
                capsize=3,
                color=item["color"],
                ecolor=item["color"],
                linewidth=1.2,
                zorder=2,
            )
        ax.scatter(
            item["x"],
            item["y"],
            marker=item["marker"],
            s=34,
            color=item["color"],
            edgecolor="white",
            linewidths=0.6,
            zorder=3,
        )
    elif item["yerr"] is not None:
        ax.errorbar(
            item["x"],
            item["y"],
            yerr=item["yerr"],
            fmt=item["marker"],
            capsize=3,
            markersize=5.2,
            color=item["color"],
            ecolor=item["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            linestyle="none",
        )
    else:
        ax.scatter(
            item["x"],
            item["y"],
            marker=item["marker"],
            s=34,
            color=item["color"],
            edgecolor="white",
            linewidths=0.6,
        )

    ax.plot(
        item["x_fit"],
        item["y_fit"],
        linestyle=item["linestyle"],
        linewidth=2.0,
        color=item["color"],
        alpha=0.95,
        label=label,
    )
    infinite_ci_low = item.get("infinite_ci_low")
    infinite_ci_high = item.get("infinite_ci_high")
    if infinite_ci_low is not None and infinite_ci_high is not None:
        infinite_ci_mid = 0.5 * (infinite_ci_low + infinite_ci_high)
        ax.errorbar(
            [0.0],
            [infinite_ci_mid],
            yerr=np.asarray(
                [
                    [infinite_ci_mid - infinite_ci_low],
                    [infinite_ci_high - infinite_ci_mid],
                ]
            ),
            fmt="none",
            capsize=5,
            capthick=1.8,
            color=item["color"],
            ecolor=item["color"],
            linewidth=2.0,
            zorder=5,
        )
    ax.scatter(
        [0.0],
        [item["zz_infinite"]],
        marker="*",
        s=150,
        color=item["color"],
        edgecolor="black",
        linewidths=0.8,
        zorder=6,
    )
    if annotate_infinite:
        annotation_offset = INFINITE_ANNOTATION_OFFSETS.get(
            item["g_index"], DEFAULT_INFINITE_ANNOTATION_OFFSET
        )
        ax.annotate(
            item["infinite_annotation"],
            xy=(0.0, item["zz_infinite"]),
            xytext=annotation_offset,
            textcoords="offset points",
            ha="left" if annotation_offset[0] >= 0 else "right",
            va="bottom" if annotation_offset[1] >= 0 else "top",
            fontsize=8.5,
            color=item["color"],
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": item["color"],
                "alpha": 0.90,
                "linewidth": 0.8,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": item["color"],
                "linewidth": 0.8,
            },
            annotation_clip=True,
            zorder=7,
        )

    if annotate_n:
        y_low, y_high = sorted(ax.get_ylim())
        for xi, yi, n in zip(item["x"], item["y"], item["ns"]):
            if y_low <= yi <= y_high:
                ax.annotate(
                    f"N={n}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=8,
                )


def setup_common_axis(ax, args, *, xlabel=True, ylabel=True, title=True):
    if xlabel:
        ax.set_xlabel(r"$1/N$")
    if ylabel:
        ax.set_ylabel(r"$\langle Z_i Z_{i+r} \rangle$")
    # if title:
    #     title_text = "ZZ finite-size extrapolation"
    #     if args.bootstrap_files and not args.no_errorbar:
    #         title_text += f" ({100.0 * args.confidence:g}% bootstrap CI)"
    #     ax.set_title(title_text)
    right_limit = ax.get_xlim()[1]
    ax.set_xlim(left=-0.035 * right_limit)
    ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.9, zorder=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)


def add_deduplicated_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys(), loc='upper right',bbox_to_anchor=(0.94, 0.98))


def plot_combined(results, bootstrap_results, g_indices, distances, args, rng):
    items, fit_records, values_by_g = build_plot_items(
        results, bootstrap_results, g_indices, distances, args, rng
    )
    y_segments = choose_y_segments(values_by_g, args)

    if y_segments is None:
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        for item in items:
            draw_item(
                ax,
                item,
                label=item["label"],
                annotate_n=args.annotate_n,
                annotate_infinite=args.annotate_infinite,
            )
        setup_common_axis(ax, args)
        add_deduplicated_legend(ax)
        fig.tight_layout()
        return fig, fit_records

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(args.fig_width, args.fig_height),
        gridspec_kw={"height_ratios": y_segments["height_ratios"], "hspace": 0.06},
    )
    ax_top.set_ylim(*y_segments["upper_ylim"])
    ax_bottom.set_ylim(*y_segments["lower_ylim"])

    for item in items:
        draw_item(
            ax_top,
            item,
            label=item["label"],
            annotate_n=args.annotate_n,
            annotate_infinite=args.annotate_infinite,
        )
        draw_item(
            ax_bottom,
            item,
            label=None,
            annotate_n=args.annotate_n,
            annotate_infinite=args.annotate_infinite,
        )

    setup_common_axis(ax_top, args, xlabel=False, ylabel=False, title=True)
    setup_common_axis(ax_bottom, args, xlabel=True, ylabel=False, title=False)
    fig.text(0.02, 0.5, r"$\langle Z_i Z_{i+r} \rangle$", va="center", rotation="vertical")
    draw_axis_break_marks(ax_top, ax_bottom)
    add_deduplicated_legend(ax_top)
    fig.subplots_adjust(left=0.16, hspace=0.06)
    return fig, fit_records


def save_bootstrap_fit_records(fit_records, output_dir, csv_name):
    csv_path = os.path.join(output_dir, csv_name)
    fieldnames = [
        "source_mode",
        "distance",
        "g_index",
        "g_value",
        "fit_of_finite_mean",
        "ci_low",
        "ci_high",
        "bootstrap_mean",
        "bootstrap_median",
        "bootstrap_std",
        "confidence",
        "fit_degree",
        "central_slope",
        "central_r_squared",
        "bootstrap_repetitions",
        "pairing",
        "num_points",
        "qubits",
        "inverse_qubits",
        "finite_size_means",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in fit_records:
            writer.writerow(
                {
                    "source_mode": record.source_mode,
                    "distance": record.distance,
                    "g_index": record.g_index,
                    "g_value": f"{record.g_value:.12g}",
                    "fit_of_finite_mean": f"{record.zz_infinite:.12g}",
                    "ci_low": f"{record.ci_low:.12g}",
                    "ci_high": f"{record.ci_high:.12g}",
                    "bootstrap_mean": f"{record.bootstrap_mean:.12g}",
                    "bootstrap_median": f"{record.bootstrap_median:.12g}",
                    "bootstrap_std": f"{record.bootstrap_std:.12g}",
                    "confidence": f"{record.confidence:.12g}",
                    "fit_degree": record.fit_degree,
                    "central_slope": f"{record.slope:.12g}",
                    "central_r_squared": f"{record.r_squared:.12g}",
                    "bootstrap_repetitions": record.bootstrap_repetitions,
                    "pairing": record.pairing,
                    "num_points": record.num_points,
                    "qubits": record.qubits,
                    "inverse_qubits": record.inverse_qubits,
                    "finite_size_means": record.zz_values,
                }
            )
    return csv_path


def main():
    global plt
    args = parse_args()
    if args.fit_degree < 1:
        raise ValueError("--fit_degree must be at least 1.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between 0 and 1.")
    if args.match_tol < 0.0:
        raise ValueError("--match_tol must be non-negative.")
    if args.bootstrap_repetitions is not None and args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap_repetitions/--B must be positive.")
    if any(index < 0 for index in args.g_indices):
        raise ValueError("--g-index values must be non-negative.")
    plt = import_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.files is not None:
        if args.bootstrap_num_qubits is not None:
            raise ValueError(
                "--bootstrap_num_qubits can only be used with --bootstrap_files."
            )
        results = load_all_results(args)
        bootstrap_results = {}
        distances = select_distances(results, args.distances)
    else:
        if args.num_qubits is not None:
            raise ValueError("--num_qubits can only be used with --files.")
        results = []
        bootstrap_results = load_bootstrap_results(args)
        distances = select_bootstrap_distances(bootstrap_results, args.distances)

    g_indices = np.asarray(args.g_indices, dtype=int)
    rng = np.random.default_rng(args.seed)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 10,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig, fit_records = plot_combined(
        results, bootstrap_results, g_indices, distances, args, rng
    )
    save_path = args.output if args.output is not None else default_output_path(args)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    csv_path = save_bootstrap_fit_records(fit_records, args.output_dir, args.fit_csv)

    if bootstrap_results:
        bootstrap_ns_text = ", ".join(str(value) for value in sorted(bootstrap_results))
        print(
            f"Loaded bootstrap fit sizes N: {bootstrap_ns_text} "
            f"({100.0 * args.confidence:g}% percentile interval, "
            f"pairing={args.pairing})"
        )
    else:
        ns_text = ", ".join(str(result.num_qubits) for result in results)
        print(f"Loaded eval fit sizes N: {ns_text}")
    print("Fitted N -> infinity ZZ values:")
    for record in fit_records:
        prefix = (
            f"  d={record.distance}, g_index={record.g_index}, "
            f"{args.param_label}={record.g_value:g}: "
            f"zz_inf={record.zz_infinite:.12g}, "
        )
        if record.source_mode == "bootstrap":
            print(
                prefix
                + f"CI=[{record.ci_low:.12g}, {record.ci_high:.12g}], "
                + f"slope={record.slope:.12g}, R^2={record.r_squared:.6g}, "
                + f"B={record.bootstrap_repetitions}"
            )
        else:
            print(
                prefix
                + f"slope={record.slope:.12g}, R^2={record.r_squared:.6g}"
            )
    print(f"Fit values saved to: {csv_path}")
    print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    main()
