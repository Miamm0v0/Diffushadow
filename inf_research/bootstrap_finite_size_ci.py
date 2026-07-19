#!/usr/bin/env python3
"""Finite-size extrapolation with snapshot-bootstrap confidence intervals.

This is the uncertainty-propagating companion to the existing ``inf_research``
plot scripts.  For every bootstrap replicate it fits the finite-N observable as
a polynomial in 1/N, exactly as those scripts do with ``numpy.polyfit``.  The
constant coefficient is the N -> infinity estimate.  Percentiles of all fitted
intercepts form the reported confidence interval.

Inputs are the per-size NPZ files produced by
``Diffushadow_run/bootstrap_snapshot_observables.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PARAM_KEYS = ("params", "hs_gpt", "hs", "Js", "Deltas", "J2s", "alphas")


@dataclass
class BootstrapResult:
    path: str
    num_qubits: int
    params: np.ndarray
    d_values: Optional[np.ndarray]
    arrays: Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit bootstrap physical observables versus 1/N and report N->infinity "
            "percentile confidence intervals."
        )
    )
    parser.add_argument("--files", nargs="+", required=True, help="One bootstrap NPZ per N.")
    parser.add_argument(
        "--num_qubits",
        nargs="+",
        type=int,
        default=None,
        help="Optional N values matching --files; normally read from each NPZ.",
    )
    parser.add_argument(
        "--observable_keys",
        nargs="+",
        default=None,
        help=(
            "Bootstrap keys to fit, e.g. bootstrap_zz bootstrap_corr_spin_dot. "
            "Default: all common bootstrap_* arrays with shape [B,H] or [B,D,H]."
        ),
    )
    parser.add_argument(
        "--parameter_values",
        "--g_values",
        dest="parameter_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of scan values. Default: all values in the first file.",
    )
    parser.add_argument(
        "--components",
        "--distances",
        dest="components",
        nargs="+",
        type=int,
        default=None,
        help="Optional d/component values for [B,D,H] observables.",
    )
    parser.add_argument("--fit_degree", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--match_tol", type=float, default=1e-6)
    parser.add_argument(
        "--pairing",
        choices=["independent", "index"],
        default="independent",
        help=(
            "independent randomly permutes bootstrap replicates at every N before "
            "forming a finite-size fit; index preserves stored replicate indices."
        ),
    )
    parser.add_argument(
        "--bootstrap_repetitions",
        "--B",
        dest="bootstrap_repetitions",
        type=int,
        default=None,
        help="Optional number of stored replicates to use. Default: require and use all.",
    )
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--output_dir", default="bootstrap_finite_size_results")
    parser.add_argument("--output_prefix", default="bootstrap_infinite")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save N->infinity curves with confidence bands.",
    )
    parser.add_argument("--param_label", default="parameter")
    parser.add_argument("--save_format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def first_existing(data, candidates: Sequence[str], description: str) -> str:
    for key in candidates:
        if key in data:
            return key
    raise KeyError("Could not find {}. Tried: {}".format(description, ", ".join(candidates)))


def infer_num_qubits(path: str, data, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    if "N" in data:
        return int(np.asarray(data["N"]).reshape(-1)[0])
    basename = os.path.basename(path)
    patterns = (
        r"(?i)(?:^|[_-])N(\d+)(?:[_\-.]|$)",
        r"(?i)(?:^|[_-])(\d+)q(?:ubit)?s?(?:[_\-.]|$)",
        r"(?i)(?:^|[_-])qubits?[_-]?(\d+)(?:[_\-.]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            return int(match.group(1))
    raise ValueError("Cannot infer N for '{}'; pass --num_qubits.".format(path))


def discover_observable_keys(paths: Sequence[str]) -> List[str]:
    common: Optional[set[str]] = None
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            keys = {
                key
                for key in data.files
                if key.startswith("bootstrap_")
                and np.asarray(data[key]).ndim in (2, 3)
            }
        common = keys if common is None else common.intersection(keys)
    if not common:
        raise ValueError("No common [B,H] or [B,D,H] bootstrap_* arrays were found.")
    return sorted(common)


def load_results(args: argparse.Namespace, observable_keys: Sequence[str]) -> List[BootstrapResult]:
    if args.num_qubits is not None and len(args.num_qubits) != len(args.files):
        raise ValueError("--num_qubits must have the same length as --files.")

    results: List[BootstrapResult] = []
    for index, path in enumerate(args.files):
        explicit_n = args.num_qubits[index] if args.num_qubits is not None else None
        with np.load(path, allow_pickle=True) as data:
            param_key = first_existing(data, PARAM_KEYS, "parameter array")
            params = np.asarray(data[param_key], dtype=np.float64).reshape(-1)
            if params.size == 0:
                raise ValueError("{}: parameter array is empty.".format(path))
            num_qubits = infer_num_qubits(path, data, explicit_n)
            d_values = (
                np.asarray(data["d_values"], dtype=np.int64).reshape(-1)
                if "d_values" in data
                else None
            )
            arrays: Dict[str, np.ndarray] = {}
            for key in observable_keys:
                if key not in data:
                    raise KeyError("{}: missing observable key '{}'.".format(path, key))
                values = np.asarray(data[key], dtype=np.float64)
                if values.ndim not in (2, 3):
                    raise ValueError(
                        "{}:{} must have shape [B,H] or [B,D,H], got {}.".format(
                            path, key, values.shape
                        )
                    )
                if values.shape[-1] != params.size:
                    raise ValueError(
                        "{}:{} last dimension {} != parameter count {}.".format(
                            path, key, values.shape[-1], params.size
                        )
                    )
                arrays[key] = values
        results.append(
            BootstrapResult(
                path=path,
                num_qubits=num_qubits,
                params=params,
                d_values=d_values,
                arrays=arrays,
            )
        )

    results.sort(key=lambda item: item.num_qubits)
    sizes = [item.num_qubits for item in results]
    if len(sizes) != len(set(sizes)):
        raise ValueError("Duplicate qubit sizes are not allowed: {}.".format(sizes))
    return results


def nearest_indices(source: np.ndarray, targets: np.ndarray, tol: float, path: str) -> np.ndarray:
    indices = []
    for target in targets:
        differences = np.abs(source - target)
        index = int(np.argmin(differences))
        if differences[index] > tol:
            raise ValueError(
                "{}: parameter {} not found; nearest is {} (difference {}).".format(
                    path, target, source[index], differences[index]
                )
            )
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def component_indices(
    result: BootstrapResult, key: str, requested: Optional[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    values = result.arrays[key]
    if values.ndim == 2:
        if requested is not None:
            raise ValueError("--components/--distances cannot be used with scalar key '{}'.".format(key))
        return np.asarray([0], dtype=np.int64), np.asarray([-1], dtype=np.int64)

    count = values.shape[1]
    labels = (
        result.d_values
        if result.d_values is not None and result.d_values.size == count
        else np.arange(count, dtype=np.int64)
    )
    if requested is None:
        return np.arange(count, dtype=np.int64), labels.copy()
    indices = []
    for component in requested:
        matches = np.flatnonzero(labels == int(component))
        if matches.size == 0:
            raise ValueError(
                "Component {} is unavailable for '{}'; choices are {}.".format(
                    component, key, labels.tolist()
                )
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64), np.asarray(requested, dtype=np.int64)


def select_bootstrap_count(
    results: Sequence[BootstrapResult], key: str, requested: Optional[int]
) -> int:
    counts = [item.arrays[key].shape[0] for item in results]
    if requested is None:
        if len(set(counts)) != 1:
            raise ValueError(
                "Stored bootstrap counts differ for '{}': {}. Pass --B no larger than "
                "the minimum to select an explicit common count.".format(key, counts)
            )
        return counts[0]
    if requested <= 0:
        raise ValueError("--bootstrap_repetitions/--B must be positive.")
    if requested > min(counts):
        raise ValueError("Requested B={} exceeds stored counts {}.".format(requested, counts))
    return requested


def align_finite_bootstrap(
    results: Sequence[BootstrapResult],
    key: str,
    target_params: np.ndarray,
    selected_components: np.ndarray,
    bootstrap_count: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    # Output order: [N, B, component, parameter].
    aligned = []
    expected_component_count = results[0].arrays[key].shape[1] if results[0].arrays[key].ndim == 3 else 1
    for result in results:
        values = result.arrays[key]
        component_count = values.shape[1] if values.ndim == 3 else 1
        if component_count != expected_component_count:
            raise ValueError(
                "{}:{} component count {} differs from {}.".format(
                    result.path, key, component_count, expected_component_count
                )
            )
        param_indices = nearest_indices(result.params, target_params, args.match_tol, result.path)
        if values.ndim == 2:
            selected = values[:bootstrap_count, param_indices][:, np.newaxis, :]
        else:
            selected = values[:bootstrap_count][:, selected_components][:, :, param_indices]
        if args.pairing == "independent":
            selected = selected[rng.permutation(bootstrap_count)]
        aligned.append(selected)
    return np.stack(aligned, axis=0)


def r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else float("nan")
    return 1.0 - residual / total


def fit_bootstrap_intercepts(
    inverse_sizes: np.ndarray, finite: np.ndarray, fit_degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # finite: [N, B, C, H]
    _, bootstrap_count, component_count, param_count = finite.shape
    intercepts = np.empty((bootstrap_count, component_count, param_count), dtype=np.float64)
    fit_of_mean = np.empty((component_count, param_count), dtype=np.float64)
    slopes = np.empty_like(fit_of_mean)
    r2_values = np.empty_like(fit_of_mean)

    for component_index in range(component_count):
        for param_index in range(param_count):
            finite_mean = np.mean(finite[:, :, component_index, param_index], axis=1)
            coefficients = np.polyfit(inverse_sizes, finite_mean, fit_degree)
            fit_of_mean[component_index, param_index] = coefficients[-1]
            slopes[component_index, param_index] = coefficients[-2] if fit_degree >= 1 else 0.0
            r2_values[component_index, param_index] = r_squared(
                finite_mean, np.polyval(coefficients, inverse_sizes)
            )
            for bootstrap_index in range(bootstrap_count):
                y_values = finite[:, bootstrap_index, component_index, param_index]
                if not np.all(np.isfinite(y_values)):
                    intercepts[bootstrap_index, component_index, param_index] = np.nan
                    continue
                intercepts[bootstrap_index, component_index, param_index] = np.polyfit(
                    inverse_sizes, y_values, fit_degree
                )[-1]
    return intercepts, fit_of_mean, slopes, r2_values


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def format_values(values: np.ndarray) -> str:
    return ";".join("{:.12g}".format(float(value)) for value in values)


def save_plot(
    path: str,
    params: np.ndarray,
    component_labels: np.ndarray,
    fit_of_mean: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    key: str,
    args: argparse.Namespace,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("--plot requires matplotlib.") from exc

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for component_index, component in enumerate(component_labels):
        label = key if component < 0 else "{} (component={})".format(key, int(component))
        axis.plot(params, fit_of_mean[component_index], marker="o", linewidth=1.8, label=label)
        axis.fill_between(
            params,
            ci_low[component_index],
            ci_high[component_index],
            alpha=0.22,
        )
    axis.set_xlabel(args.param_label)
    axis.set_ylabel(r"$N\to\infty$ estimate")
    axis.set_title("Bootstrap finite-size extrapolation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def validate_args(args: argparse.Namespace) -> None:
    if args.fit_degree < 1:
        raise ValueError("--fit_degree must be at least 1, matching inf_research fits.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between 0 and 1.")
    if args.match_tol < 0.0:
        raise ValueError("--match_tol must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    observable_keys = args.observable_keys or discover_observable_keys(args.files)
    results = load_results(args, observable_keys)
    if len(results) < args.fit_degree + 1:
        raise ValueError(
            "Degree-{} finite-size fit needs at least {} distinct N values; got {}.".format(
                args.fit_degree, args.fit_degree + 1, len(results)
            )
        )

    sizes = np.asarray([item.num_qubits for item in results], dtype=np.int64)
    inverse_sizes = 1.0 / sizes.astype(np.float64)
    target_params = (
        np.asarray(args.parameter_values, dtype=np.float64)
        if args.parameter_values is not None
        else results[0].params.copy()
    )
    if target_params.size == 0:
        raise ValueError("No parameter values selected.")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.output_prefix + ".csv")
    npz_path = os.path.join(args.output_dir, args.output_prefix + ".npz")
    rng = np.random.default_rng(args.seed)
    alpha = (1.0 - args.confidence) / 2.0

    rows: List[Dict[str, object]] = []
    npz_output: Dict[str, np.ndarray] = {
        "N_values": sizes,
        "inverse_N_values": inverse_sizes,
        "params": target_params,
        "fit_degree": np.asarray(args.fit_degree),
        "confidence": np.asarray(args.confidence),
        "pairing": np.asarray(args.pairing),
        "source_files": np.asarray([os.path.abspath(item.path) for item in results]),
    }

    for key in observable_keys:
        selected_components, component_labels = component_indices(results[0], key, args.components)
        bootstrap_count = select_bootstrap_count(results, key, args.bootstrap_repetitions)
        finite = align_finite_bootstrap(
            results,
            key,
            target_params,
            selected_components,
            bootstrap_count,
            args,
            rng,
        )
        intercepts, fit_of_mean, slopes, central_r2 = fit_bootstrap_intercepts(
            inverse_sizes, finite, args.fit_degree
        )
        ci_low = np.nanquantile(intercepts, alpha, axis=0)
        ci_high = np.nanquantile(intercepts, 1.0 - alpha, axis=0)
        bootstrap_mean = np.nanmean(intercepts, axis=0)
        bootstrap_median = np.nanmedian(intercepts, axis=0)
        bootstrap_std = np.nanstd(intercepts, axis=0, ddof=1) if bootstrap_count > 1 else np.zeros_like(fit_of_mean)
        finite_means = np.mean(finite, axis=1)

        suffix = safe_name(key)
        npz_output["{}_component_values".format(suffix)] = component_labels
        npz_output["{}_finite_bootstrap".format(suffix)] = finite
        npz_output["{}_finite_mean".format(suffix)] = finite_means
        npz_output["{}_infinite_bootstrap".format(suffix)] = intercepts
        npz_output["{}_fit_of_finite_mean".format(suffix)] = fit_of_mean
        npz_output["{}_bootstrap_mean".format(suffix)] = bootstrap_mean
        npz_output["{}_bootstrap_median".format(suffix)] = bootstrap_median
        npz_output["{}_bootstrap_std".format(suffix)] = bootstrap_std
        npz_output["{}_ci_low".format(suffix)] = ci_low
        npz_output["{}_ci_high".format(suffix)] = ci_high
        npz_output["{}_central_slope".format(suffix)] = slopes
        npz_output["{}_central_r_squared".format(suffix)] = central_r2

        for component_index, component in enumerate(component_labels):
            for param_index, param in enumerate(target_params):
                rows.append({
                    "observable_key": key,
                    "component": "" if component < 0 else int(component),
                    "parameter": "{:.12g}".format(float(param)),
                    "fit_of_finite_mean": "{:.12g}".format(fit_of_mean[component_index, param_index]),
                    "bootstrap_mean": "{:.12g}".format(bootstrap_mean[component_index, param_index]),
                    "bootstrap_median": "{:.12g}".format(bootstrap_median[component_index, param_index]),
                    "bootstrap_std": "{:.12g}".format(bootstrap_std[component_index, param_index]),
                    "ci_low": "{:.12g}".format(ci_low[component_index, param_index]),
                    "ci_high": "{:.12g}".format(ci_high[component_index, param_index]),
                    "confidence": "{:.12g}".format(args.confidence),
                    "fit_degree": args.fit_degree,
                    "central_slope": "{:.12g}".format(slopes[component_index, param_index]),
                    "central_r_squared": "{:.12g}".format(central_r2[component_index, param_index]),
                    "bootstrap_repetitions": bootstrap_count,
                    "qubits": format_values(sizes),
                    "inverse_qubits": format_values(inverse_sizes),
                    "finite_size_means": format_values(finite_means[:, component_index, param_index]),
                })

        if args.plot:
            plot_path = os.path.join(
                args.output_dir, "{}_{}.{}".format(args.output_prefix, suffix, args.save_format)
            )
            save_plot(
                plot_path,
                target_params,
                component_labels,
                fit_of_mean,
                ci_low,
                ci_high,
                key,
                args,
            )
            print("Saved plot: {}".format(plot_path))

        print(
            "Fitted {}: B={}, components={}, parameters={}".format(
                key, bootstrap_count, len(component_labels), len(target_params)
            )
        )

    fieldnames = [
        "observable_key",
        "component",
        "parameter",
        "fit_of_finite_mean",
        "bootstrap_mean",
        "bootstrap_median",
        "bootstrap_std",
        "ci_low",
        "ci_high",
        "confidence",
        "fit_degree",
        "central_slope",
        "central_r_squared",
        "bootstrap_repetitions",
        "qubits",
        "inverse_qubits",
        "finite_size_means",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(npz_path, **npz_output)

    print("Finite-size N values: {}".format(", ".join(str(value) for value in sizes)))
    print("Saved confidence intervals: {}".format(csv_path))
    print("Saved full fitted bootstrap distributions: {}".format(npz_path))


if __name__ == "__main__":
    main()
