"""Shared eval/bootstrap finite-size plots for J1-J2 and ANNNI observables."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass

import numpy as np

from bootstrap_inf import fit_bootstrap_intercepts


PALETTE = ["#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":"]


@dataclass(frozen=True)
class ModeConfig:
    mode: str
    description: str
    default_output_dir: str
    default_csv: str
    default_param_label: str
    param_candidates: tuple[str, ...]


CONFIGS = {
    "j1j2": ModeConfig(
        mode="j1j2",
        description="Plot J1-J2 spin-dot versus 1/N for selected parameter indices.",
        default_output_dir="j1j2_spin_dot_invN_gidx_figs",
        default_csv="j1j2_spin_dot_gidx_infinite_fit_values.csv",
        default_param_label="J2",
        param_candidates=("params", "J2s", "alphas", "J2_values", "J2_values_dense"),
    ),
    "annni": ModeConfig(
        mode="annni",
        description="Plot ANNNI structure factors versus 1/N for selected parameter indices.",
        default_output_dir="annni_sf_invN_gidx_figs",
        default_csv="annni_sf_gidx_infinite_fit_values.csv",
        default_param_label="h",
        param_candidates=("params", "hs", "hs_gpt", "h_values", "h_values_dense"),
    ),
}

Q_CONFIG = {
    "pi": {
        "label": r"$S_Z(\pi)$",
        "name": "qpi",
        "mean": ("structure_factor_pi_mean", "sf_pi_mean"),
        "std": ("structure_factor_pi_std", "sf_pi_std"),
        "bootstrap": "bootstrap_structure_factor_pi",
    },
    "pi_over_2": {
        "label": r"$S_Z(\pi/2)$",
        "name": "qpi_over_2",
        "mean": ("structure_factor_pi_over_2_mean", "sf_pi_over_2_mean"),
        "std": ("structure_factor_pi_over_2_std", "sf_pi_over_2_std"),
        "bootstrap": "bootstrap_structure_factor_pi_over_2",
    },
}


@dataclass
class Series:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    bootstrap: np.ndarray | None = None


@dataclass
class SizeResult:
    path: str
    num_qubits: int
    params: np.ndarray
    series: dict[str, Series]


@dataclass
class FitRecord:
    source_mode: str
    component: str
    g_index: int
    g_value: float
    infinite: float
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
    finite_size_means: str


def _normalize_q_values(values):
    aliases = {
        "pi": "pi",
        "qpi": "pi",
        "pi_over_2": "pi_over_2",
        "pi/2": "pi_over_2",
        "piover2": "pi_over_2",
        "qpi_over_2": "pi_over_2",
    }
    output = []
    for value in values:
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        canonical = aliases.get(key)
        if canonical is None:
            raise ValueError("Unknown q value {!r}; use pi and/or pi_over_2.".format(value))
        if canonical not in output:
            output.append(canonical)
    return output


def build_parser(config: ModeConfig):
    parser = argparse.ArgumentParser(description=config.description)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--files",
        nargs="+",
        help="Per-N eval_new.py NPZ files for the original mean/std fit.",
    )
    inputs.add_argument(
        "--bootstrap_files",
        nargs="+",
        help=(
            "Per-N NPZ files from bootstrap_snapshot_observables.py; finite-N "
            "percentile intervals and the N->infinity CI are plotted."
        ),
    )
    parser.add_argument("--num_qubits", nargs="+", type=int, default=None)
    parser.add_argument("--bootstrap_num_qubits", nargs="+", type=int, default=None)
    parser.add_argument(
        "--g_indices",
        "--g-index",
        "--g_index",
        "--j2_indices",
        dest="g_indices",
        nargs=2,
        type=int,
        required=True,
        help="Exactly two saved parameter indices.",
    )
    parser.add_argument("--index_base", choices=[0, 1], type=int, default=0)
    parser.add_argument("--param_key", default="auto")
    parser.add_argument("--param_label", default=config.default_param_label)
    parser.add_argument("--fit_degree", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--pairing",
        choices=["independent", "index"],
        default="independent",
        help="Bootstrap pairing across N, identical to bootstrap_finite_size_ci.py.",
    )
    parser.add_argument(
        "--bootstrap_repetitions", "--B", dest="bootstrap_repetitions", type=int
    )
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--match_tol", "--param_mismatch_tol", type=float, default=1e-6)
    parser.add_argument("--output_dir", default=config.default_output_dir)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fit_csv", default=config.default_csv)
    parser.add_argument("--save_format", choices=["png", "pdf", "svg", "eps"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig_width", type=float, default=7.2)
    parser.add_argument("--fig_height", type=float, default=5.2)
    parser.add_argument("--no_errorbar", action="store_true")
    parser.add_argument("--annotate_n", action="store_true")
    annotations = parser.add_mutually_exclusive_group()
    annotations.add_argument(
        "--annotate_infinite", dest="annotate_infinite", action="store_true"
    )
    annotations.add_argument(
        "--no_annotate_infinite", dest="annotate_infinite", action="store_false"
    )
    parser.set_defaults(annotate_infinite=True)
    parser.add_argument("--no_y_break", action="store_true")
    parser.add_argument("--y_break_min_gap_fraction", type=float, default=0.05)
    parser.add_argument("--y_pad_fraction", type=float, default=0.12)

    if config.mode == "j1j2":
        parser.add_argument("--distances", nargs="+", type=int, default=None)
        parser.add_argument("--spin_dot_key", default="auto")
        parser.add_argument("--spin_dot_std_key", default="auto")
        parser.add_argument(
            "--bootstrap_key",
            default="bootstrap_corr_spin_dot",
            help="Default: bootstrap_corr_spin_dot.",
        )
    else:
        parser.add_argument("--q_values", nargs="+", default=["pi", "pi_over_2"])
        parser.add_argument("--sf_pi_key", default="auto")
        parser.add_argument("--sf_pi_std_key", default="auto")
        parser.add_argument("--sf_pi_over_2_key", default="auto")
        parser.add_argument("--sf_pi_over_2_std_key", default="auto")
        parser.add_argument("--sf_pi_bootstrap_key", default=Q_CONFIG["pi"]["bootstrap"])
        parser.add_argument(
            "--sf_pi_over_2_bootstrap_key",
            default=Q_CONFIG["pi_over_2"]["bootstrap"],
        )
    return parser


def _infer_num_qubits(path, data, explicit=None):
    if explicit is not None:
        return int(explicit)
    if "N" in data:
        return int(np.asarray(data["N"]).reshape(-1)[0])
    name = os.path.basename(path)
    for pattern in (
        r"(?i)(?:^|[_-])N(\d+)(?:[_\-.]|$)",
        r"(?i)(?:^|[_-])(\d+)q(?:ubit)?s?(?:[_\-.]|$)",
        r"(?i)(?:^|[_-])qubits?[_-]?(\d+)(?:[_\-.]|$)",
    ):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    raise ValueError("Cannot infer N for {!r}; pass the matching N option.".format(path))


def _key(data, requested, candidates, description, optional=False):
    if requested == "none" and optional:
        return None
    if requested != "auto":
        if requested not in data:
            raise KeyError("{} key {!r} not found.".format(description, requested))
        return requested
    for candidate in candidates:
        if candidate in data:
            return candidate
    if optional:
        return None
    raise KeyError("Could not infer {} key; tried {}.".format(description, ", ".join(candidates)))


def _load_files(config, args):
    paths = args.bootstrap_files if args.bootstrap_files is not None else args.files
    ns = args.bootstrap_num_qubits if args.bootstrap_files is not None else args.num_qubits
    if ns is not None and len(ns) != len(paths):
        option = "--bootstrap_num_qubits" if args.bootstrap_files is not None else "--num_qubits"
        raise ValueError("{} must have the same length as the input files.".format(option))
    output = []
    for file_index, path in enumerate(paths):
        explicit_n = ns[file_index] if ns is not None else None
        with np.load(path, allow_pickle=True) as data:
            param_key = _key(data, args.param_key, config.param_candidates, "parameter")
            params = np.asarray(data[param_key], dtype=float).reshape(-1)
            num_qubits = _infer_num_qubits(path, data, explicit_n)
            if config.mode == "j1j2":
                series = _load_j1j2_series(data, params, args)
            else:
                series = _load_annni_series(data, params, args)
        output.append(SizeResult(path, num_qubits, params, series))
    output.sort(key=lambda value: value.num_qubits)
    sizes = [value.num_qubits for value in output]
    if len(sizes) != len(set(sizes)):
        raise ValueError("Duplicate qubit sizes are not allowed: {}.".format(sizes))
    return output


def _load_j1j2_series(data, params, args):
    output = {}
    if args.bootstrap_files is not None:
        key = args.bootstrap_key
        if key not in data:
            raise KeyError("bootstrap key {!r} not found.".format(key))
        values = np.asarray(data[key], dtype=float)
        if values.ndim == 2:
            values = values[:, np.newaxis, :]
        if values.ndim != 3 or values.shape[-1] != params.size:
            raise ValueError("{} must have shape [B,H] or [B,D,H], got {}.".format(key, values.shape))
        distances = (
            np.asarray(data["d_values"], dtype=int).reshape(-1)
            if "d_values" in data
            else np.arange(1, values.shape[1] + 1)
        )
        if distances.size != values.shape[1]:
            raise ValueError("d_values does not match {} component count.".format(key))
        for row, distance in enumerate(distances):
            output["d{}".format(int(distance))] = Series(bootstrap=values[:, row, :])
    else:
        mean_key = _key(
            data,
            args.spin_dot_key,
            ("corr_spin_dot_mean", "spin_dot_mean", "correlations_spin_dot_mean"),
            "spin-dot mean",
        )
        std_key = _key(
            data,
            args.spin_dot_std_key,
            ("corr_spin_dot_std", "spin_dot_std", "correlations_spin_dot_std"),
            "spin-dot std",
            optional=True,
        )
        mean = np.asarray(data[mean_key], dtype=float)
        if mean.ndim == 1:
            mean = mean[np.newaxis, :]
        if mean.ndim != 2 or mean.shape[-1] != params.size:
            raise ValueError("{} must have shape [D,H], got {}.".format(mean_key, mean.shape))
        std = np.asarray(data[std_key], dtype=float) if std_key is not None else None
        if std is not None:
            if std.ndim == 1:
                std = std[np.newaxis, :]
            if std.shape != mean.shape:
                raise ValueError("{} shape does not match {}.".format(std_key, mean_key))
        distances = (
            np.asarray(data["d_values"], dtype=int).reshape(-1)[: mean.shape[0]]
            if "d_values" in data
            else np.arange(1, mean.shape[0] + 1)
        )
        for row, distance in enumerate(distances):
            output["d{}".format(int(distance))] = Series(
                mean=mean[row], std=None if std is None else std[row]
            )
    return output


def _load_annni_series(data, params, args):
    output = {}
    for q_value in args.q_values:
        config = Q_CONFIG[q_value]
        if args.bootstrap_files is not None:
            arg_name = "sf_{}_bootstrap_key".format(q_value)
            key = getattr(args, arg_name)
            if key not in data:
                raise KeyError("bootstrap key {!r} not found for q={}.".format(key, q_value))
            values = np.asarray(data[key], dtype=float)
            if values.ndim != 2 or values.shape[-1] != params.size:
                raise ValueError("{} must have shape [B,H], got {}.".format(key, values.shape))
            output[q_value] = Series(bootstrap=values)
        else:
            key_stem = "sf_{}".format(q_value)
            mean_key = _key(data, getattr(args, key_stem + "_key"), config["mean"], config["label"])
            std_key = _key(
                data,
                getattr(args, key_stem + "_std_key"),
                config["std"],
                config["label"] + " std",
                optional=True,
            )
            mean = np.asarray(data[mean_key], dtype=float).reshape(-1)
            std = np.asarray(data[std_key], dtype=float).reshape(-1) if std_key else None
            if mean.size != params.size or (std is not None and std.shape != mean.shape):
                raise ValueError("ANNNI structure-factor array shape does not match parameter grid.")
            output[q_value] = Series(mean=mean, std=std)
    return output


def _selected_components(config, results, args):
    if config.mode == "annni":
        return list(args.q_values)
    available = sorted(
        {int(key[1:]) for result in results for key in result.series if key.startswith("d")}
    )
    requested = available if args.distances is None else list(dict.fromkeys(args.distances))
    missing = [value for value in requested if value not in available]
    if missing:
        raise ValueError("Requested distances {} are unavailable; choices are {}.".format(missing, available))
    return ["d{}".format(int(value)) for value in requested]


def _normalize_indices(results, args):
    length = results[0].params.size
    output = []
    for raw in args.g_indices:
        index = int(raw) - args.index_base
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise ValueError("g index {} is outside a parameter grid of length {}.".format(raw, length))
        output.append(index)
    if output[0] == output[1]:
        raise ValueError("--g_indices must select two distinct parameter points.")
    return output


def _collect_eval(results, component, g_index, args):
    rows = []
    reference = None
    missing = []
    for result in results:
        if component not in result.series:
            missing.append(result.num_qubits)
            continue
        if g_index >= result.params.size:
            raise IndexError("{}: g index {} is out of range.".format(result.path, g_index))
        value = float(result.params[g_index])
        if reference is None:
            reference = value
        elif abs(value - reference) > args.match_tol:
            raise ValueError("g index {} maps to inconsistent parameter values across N.".format(g_index))
        series = result.series[component]
        y = float(series.mean[g_index])
        if np.isfinite(y):
            error = float(series.std[g_index]) if series.std is not None else np.nan
            rows.append((result.num_qubits, y, error))
    if len(rows) < args.fit_degree + 1:
        raise ValueError("Degree-{} fit needs at least {} usable N values.".format(args.fit_degree, args.fit_degree + 1))
    ns = np.asarray([row[0] for row in rows], dtype=int)
    x = 1.0 / ns.astype(float)
    y = np.asarray([row[1] for row in rows])
    yerr = np.asarray([row[2] for row in rows])
    order = np.argsort(x)
    x, y, yerr, ns = x[order], y[order], yerr[order], ns[order]
    coeffs = np.polyfit(x, y, args.fit_degree)
    fitted = np.polyval(coeffs, x)
    total = float(np.sum((y - np.mean(y)) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    return {
        "x": x, "y": y, "yerr": None if np.any(~np.isfinite(yerr)) else yerr,
        "ns": ns, "g_value": float(reference), "coefficients": coeffs,
        "infinite": float(coeffs[-1]), "ci_low": np.nan, "ci_high": np.nan,
        "bootstrap_mean": np.nan, "bootstrap_median": np.nan, "bootstrap_std": np.nan,
        "slope": float(coeffs[-2]) if args.fit_degree else 0.0,
        "r_squared": 1.0 - residual / total if total > 0 else np.nan,
        "bootstrap_count": 0, "finite_ci_low": None, "finite_ci_high": None,
        "missing_ns": missing,
    }


def _collect_bootstrap(results, component, g_index, args, rng):
    rows = []
    reference = None
    missing = []
    for result in results:
        if component not in result.series:
            missing.append(result.num_qubits)
            continue
        if g_index >= result.params.size:
            raise IndexError("{}: g index {} is out of range.".format(result.path, g_index))
        value = float(result.params[g_index])
        if reference is None:
            reference = value
        elif abs(value - reference) > args.match_tol:
            raise ValueError("g index {} maps to inconsistent parameter values across N.".format(g_index))
        rows.append((result.num_qubits, result.series[component].bootstrap[:, g_index]))
    if len(rows) < args.fit_degree + 1:
        raise ValueError("Degree-{} bootstrap fit needs at least {} usable N values.".format(args.fit_degree, args.fit_degree + 1))
    counts = [samples.size for _, samples in rows]
    if args.bootstrap_repetitions is None:
        if len(set(counts)) != 1:
            raise ValueError("Stored bootstrap counts differ: {}; pass --B.".format(counts))
        count = counts[0]
    else:
        count = args.bootstrap_repetitions
        if count > min(counts):
            raise ValueError("Requested B={} exceeds stored counts {}.".format(count, counts))
    aligned = []
    for _, samples in rows:
        samples = samples[:count]
        if args.pairing == "independent":
            samples = samples[rng.permutation(count)]
        aligned.append(samples)
    finite = np.stack(aligned, axis=0)[:, :, np.newaxis, np.newaxis]
    ns = np.asarray([row[0] for row in rows], dtype=int)
    x = 1.0 / ns.astype(float)
    intercepts, fit_mean, slopes, r2 = fit_bootstrap_intercepts(x, finite, args.fit_degree)
    y = np.mean(finite[:, :, 0, 0], axis=1)
    coeffs = np.polyfit(x, y, args.fit_degree)
    alpha = (1.0 - args.confidence) / 2.0
    infinite_samples = intercepts[:, 0, 0]
    ci = np.nanquantile(infinite_samples, [alpha, 1.0 - alpha])
    low = np.nanquantile(finite[:, :, 0, 0], alpha, axis=1)
    high = np.nanquantile(finite[:, :, 0, 0], 1.0 - alpha, axis=1)
    order = np.argsort(x)
    return {
        "x": x[order], "y": y[order], "yerr": None, "ns": ns[order],
        "g_value": float(reference), "coefficients": coeffs,
        "infinite": float(fit_mean[0, 0]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        "bootstrap_mean": float(np.nanmean(infinite_samples)),
        "bootstrap_median": float(np.nanmedian(infinite_samples)),
        "bootstrap_std": float(np.nanstd(infinite_samples, ddof=1)) if count > 1 else 0.0,
        "slope": float(slopes[0, 0]), "r_squared": float(r2[0, 0]),
        "bootstrap_count": int(count), "finite_ci_low": low[order],
        "finite_ci_high": high[order], "missing_ns": missing,
    }


def _format_array(values):
    return ";".join("{:.12g}".format(float(value)) for value in np.asarray(values).reshape(-1))


def _component_label(config, component):
    if config.mode == "j1j2":
        return "d={}".format(component[1:])
    return Q_CONFIG[component]["label"]


def _build_items(config, results, components, indices, args, rng):
    items, records = [], []
    bootstrap_mode = args.bootstrap_files is not None
    for curve_index, g_index in enumerate(indices):
        color = PALETTE[curve_index % len(PALETTE)]
        for component_index, component in enumerate(components):
            series = (
                _collect_bootstrap(results, component, g_index, args, rng)
                if bootstrap_mode
                else _collect_eval(results, component, g_index, args)
            )
            if series["missing_ns"]:
                print("Warning: excluded N={} for component {}.".format(series["missing_ns"], component))
            record = FitRecord(
                source_mode="bootstrap" if bootstrap_mode else "eval",
                component=component,
                g_index=int(g_index),
                g_value=series["g_value"],
                infinite=series["infinite"],
                ci_low=series["ci_low"], ci_high=series["ci_high"],
                bootstrap_mean=series["bootstrap_mean"],
                bootstrap_median=series["bootstrap_median"],
                bootstrap_std=series["bootstrap_std"],
                slope=series["slope"], r_squared=series["r_squared"],
                fit_degree=args.fit_degree,
                bootstrap_repetitions=series["bootstrap_count"],
                confidence=args.confidence if bootstrap_mode else np.nan,
                pairing=args.pairing if bootstrap_mode else "",
                num_points=len(series["x"]), qubits=_format_array(series["ns"]),
                inverse_qubits=_format_array(series["x"]),
                finite_size_means=_format_array(series["y"]),
            )
            records.append(record)
            x_fit = np.linspace(0.0, float(np.max(series["x"])), 240)
            label = r"${}={:.2f}$".format(args.param_label, series["g_value"])
            if len(components) > 1:
                label += ", " + _component_label(config, component)
            items.append({
                **series, "component": component, "g_index": int(g_index),
                "color": color, "marker": MARKERS[component_index % len(MARKERS)],
                "linestyle": LINESTYLES[component_index % len(LINESTYLES)],
                "x_fit": x_fit, "y_fit": np.polyval(series["coefficients"], x_fit),
                "label": label,
            })
    return items, records


def _style_axis(ax, config, components, args, xlabel=True, ylabel=True):
    if xlabel:
        ax.set_xlabel(r"$1/N$")
    if ylabel:
        if config.mode == "j1j2":
            ax.set_ylabel(r"$\langle \boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_{i+d}\rangle$")
        elif len(components) == 1:
            ax.set_ylabel(Q_CONFIG[components[0]]["label"])
        else:
            ax.set_ylabel(r"$S_Z(q)$")
    right = ax.get_xlim()[1]
    ax.set_xlim(left=-0.035 * right)
    if config.mode == "annni":
        ax.set_ylim(top=1.4)
    ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.9, zorder=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_item(ax, item, args, label=None):
    low, high = item["finite_ci_low"], item["finite_ci_high"]
    if low is not None and not args.no_errorbar:
        midpoint = 0.5 * (low + high)
        ax.errorbar(
            item["x"], midpoint,
            yerr=np.vstack([midpoint - low, high - midpoint]), fmt="none",
            capsize=3, color=item["color"], linewidth=1.2, zorder=2,
        )
        ax.scatter(item["x"], item["y"], marker=item["marker"], s=38,
                   color=item["color"], edgecolor="white", linewidths=0.6, zorder=3)
    elif item["yerr"] is not None and not args.no_errorbar:
        ax.errorbar(item["x"], item["y"], yerr=item["yerr"], fmt=item["marker"],
                    capsize=3, color=item["color"], linestyle="none")
    else:
        ax.scatter(item["x"], item["y"], marker=item["marker"], s=38,
                   color=item["color"], edgecolor="white", linewidths=0.6)
    ax.plot(item["x_fit"], item["y_fit"], item["linestyle"], color=item["color"],
            linewidth=2.0, label=label)
    if np.isfinite(item["ci_low"]) and not args.no_errorbar:
        midpoint = 0.5 * (item["ci_low"] + item["ci_high"])
        ax.errorbar([0.0], [midpoint],
                    yerr=[[midpoint - item["ci_low"]], [item["ci_high"] - midpoint]],
                    fmt="none", capsize=5, color=item["color"], linewidth=2.0, zorder=5)
    ax.scatter([0.0], [item["infinite"]], marker="*", s=150, color=item["color"],
               edgecolor="black", linewidths=0.8, zorder=6)
    if args.annotate_infinite:
        text = "extrapolated = {:.6g}".format(item["infinite"])
        if np.isfinite(item["ci_low"]):
            text += "\n{:g}% CI = [{:.6g}, {:.6g}]".format(
                100.0 * args.confidence, item["ci_low"], item["ci_high"]
            )
        offset = (10, 10) if item["g_index"] == min(args._zero_indices) else (-10, -18)
        ax.annotate(text, (0.0, item["infinite"]), xytext=offset,
                    textcoords="offset points", fontsize=8.5, color=item["color"],
                    ha="left" if offset[0] >= 0 else "right",
                    va="bottom" if offset[1] >= 0 else "top",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                          "edgecolor": item["color"], "alpha": 0.9})
    if args.annotate_n:
        for x, y, n in zip(item["x"], item["y"], item["ns"]):
            ax.annotate("N={}".format(n), (x, y), xytext=(4, 5),
                        textcoords="offset points", fontsize=8)


def _plot(config, items, components, args, plt):
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    for item in items:
        _draw_item(ax, item, args, item["label"])
    _style_axis(ax, config, components, args)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), loc="upper right", bbox_to_anchor=(0.96, 1.0),)
    fig.tight_layout()
    return fig


def _output_path(config, args, components, suffix=None):
    if args.output is not None:
        if suffix is None:
            return args.output
        root, ext = os.path.splitext(args.output)
        return root + "_" + suffix + (ext or "." + args.save_format)
    indices = "_".join("idx{}".format(value) for value in args.g_indices)
    component_part = "_".join(components)
    name = "{}_vs_invN_{}_{}".format(
        "j1j2_spin_dot" if config.mode == "j1j2" else "annni_sf",
        indices,
        component_part,
    )
    if suffix is not None:
        name += "_" + suffix
    return os.path.join(args.output_dir, name + "." + args.save_format)


def _save_csv(records, args):
    path = os.path.join(args.output_dir, args.fit_csv)
    fields = list(FitRecord.__dataclass_fields__)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
    return path


def run(mode):
    config = CONFIGS[mode]
    args = build_parser(config).parse_args()
    if config.mode == "annni":
        args.q_values = _normalize_q_values(args.q_values)
    if args.fit_degree < 1:
        raise ValueError("--fit_degree must be at least 1.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between 0 and 1.")
    if args.match_tol < 0:
        raise ValueError("--match_tol must be non-negative.")
    if args.bootstrap_repetitions is not None and args.bootstrap_repetitions <= 0:
        raise ValueError("--B must be positive.")
    if args.files is not None and args.bootstrap_num_qubits is not None:
        raise ValueError("--bootstrap_num_qubits requires --bootstrap_files.")
    if args.bootstrap_files is not None and args.num_qubits is not None:
        raise ValueError("--num_qubits requires --files.")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc


    plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
    })


    results = _load_files(config, args)
    components = _selected_components(config, results, args)
    args._zero_indices = _normalize_indices(results, args)
    rng = np.random.default_rng(args.seed)
    items, records = _build_items(
        config, results, components, args._zero_indices, args, rng
    )
    os.makedirs(args.output_dir, exist_ok=True)

    paths = []
    for component in components:
        selected = [item for item in items if item["component"] == component]
        fig = _plot(config, selected, [component], args, plt)
        path = _output_path(
            config, args, [component], component if len(components) > 1 else None
        )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    if len(components) > 1:
        fig = _plot(config, items, components, args, plt)
        path = _output_path(config, args, components, "combined")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    csv_path = _save_csv(records, args)
    source = "bootstrap" if args.bootstrap_files is not None else "eval"
    print("Loaded {} sizes N: {}".format(source, ", ".join(str(r.num_qubits) for r in results)))
    print("Fitted N -> infinity values:")
    for record in records:
        message = (
            "  {}, g_index={}, {}={:.12g}: infinite={:.12g}".format(
                record.component, record.g_index, args.param_label,
                record.g_value, record.infinite
            )
        )
        if record.source_mode == "bootstrap":
            message += ", CI=[{:.12g}, {:.12g}], B={}".format(
                record.ci_low, record.ci_high, record.bootstrap_repetitions
            )
        print(message)
    print("Fit values saved to: {}".format(csv_path))
    for path in paths:
        print("Figure saved to: {}".format(path))

