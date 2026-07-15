"""
Plot J1-J2 spin-dot correlations versus 1/N from eval_new.py .npz files.

Example:
  python inf_research/plot_j1j2_spin_dot_inverse_qubit.py `
    --files j1j2_N8.npz j1j2_N10.npz j1j2_N12.npz `
    --num_qubits 8 10 12 `
    --j2_values 0.2 0.5 0.8 `
    --distances 1 2 3 `
    --layout by_distance `
    --output_dir j1j2_spin_dot_invN_figs
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass

import numpy as np


plt = None
NATURE_PALETTE = [
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#7E6148",
    "#B09C85",
]
FIT_LINE_COLOR = "#E64B35"
EXTRAPOLATED_COLOR = "#333333"
FIT_LINE_STYLES = ["-", "--", "-.", ":"]



def import_matplotlib():
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it first, for example: "
            "pip install matplotlib"
        ) from exc
    return pyplot


def infer_key(data, requested_key, candidates, field_name):
    if requested_key != "auto":
        if requested_key not in data:
            raise KeyError(f"{field_name} key '{requested_key}' not found.")
        return requested_key

    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Could not infer {field_name} key. Tried: {', '.join(candidates)}")


def infer_num_qubits(file_path, data, explicit_n=None):
    if explicit_n is not None:
        return int(explicit_n)

    if "N" in data:
        value = np.asarray(data["N"]).reshape(-1)[0]
        return int(value)

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

    raise ValueError(
        f"Cannot infer num_qubits for '{file_path}'. "
        "Pass --num_qubits with one value per file."
    )


def nearest_param_index(params, target, tol):
    distances = np.abs(params - target)
    index = int(np.argmin(distances))
    if distances[index] > tol:
        raise ValueError(
            f"Requested parameter {target:g} was not found. "
            f"Nearest saved value is {params[index]:g}, diff={distances[index]:g}."
        )
    return index


def format_float(value):
    text = f"{value:.8g}"
    return text.replace("-", "m").replace(".", "p")


def format_array(values):
    return ";".join(f"{value:.12g}" for value in values)


def palette_color(index):
    return NATURE_PALETTE[index % len(NATURE_PALETTE)]


def split_group_spec(spec):
    for separator in [":", "="]:
        if separator in spec:
            left, right = spec.split(separator, 1)
            return left.strip(), right.strip()
    raise ValueError(
        f"Invalid --qubit_groups spec '{spec}'. Expected 'label:6,8,10' "
        "or '6,8,10:label'."
    )


def parse_qubit_list(text):
    values = []
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        if not re.fullmatch(r"\d+", token):
            raise ValueError(f"Invalid qubit value '{token}' in --qubit_groups.")
        values.append(int(token))
    if not values:
        raise ValueError(f"Empty qubit list in --qubit_groups item '{text}'.")
    return values


def is_qubit_list(text):
    try:
        parse_qubit_list(text)
    except ValueError:
        return False
    return True


def build_qubit_group_styles(results, group_specs):
    available_ns = {result.num_qubits for result in results}
    group_by_n = {}
    group_labels = []

    if group_specs is None:
        return {result.num_qubits: "data" for result in results}, {"data": palette_color(0)}

    for spec in group_specs:
        left, right = split_group_spec(spec)
        left_is_qubits = is_qubit_list(left)
        right_is_qubits = is_qubit_list(right)

        if left_is_qubits and not right_is_qubits:
            qubits = parse_qubit_list(left)
            label = right
        elif right_is_qubits and not left_is_qubits:
            label = left
            qubits = parse_qubit_list(right)
        else:
            raise ValueError(
                f"Cannot parse --qubit_groups spec '{spec}'. Use exactly one qubit list "
                "and one group label."
            )

        if not label:
            raise ValueError(f"Empty group label in --qubit_groups spec '{spec}'.")
        if label not in group_labels:
            group_labels.append(label)

        for n in qubits:
            if n not in available_ns:
                raise ValueError(
                    f"--qubit_groups contains N={n}, but loaded data only has "
                    f"N={sorted(available_ns)}."
                )
            if n in group_by_n:
                raise ValueError(f"N={n} appears in more than one --qubit_groups spec.")
            group_by_n[n] = label

    ungrouped_ns = sorted(available_ns.difference(group_by_n))
    if ungrouped_ns:
        group_labels.append("Ungrouped")
        for n in ungrouped_ns:
            group_by_n[n] = "Ungrouped"

    group_colors = {label: palette_color(i) for i, label in enumerate(group_labels)}
    return group_by_n, group_colors


def plot_grouped_points(ax, x, y, yerr, ns, group_by_n, group_colors, used_labels, default_label="data"):
    groups_for_points = np.asarray([group_by_n.get(int(n), default_label) for n in ns], dtype=object)
    ordered_labels = []
    for label in groups_for_points:
        if label not in ordered_labels:
            ordered_labels.append(label)

    for label in ordered_labels:
        mask = groups_for_points == label
        color = group_colors.get(label, palette_color(len(group_colors)))
        legend_label = label if label not in used_labels else None
        if legend_label is not None:
            used_labels.add(label)

        if yerr is not None:
            ax.errorbar(
                x[mask],
                y[mask],
                yerr=yerr[mask],
                fmt="o",
                capsize=3,
                markersize=5,
                color=color,
                ecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=legend_label,
            )
        else:
            ax.scatter(
                x[mask],
                y[mask],
                s=28,
                color=color,
                edgecolor="white",
                linewidths=0.6,
                label=legend_label,
            )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)

@dataclass
class EvalResult:
    file_path: str
    num_qubits: int
    param_key: str
    params: np.ndarray
    spin_dot_mean: np.ndarray
    spin_dot_std: np.ndarray | None
    d_values: np.ndarray


@dataclass
class FitRecord:
    distance: int
    j2_value: float
    spin_dot_infinite: float
    slope: float
    r_squared: float
    fit_degree: int
    num_points: int
    qubits: str
    inverse_qubits: str
    spin_dot_values: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot J1-J2 spin-dot correlation versus 1 / num_qubits from "
            "multiple eval_new.py .npz result files."
        )
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="J1-J2 eval_new.py output .npz files. Each file should correspond to one qubit size.",
    )
    parser.add_argument(
        "--num_qubits",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional qubit sizes matching --files. Needed when N is not saved in "
            "the .npz and cannot be parsed from the filename."
        ),
    )
    parser.add_argument(
        "--j2_values",
        "--g_values",
        dest="j2_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional J2 values to plot. By default all parameter points in the first file are used.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=None,
        help="Optional correlation distances d to plot. By default all saved distances are used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="j1j2_spin_dot_inverse_qubit_plots",
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--layout",
        choices=["separate", "by_distance"],
        default="separate",
        help=(
            "separate: one figure for every (J2, d). "
            "by_distance: one figure for every d, with different J2 values in the legend."
        ),
    )
    parser.add_argument(
        "--qubit_groups",
        nargs="+",
        default=None,
        help=(
            "Optional qubit-size groups for point colors. Use specs like "
            "'Diffushadow1:6,8,10,12' 'Diffushadow2:18,20,22,24'. "
            "The reversed form '6,8,10,12:Diffushadow1' is also accepted."
        ),
    )
    parser.add_argument(
        "--param_key",
        type=str,
        default="auto",
        help="Parameter key in .npz. Use auto for J2s, hs_gpt, Js, Deltas, or J2_values.",
    )
    parser.add_argument(
        "--spin_dot_key",
        type=str,
        default="auto",
        help="Spin-dot mean key in .npz. Use auto for corr_spin_dot_mean or spin_dot_mean.",
    )
    parser.add_argument(
        "--spin_dot_std_key",
        type=str,
        default="auto",
        help="Spin-dot std key in .npz. Use auto for corr_spin_dot_std or spin_dot_std.",
    )
    parser.add_argument(
        "--param_label",
        type=str,
        default="J2",
        help="Label used in titles, legends, and filenames for the scanned parameter.",
    )
    parser.add_argument(
        "--match_tol",
        type=float,
        default=1e-6,
        help="Tolerance for matching requested J2 values to saved parameter points.",
    )
    parser.add_argument(
        "--fit_degree",
        type=int,
        default=1,
        help="Polynomial degree for fitting spin-dot as a function of 1/N. Default is linear.",
    )
    parser.add_argument(
        "--fit_csv",
        type=str,
        default="j1j2_spin_dot_infinite_fit_values.csv",
        help="CSV filename for fitted N -> infinity spin-dot values.",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Figure file format.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--fig_width", type=float, default=6.5, help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=4.8, help="Figure height in inches.")
    parser.add_argument(
        "--no_errorbar",
        action="store_true",
        help="Do not draw std error bars even if std values are present.",
    )
    parser.add_argument(
        "--annotate_n",
        action="store_true",
        help="Annotate each point with its qubit size N.",
    )
    return parser.parse_args()


def infer_optional_key(data, requested_key, candidates, field_name):
    if requested_key == "none":
        return None
    if requested_key != "auto":
        if requested_key not in data:
            raise KeyError(f"{field_name} key '{requested_key}' not found in file.")
        return requested_key

    for key in candidates:
        if key in data:
            return key
    return None


def as_2d_array(array, key):
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim != 2:
        raise ValueError(f"{key} must be 1D or 2D, got shape {values.shape}.")
    if values.size == 0:
        raise ValueError(f"{key} is empty.")
    return values


def load_d_values(data, row_count):
    if "d_values" not in data:
        return np.arange(1, row_count + 1, dtype=int)

    values = np.asarray(data["d_values"], dtype=int).reshape(-1)
    if values.shape[0] < row_count:
        raise ValueError(
            f"d_values has {values.shape[0]} entries, but spin-dot data has {row_count} rows."
        )
    return values[:row_count]


def find_distance_index(result, distance):
    matches = np.where(result.d_values == int(distance))[0]
    if matches.size == 0:
        raise ValueError(
            f"{result.file_path}: requested d={distance}, "
            f"but saved distances are {result.d_values.tolist()}."
        )
    return int(matches[0])


def maybe_find_distance_index(result, distance):
    matches = np.where(result.d_values == int(distance))[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def load_eval_result(file_path, args, explicit_n=None):
    with np.load(file_path, allow_pickle=True) as data:
        param_key = infer_key(
            data,
            args.param_key,
            ["J2s", "hs_gpt", "Js", "Deltas", "J2_values", "J2_values_dense", "hs"],
            "parameter",
        )
        spin_dot_key = infer_key(
            data,
            args.spin_dot_key,
            ["corr_spin_dot_mean", "spin_dot_mean", "correlations_spin_dot_mean"],
            "spin-dot mean",
        )
        spin_dot_std_key = infer_optional_key(
            data,
            args.spin_dot_std_key,
            ["corr_spin_dot_std", "spin_dot_std", "correlations_spin_dot_std"],
            "spin-dot std",
        )

        params = np.asarray(data[param_key], dtype=float).reshape(-1)
        spin_dot_mean = as_2d_array(data[spin_dot_key], spin_dot_key)
        spin_dot_std = as_2d_array(data[spin_dot_std_key], spin_dot_std_key) if spin_dot_std_key else None
        d_values = load_d_values(data, spin_dot_mean.shape[0])
        num_qubits = infer_num_qubits(file_path, data, explicit_n)

    if spin_dot_mean.shape[1] != params.shape[0]:
        raise ValueError(
            f"{file_path}: {spin_dot_key}.shape[1] ({spin_dot_mean.shape[1]}) does not match "
            f"{param_key} length ({params.shape[0]})."
        )
    if spin_dot_std is not None and spin_dot_std.shape != spin_dot_mean.shape:
        raise ValueError(
            f"{file_path}: {spin_dot_std_key}.shape {spin_dot_std.shape} != "
            f"{spin_dot_key}.shape {spin_dot_mean.shape}."
        )

    return EvalResult(
        file_path=file_path,
        num_qubits=num_qubits,
        param_key=param_key,
        params=params,
        spin_dot_mean=spin_dot_mean,
        spin_dot_std=spin_dot_std,
        d_values=d_values,
    )


def load_all_results(args):
    if args.num_qubits is not None and len(args.num_qubits) != len(args.files):
        raise ValueError("--num_qubits must have the same length as --files.")

    results = []
    for i, file_path in enumerate(args.files):
        explicit_n = args.num_qubits[i] if args.num_qubits is not None else None
        results.append(load_eval_result(file_path, args, explicit_n))

    seen = {}
    for result in results:
        if result.num_qubits in seen:
            raise ValueError(
                f"Duplicate qubit size N={result.num_qubits}: "
                f"{seen[result.num_qubits]} and {result.file_path}"
            )
        seen[result.num_qubits] = result.file_path

    return sorted(results, key=lambda item: item.num_qubits)


def select_j2_values(results, requested_j2_values):
    if requested_j2_values is not None:
        return np.asarray(requested_j2_values, dtype=float)
    return np.asarray(results[0].params, dtype=float)


def select_distances(results, requested_distances):
    available = sorted({int(d) for result in results for d in result.d_values})

    if requested_distances is None:
        if not available:
            raise ValueError("No saved spin-dot distances found.")
        return available

    distances = []
    for distance in requested_distances:
        value = int(distance)
        if value < 1:
            raise ValueError("--distances values must be >= 1.")
        if value not in distances:
            distances.append(value)

    missing = [d for d in distances if d not in available]
    if missing:
        raise ValueError(
            f"Requested distances {missing} are absent from all files. "
            f"Available distances: {available}"
        )
    return distances


def format_param_values(values):
    return np.array2string(
        np.asarray(values, dtype=float),
        precision=12,
        separator=", ",
        threshold=np.inf,
        max_line_width=200,
    )


def print_loaded_param_axes(results, selected_j2_values, requested_j2_values):
    print("Loaded parameter axes from npz files:")
    for result in results:
        params = np.asarray(result.params, dtype=float).reshape(-1)
        print(f"  {result.file_path}")
        print(
            f"    N={result.num_qubits}, key={result.param_key}, "
            f"count={params.size}, min={np.nanmin(params):.12g}, "
            f"max={np.nanmax(params):.12g}"
        )
        print(f"    params={format_param_values(params)}")

    if requested_j2_values is None:
        print(
            "No --j2_values/--g_values provided; selected values are taken from "
            "the first loaded result after sorting by N:"
        )
    else:
        print("Explicit --j2_values/--g_values provided:")
    print(f"  selected_values={format_param_values(selected_j2_values)}")


def collect_series(results, j2_value, distance, tol, include_std):
    xs = []
    ys = []
    yerrs = []
    ns = []

    for result in results:
        distance_index = maybe_find_distance_index(result, distance)
        if distance_index is None:
            continue
        index = nearest_param_index(result.params, j2_value, tol)
        y_value = float(result.spin_dot_mean[distance_index, index])
        if not np.isfinite(y_value):
            continue
        xs.append(1.0 / result.num_qubits)
        ys.append(y_value)
        if include_std and result.spin_dot_std is not None:
            yerr_value = float(result.spin_dot_std[distance_index, index])
            yerrs.append(yerr_value if np.isfinite(yerr_value) else np.nan)
        else:
            yerrs.append(np.nan)
        ns.append(result.num_qubits)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yerrs = np.asarray(yerrs, dtype=float)
    ns = np.asarray(ns, dtype=int)
    order = np.argsort(xs)

    yerr_out = yerrs[order]
    if np.any(np.isnan(yerr_out)):
        yerr_out = None

    return xs[order], ys[order], yerr_out, ns[order]


def fit_series(x, y, ns, j2_value, distance, fit_degree):
    if fit_degree < 1:
        raise ValueError("--fit_degree must be >= 1.")

    unique_x_count = len(np.unique(x))
    if unique_x_count < fit_degree + 1:
        raise ValueError(
            f"Need at least {fit_degree + 1} distinct qubit sizes to fit degree "
            f"{fit_degree}, but got {unique_x_count} for distance={distance}, "
            f"j2_value={j2_value}."
        )

    coeffs = np.polyfit(x, y, fit_degree)
    fitted_y = np.polyval(coeffs, x)
    residual = y - fitted_y
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitRecord(
        distance=int(distance),
        j2_value=float(j2_value),
        spin_dot_infinite=float(coeffs[-1]),
        slope=float(coeffs[-2]),
        r_squared=r_squared,
        fit_degree=fit_degree,
        num_points=len(x),
        qubits=format_array(ns),
        inverse_qubits=format_array(x),
        spin_dot_values=format_array(y),
    ), coeffs


def spin_dot_ylabel():
    return r"$\langle \sigma_i\cdot\sigma_{i+r}\rangle$"


def save_separate_plot(results, j2_value, distance, args, group_by_n, group_colors):
    x, y, yerr, ns = collect_series(
        results,
        j2_value,
        distance,
        args.match_tol,
        include_std=not args.no_errorbar,
    )
    fit_record, coeffs = fit_series(x, y, ns, j2_value, distance, args.fit_degree)
    x_fit = np.linspace(0.0, float(np.max(x)), 200)
    y_fit = np.polyval(coeffs, x_fit)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    used_labels = set()
    plot_grouped_points(ax, x, y, yerr, ns, group_by_n, group_colors, used_labels)

    ax.plot(
        x_fit,
        y_fit,
        "-",
        linewidth=2.2,
        color=FIT_LINE_COLOR,
        label=rf"fit, $N\to\infty$={fit_record.spin_dot_infinite:.6g}",
    )
    ax.scatter(
        [0.0],
        [fit_record.spin_dot_infinite],
        marker="x",
        s=45,
        color=EXTRAPOLATED_COLOR,
        linewidths=1.8,
        zorder=4,
        label=r"$N\to\infty$",
    )

    if args.annotate_n:
        for xi, yi, n in zip(x, y, ns):
            ax.annotate(f"N={n}", (xi, yi), textcoords="offset points", xytext=(4, 5), fontsize=8)

    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(spin_dot_ylabel())
    ax.set_title(f"Spin-dot vs 1/N ({args.param_label}={j2_value:g}, d={distance})")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)
    ax.legend()
    fig.tight_layout()

    filename = (
        f"j1j2_spin_dot_vs_invN_{args.param_label}{format_float(j2_value)}_"
        f"d{distance}.{args.save_format}"
    )
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, fit_record


def save_by_distance_plot(results, j2_values, distance, args, group_by_n, group_colors):
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    fit_records = []
    used_group_labels = set()

    for i, j2_value in enumerate(j2_values):
        x, y, yerr, ns = collect_series(
            results,
            float(j2_value),
            distance,
            args.match_tol,
            include_std=not args.no_errorbar,
        )
        fit_record, coeffs = fit_series(x, y, ns, float(j2_value), distance, args.fit_degree)
        fit_records.append(fit_record)
        x_fit = np.linspace(0.0, float(np.max(x)), 200)
        y_fit = np.polyval(coeffs, x_fit)
        label = f"{args.param_label}={j2_value:g}, inf={fit_record.spin_dot_infinite:.6g}"

        plot_grouped_points(
            ax,
            x,
            y,
            yerr,
            ns,
            group_by_n,
            group_colors,
            used_group_labels,
        )

        ax.plot(
            x_fit,
            y_fit,
            linestyle=FIT_LINE_STYLES[i % len(FIT_LINE_STYLES)],
            linewidth=2.0,
            color=FIT_LINE_COLOR,
            alpha=0.9,
            label=label,
        )
        ax.scatter(
            [0.0],
            [fit_record.spin_dot_infinite],
            marker="x",
            s=35,
            color=EXTRAPOLATED_COLOR,
            linewidths=1.6,
            zorder=4,
        )

        if args.annotate_n and i == 0:
            for xi, yi, n in zip(x, y, ns):
                ax.annotate(f"N={n}", (xi, yi), textcoords="offset points", xytext=(4, 5), fontsize=8)

    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(spin_dot_ylabel())
    ax.set_title(f"Spin-dot vs 1/N (d={distance})")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)
    ax.legend()
    fig.tight_layout()

    filename = f"j1j2_spin_dot_vs_invN_d{distance}.{args.save_format}"
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, fit_records


def save_fit_records(fit_records, output_dir, csv_name):
    csv_path = os.path.join(output_dir, csv_name)
    fieldnames = [
        "distance",
        "j2_value",
        "spin_dot_infinite",
        "slope",
        "r_squared",
        "fit_degree",
        "num_points",
        "qubits",
        "inverse_qubits",
        "spin_dot_values",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in fit_records:
            writer.writerow(
                {
                    "distance": record.distance,
                    "j2_value": f"{record.j2_value:.12g}",
                    "spin_dot_infinite": f"{record.spin_dot_infinite:.12g}",
                    "slope": f"{record.slope:.12g}",
                    "r_squared": f"{record.r_squared:.12g}",
                    "fit_degree": record.fit_degree,
                    "num_points": record.num_points,
                    "qubits": record.qubits,
                    "inverse_qubits": record.inverse_qubits,
                    "spin_dot_values": record.spin_dot_values,
                }
            )
    return csv_path


def main():
    global plt
    args = parse_args()
    plt = import_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_all_results(args)
    group_by_n, group_colors = build_qubit_group_styles(results, args.qubit_groups)
    j2_values = select_j2_values(results, args.j2_values)
    distances = select_distances(results, args.distances)
    print_loaded_param_axes(results, j2_values, args.j2_values)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    save_paths = []
    fit_records = []
    if args.layout == "separate":
        for distance in distances:
            for j2_value in j2_values:
                save_path, fit_record = save_separate_plot(
                    results,
                    float(j2_value),
                    distance,
                    args,
                    group_by_n,
                    group_colors,
                )
                save_paths.append(save_path)
                fit_records.append(fit_record)
    else:
        for distance in distances:
            save_path, distance_fit_records = save_by_distance_plot(
                results,
                j2_values,
                distance,
                args,
                group_by_n,
                group_colors,
            )
            save_paths.append(save_path)
            fit_records.extend(distance_fit_records)

    csv_path = save_fit_records(fit_records, args.output_dir, args.fit_csv)

    ns_text = ", ".join(str(result.num_qubits) for result in results)
    print(f"Loaded qubit sizes N: {ns_text}")
    print("Fitted N -> infinity J1-J2 spin-dot values:")
    for record in fit_records:
        print(
            f"  d={record.distance}, {args.param_label}={record.j2_value:g}: "
            f"spin_dot_inf={record.spin_dot_infinite:.12g}, "
            f"slope={record.slope:.12g}, R^2={record.r_squared:.6g}"
        )
    print(f"Fit values saved to: {csv_path}")
    print(f"Saved {len(save_paths)} figure(s) to: {args.output_dir}")
    for save_path in save_paths:
        print(f"  {save_path}")


if __name__ == "__main__":
    main()
