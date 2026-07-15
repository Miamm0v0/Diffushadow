"""
Plot ANNNI structure factors versus 1/N from eval_new.py .npz outputs.

Example:
  python inf_research/plot_annni_sf_inverse_qubit.py `
    --files annni_N12.npz annni_N16.npz annni_N20.npz `
    --num_qubits 12 16 20 `
    --g_values 0.3 0.9 `
    --q_values pi pi_over_2 `
    --layout by_q `
    --output_dir annni_sf_invN_figs
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
    "#4DBBD5",  # blue
    "#00A087",  # green
    "#3C5488",  # navy
    "#F39B7F",  # salmon
    "#8491B4",  # lavender gray
    "#91D1C2",  # mint
    "#7E6148",  # brown
    "#B09C85",  # taupe
]
FIT_LINE_COLOR = "#E64B35"
EXTRAPOLATED_COLOR = "#333333"
FIT_LINE_STYLES = ["-", "--", "-.", ":"]

Q_CONFIG = {
    "pi": {
        "mean_candidates": ["structure_factor_pi_mean", "sf_pi_mean"],
        "std_candidates": ["structure_factor_pi_std", "sf_pi_std"],
        "arg_key": "sf_pi_key",
        "arg_std_key": "sf_pi_std_key",
        "label": r"$S_Z(\pi)$",
        "title": r"Z structure factor $q=\pi$",
        "name": "qpi",
    },
    "pi_over_2": {
        "mean_candidates": ["structure_factor_pi_over_2_mean", "sf_pi_over_2_mean"],
        "std_candidates": ["structure_factor_pi_over_2_std", "sf_pi_over_2_std"],
        "arg_key": "sf_pi_over_2_key",
        "arg_std_key": "sf_pi_over_2_std_key",
        "label": r"$S_Z(\pi/2)$",
        "title": r"Z structure factor $q=\pi/2$",
        "name": "qpi_over_2",
    },
}


@dataclass
class EvalResult:
    file_path: str
    num_qubits: int
    param_key: str
    params: np.ndarray
    sf_mean: dict[str, np.ndarray]
    sf_std: dict[str, np.ndarray | None]


@dataclass
class FitRecord:
    q_value: str
    g_value: float
    sf_infinite: float
    slope: float
    r_squared: float
    fit_degree: int
    num_points: int
    qubits: str
    inverse_qubits: str
    sf_values: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ANNNI structure factor versus 1 / num_qubits from multiple "
            "eval_new.py .npz result files."
        )
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="ANNNI eval_new.py output .npz files. Each file should correspond to one qubit size.",
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
        "--g_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional g/h values to plot. By default all parameter points in the first file are used.",
    )
    parser.add_argument(
        "--q_values",
        nargs="+",
        default=["pi", "pi_over_2"],
        help="Structure-factor momenta to plot: pi and/or pi_over_2. Also accepts pi/2.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="annni_sf_vs_inverse_qubit_plots",
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--layout",
        choices=["separate", "by_q"],
        default="separate",
        help=(
            "separate: one figure for every (g, q). "
            "by_q: one figure for every q, with different g values in the legend."
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
        help="Parameter key in .npz. Use auto for hs, hs_gpt, h_values, or h_values_dense.",
    )
    parser.add_argument(
        "--sf_pi_key",
        type=str,
        default="auto",
        help="Mean key for S_Z(pi). Use auto for structure_factor_pi_mean or sf_pi_mean.",
    )
    parser.add_argument(
        "--sf_pi_std_key",
        type=str,
        default="auto",
        help="Std key for S_Z(pi). Use auto for structure_factor_pi_std or sf_pi_std.",
    )
    parser.add_argument(
        "--sf_pi_over_2_key",
        type=str,
        default="auto",
        help="Mean key for S_Z(pi/2). Use auto for structure_factor_pi_over_2_mean or sf_pi_over_2_mean.",
    )
    parser.add_argument(
        "--sf_pi_over_2_std_key",
        type=str,
        default="auto",
        help="Std key for S_Z(pi/2). Use auto for structure_factor_pi_over_2_std or sf_pi_over_2_std.",
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
        help="Tolerance for matching requested g_values to saved parameter points.",
    )
    parser.add_argument(
        "--fit_degree",
        type=int,
        default=1,
        help="Polynomial degree for fitting S_Z(q) as a function of 1/N. Default is linear.",
    )
    parser.add_argument(
        "--fit_csv",
        type=str,
        default="annni_sf_infinite_fit_values.csv",
        help="CSV filename for fitted N -> infinity structure-factor values.",
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
    args = parser.parse_args()
    args.q_values = normalize_q_values(args.q_values)
    return args


def import_matplotlib():
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it first, for example: "
            "pip install matplotlib"
        ) from exc
    return pyplot


def normalize_q_values(raw_q_values):
    q_values = []
    aliases = {
        "pi": "pi",
        "qpi": "pi",
        "pi_over_2": "pi_over_2",
        "pi/2": "pi_over_2",
        "piover2": "pi_over_2",
        "qpi_over_2": "pi_over_2",
        "qpi/2": "pi_over_2",
    }
    for raw in raw_q_values:
        key = str(raw).strip().lower().replace("-", "_")
        key = key.replace(" ", "_")
        canonical = aliases.get(key)
        if canonical is None:
            raise ValueError(
                f"Unknown q value '{raw}'. Supported values are: pi, pi_over_2."
            )
        if canonical not in q_values:
            q_values.append(canonical)
    if not q_values:
        raise ValueError("--q_values cannot be empty.")
    return q_values


def infer_key(data, requested_key, candidates, field_name):
    if requested_key != "auto":
        if requested_key not in data:
            raise KeyError(f"{field_name} key '{requested_key}' not found in file.")
        return requested_key

    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Could not infer {field_name} key. Tried: {', '.join(candidates)}")


def infer_optional_key(data, requested_key, candidates, field_name):
    if requested_key != "auto":
        if requested_key not in data:
            raise KeyError(f"{field_name} key '{requested_key}' not found in file.")
        return requested_key

    for key in candidates:
        if key in data:
            return key
    return None


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


def as_1d_array(array, key):
    values = np.asarray(array, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{key} is empty.")
    return values


def load_eval_result(file_path, args, explicit_n=None):
    with np.load(file_path, allow_pickle=True) as data:
        param_key = infer_key(
            data,
            args.param_key,
            ["hs", "hs_gpt", "h_values", "h_values_dense"],
            "parameter",
        )
        params = as_1d_array(data[param_key], param_key)
        num_qubits = infer_num_qubits(file_path, data, explicit_n)

        sf_mean = {}
        sf_std = {}
        for q_value in args.q_values:
            config = Q_CONFIG[q_value]
            mean_requested_key = getattr(args, config["arg_key"])
            std_requested_key = getattr(args, config["arg_std_key"])
            mean_key = infer_key(data, mean_requested_key, config["mean_candidates"], config["label"])
            std_key = infer_optional_key(data, std_requested_key, config["std_candidates"], f"{config['label']} std")

            mean_values = as_1d_array(data[mean_key], mean_key)
            std_values = as_1d_array(data[std_key], std_key) if std_key is not None else None
            if mean_values.shape[0] != params.shape[0]:
                raise ValueError(
                    f"{file_path}: {mean_key} length ({mean_values.shape[0]}) does not match "
                    f"{param_key} length ({params.shape[0]})."
                )
            if std_values is not None and std_values.shape != mean_values.shape:
                raise ValueError(
                    f"{file_path}: {std_key}.shape {std_values.shape} != "
                    f"{mean_key}.shape {mean_values.shape}."
                )
            sf_mean[q_value] = mean_values
            sf_std[q_value] = std_values

    return EvalResult(
        file_path=file_path,
        num_qubits=num_qubits,
        param_key=param_key,
        params=params,
        sf_mean=sf_mean,
        sf_std=sf_std,
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


def select_g_values(results, requested_g_values):
    if requested_g_values is not None:
        return np.asarray(requested_g_values, dtype=float)
    return np.asarray(results[0].params, dtype=float)


def format_param_values(values):
    return np.array2string(
        np.asarray(values, dtype=float),
        precision=12,
        separator=', ',
        threshold=np.inf,
        max_line_width=200,
    )


def print_loaded_param_axes(results, selected_g_values, requested_g_values):
    print('Loaded parameter axes from npz files:')
    for result in results:
        params = np.asarray(result.params, dtype=float).reshape(-1)
        print(f'  {result.file_path}')
        print(
            f'    N={result.num_qubits}, key={result.param_key}, '
            f'count={params.size}, min={np.nanmin(params):.12g}, '
            f'max={np.nanmax(params):.12g}'
        )
        print(f'    hs={format_param_values(params)}')

    if requested_g_values is None:
        print(
            'No --g_values provided; selected g_values are taken from the first '
            'loaded result after sorting by N:'
        )
    else:
        print('Explicit --g_values provided:')
    print(f'  selected_g_values={format_param_values(selected_g_values)}')


def nearest_param_index(params, target, tol):
    distances = np.abs(params - target)
    index = int(np.argmin(distances))
    if distances[index] > tol:
        raise ValueError(
            f"Requested parameter {target:g} was not found. "
            f"Nearest saved value is {params[index]:g}, diff={distances[index]:g}."
        )
    return index


def collect_series(results, q_value, g_value, tol, include_std):
    xs = []
    ys = []
    yerrs = []
    ns = []

    for result in results:
        index = nearest_param_index(result.params, g_value, tol)
        y_value = float(result.sf_mean[q_value][index])
        if not np.isfinite(y_value):
            continue
        xs.append(1.0 / result.num_qubits)
        ys.append(y_value)
        std_values = result.sf_std.get(q_value)
        if include_std and std_values is not None:
            yerr_value = float(std_values[index])
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


def fit_series(x, y, ns, g_value, q_value, fit_degree):
    if fit_degree < 1:
        raise ValueError("--fit_degree must be >= 1.")

    unique_x_count = len(np.unique(x))
    if unique_x_count < fit_degree + 1:
        raise ValueError(
            f"Need at least {fit_degree + 1} distinct qubit sizes to fit degree "
            f"{fit_degree}, but got {unique_x_count} for q={q_value}, "
            f"{g_value=}."
        )

    coeffs = np.polyfit(x, y, fit_degree)
    fitted_y = np.polyval(coeffs, x)
    residual = y - fitted_y
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitRecord(
        q_value=q_value,
        g_value=float(g_value),
        sf_infinite=float(coeffs[-1]),
        slope=float(coeffs[-2]),
        r_squared=r_squared,
        fit_degree=fit_degree,
        num_points=len(x),
        qubits=format_array(ns),
        inverse_qubits=format_array(x),
        sf_values=format_array(y),
    ), coeffs


def save_separate_plot(results, q_value, g_value, args, group_by_n, group_colors):
    x, y, yerr, ns = collect_series(
        results,
        q_value,
        g_value,
        args.match_tol,
        include_std=not args.no_errorbar,
    )
    fit_record, coeffs = fit_series(x, y, ns, g_value, q_value, args.fit_degree)
    x_fit = np.linspace(0.0, float(np.max(x)), 200)
    y_fit = np.polyval(coeffs, x_fit)
    q_config = Q_CONFIG[q_value]

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    used_labels = set()
    plot_grouped_points(ax, x, y, yerr, ns, group_by_n, group_colors, used_labels)

    ax.plot(
        x_fit,
        y_fit,
        "-",
        linewidth=2.2,
        color=FIT_LINE_COLOR,
        label=rf"fit, $N\to\infty$={fit_record.sf_infinite:.6g}",
    )
    ax.scatter(
        [0.0],
        [fit_record.sf_infinite],
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
    ax.set_ylabel(q_config["label"])
    ax.set_title(f"{q_config['title']} vs 1/N ({args.param_label}={g_value:g})")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)
    ax.legend()
    fig.tight_layout()

    filename = (
        f"annni_sf_vs_invN_{args.param_label}{format_float(g_value)}_"
        f"{q_config['name']}.{args.save_format}"
    )
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, fit_record


def save_by_q_plot(results, q_value, g_values, args, group_by_n, group_colors):
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    fit_records = []
    used_group_labels = set()
    q_config = Q_CONFIG[q_value]

    for i, g_value in enumerate(g_values):
        x, y, yerr, ns = collect_series(
            results,
            q_value,
            float(g_value),
            args.match_tol,
            include_std=not args.no_errorbar,
        )
        fit_record, coeffs = fit_series(x, y, ns, float(g_value), q_value, args.fit_degree)
        fit_records.append(fit_record)
        x_fit = np.linspace(0.0, float(np.max(x)), 200)
        y_fit = np.polyval(coeffs, x_fit)
        label = f"{args.param_label}={g_value:g}, inf={fit_record.sf_infinite:.6g}"

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
            [fit_record.sf_infinite],
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
    ax.set_ylabel(q_config["label"])
    ax.set_title(f"{q_config['title']} vs 1/N")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)
    ax.legend()
    fig.tight_layout()

    filename = f"annni_sf_vs_invN_{q_config['name']}.{args.save_format}"
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, fit_records


def save_fit_records(fit_records, output_dir, csv_name):
    csv_path = os.path.join(output_dir, csv_name)
    fieldnames = [
        "q_value",
        "g_value",
        "sf_infinite",
        "slope",
        "r_squared",
        "fit_degree",
        "num_points",
        "qubits",
        "inverse_qubits",
        "sf_values",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in fit_records:
            writer.writerow(
                {
                    "q_value": record.q_value,
                    "g_value": f"{record.g_value:.12g}",
                    "sf_infinite": f"{record.sf_infinite:.12g}",
                    "slope": f"{record.slope:.12g}",
                    "r_squared": f"{record.r_squared:.12g}",
                    "fit_degree": record.fit_degree,
                    "num_points": record.num_points,
                    "qubits": record.qubits,
                    "inverse_qubits": record.inverse_qubits,
                    "sf_values": record.sf_values,
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
    g_values = select_g_values(results, args.g_values)
    print_loaded_param_axes(results, g_values, args.g_values)

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
        for q_value in args.q_values:
            for g_value in g_values:
                save_path, fit_record = save_separate_plot(
                    results,
                    q_value,
                    float(g_value),
                    args,
                    group_by_n,
                    group_colors,
                )
                save_paths.append(save_path)
                fit_records.append(fit_record)
    elif args.layout == "by_q":
        for q_value in args.q_values:
            save_path, q_fit_records = save_by_q_plot(
                results,
                q_value,
                g_values,
                args,
                group_by_n,
                group_colors,
            )
            save_paths.append(save_path)
            fit_records.extend(q_fit_records)

    csv_path = save_fit_records(fit_records, args.output_dir, args.fit_csv)

    ns_text = ", ".join(str(result.num_qubits) for result in results)
    print(f"Loaded qubit sizes N: {ns_text}")
    if args.qubit_groups is not None:
        print("Qubit groups:")
        for label in dict.fromkeys(group_by_n.values()):
            grouped_ns = sorted(n for n, group_label in group_by_n.items() if group_label == label)
            print(f"  {label}: {grouped_ns}")
    print("Fitted N -> infinity ANNNI structure-factor values:")
    for record in fit_records:
        print(
            f"  q={record.q_value}, {args.param_label}={record.g_value:g}: "
            f"sf_inf={record.sf_infinite:.12g}, slope={record.slope:.12g}, "
            f"R^2={record.r_squared:.6g}"
        )
    print(f"Fit values saved to: {csv_path}")
    print(f"Saved {len(save_paths)} figure(s) to: {args.output_dir}")
    for save_path in save_paths:
        print(f"  {save_path}")


if __name__ == "__main__":
    main()
