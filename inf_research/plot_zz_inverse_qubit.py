"""
python plot_zz_vs_inverse_qubit.py `
  --files result_N8.npz result_N10.npz result_N12.npz `
  --num_qubits 8 10 12 `
  --g_values 0.2 0.5 0.8 `
  --distances 1 2 3 `
  --output_dir zz_invN_figs
"""





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


@dataclass
class EvalResult:
    file_path: str
    num_qubits: int
    params: np.ndarray
    zz_mean: np.ndarray
    zz_std: np.ndarray | None


@dataclass
class FitRecord:
    distance: int
    g_value: float
    zz_infinite: float
    slope: float
    r_squared: float
    fit_degree: int
    num_points: int
    qubits: str
    inverse_qubits: str
    zz_values: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ZZ correlation versus 1 / num_qubits from multiple eval_new.py "
            ".npz result files."
        )
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="eval_new.py output .npz files. Each file should correspond to one qubit size.",
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
        help="Optional g/h/J values to plot. By default all parameter points in the first file are used.",
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
        default="zz_vs_inverse_qubit_plots",
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--layout",
        choices=["separate", "by_distance"],
        default="separate",
        help=(
            "separate: one figure for every (g, d). "
            "by_distance: one figure for every d, with different g values in the legend."
        ),
    )
    parser.add_argument(
        "--param_key",
        type=str,
        default="auto",
        help="Parameter key in .npz. Use auto for hs_gpt or Js.",
    )
    parser.add_argument(
        "--zz_key",
        type=str,
        default="auto",
        help="ZZ mean key in .npz. Use auto for zz_mean or corr_ZZ_mean.",
    )
    parser.add_argument(
        "--zz_std_key",
        type=str,
        default="auto",
        help="ZZ std key in .npz. Use auto for zz_std or corr_ZZ_std.",
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
        help="Polynomial degree for fitting ZZ as a function of 1/N. Default is linear.",
    )
    parser.add_argument(
        "--fit_csv",
        type=str,
        default="zz_infinite_fit_values.csv",
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
            raise KeyError(f"{field_name} key '{requested_key}' not found in file.")
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


def as_2d_zz(array, key):
    values = np.asarray(array, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{key} must have shape (num_distances, num_params), got {values.shape}.")
    return values


def load_eval_result(file_path, args, explicit_n=None):
    data = np.load(file_path, allow_pickle=True)
    param_key = infer_key(data, args.param_key, ["hs_gpt", "Js", "Deltas"], "parameter")
    zz_key = infer_key(data, args.zz_key, ["zz_mean", "corr_ZZ_mean"], "ZZ mean")

    if args.zz_std_key == "auto":
        zz_std_key = None
        for candidate in ["zz_std", "corr_ZZ_std"]:
            if candidate in data:
                zz_std_key = candidate
                break
    else:
        zz_std_key = args.zz_std_key
        if zz_std_key not in data:
            raise KeyError(f"ZZ std key '{zz_std_key}' not found in file.")

    params = np.asarray(data[param_key], dtype=float).reshape(-1)
    zz_mean = as_2d_zz(data[zz_key], zz_key)
    zz_std = as_2d_zz(data[zz_std_key], zz_std_key) if zz_std_key is not None else None
    num_qubits = infer_num_qubits(file_path, data, explicit_n)

    if zz_mean.shape[1] != params.shape[0]:
        raise ValueError(
            f"{file_path}: {zz_key}.shape[1] ({zz_mean.shape[1]}) does not match "
            f"{param_key} length ({params.shape[0]})."
        )
    if zz_std is not None and zz_std.shape != zz_mean.shape:
        raise ValueError(f"{file_path}: {zz_std_key}.shape {zz_std.shape} != {zz_key}.shape {zz_mean.shape}.")

    return EvalResult(
        file_path=file_path,
        num_qubits=num_qubits,
        params=params,
        zz_mean=zz_mean,
        zz_std=zz_std,
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


def select_distances(results, requested_distances):
    max_distance = min(result.zz_mean.shape[0] for result in results)
    if requested_distances is None:
        return list(range(1, max_distance + 1))

    distances = [int(d) for d in requested_distances]
    invalid = [d for d in distances if d < 1 or d > max_distance]
    if invalid:
        raise ValueError(f"Invalid distances {invalid}. Available range is 1..{max_distance}.")
    return distances


def nearest_param_index(params, target, tol):
    distances = np.abs(params - target)
    index = int(np.argmin(distances))
    if distances[index] > tol:
        raise ValueError(
            f"Requested parameter {target:g} was not found. "
            f"Nearest saved value is {params[index]:g}, diff={distances[index]:g}."
        )
    return index


def collect_series(results, g_value, distance, tol, include_std):
    xs = []
    ys = []
    yerrs = []
    ns = []

    for result in results:
        index = nearest_param_index(result.params, g_value, tol)
        xs.append(1.0 / result.num_qubits)
        ys.append(result.zz_mean[distance - 1, index])
        if include_std and result.zz_std is not None:
            yerrs.append(result.zz_std[distance - 1, index])
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


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)


def fit_series(x, y, ns, g_value, distance, fit_degree):
    if fit_degree < 1:
        raise ValueError("--fit_degree must be >= 1.")

    unique_x_count = len(np.unique(x))
    if unique_x_count < fit_degree + 1:
        raise ValueError(
            f"Need at least {fit_degree + 1} distinct qubit sizes to fit degree "
            f"{fit_degree}, but got {unique_x_count} for {distance=}, "
            f"{g_value=}."
        )

    coeffs = np.polyfit(x, y, fit_degree)
    fitted_y = np.polyval(coeffs, x)
    residual = y - fitted_y
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitRecord(
        distance=distance,
        g_value=float(g_value),
        zz_infinite=float(coeffs[-1]),
        slope=float(coeffs[-2]),
        r_squared=r_squared,
        fit_degree=fit_degree,
        num_points=len(x),
        qubits=format_array(ns),
        inverse_qubits=format_array(x),
        zz_values=format_array(y),
    ), coeffs


def save_separate_plot(results, g_value, distance, args):
    x, y, yerr, ns = collect_series(
        results,
        g_value,
        distance,
        args.match_tol,
        include_std=not args.no_errorbar,
    )
    fit_record, coeffs = fit_series(x, y, ns, g_value, distance, args.fit_degree)
    x_fit = np.linspace(0.0, float(np.max(x)), 200)
    y_fit = np.polyval(coeffs, x_fit)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    if yerr is not None:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=3,
            markersize=5,
            color=palette_color(0),
            ecolor=palette_color(0),
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="data",
        )
    else:
        ax.scatter(
            x,
            y,
            s=28,
            color=palette_color(0),
            edgecolor="white",
            linewidths=0.6,
            label="data",
        )

    ax.plot(
        x_fit,
        y_fit,
        "-",
        linewidth=2.2,
        color=FIT_LINE_COLOR,
        label=rf"fit, $N\to\infty$={fit_record.zz_infinite:.6g}",
    )
    ax.scatter(
        [0.0],
        [fit_record.zz_infinite],
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
    ax.set_ylabel(r"$\langle Z_i Z_{i+d} \rangle$")
    ax.set_title(f"ZZ vs 1/N ({args.param_label}={g_value:g}, d={distance})")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)
    ax.legend()
    fig.tight_layout()

    filename = f"zz_vs_invN_{args.param_label}{format_float(g_value)}_d{distance}.{args.save_format}"
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, fit_record


def save_by_distance_plot(results, g_values, distance, args):
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    fit_records = []

    for i, g_value in enumerate(g_values):
        x, y, yerr, ns = collect_series(
            results,
            g_value,
            distance,
            args.match_tol,
            include_std=not args.no_errorbar,
        )
        color = palette_color(i)
        fit_record, coeffs = fit_series(x, y, ns, float(g_value), distance, args.fit_degree)
        fit_records.append(fit_record)
        x_fit = np.linspace(0.0, float(np.max(x)), 200)
        y_fit = np.polyval(coeffs, x_fit)
        label = f"{args.param_label}={g_value:g}, inf={fit_record.zz_infinite:.6g}"

        if yerr is not None:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                capsize=3,
                markersize=4.5,
                color=color,
                ecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=label,
            )
        else:
            ax.scatter(x, y, s=24, color=color, edgecolor="white", linewidths=0.6, label=label)

        ax.plot(
            x_fit,
            y_fit,
            linestyle=FIT_LINE_STYLES[i % len(FIT_LINE_STYLES)],
            linewidth=2.0,
            color=FIT_LINE_COLOR,
            alpha=0.9,
        )
        ax.scatter(
            [0.0],
            [fit_record.zz_infinite],
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
    ax.set_ylabel(r"$\langle Z_i Z_{i+d} \rangle$")
    ax.set_title(f"ZZ vs 1/N (d={distance})")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)
    ax.legend()
    fig.tight_layout()

    filename = f"zz_vs_invN_d{distance}.{args.save_format}"
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path, fit_records


def save_fit_records(fit_records, output_dir, csv_name):
    csv_path = os.path.join(output_dir, csv_name)
    fieldnames = [
        "distance",
        "g_value",
        "zz_infinite",
        "slope",
        "r_squared",
        "fit_degree",
        "num_points",
        "qubits",
        "inverse_qubits",
        "zz_values",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in fit_records:
            writer.writerow(
                {
                    "distance": record.distance,
                    "g_value": f"{record.g_value:.12g}",
                    "zz_infinite": f"{record.zz_infinite:.12g}",
                    "slope": f"{record.slope:.12g}",
                    "r_squared": f"{record.r_squared:.12g}",
                    "fit_degree": record.fit_degree,
                    "num_points": record.num_points,
                    "qubits": record.qubits,
                    "inverse_qubits": record.inverse_qubits,
                    "zz_values": record.zz_values,
                }
            )
    return csv_path


def main():
    global plt
    args = parse_args()
    plt = import_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_all_results(args)
    g_values = select_g_values(results, args.g_values)
    distances = select_distances(results, args.distances)

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
            for g_value in g_values:
                save_path, fit_record = save_separate_plot(results, float(g_value), distance, args)
                save_paths.append(save_path)
                fit_records.append(fit_record)
    elif args.layout == "by_distance":
        for distance in distances:
            save_path, distance_fit_records = save_by_distance_plot(results, g_values, distance, args)
            save_paths.append(save_path)
            fit_records.extend(distance_fit_records)

    csv_path = save_fit_records(fit_records, args.output_dir, args.fit_csv)

    ns_text = ", ".join(str(result.num_qubits) for result in results)
    print(f"Loaded qubit sizes N: {ns_text}")
    print("Fitted N -> infinity ZZ values:")
    for record in fit_records:
        print(
            f"  d={record.distance}, {args.param_label}={record.g_value:g}: "
            f"zz_inf={record.zz_infinite:.12g}, slope={record.slope:.12g}, "
            f"R^2={record.r_squared:.6g}"
        )
    print(f"Fit values saved to: {csv_path}")
    print(f"Saved {len(save_paths)} figure(s) to: {args.output_dir}")
    for save_path in save_paths:
        print(f"  {save_path}")


if __name__ == "__main__":
    main()
