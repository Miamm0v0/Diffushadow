import argparse
import csv
import os
import re
from dataclasses import dataclass

import numpy as np


plt = None

EXTRAPOLATED_COLOR = "#3C5488"
EXACT_COLOR = "#7A7A7A"
NATURE_COLORS = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
    "#3B4992",
    "#008B45",
    "#631879",
    "#008280",
    "#BB0021",
    "#5F559B",
    "#A20056",
    "#D39200",
    "#93AA00",
    "#00B9E3",
]
OVERLAY_MARKER = "o"


@dataclass
class EvalData:
    file_path: str
    num_qubits: int | None
    params: np.ndarray
    zz_mean: np.ndarray
    exact_params: np.ndarray | None
    exact_zz: np.ndarray | None


@dataclass
class ExtrapolatedPoint:
    distance: int
    g_value: float
    zz_infinite: float
    slope: float
    r_squared: float
    fit_degree: int
    num_points: int
    qubits: str
    finite_zz_values: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot extrapolated N -> infinity ZZ versus g. By default this reads "
            "the fit CSV produced by plot_zz_vs_inverse_qubit.py; it can also "
            "fall back to fitting finite-size .npz files directly."
        )
    )
    parser.add_argument(
        "--fit_csv",
        default=None,
        help=(
            "CSV produced by plot_zz_vs_inverse_qubit.py, usually "
            "zz_infinite_fit_values.csv. If provided, no refit is performed."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=(
            "Optional eval_new.py .npz files. Used for fallback fitting when "
            "--fit_csv is absent, or as exact-curve sources when --fit_csv is present."
        ),
    )
    parser.add_argument(
        "--num_qubits",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Qubit sizes matching --files. Use this when N is absent from the "
            ".npz and cannot be parsed from the filename."
        ),
    )
    parser.add_argument(
        "--overlay_files",
        nargs="+",
        default=None,
        help="Optional finite-N .npz files to draw on the same ZZ-vs-g figures.",
    )
    parser.add_argument(
        "--overlay_num_qubits",
        nargs="+",
        type=int,
        default=None,
        help="Qubit sizes matching --overlay_files.",
    )
    parser.add_argument(
        "--overlay_labels",
        nargs="+",
        default=None,
        help="Labels matching --overlay_files. Defaults to N=<num_qubits> or filename.",
    )
    parser.add_argument(
        "--g_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional g values to draw. Defaults to all g points in --fit_csv or the first .npz file.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=None,
        help="Optional correlation distances d. Defaults to all saved distances.",
    )
    parser.add_argument(
        "--output_dir",
        default="zz_extrapolated_vs_g_plots",
        help="Directory where figures and CSV values are saved.",
    )
    parser.add_argument(
        "--param_key",
        default="auto",
        help="Parameter key in .npz. Use auto for hs_gpt, Js, or Deltas.",
    )
    parser.add_argument(
        "--zz_key",
        default="auto",
        help="ZZ mean key in .npz. Use auto for zz_mean or corr_ZZ_mean.",
    )
    parser.add_argument(
        "--param_label",
        default="g",
        help="x-axis label for the scanned parameter.",
    )
    parser.add_argument(
        "--fit_degree",
        type=int,
        default=1,
        help="Fallback .npz mode only: polynomial degree for fitting ZZ as a function of 1/N.",
    )
    parser.add_argument(
        "--match_tol",
        type=float,
        default=1e-6,
        help="Tolerance for matching requested g values to saved grid points.",
    )
    parser.add_argument(
        "--csv_name",
        default="zz_extrapolated_vs_g_values.csv",
        help="CSV filename for extrapolated ZZ values.",
    )
    parser.add_argument(
        "--save_format",
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Figure file format.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--fig_width", type=float, default=7.0, help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=5.0, help="Figure height in inches.")
    parser.add_argument(
        "--overlay_style",
        choices=["scatter", "line"],
        default="scatter",
        help="How to draw optional finite-N overlay files.",
    )
    parser.add_argument(
        "--palette",
        choices=["nature", "tab10"],
        default="nature",
        help="Color palette for finite-N overlays.",
    )
    parser.add_argument(
        "--no_exact",
        action="store_true",
        help="Do not draw exact ZZ curves even if they are present in the .npz.",
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
            raise KeyError(f"{field_name} key '{requested_key}' not found.")
        return requested_key

    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Could not infer {field_name} key. Tried: {', '.join(candidates)}")


def none_if_scalar_none(value):
    if isinstance(value, np.ndarray) and value.shape == () and value.item() is None:
        return None
    return value


def infer_num_qubits(file_path, data, explicit_n=None, required=True):
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

    if required:
        raise ValueError(
            f"Cannot infer num_qubits for '{file_path}'. "
            "Pass --num_qubits or --overlay_num_qubits."
        )
    return None


def as_2d(values, key):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{key} must have shape (num_distances, num_params), got {array.shape}.")
    return array


def maybe_load_exact(data):
    exact_param_key = None
    for key in ["h_values_true", "J_values_dense", "Delta_values_dense"]:
        if key in data:
            exact_param_key = key
            break
    if exact_param_key is None:
        return None, None

    exact_zz_key = None
    for key in ["ZZ_curves", "exact_corr_ZZ", "exact_zz_correlation"]:
        if key in data:
            exact_zz_key = key
            break
    if exact_zz_key is None:
        return None, None

    exact_params = none_if_scalar_none(data[exact_param_key])
    exact_zz = none_if_scalar_none(data[exact_zz_key])
    if exact_params is None or exact_zz is None:
        return None, None

    exact_params = np.asarray(exact_params, dtype=float).reshape(-1)
    exact_zz = as_2d(exact_zz, exact_zz_key)
    if exact_zz.shape[1] != exact_params.shape[0]:
        return None, None
    return exact_params, exact_zz


def load_eval_data(file_path, args, explicit_n=None, require_n=True):
    data = np.load(file_path, allow_pickle=True)
    param_key = infer_key(data, args.param_key, ["hs_gpt", "Js", "Deltas"], "parameter")
    zz_key = infer_key(data, args.zz_key, ["zz_mean", "corr_ZZ_mean"], "ZZ mean")

    params = np.asarray(data[param_key], dtype=float).reshape(-1)
    zz_mean = as_2d(data[zz_key], zz_key)
    if zz_mean.shape[1] != params.shape[0]:
        raise ValueError(
            f"{file_path}: {zz_key}.shape[1] ({zz_mean.shape[1]}) does not match "
            f"{param_key} length ({params.shape[0]})."
        )

    exact_params, exact_zz = maybe_load_exact(data)
    num_qubits = infer_num_qubits(file_path, data, explicit_n, required=require_n)
    return EvalData(file_path, num_qubits, params, zz_mean, exact_params, exact_zz)


def load_fit_data(args):
    if args.files is None:
        raise ValueError("Pass --fit_csv, or pass --files for fallback finite-size fitting.")
    if args.num_qubits is not None and len(args.num_qubits) != len(args.files):
        raise ValueError("--num_qubits must have the same length as --files.")

    results = []
    for i, file_path in enumerate(args.files):
        explicit_n = args.num_qubits[i] if args.num_qubits is not None else None
        results.append(load_eval_data(file_path, args, explicit_n=explicit_n, require_n=True))

    seen = {}
    for result in results:
        if result.num_qubits in seen:
            raise ValueError(
                f"Duplicate fit qubit size N={result.num_qubits}: "
                f"{seen[result.num_qubits]} and {result.file_path}"
            )
        seen[result.num_qubits] = result.file_path
    return sorted(results, key=lambda item: item.num_qubits)


def load_optional_file_data(args):
    if args.files is None:
        return []
    if args.num_qubits is not None and len(args.num_qubits) != len(args.files):
        raise ValueError("--num_qubits must have the same length as --files.")

    results = []
    for i, file_path in enumerate(args.files):
        explicit_n = args.num_qubits[i] if args.num_qubits is not None else None
        results.append(load_eval_data(file_path, args, explicit_n=explicit_n, require_n=False))
    return results


def load_overlay_data(args):
    if args.overlay_files is None:
        return []
    if args.overlay_num_qubits is not None and len(args.overlay_num_qubits) != len(args.overlay_files):
        raise ValueError("--overlay_num_qubits must have the same length as --overlay_files.")
    if args.overlay_labels is not None and len(args.overlay_labels) != len(args.overlay_files):
        raise ValueError("--overlay_labels must have the same length as --overlay_files.")

    overlays = []
    for i, file_path in enumerate(args.overlay_files):
        explicit_n = args.overlay_num_qubits[i] if args.overlay_num_qubits is not None else None
        overlays.append(load_eval_data(file_path, args, explicit_n=explicit_n, require_n=False))
    return overlays


def select_g_values(results, requested_g_values):
    if requested_g_values is not None:
        return np.asarray(requested_g_values, dtype=float)
    return np.asarray(results[0].params, dtype=float)


def select_distances(fit_results, overlay_results, requested_distances):
    max_distance = min(result.zz_mean.shape[0] for result in fit_results)
    for result in overlay_results:
        max_distance = min(max_distance, result.zz_mean.shape[0])

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
            f"Requested g={target:g} was not found. "
            f"Nearest saved value is {params[index]:g}, diff={distances[index]:g}."
        )
    return index


def format_array(values):
    return ";".join(f"{value:.12g}" for value in values)


def fit_one_g(fit_results, g_value, distance, args):
    xs = []
    ys = []
    qubits = []
    for result in fit_results:
        index = nearest_param_index(result.params, g_value, args.match_tol)
        xs.append(1.0 / result.num_qubits)
        ys.append(result.zz_mean[distance - 1, index])
        qubits.append(result.num_qubits)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    qubits = np.asarray(qubits, dtype=int)

    unique_x_count = len(np.unique(xs))
    if unique_x_count < args.fit_degree + 1:
        raise ValueError(
            f"Need at least {args.fit_degree + 1} distinct qubit sizes to fit degree "
            f"{args.fit_degree}, but got {unique_x_count} for d={distance}, g={g_value:g}."
        )

    coeffs = np.polyfit(xs, ys, args.fit_degree)
    fitted = np.polyval(coeffs, xs)
    ss_res = float(np.sum((ys - fitted) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return ExtrapolatedPoint(
        distance=distance,
        g_value=float(g_value),
        zz_infinite=float(coeffs[-1]),
        slope=float(coeffs[-2]),
        r_squared=r_squared,
        fit_degree=args.fit_degree,
        num_points=len(xs),
        qubits=format_array(qubits),
        finite_zz_values=format_array(ys),
    )


def extrapolate_curve(fit_results, g_values, distance, args):
    points = [fit_one_g(fit_results, float(g), distance, args) for g in g_values]
    points.sort(key=lambda point: point.g_value)
    return points


def parse_optional_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def parse_optional_int(value):
    if value is None or value == "":
        return 0
    return int(float(value))


def load_points_from_fit_csv(csv_path):
    points_by_distance = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"distance", "g_value", "zz_infinite"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            distance = int(float(row["distance"]))
            point = ExtrapolatedPoint(
                distance=distance,
                g_value=float(row["g_value"]),
                zz_infinite=float(row["zz_infinite"]),
                slope=parse_optional_float(row.get("slope")),
                r_squared=parse_optional_float(row.get("r_squared")),
                fit_degree=parse_optional_int(row.get("fit_degree")),
                num_points=parse_optional_int(row.get("num_points")),
                qubits=row.get("qubits", ""),
                finite_zz_values=row.get("finite_zz_values", row.get("zz_values", "")),
            )
            points_by_distance.setdefault(distance, []).append(point)

    for distance in points_by_distance:
        points_by_distance[distance].sort(key=lambda point: point.g_value)
    return points_by_distance


def filter_points_by_args(points_by_distance, args):
    available_distances = sorted(points_by_distance)
    if args.distances is None:
        selected_distances = available_distances
    else:
        selected_distances = [int(d) for d in args.distances]
        missing = [d for d in selected_distances if d not in points_by_distance]
        if missing:
            raise ValueError(
                f"Requested distances {missing} are absent from --fit_csv. "
                f"Available distances: {available_distances}"
            )

    filtered = {}
    for distance in selected_distances:
        points = points_by_distance[distance]
        if args.g_values is None:
            filtered[distance] = points
            continue

        selected_points = []
        point_g = np.asarray([point.g_value for point in points], dtype=float)
        for g_value in args.g_values:
            diffs = np.abs(point_g - g_value)
            index = int(np.argmin(diffs))
            if diffs[index] > args.match_tol:
                raise ValueError(
                    f"Requested g={g_value:g} is absent from --fit_csv for d={distance}. "
                    f"Nearest saved value is {point_g[index]:g}, diff={diffs[index]:g}."
                )
            selected_points.append(points[index])
        filtered[distance] = selected_points
    return filtered


def overlay_label(result, index, args):
    if args.overlay_labels is not None:
        return args.overlay_labels[index]
    if result.num_qubits is not None:
        return f"N={result.num_qubits}"
    return os.path.basename(result.file_path).replace(".npz", "")


def overlay_color(index, args):
    if args.palette == "tab10":
        return plt.colormaps["tab10"](index % 10)
    return NATURE_COLORS[index % len(NATURE_COLORS)]


def plot_distance(distance, points, exact_sources, overlay_results, args):
    g = np.asarray([point.g_value for point in points], dtype=float)
    zz_inf = np.asarray([point.zz_infinite for point in points], dtype=float)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    if not args.no_exact:
        for result in exact_sources + overlay_results:
            if result.exact_params is not None and result.exact_zz is not None:
                if distance - 1 < result.exact_zz.shape[0]:
                    ax.plot(
                        result.exact_params,
                        result.exact_zz[distance - 1],
                        color=EXACT_COLOR,
                        linewidth=1.8,
                        alpha=0.65,
                        label="Exact",
                    )
                    break

    ax.plot(
        g,
        zz_inf,
        "o-",
        color=EXTRAPOLATED_COLOR,
        linewidth=2.4,
        markersize=5.5,
        label=r"$N\to\infty$",
    )

    for i, result in enumerate(overlay_results):
        if distance - 1 >= result.zz_mean.shape[0]:
            continue
        color = overlay_color(i, args)
        label = overlay_label(result, i, args)
        y = result.zz_mean[distance - 1]
        if args.overlay_style == "line":
            ax.plot(
                result.params,
                y,
                marker=OVERLAY_MARKER,
                linestyle="-",
                color=color,
                linewidth=1.35,
                markersize=3.8,
                alpha=0.9,
                label=label,
            )
        else:
            ax.scatter(
                result.params,
                y,
                color=color,
                marker=OVERLAY_MARKER,
                s=24,
                alpha=0.86,
                label=label,
            )

    ax.set_xlabel(args.param_label)
    ax.set_ylabel(r"$\langle Z_i Z_{i+d} \rangle$")
    ax.set_title(f"ZZ correlation, d={distance}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    filename = f"zz_extrapolated_vs_{args.param_label}_d{distance}.{args.save_format}"
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_csv(points_by_distance, args):
    csv_path = os.path.join(args.output_dir, args.csv_name)
    fieldnames = [
        "distance",
        "g_value",
        "zz_infinite",
        "slope",
        "r_squared",
        "fit_degree",
        "num_points",
        "qubits",
        "finite_zz_values",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for distance in sorted(points_by_distance):
            for point in points_by_distance[distance]:
                writer.writerow(
                    {
                        "distance": point.distance,
                        "g_value": f"{point.g_value:.12g}",
                        "zz_infinite": f"{point.zz_infinite:.12g}",
                        "slope": f"{point.slope:.12g}",
                        "r_squared": f"{point.r_squared:.12g}",
                        "fit_degree": point.fit_degree,
                        "num_points": point.num_points,
                        "qubits": point.qubits,
                        "finite_zz_values": point.finite_zz_values,
                    }
                )
    return csv_path


def main():
    global plt
    args = parse_args()
    plt = import_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.fit_csv is None and args.files is None:
        raise ValueError("Pass --fit_csv to read existing extrapolated values, or --files to refit .npz data.")

    exact_sources = load_optional_file_data(args) if args.fit_csv is not None else []
    fit_results = [] if args.fit_csv is not None else load_fit_data(args)
    overlay_results = load_overlay_data(args)

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

    if args.fit_csv is not None:
        points_by_distance = filter_points_by_args(load_points_from_fit_csv(args.fit_csv), args)
    else:
        g_values = select_g_values(fit_results, args.g_values)
        distances = select_distances(fit_results, overlay_results, args.distances)
        points_by_distance = {}
        for distance in distances:
            points_by_distance[distance] = extrapolate_curve(fit_results, g_values, distance, args)
        exact_sources = fit_results

    save_paths = []
    for distance in sorted(points_by_distance):
        save_paths.append(plot_distance(distance, points_by_distance[distance], exact_sources, overlay_results, args))

    csv_path = save_csv(points_by_distance, args)

    if args.fit_csv is not None:
        print(f"Loaded extrapolated values from: {args.fit_csv}")
    else:
        fit_ns = ", ".join(str(result.num_qubits) for result in fit_results)
        print(f"Fit qubit sizes N: {fit_ns}")
    print("Extrapolated N -> infinity ZZ values:")
    for distance in sorted(points_by_distance):
        for point in points_by_distance[distance]:
            print(
                f"  d={point.distance}, {args.param_label}={point.g_value:g}: "
                f"zz_inf={point.zz_infinite:.12g}, R^2={point.r_squared:.6g}"
            )
    print(f"Values saved to: {csv_path}")
    print(f"Saved {len(save_paths)} figure(s) to: {args.output_dir}")
    for save_path in save_paths:
        print(f"  {save_path}")


if __name__ == "__main__":
    main()
