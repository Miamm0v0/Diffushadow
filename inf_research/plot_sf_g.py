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

Q_CONFIG = {
    "pi": {
        "mean_keys": ["structure_factor_pi_mean", "sf_pi_mean"],
        "exact_keys": ["exact_structure_factor_pi", "structure_factor_pi"],
        "label": r"$S_Z(\pi)$",
        "title": r"Z structure factor, $q=\pi$",
        "filename": "qpi",
    },
    "pi_over_2": {
        "mean_keys": ["structure_factor_pi_over_2_mean", "sf_pi_over_2_mean"],
        "exact_keys": ["exact_structure_factor_pi_over_2", "structure_factor_pi_over_2"],
        "label": r"$S_Z(\pi/2)$",
        "title": r"Z structure factor, $q=\pi/2$",
        "filename": "qpi_over_2",
    },
}


@dataclass
class EvalData:
    file_path: str
    num_qubits: int | None
    params: np.ndarray
    sf_mean: dict[str, np.ndarray]
    exact_params: np.ndarray | None
    exact_sf: dict[str, np.ndarray | None]


@dataclass
class ExtrapolatedPoint:
    q_value: str
    g_value: float
    sf_infinite: float
    slope: float
    r_squared: float
    fit_degree: int
    num_points: int
    qubits: str
    finite_sf_values: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot extrapolated N -> infinity ANNNI structure factor versus g. "
            "By default this reads the fit CSV produced by "
            "plot_annni_sf_inverse_qubit.py; it can also fall back to fitting "
            "finite-size .npz files directly."
        )
    )
    parser.add_argument(
        "--fit_csv",
        default=None,
        help=(
            "CSV produced by plot_annni_sf_inverse_qubit.py, usually "
            "annni_sf_infinite_fit_values.csv. If provided, no refit is performed."
        ),
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=(
            "Optional ANNNI eval_new.py .npz files. Used for fallback fitting when "
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
        help="Optional finite-N ANNNI .npz files to draw on the same SF-vs-g figures.",
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
        "--q_values",
        nargs="+",
        default=None,
        help="Optional q values: pi and/or pi_over_2. Defaults to all q values in --fit_csv or both npz fields.",
    )
    parser.add_argument(
        "--output_dir",
        default="sf_extrapolated_vs_g_plots",
        help="Directory where figures and CSV values are saved.",
    )
    parser.add_argument(
        "--param_key",
        default="auto",
        help="Parameter key in .npz. Use auto for hs, hs_gpt, h_values, or h_values_dense.",
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
        help="Fallback .npz mode only: polynomial degree for fitting SF as a function of 1/N.",
    )
    parser.add_argument(
        "--match_tol",
        type=float,
        default=1e-6,
        help="Tolerance for matching requested g values to saved grid points.",
    )
    parser.add_argument(
        "--csv_name",
        default="sf_extrapolated_vs_g_values.csv",
        help="CSV filename for extrapolated SF values.",
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
        help="Do not draw exact SF curves even if they are present in the .npz.",
    )
    args = parser.parse_args()
    args.q_values = normalize_q_values(args.q_values) if args.q_values is not None else None
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
    aliases = {
        "pi": "pi",
        "qpi": "pi",
        "pi_over_2": "pi_over_2",
        "pi/2": "pi_over_2",
        "piover2": "pi_over_2",
        "qpi_over_2": "pi_over_2",
        "qpi/2": "pi_over_2",
    }
    q_values = []
    for raw in raw_q_values:
        key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        canonical = aliases.get(key)
        if canonical is None:
            raise ValueError(f"Unknown q value '{raw}'. Supported values are: pi, pi_over_2.")
        if canonical not in q_values:
            q_values.append(canonical)
    if not q_values:
        raise ValueError("--q_values cannot be empty.")
    return q_values


def infer_key(data, requested_key, candidates, field_name):
    if requested_key != "auto":
        if requested_key not in data:
            raise KeyError(f"{field_name} key '{requested_key}' not found.")
        return requested_key

    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Could not infer {field_name} key. Tried: {', '.join(candidates)}")


def maybe_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


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


def as_1d(values, key):
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{key} is empty.")
    return array


def maybe_load_exact(data, q_values):
    exact_param_key = maybe_key(data, ["h_values_dense", "h_values", "h_values_true"])
    if exact_param_key is None:
        return None, {q_value: None for q_value in q_values}

    exact_params = none_if_scalar_none(data[exact_param_key])
    if exact_params is None:
        return None, {q_value: None for q_value in q_values}

    exact_params = np.asarray(exact_params, dtype=float).reshape(-1)
    exact_sf = {}
    for q_value in q_values:
        exact_key = maybe_key(data, Q_CONFIG[q_value]["exact_keys"])
        if exact_key is None:
            exact_sf[q_value] = None
            continue
        values = none_if_scalar_none(data[exact_key])
        if values is None:
            exact_sf[q_value] = None
            continue
        values = as_1d(values, exact_key)
        exact_sf[q_value] = values if values.shape[0] == exact_params.shape[0] else None
    return exact_params, exact_sf


def available_q_values(data, requested_q_values):
    if requested_q_values is not None:
        return requested_q_values
    q_values = []
    for q_value, config in Q_CONFIG.items():
        if any(key in data for key in config["mean_keys"]):
            q_values.append(q_value)
    if not q_values:
        raise KeyError("No ANNNI structure-factor mean keys found in npz file.")
    return q_values


def load_eval_data(file_path, args, explicit_n=None, require_n=True, q_values=None):
    data = np.load(file_path, allow_pickle=True)
    loaded_q_values = q_values if q_values is not None else available_q_values(data, args.q_values)
    param_key = infer_key(data, args.param_key, ["hs", "hs_gpt", "h_values", "h_values_dense"], "parameter")
    params = np.asarray(data[param_key], dtype=float).reshape(-1)

    sf_mean = {}
    for q_value in loaded_q_values:
        mean_key = infer_key(data, "auto", Q_CONFIG[q_value]["mean_keys"], Q_CONFIG[q_value]["label"])
        values = as_1d(data[mean_key], mean_key)
        if values.shape[0] != params.shape[0]:
            raise ValueError(
                f"{file_path}: {mean_key} length ({values.shape[0]}) does not match "
                f"{param_key} length ({params.shape[0]})."
            )
        sf_mean[q_value] = values

    exact_params, exact_sf = maybe_load_exact(data, loaded_q_values)
    num_qubits = infer_num_qubits(file_path, data, explicit_n, required=require_n)
    return EvalData(file_path, num_qubits, params, sf_mean, exact_params, exact_sf)


def load_fit_data(args):
    if args.files is None:
        raise ValueError("Pass --fit_csv, or pass --files for fallback finite-size fitting.")
    if args.num_qubits is not None and len(args.num_qubits) != len(args.files):
        raise ValueError("--num_qubits must have the same length as --files.")

    results = []
    q_values = args.q_values
    for i, file_path in enumerate(args.files):
        explicit_n = args.num_qubits[i] if args.num_qubits is not None else None
        result = load_eval_data(file_path, args, explicit_n=explicit_n, require_n=True, q_values=q_values)
        if q_values is None:
            q_values = list(result.sf_mean)
        results.append(result)

    seen = {}
    for result in results:
        if result.num_qubits in seen:
            raise ValueError(
                f"Duplicate fit qubit size N={result.num_qubits}: "
                f"{seen[result.num_qubits]} and {result.file_path}"
            )
        seen[result.num_qubits] = result.file_path
    return sorted(results, key=lambda item: item.num_qubits)


def load_optional_file_data(args, q_values=None):
    if args.files is None:
        return []
    if args.num_qubits is not None and len(args.num_qubits) != len(args.files):
        raise ValueError("--num_qubits must have the same length as --files.")

    results = []
    for i, file_path in enumerate(args.files):
        explicit_n = args.num_qubits[i] if args.num_qubits is not None else None
        results.append(load_eval_data(file_path, args, explicit_n=explicit_n, require_n=False, q_values=q_values))
    return results


def load_overlay_data(args, q_values=None):
    if args.overlay_files is None:
        return []
    if args.overlay_num_qubits is not None and len(args.overlay_num_qubits) != len(args.overlay_files):
        raise ValueError("--overlay_num_qubits must have the same length as --overlay_files.")
    if args.overlay_labels is not None and len(args.overlay_labels) != len(args.overlay_files):
        raise ValueError("--overlay_labels must have the same length as --overlay_files.")

    overlays = []
    for i, file_path in enumerate(args.overlay_files):
        explicit_n = args.overlay_num_qubits[i] if args.overlay_num_qubits is not None else None
        overlays.append(load_eval_data(file_path, args, explicit_n=explicit_n, require_n=False, q_values=q_values))
    return overlays


def select_g_values(results, requested_g_values):
    if requested_g_values is not None:
        return np.asarray(requested_g_values, dtype=float)
    return np.asarray(results[0].params, dtype=float)


def select_q_values(fit_results, requested_q_values):
    if requested_q_values is not None:
        return requested_q_values
    q_values = []
    for result in fit_results:
        for q_value in result.sf_mean:
            if q_value not in q_values:
                q_values.append(q_value)
    if not q_values:
        raise ValueError("No structure-factor q values are available in --files.")
    return q_values


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


def fit_one_g(fit_results, g_value, q_value, args):
    xs = []
    ys = []
    qubits = []
    for result in fit_results:
        if q_value not in result.sf_mean:
            continue
        index = nearest_param_index(result.params, g_value, args.match_tol)
        xs.append(1.0 / result.num_qubits)
        ys.append(result.sf_mean[q_value][index])
        qubits.append(result.num_qubits)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    qubits = np.asarray(qubits, dtype=int)

    unique_x_count = len(np.unique(xs))
    if unique_x_count < args.fit_degree + 1:
        raise ValueError(
            f"Need at least {args.fit_degree + 1} distinct qubit sizes to fit degree "
            f"{args.fit_degree}, but got {unique_x_count} for q={q_value}, g={g_value:g}."
        )

    coeffs = np.polyfit(xs, ys, args.fit_degree)
    fitted = np.polyval(coeffs, xs)
    ss_res = float(np.sum((ys - fitted) ** 2))
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return ExtrapolatedPoint(
        q_value=q_value,
        g_value=float(g_value),
        sf_infinite=float(coeffs[-1]),
        slope=float(coeffs[-2]),
        r_squared=r_squared,
        fit_degree=args.fit_degree,
        num_points=len(xs),
        qubits=format_array(qubits),
        finite_sf_values=format_array(ys),
    )


def extrapolate_curve(fit_results, g_values, q_value, args):
    points = [fit_one_g(fit_results, float(g), q_value, args) for g in g_values]
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
    points_by_q = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"q_value", "g_value", "sf_infinite"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            q_value = normalize_q_values([row["q_value"]])[0]
            point = ExtrapolatedPoint(
                q_value=q_value,
                g_value=float(row["g_value"]),
                sf_infinite=float(row["sf_infinite"]),
                slope=parse_optional_float(row.get("slope")),
                r_squared=parse_optional_float(row.get("r_squared")),
                fit_degree=parse_optional_int(row.get("fit_degree")),
                num_points=parse_optional_int(row.get("num_points")),
                qubits=row.get("qubits", ""),
                finite_sf_values=row.get("finite_sf_values", row.get("sf_values", "")),
            )
            points_by_q.setdefault(q_value, []).append(point)

    for q_value in points_by_q:
        points_by_q[q_value].sort(key=lambda point: point.g_value)
    return points_by_q


def filter_points_by_args(points_by_q, args):
    available_q_values = sorted(points_by_q)
    if args.q_values is None:
        selected_q_values = available_q_values
    else:
        selected_q_values = args.q_values
        missing = [q_value for q_value in selected_q_values if q_value not in points_by_q]
        if missing:
            raise ValueError(
                f"Requested q values {missing} are absent from --fit_csv. "
                f"Available q values: {available_q_values}"
            )

    filtered = {}
    for q_value in selected_q_values:
        points = points_by_q[q_value]
        if args.g_values is None:
            filtered[q_value] = points
            continue

        selected_points = []
        point_g = np.asarray([point.g_value for point in points], dtype=float)
        for g_value in args.g_values:
            diffs = np.abs(point_g - g_value)
            index = int(np.argmin(diffs))
            if diffs[index] > args.match_tol:
                raise ValueError(
                    f"Requested g={g_value:g} is absent from --fit_csv for q={q_value}. "
                    f"Nearest saved value is {point_g[index]:g}, diff={diffs[index]:g}."
                )
            selected_points.append(points[index])
        filtered[q_value] = selected_points
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


def plot_q(q_value, points, exact_sources, overlay_results, args):
    config = Q_CONFIG[q_value]
    g = np.asarray([point.g_value for point in points], dtype=float)
    sf_inf = np.asarray([point.sf_infinite for point in points], dtype=float)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    if not args.no_exact:
        for result in exact_sources + overlay_results:
            exact_values = result.exact_sf.get(q_value) if result.exact_sf is not None else None
            if result.exact_params is not None and exact_values is not None:
                ax.plot(
                    result.exact_params,
                    exact_values,
                    color=EXACT_COLOR,
                    linewidth=1.8,
                    alpha=0.65,
                    label="Exact",
                )
                break

    ax.plot(
        g,
        sf_inf,
        "o-",
        color=EXTRAPOLATED_COLOR,
        linewidth=2.4,
        markersize=5.5,
        label=r"Extrapolated $N\to\infty$",
    )

    for i, result in enumerate(overlay_results):
        if q_value not in result.sf_mean:
            continue
        color = overlay_color(i, args)
        label = overlay_label(result, i, args)
        y = result.sf_mean[q_value]
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
    ax.set_ylabel(config["label"])
    # ax.set_title(config["title"])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    filename = f"sf_extrapolated_vs_{args.param_label}_{config['filename']}.{args.save_format}"
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi)
    plt.close(fig)
    return save_path



def plot_combined_q(points_by_q, exact_sources, overlay_results, args):
    q_values = sorted(points_by_q)
    q_markers = ["o", "s", "^", "D", "v", "P", "X"]
    q_linestyles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    for q_index, q_value in enumerate(q_values):
        config = Q_CONFIG[q_value]
        q_label = config["label"]
        q_color = NATURE_COLORS[q_index % len(NATURE_COLORS)]
        q_marker = q_markers[q_index % len(q_markers)]
        q_linestyle = q_linestyles[q_index % len(q_linestyles)]
        points = points_by_q[q_value]
        g = np.asarray([point.g_value for point in points], dtype=float)
        sf_inf = np.asarray([point.sf_infinite for point in points], dtype=float)

        if not args.no_exact:
            for result in exact_sources + overlay_results:
                exact_values = result.exact_sf.get(q_value) if result.exact_sf is not None else None
                if result.exact_params is not None and exact_values is not None:
                    ax.plot(
                        result.exact_params,
                        exact_values,
                        color=EXACT_COLOR,
                        linestyle=q_linestyle,
                        linewidth=1.7,
                        alpha=0.7,
                        label=f"Exact {q_label}",
                    )
                    break

        ax.plot(
            g,
            sf_inf,
            marker=q_marker,
            linestyle=q_linestyle,
            color=q_color,
            linewidth=2.4,
            markersize=5.5,
            label=rf"Extrapolated {q_label}, $N\to\infty$",
        )

        for i, result in enumerate(overlay_results):
            if q_value not in result.sf_mean:
                continue
            color = overlay_color(i, args)
            label = f"{overlay_label(result, i, args)} {q_label}"
            y = result.sf_mean[q_value]
            if args.overlay_style == "line":
                ax.plot(
                    result.params,
                    y,
                    marker=q_marker,
                    linestyle=q_linestyle,
                    color=color,
                    linewidth=1.25,
                    markersize=3.8,
                    alpha=0.82,
                    label=label,
                )
            else:
                ax.scatter(
                    result.params,
                    y,
                    color=color,
                    marker=q_marker,
                    s=24,
                    alpha=0.82,
                    label=label,
                )

    ax.set_xlabel(args.param_label)
    ax.set_ylabel(r"$S_Z(q)$")
    # ax.set_title("ANNNI Z structure factor")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 0.5), fontsize=4)
    fig.tight_layout()

    q_part = "_".join(Q_CONFIG[q_value]["filename"] for q_value in q_values)
    filename = f"sf_extrapolated_vs_{args.param_label}_{q_part}_combined.{args.save_format}"
    save_path = os.path.join(args.output_dir, filename)
    fig.savefig(save_path, dpi=args.dpi)
    plt.close(fig)
    return save_path

def save_csv(points_by_q, args):
    csv_path = os.path.join(args.output_dir, args.csv_name)
    fieldnames = [
        "q_value",
        "g_value",
        "sf_infinite",
        "slope",
        "r_squared",
        "fit_degree",
        "num_points",
        "qubits",
        "finite_sf_values",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q_value in sorted(points_by_q):
            for point in points_by_q[q_value]:
                writer.writerow(
                    {
                        "q_value": point.q_value,
                        "g_value": f"{point.g_value:.12g}",
                        "sf_infinite": f"{point.sf_infinite:.12g}",
                        "slope": f"{point.slope:.12g}",
                        "r_squared": f"{point.r_squared:.12g}",
                        "fit_degree": point.fit_degree,
                        "num_points": point.num_points,
                        "qubits": point.qubits,
                        "finite_sf_values": point.finite_sf_values,
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

    if args.fit_csv is not None:
        points_by_q = filter_points_by_args(load_points_from_fit_csv(args.fit_csv), args)
        q_values = list(points_by_q)
        exact_sources = load_optional_file_data(args, q_values=q_values)
        fit_results = []
    else:
        fit_results = load_fit_data(args)
        g_values = select_g_values(fit_results, args.g_values)
        q_values = select_q_values(fit_results, args.q_values)
        points_by_q = {}
        for q_value in q_values:
            points_by_q[q_value] = extrapolate_curve(fit_results, g_values, q_value, args)
        exact_sources = fit_results

    overlay_results = load_overlay_data(args, q_values=list(points_by_q))

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    save_paths = []
    for q_value in sorted(points_by_q):
        save_paths.append(plot_q(q_value, points_by_q[q_value], exact_sources, overlay_results, args))
    if len(points_by_q) > 1:
        save_paths.append(plot_combined_q(points_by_q, exact_sources, overlay_results, args))

    csv_path = save_csv(points_by_q, args)

    if args.fit_csv is not None:
        print(f"Loaded extrapolated values from: {args.fit_csv}")
    else:
        fit_ns = ", ".join(str(result.num_qubits) for result in fit_results)
        print(f"Fit qubit sizes N: {fit_ns}")
    print("Extrapolated N -> infinity ANNNI structure-factor values:")
    for q_value in sorted(points_by_q):
        for point in points_by_q[q_value]:
            print(
                f"  q={point.q_value}, {args.param_label}={point.g_value:g}: "
                f"sf_inf={point.sf_infinite:.12g}, R^2={point.r_squared:.6g}"
            )
    print(f"Values saved to: {csv_path}")
    print(f"Saved {len(save_paths)} figure(s) to: {args.output_dir}")
    for save_path in save_paths:
        print(f"  {save_path}")


if __name__ == "__main__":
    main()
