"""
Plot J1-J2 spin-dot correlations versus 1/N for two selected parameter indices.

This mirrors plot_annni_sf_inverse_qubit_g_indices.py, but reads the J1-J2
spin-dot outputs handled by plot_j1j2_spin_dot_inverse_qubit.py.

Example:
  python inf_research/plot_j1j2_spin_dot_inverse_qubit_g_indices.py `
    --files j1j2_N12.npz j1j2_N16.npz j1j2_N20.npz `
    --num_qubits 12 16 20 `
    --g_indices 6 18 `
    --distances 1 2 `
    --output_dir j1j2_spin_dot_invN_gidx_figs
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from plot_j1j2_spin_dot_inverse_qubit import (
    FIT_LINE_STYLES,
    NATURE_PALETTE,
    fit_series,
    import_matplotlib,
    load_all_results,
    maybe_find_distance_index,
    save_fit_records,
    select_distances,
    spin_dot_ylabel,
    style_axis,
)


plt = None
POINT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two parameter-index J1-J2 spin-dot finite-size curves together."
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
        help="Optional qubit sizes matching --files.",
    )
    parser.add_argument(
        "--g_indices",
        "--j2_indices",
        dest="g_indices",
        nargs=2,
        type=int,
        required=True,
        help="Exactly two saved parameter indices to plot. Default indexing is zero-based.",
    )
    parser.add_argument(
        "--index_base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use 0 for Python-style indices, or 1 for one-based indices.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=None,
        help="Correlation distances d to plot. Defaults to all saved spin-dot distances.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="j1j2_spin_dot_invN_gidx_figs",
        help="Directory where figures and fit CSV are saved.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional full output path. With multiple distances, suffixes are appended.",
    )
    parser.add_argument(
        "--param_key",
        type=str,
        default="auto",
        help="Parameter key in .npz. Use auto for J2s, hs_gpt, Js, Deltas, J2_values, or hs.",
    )
    parser.add_argument(
        "--spin_dot_key",
        type=str,
        default="auto",
        help="Mean key for spin-dot. Use auto for corr_spin_dot_mean or spin_dot_mean.",
    )
    parser.add_argument(
        "--spin_dot_std_key",
        type=str,
        default="auto",
        help="Std key for spin-dot. Use auto for corr_spin_dot_std or spin_dot_std.",
    )
    parser.add_argument(
        "--param_label",
        type=str,
        default="J2",
        help="Label used in titles and legends for the scanned parameter.",
    )
    parser.add_argument(
        "--fit_degree",
        type=int,
        default=1,
        help="Polynomial degree for fitting spin-dot as a function of 1/N.",
    )
    parser.add_argument(
        "--fit_csv",
        type=str,
        default="j1j2_spin_dot_gidx_infinite_fit_values.csv",
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
    parser.add_argument("--fig_width", type=float, default=7.2, help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=5.2, help="Figure height in inches.")
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
    parser.add_argument(
        "--no_y_break",
        action="store_true",
        help="Disable automatic broken y-axis even when the two selected index ranges are separated.",
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
    parser.add_argument(
        "--param_mismatch_tol",
        type=float,
        default=1e-8,
        help="Warn when the same index maps to parameter values differing by more than this across files.",
    )
    return parser.parse_args()


def normalize_index(raw_index, length, index_base):
    index = int(raw_index)
    if index_base == 1:
        if index <= 0:
            raise ValueError("With --index_base 1, --g_indices must be positive.")
        index -= 1
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise ValueError(f"g index {raw_index} is out of range for reference grid length {length}.")
    return index


def select_g_indices(results, args):
    reference = results[0]
    length = reference.params.shape[0]
    zero_based = [normalize_index(index, length, args.index_base) for index in args.g_indices]
    if zero_based[0] == zero_based[1]:
        raise ValueError("--g_indices must refer to two distinct grid points.")
    display = list(args.g_indices)
    reference_g = np.asarray([reference.params[index] for index in zero_based], dtype=float)
    return zero_based, display, reference_g


def format_float_for_name(value):
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def distance_name(distance):
    return f"r{int(distance)}"


def default_output_path(args, display_indices, reference_g):
    idx_part = "_".join(f"idx{index}" for index in display_indices)
    value_part = "_".join(f"g{format_float_for_name(g)}" for g in reference_g)
    distance_part = "_".join(distance_name(distance) for distance in args.distances)
    return os.path.join(
        args.output_dir,
        f"j1j2_spin_dot_vs_invN_{idx_part}_{value_part}_{distance_part}.{args.save_format}",
    )


def output_path_for_distance(args, distance, display_indices, reference_g):
    if args.output is None:
        plot_args = argparse.Namespace(**vars(args))
        plot_args.distances = [distance]
        return default_output_path(plot_args, display_indices, reference_g)

    if len(args.distances) == 1:
        return args.output

    root, ext = os.path.splitext(args.output)
    if not ext:
        ext = f".{args.save_format}"
    return f"{root}_{distance_name(distance)}{ext}"


def output_path_for_combined(args, display_indices, reference_g):
    if args.output is None:
        return default_output_path(args, display_indices, reference_g)

    if len(args.distances) == 1:
        return args.output

    root, ext = os.path.splitext(args.output)
    if not ext:
        ext = f".{args.save_format}"
    distance_part = "_".join(distance_name(distance) for distance in args.distances)
    return f"{root}_{distance_part}{ext}"


def collect_series_by_index(results, distance, g_index, args):
    xs = []
    ys = []
    yerrs = []
    ns = []
    params_at_index = []

    for result in results:
        index = g_index
        if index >= result.params.shape[0]:
            raise ValueError(
                f"{result.file_path}: requested zero-based g index {g_index}, "
                f"but parameter grid has length {result.params.shape[0]}."
            )
        distance_index = maybe_find_distance_index(result, distance)
        if distance_index is None:
            continue

        y_value = float(result.spin_dot_mean[distance_index, index])
        if not np.isfinite(y_value):
            continue
        xs.append(1.0 / result.num_qubits)
        ys.append(y_value)

        if (not args.no_errorbar) and result.spin_dot_std is not None:
            yerr_value = float(result.spin_dot_std[distance_index, index])
            yerrs.append(yerr_value if np.isfinite(yerr_value) else np.nan)
        else:
            yerrs.append(np.nan)

        ns.append(result.num_qubits)
        params_at_index.append(float(result.params[index]))

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yerrs = np.asarray(yerrs, dtype=float)
    ns = np.asarray(ns, dtype=int)
    params_at_index = np.asarray(params_at_index, dtype=float)
    order = np.argsort(xs)

    yerr_out = yerrs[order]
    if np.any(np.isnan(yerr_out)):
        yerr_out = None
    return xs[order], ys[order], yerr_out, ns[order], params_at_index[order]


def update_range(values_by_index, display_index, *arrays):
    values = []
    for array in arrays:
        if array is None:
            continue
        arr = np.asarray(array, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if values:
        values_by_index.setdefault(display_index, []).append(np.concatenate(values))


def warn_if_param_mismatch(display_index, reference_g, params_at_index, ns, args):
    diffs = np.abs(params_at_index - float(reference_g))
    if diffs.size and float(np.nanmax(diffs)) > args.param_mismatch_tol:
        pairs = ", ".join(
            f"N={int(n)}:{g_value:.12g}" for n, g_value in zip(ns, params_at_index)
        )
        print(
            f"Warning: g index {display_index} maps to different {args.param_label} values "
            f"across files. Reference={float(reference_g):.12g}; per-file values: {pairs}"
        )


def build_plot_items(results, zero_indices, display_indices, reference_g_values, args):
    items = []
    fit_records = []
    values_by_index = {}

    for g_order, (zero_index, display_index, reference_g) in enumerate(
        zip(zero_indices, display_indices, reference_g_values)
    ):
        color = NATURE_PALETTE[g_order % len(NATURE_PALETTE)]
        for distance_index, distance in enumerate(args.distances):
            x, y, yerr, ns, params_at_index = collect_series_by_index(
                results,
                distance,
                zero_index,
                args,
            )
            warn_if_param_mismatch(display_index, reference_g, params_at_index, ns, args)
            fit_record, coeffs = fit_series(
                x,
                y,
                ns,
                float(reference_g),
                distance,
                args.fit_degree,
            )
            fit_records.append(fit_record)

            x_fit = np.linspace(0.0, float(np.max(x)), 240)
            y_fit = np.polyval(coeffs, x_fit)
            marker = POINT_MARKERS[distance_index % len(POINT_MARKERS)]
            linestyle = FIT_LINE_STYLES[distance_index % len(FIT_LINE_STYLES)]
            series_label = (
                rf"${args.param_label}\approx{float(reference_g):g}$, "
                rf"r={int(distance)}, $N\to\infty={fit_record.spin_dot_infinite:.6g}$"
            )

            if yerr is not None:
                update_range(
                    values_by_index,
                    display_index,
                    y - yerr,
                    y + yerr,
                    y_fit,
                    [fit_record.spin_dot_infinite],
                )
            else:
                update_range(values_by_index, display_index, y, y_fit, [fit_record.spin_dot_infinite])

            items.append(
                {
                    "display_index": display_index,
                    "zero_index": zero_index,
                    "distance": int(distance),
                    "reference_g": float(reference_g),
                    "x": x,
                    "y": y,
                    "yerr": yerr,
                    "ns": ns,
                    "params_at_index": params_at_index,
                    "x_fit": x_fit,
                    "y_fit": y_fit,
                    "spin_dot_infinite": fit_record.spin_dot_infinite,
                    "color": color,
                    "marker": marker,
                    "linestyle": linestyle,
                    "label": series_label,
                }
            )

    return items, fit_records, values_by_index


def expand_range(low, high, pad_fraction):
    span = float(high - low)
    if span <= 0:
        span = max(abs(float(low)), 1.0) * 0.02
    pad = span * float(pad_fraction)
    return float(low - pad), float(high + pad)


def choose_y_segments(values_by_index, args):
    if args.no_y_break or len(values_by_index) != 2:
        return None

    ranges = []
    for display_index, chunks in values_by_index.items():
        values = np.concatenate(chunks)
        ranges.append((float(np.min(values)), float(np.max(values)), display_index))
    ranges.sort(key=lambda item: item[0])

    lower_min, lower_max, lower_index = ranges[0]
    upper_min, upper_max, upper_index = ranges[1]
    gap = upper_min - lower_max
    total_span = max(upper_max - lower_min, 1e-12)
    if gap <= args.y_break_min_gap_fraction * total_span:
        return None

    lower_ylim = expand_range(lower_min, lower_max, args.y_pad_fraction)
    upper_ylim = expand_range(upper_min, upper_max, args.y_pad_fraction)
    height_low = max(lower_ylim[1] - lower_ylim[0], 1e-12)
    height_high = max(upper_ylim[1] - upper_ylim[0], 1e-12)

    return {
        "lower_index": lower_index,
        "upper_index": upper_index,
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


def draw_item(ax, item, label=None, annotate_n=False):
    if item["yerr"] is not None:
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
    ax.scatter(
        [0.0],
        [item["spin_dot_infinite"]],
        marker="x",
        s=42,
        color=item["color"],
        linewidths=1.7,
        zorder=4,
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


def setup_common_axis(ax, args, *, xlabel=True, ylabel=True, title=False):
    if xlabel:
        ax.set_xlabel(r"$1/N$")
    if ylabel:
        ax.set_ylabel(spin_dot_ylabel())
    if title:
        if len(args.distances) == 1:
            ax.set_title(f"J1-J2 spin-dot finite-size extrapolation, d={int(args.distances[0])}")
        else:
            ax.set_title("J1-J2 spin-dot finite-size extrapolation")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    style_axis(ax)


def plot_combined(results, zero_indices, display_indices, reference_g_values, args):
    items, fit_records, values_by_index = build_plot_items(
        results,
        zero_indices,
        display_indices,
        reference_g_values,
        args,
    )
    y_segments = choose_y_segments(values_by_index, args)

    if y_segments is None:
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        for item in items:
            draw_item(ax, item, label=item["label"], annotate_n=args.annotate_n)
        setup_common_axis(ax, args)
        ax.legend(loc="upper right", bbox_to_anchor=(0.96, 0.88))
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
        draw_item(ax_top, item, label=item["label"], annotate_n=args.annotate_n)
        draw_item(ax_bottom, item, label=None, annotate_n=args.annotate_n)

    setup_common_axis(ax_top, args, xlabel=False, ylabel=False, title=True)
    setup_common_axis(ax_bottom, args, xlabel=True, ylabel=False, title=False)
    fig.text(0.02, 0.5, spin_dot_ylabel(), va="center", rotation="vertical")
    draw_axis_break_marks(ax_top, ax_bottom)
    ax_top.legend()
    fig.subplots_adjust(left=0.12, hspace=0.06)
    return fig, fit_records


def print_selected_index_summary(results, zero_indices, display_indices):
    print("Selected parameter indices:")
    for zero_index, display_index in zip(zero_indices, display_indices):
        pieces = []
        for result in results:
            if zero_index < result.params.shape[0]:
                pieces.append(f"N={result.num_qubits}: {result.params[zero_index]:.12g}")
            else:
                pieces.append(f"N={result.num_qubits}: out-of-range")
        print(f"  index {display_index} (zero-based {zero_index}): " + ", ".join(pieces))


def main():
    global plt
    args = parse_args()
    plt = import_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_all_results(args)
    zero_indices, display_indices, reference_g_values = select_g_indices(results, args)
    distances = select_distances(results, args.distances)
    args.distances = distances
    print_selected_index_summary(results, zero_indices, display_indices)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    save_paths = []
    fit_records = []
    for distance in args.distances:
        plot_args = argparse.Namespace(**vars(args))
        plot_args.distances = [distance]
        fig, distance_fit_records = plot_combined(
            results,
            zero_indices,
            display_indices,
            reference_g_values,
            plot_args,
        )
        save_path = output_path_for_distance(args, distance, display_indices, reference_g_values)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        save_paths.append(save_path)
        fit_records.extend(distance_fit_records)

    if len(args.distances) > 1:
        fig, _ = plot_combined(results, zero_indices, display_indices, reference_g_values, args)
        save_path = output_path_for_combined(args, display_indices, reference_g_values)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        save_paths.append(save_path)

    csv_path = save_fit_records(fit_records, args.output_dir, args.fit_csv)

    ns_text = ", ".join(str(result.num_qubits) for result in results)
    print(f"Loaded qubit sizes N: {ns_text}")
    print("Fitted N -> infinity J1-J2 spin-dot values:")
    for record in fit_records:
        print(
            f"  r={record.distance}, reference {args.param_label}={record.j2_value:g}: "
            f"spin_dot_inf={record.spin_dot_infinite:.12g}, slope={record.slope:.12g}, "
            f"R^2={record.r_squared:.6g}"
        )
    print(f"Fit values saved to: {csv_path}")
    print(f"Saved {len(save_paths)} figure(s):")
    for save_path in save_paths:
        print(f"  {save_path}")


if __name__ == "__main__":
    main()
