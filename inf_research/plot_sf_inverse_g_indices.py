"""
Plot ANNNI structure factors versus 1/N for two selected g indices.

This mirrors plot_zz_inverse_qubit_g0.3_g0.9.py, but reads the ANNNI
structure-factor outputs handled by plot_annni_sf_inverse_qubit.py.

Example:
  python inf_research/plot_annni_sf_inverse_qubit_g_indices.py `
    --files annni_N12.npz annni_N16.npz annni_N20.npz `
    --num_qubits 12 16 20 `
    --g_indices 6 18 `
    --q_values pi pi_over_2 `
    --output_dir annni_sf_invN_gidx_figs
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from plot_sf_inverse_qubit import (
    FIT_LINE_STYLES,
    NATURE_PALETTE,
    Q_CONFIG,
    fit_series,
    import_matplotlib,
    load_all_results,
    normalize_q_values,
    save_fit_records,
    style_axis,
)


plt = None
POINT_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two g-index ANNNI structure-factor finite-size curves together."
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
        help="Optional qubit sizes matching --files.",
    )
    parser.add_argument(
        "--g_indices",
        nargs=2,
        type=int,
        required=True,
        help="Exactly two g indices to plot. Default indexing is zero-based.",
    )
    parser.add_argument(
        "--index_base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use 0 for Python-style indices, or 1 for one-based indices.",
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
        default="annni_sf_invN_gidx_figs",
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
        "--fit_degree",
        type=int,
        default=1,
        help="Polynomial degree for fitting S_Z(q) as a function of 1/N.",
    )
    parser.add_argument(
        "--fit_csv",
        type=str,
        default="annni_sf_gidx_infinite_fit_values.csv",
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
        help="Warn when the same g index maps to parameter values differing by more than this across files.",
    )
    args = parser.parse_args()
    args.q_values = normalize_q_values(args.q_values)
    return args


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


def default_output_path(args, display_indices, reference_g):
    idx_part = "_".join(f"idx{index}" for index in display_indices)
    g_part = "_".join(f"g{format_float_for_name(g)}" for g in reference_g)
    q_part = "_".join(Q_CONFIG[q_value]["name"] for q_value in args.q_values)
    return os.path.join(args.output_dir, f"annni_sf_vs_invN_{idx_part}_{g_part}_{q_part}.{args.save_format}")



def output_path_for_q(args, q_value, display_indices, reference_g):
    if args.output is None:
        plot_args = argparse.Namespace(**vars(args))
        plot_args.q_values = [q_value]
        return default_output_path(plot_args, display_indices, reference_g)

    if len(args.q_values) == 1:
        return args.output

    root, ext = os.path.splitext(args.output)
    if not ext:
        ext = f".{args.save_format}"
    return f"{root}_{Q_CONFIG[q_value]['name']}{ext}"


def output_path_for_combined(args, display_indices, reference_g):
    if args.output is None:
        return default_output_path(args, display_indices, reference_g)

    if len(args.q_values) == 1:
        return args.output

    root, ext = os.path.splitext(args.output)
    if not ext:
        ext = f".{args.save_format}"
    q_part = "_".join(Q_CONFIG[q_value]["name"] for q_value in args.q_values)
    return f"{root}_{q_part}{ext}"

def collect_series_by_index(results, q_value, g_index, args):
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
        if q_value not in result.sf_mean:
            raise ValueError(f"{result.file_path}: missing loaded q={q_value} structure-factor values.")

        y_value = float(result.sf_mean[q_value][index])
        if not np.isfinite(y_value):
            continue
        xs.append(1.0 / result.num_qubits)
        ys.append(y_value)

        std_values = result.sf_std.get(q_value)
        if (not args.no_errorbar) and std_values is not None:
            yerr_value = float(std_values[index])
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
        for q_index, q_value in enumerate(args.q_values):
            x, y, yerr, ns, params_at_index = collect_series_by_index(results, q_value, zero_index, args)
            warn_if_param_mismatch(display_index, reference_g, params_at_index, ns, args)
            fit_record, coeffs = fit_series(
                x,
                y,
                ns,
                float(reference_g),
                q_value,
                args.fit_degree,
            )
            fit_records.append(fit_record)

            x_fit = np.linspace(0.0, float(np.max(x)), 240)
            y_fit = np.polyval(coeffs, x_fit)
            marker = POINT_MARKERS[q_index % len(POINT_MARKERS)]
            linestyle = FIT_LINE_STYLES[q_index % len(FIT_LINE_STYLES)]
            q_label = Q_CONFIG[q_value]["label"]
            series_label = (
                rf"${args.param_label}\approx{float(reference_g):g}$, "
                rf"{q_label}, $N\to\infty={fit_record.sf_infinite:.6g}$"
            )

            if yerr is not None:
                update_range(values_by_index, display_index, y - yerr, y + yerr, y_fit, [fit_record.sf_infinite])
            else:
                update_range(values_by_index, display_index, y, y_fit, [fit_record.sf_infinite])

            items.append(
                {
                    "display_index": display_index,
                    "zero_index": zero_index,
                    "q_value": q_value,
                    "reference_g": float(reference_g),
                    "x": x,
                    "y": y,
                    "yerr": yerr,
                    "ns": ns,
                    "params_at_index": params_at_index,
                    "x_fit": x_fit,
                    "y_fit": y_fit,
                    "sf_infinite": fit_record.sf_infinite,
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
        [item["sf_infinite"]],
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


def ylabel_for_q_values(q_values):
    if len(q_values) == 1:
        return Q_CONFIG[q_values[0]]["label"]
    return r"$S_Z(q)$"


def setup_common_axis(ax, args, *, xlabel=True, ylabel=True, title=True):
    if xlabel:
        ax.set_xlabel(r"$1/N$")
    if ylabel:
        ax.set_ylabel(ylabel_for_q_values(args.q_values))
    if title:
        # ax.set_title("ANNNI structure-factor finite-size extrapolation")
        pass
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
        ax.legend(loc="upper right", bbox_to_anchor=(1.00, 0.92), fontsize=10)
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
    fig.text(0.02, 0.5, ylabel_for_q_values(args.q_values), va="center", rotation="vertical")
    draw_axis_break_marks(ax_top, ax_bottom)
    ax_top.legend()
    fig.subplots_adjust(left=0.12, hspace=0.06)
    return fig, fit_records


def print_selected_index_summary(results, zero_indices, display_indices):
    print("Selected g indices:")
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
    print_selected_index_summary(results, zero_indices, display_indices)

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
    fit_records = []
    for q_value in args.q_values:
        plot_args = argparse.Namespace(**vars(args))
        plot_args.q_values = [q_value]
        fig, q_fit_records = plot_combined(results, zero_indices, display_indices, reference_g_values, plot_args)
        save_path = output_path_for_q(args, q_value, display_indices, reference_g_values)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        save_paths.append(save_path)
        fit_records.extend(q_fit_records)

    if len(args.q_values) > 1:
        fig, _ = plot_combined(results, zero_indices, display_indices, reference_g_values, args)
        save_path = output_path_for_combined(args, display_indices, reference_g_values)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        save_paths.append(save_path)

    csv_path = save_fit_records(fit_records, args.output_dir, args.fit_csv)

    ns_text = ", ".join(str(result.num_qubits) for result in results)
    print(f"Loaded qubit sizes N: {ns_text}")
    print("Fitted N -> infinity ANNNI structure-factor values:")
    for record in fit_records:
        print(
            f"  q={record.q_value}, reference {args.param_label}={record.g_value:g}: "
            f"sf_inf={record.sf_infinite:.12g}, slope={record.slope:.12g}, "
            f"R^2={record.r_squared:.6g}"
        )
    print(f"Fit values saved to: {csv_path}")
    print(f"Saved {len(save_paths)} figure(s):")
    for save_path in save_paths:
        print(f"  {save_path}")


if __name__ == "__main__":
    main()
