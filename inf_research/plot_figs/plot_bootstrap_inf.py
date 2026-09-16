#!/usr/bin/env python3
"""Plot N->infinity estimates saved by bootstrap_finite_size_ci.py.

The y-axis label is inferred independently for each selected observable. A
TFI exact-ZZ infinite-limit CSV can optionally be overlaid.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


COLORS = [
    "#4D7FD5", "#E64B35", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


@dataclass(frozen=True)
class BootstrapRow:
    observable_key: str
    component: int | None
    parameter: float
    fit_of_finite_mean: float
    bootstrap_mean: float
    bootstrap_median: float
    bootstrap_std: float
    ci_low: float
    ci_high: float
    confidence: float


@dataclass(frozen=True)
class ExactRow:
    distance: int
    g_value: float
    zz_infinite_fit: float
    zz_infinite_exact: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot bootstrap finite-size N->infinity CSV results with optional "
            "TFI exact infinite-limit curves."
        )
    )
    parser.add_argument(
        "--bootstrap_csv", "--bootstrap-csv", dest="bootstrap_csv",
        required=True,
        help="bootstrap_infinite.csv from bootstrap_finite_size_ci.py.",
    )
    parser.add_argument(
        "--exact_csv", "--exact-csv", dest="exact_csv", default="",
        help=(
            "Optional tfi_exact_zz_infinite_fit_errors.csv. Omit to draw only "
            "Diffushadow estimates."
        ),
    )
    parser.add_argument(
        "--observable_keys", "--observable-keys", dest="observable_keys",
        nargs="+", default=None,
        help="Observable keys to plot. Default: every key in the bootstrap CSV.",
    )
    parser.add_argument(
        "--components", "--distances", dest="components", nargs="+",
        type=int, default=None,
        help="Selected component/distance values. Default: all saved components.",
    )
    parser.add_argument(
        "--center",
        choices=["fit_of_finite_mean", "bootstrap_mean", "bootstrap_median"],
        default="fit_of_finite_mean",
        help=(
            "Central curve. Default: fit_of_finite_mean, matching "
            "bootstrap_finite_size_ci.py."
        ),
    )
    parser.add_argument(
        "--ci_style", "--ci-style", dest="ci_style",
        choices=["band", "errorbar", "both", "none"], default="band",
        help="How to draw percentile confidence intervals.",
    )
    parser.add_argument(
        "--exact_series", "--exact-series", dest="exact_series",
        choices=["thermodynamic", "fit", "both"], default="thermodynamic",
        help="Column(s) drawn from the optional exact TFI ZZ CSV.",
    )
    parser.add_argument(
        "--exact_observable_key", "--exact-observable-key",
        dest="exact_observable_key", default="bootstrap_zz",
        help="Bootstrap observable on which exact TFI ZZ curves are overlaid.",
    )
    parser.add_argument(
        "--param_label", "--param-label", dest="param_label", default="g"
    )
    parser.add_argument(
        "--ylabel", default="",
        help=(
            "Y-axis label override. Default: infer a physical-quantity label "
            "from each observable_key."
        ),
    )
    parser.add_argument(
        "--title", default="",
        help="Optional title. Default: generated from the observable key.",
    )
    parser.add_argument(
        "--output_dir", "--output-dir", dest="output_dir",
        default="bootstrap_infinite_plots",
    )
    parser.add_argument(
        "--output_prefix", "--output-prefix", dest="output_prefix",
        default="bootstrap_infinite",
    )
    parser.add_argument(
        "--save_format", "--save-format", dest="save_format",
        choices=["png", "pdf", "svg", "eps"], default="png",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--fig_width", "--fig-width", dest="fig_width", type=float,
        default=7.2,
    )
    parser.add_argument(
        "--fig_height", "--fig-height", dest="fig_height", type=float,
        default=5.0,
    )
    return parser.parse_args()


def require_columns(
    fieldnames: Sequence[str] | None, required: Sequence[str], path: Path
) -> None:
    available = set(fieldnames or [])
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}; available columns are "
            f"{sorted(available)}."
        )


def parse_float(row: dict[str, str], key: str, path: Path, line: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line}: invalid {key}={row.get(key)!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{path}:{line}: {key} must be finite, got {value}.")
    return value


def parse_component(raw: str, path: Path, line: int) -> int | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{path}:{line}: invalid component={raw!r}.") from exc


def load_bootstrap_rows(path_text: str) -> list[BootstrapRow]:
    path = Path(path_text).expanduser().resolve()
    required = [
        "observable_key", "component", "parameter", "fit_of_finite_mean",
        "bootstrap_mean", "bootstrap_median", "bootstrap_std", "ci_low",
        "ci_high", "confidence",
    ]
    rows: list[BootstrapRow] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, required, path)
        for line, raw in enumerate(reader, start=2):
            key = str(raw["observable_key"]).strip()
            if not key:
                raise ValueError(f"{path}:{line}: observable_key is empty.")
            row = BootstrapRow(
                observable_key=key,
                component=parse_component(raw["component"], path, line),
                parameter=parse_float(raw, "parameter", path, line),
                fit_of_finite_mean=parse_float(
                    raw, "fit_of_finite_mean", path, line
                ),
                bootstrap_mean=parse_float(raw, "bootstrap_mean", path, line),
                bootstrap_median=parse_float(
                    raw, "bootstrap_median", path, line
                ),
                bootstrap_std=parse_float(raw, "bootstrap_std", path, line),
                ci_low=parse_float(raw, "ci_low", path, line),
                ci_high=parse_float(raw, "ci_high", path, line),
                confidence=parse_float(raw, "confidence", path, line),
            )
            if not 0.0 < row.confidence < 1.0:
                raise ValueError(
                    f"{path}:{line}: confidence must be between 0 and 1."
                )
            if row.ci_low > row.ci_high:
                raise ValueError(
                    f"{path}:{line}: ci_low={row.ci_low} exceeds "
                    f"ci_high={row.ci_high}."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    identities = [(row.observable_key, row.component, row.parameter) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(
            f"{path} contains duplicate (observable_key, component, parameter) rows."
        )
    return rows


def load_exact_rows(path_text: str) -> list[ExactRow]:
    path = Path(path_text).expanduser().resolve()
    required = ["distance", "g_value", "zz_infinite_fit", "zz_infinite_exact"]
    rows: list[ExactRow] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, required, path)
        for line, raw in enumerate(reader, start=2):
            try:
                distance = int(str(raw["distance"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line}: invalid distance={raw.get('distance')!r}."
                ) from exc
            rows.append(
                ExactRow(
                    distance=distance,
                    g_value=parse_float(raw, "g_value", path, line),
                    zz_infinite_fit=parse_float(
                        raw, "zz_infinite_fit", path, line
                    ),
                    zz_infinite_exact=parse_float(
                        raw, "zz_infinite_exact", path, line
                    ),
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    identities = [(row.distance, row.g_value) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{path} contains duplicate (distance, g_value) rows.")
    return rows


def selected_observable_keys(
    rows: Sequence[BootstrapRow], requested: Sequence[str] | None
) -> list[str]:
    available = sorted({row.observable_key for row in rows})
    if requested is None:
        return available
    selected = list(dict.fromkeys(str(key) for key in requested))
    missing = [key for key in selected if key not in available]
    if missing:
        raise ValueError(
            f"Requested observable keys {missing} are unavailable; choices are "
            f"{available}."
        )
    return selected


def selected_components(
    rows: Sequence[BootstrapRow], requested: Sequence[int] | None, key: str
) -> list[int | None]:
    available = sorted(
        {row.component for row in rows if row.observable_key == key},
        key=lambda value: -1 if value is None else value,
    )
    if requested is None:
        return available
    if available == [None]:
        raise ValueError(
            f"--components/--distances cannot be used with scalar observable {key}."
        )
    selected = list(dict.fromkeys(int(value) for value in requested))
    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError(
            f"{key}: requested components {missing} are unavailable; choices are "
            f"{available}."
        )
    return selected


def component_rows(
    rows: Sequence[BootstrapRow], key: str, component: int | None
) -> list[BootstrapRow]:
    return sorted(
        [
            row for row in rows
            if row.observable_key == key and row.component == component
        ],
        key=lambda row: row.parameter,
    )


def exact_component_rows(rows: Sequence[ExactRow], distance: int) -> list[ExactRow]:
    return sorted(
        [row for row in rows if row.distance == distance],
        key=lambda row: row.g_value,
    )


def center_values(rows: Sequence[BootstrapRow], center: str) -> np.ndarray:
    return np.asarray([getattr(row, center) for row in rows], dtype=np.float64)


def normalized_observable_name(key: str) -> str:
    name = str(key).strip().lower()
    return name[len("bootstrap_") :] if name.startswith("bootstrap_") else name


def observable_ylabel(key: str) -> str:
    """Return an N->infinity label matching the observable stored in the CSV."""
    name = normalized_observable_name(key)
    labels = {
        "zz": r"$\langle Z_i Z_{i+d}\rangle_{N\to\infty}$",
        "corr_zz": r"$\langle Z_i Z_{i+d}\rangle_{N\to\infty}$",
        "corr_xx": r"$\langle X_i X_{i+d}\rangle_{N\to\infty}$",
        "corr_yy": r"$\langle Y_i Y_{i+d}\rangle_{N\to\infty}$",
        "corr_spin_dot": (
            r"$\langle \vec{\sigma}_i\!\cdot\!"
            r"\vec{\sigma}_{i+d}\rangle_{N\to\infty}$"
        ),
        "spin_dot": (
            r"$\langle \vec{\sigma}_i\!\cdot\!"
            r"\vec{\sigma}_{i+d}\rangle_{N\to\infty}$"
        ),
        "structure_factor_pi": r"$S_Z(\pi)_{N\to\infty}$",
        "structure_factor_pi_over_2": r"$S_Z(\pi/2)_{N\to\infty}$",
        "sf_pi": r"$S_Z(\pi)_{N\to\infty}$",
        "sf_pi_over_2": r"$S_Z(\pi/2)_{N\to\infty}$",
        "energy": r"$E_{N\to\infty}$",
        "dimer_proxy": r"$D_{N\to\infty}$",
        "x": r"$\langle X_i\rangle_{N\to\infty}$",
        "x_string": r"$\langle X_i\cdots X_{i+d}\rangle_{N\to\infty}$",
        "magnetization_z": r"$m_z\;(N\to\infty)$",
        "magnetization_staggered": r"$m_{\mathrm{stag}}\;(N\to\infty)$",
        "energy_derivative": r"$\partial E/\partial g\;(N\to\infty)$",
    }
    if name in labels:
        return labels[name]
    return f"{name.replace('_', ' ')} (N→∞)"


def component_label(component: int | None) -> str:
    base = "Diffushadow N→∞ estimate"
    return base if component is None else f"{base} (d={component})"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "observable"


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=4, width=0.8)
    axis.grid(True, alpha=0.25, linestyle="--")


def plot_exact_overlay(axis, exact_rows, component, color, components, args) -> None:
    if component is None:
        raise ValueError(
            "The exact TFI ZZ CSV is distance-resolved, but the selected "
            "bootstrap observable has no component column."
        )
    exact = exact_component_rows(exact_rows, int(component))
    if not exact:
        available = sorted({row.distance for row in exact_rows})
        raise ValueError(
            f"Exact CSV has no distance d={component}; available distances are "
            f"{available}."
        )
    g_values = np.asarray([row.g_value for row in exact], dtype=np.float64)
    if args.exact_series in {"thermodynamic", "both"}:
        values = np.asarray(
            [row.zz_infinite_exact for row in exact], dtype=np.float64
        )
        axis.plot(
            g_values, values, color=color, linestyle="--", linewidth=2.0,
            label=(
                "exact thermodynamic" if len(components) == 1
                else f"exact thermodynamic (d={component})"
            ),
            zorder=4,
        )
    if args.exact_series in {"fit", "both"}:
        values = np.asarray([row.zz_infinite_fit for row in exact], dtype=np.float64)
        axis.plot(
            g_values, values, color=color, linestyle=":", linewidth=2.0,
            label=(
                "exact finite-size fit" if len(components) == 1
                else f"exact finite-size fit (d={component})"
            ),
            zorder=4,
        )


def plot_observable(
    bootstrap_rows: Sequence[BootstrapRow],
    exact_rows: Sequence[ExactRow],
    key: str,
    components: Sequence[int | None],
    args: argparse.Namespace,
    pyplot,
) -> Path:
    figure, axis = pyplot.subplots(figsize=(args.fig_width, args.fig_height))
    for index, component in enumerate(components):
        rows = component_rows(bootstrap_rows, key, component)
        if not rows:
            raise ValueError(f"No rows found for {key}, component={component}.")
        parameters = np.asarray([row.parameter for row in rows], dtype=np.float64)
        central = center_values(rows, args.center)
        ci_low = np.asarray([row.ci_low for row in rows], dtype=np.float64)
        ci_high = np.asarray([row.ci_high for row in rows], dtype=np.float64)
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
        confidence_values = {round(row.confidence, 12) for row in rows}
        if len(confidence_values) != 1:
            raise ValueError(
                f"{key}, component={component}: confidence values differ across rows."
            )
        confidence = 100.0 * rows[0].confidence
        axis.plot(
            parameters, central, color=color, marker=marker, markersize=4.5,
            linewidth=1.9, label=component_label(component), zorder=3,
        )
        if args.ci_style in {"band", "both"}:
            axis.fill_between(
                parameters, ci_low, ci_high, color=color, alpha=0.20,
                linewidth=0.0,
                label=f"{confidence:g}% CI" if len(components) == 1 else None,
                zorder=1,
            )
        if args.ci_style in {"errorbar", "both"}:
            ci_mid = 0.5 * (ci_low + ci_high)
            axis.errorbar(
                parameters, ci_mid,
                yerr=np.vstack([ci_mid - ci_low, ci_high - ci_mid]),
                fmt="none", color=color, ecolor=color, capsize=2.5,
                linewidth=1.0, alpha=0.9, zorder=2,
            )
        if exact_rows and key == args.exact_observable_key:
            plot_exact_overlay(axis, exact_rows, component, color, components, args)

    axis.set_xlabel(args.param_label)
    axis.set_ylabel(args.ylabel or observable_ylabel(key))
    # axis.set_title(args.title or f"Diffushadow N→∞ estimate: {key}")
    style_axis(axis)
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{safe_name(args.output_prefix)}_{safe_name(key)}.{args.save_format}"
    )
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    pyplot.close(figure)
    return path


def import_matplotlib():
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc
    return pyplot


def validate_args(args: argparse.Namespace) -> None:
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.fig_width <= 0.0 or args.fig_height <= 0.0:
        raise ValueError("Figure dimensions must be positive.")
    if args.components is not None and any(value < 0 for value in args.components):
        raise ValueError("--components/--distances values must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    bootstrap_rows = load_bootstrap_rows(args.bootstrap_csv)
    exact_rows = load_exact_rows(args.exact_csv) if args.exact_csv else []
    observable_keys = selected_observable_keys(bootstrap_rows, args.observable_keys)
    if exact_rows and args.exact_observable_key not in observable_keys:
        raise ValueError(
            f"--exact_csv was provided, but --exact_observable_key="
            f"{args.exact_observable_key!r} is not among selected observable "
            f"keys {observable_keys}."
        )
    pyplot = import_matplotlib()
    pyplot.rcParams.update(
        {
            "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
            "legend.fontsize": 10, "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    paths = []
    for key in observable_keys:
        components = selected_components(bootstrap_rows, args.components, key)
        paths.append(
            plot_observable(
                bootstrap_rows, exact_rows, key, components, args, pyplot
            )
        )
    print(f"Loaded bootstrap CSV: {Path(args.bootstrap_csv).resolve()}")
    if exact_rows:
        print(f"Loaded exact CSV: {Path(args.exact_csv).resolve()}")
        print(f"Exact series: {args.exact_series}")
    else:
        print("Exact overlay: disabled")
    print(f"Center: {args.center}; CI style: {args.ci_style}")
    for path in paths:
        print(f"Saved figure: {path.resolve()}")


if __name__ == "__main__":
    main()
