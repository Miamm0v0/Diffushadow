#!/usr/bin/env python3
"""Bootstrap saved measurement snapshots and evaluate physical observables.

Input is the JSON written by ``eval_gen_mea_data.py``.  Every bootstrap
replicate draws M rows with replacement *within every parameter value*, so one
replicate always contains ``h_length * M`` measurement snapshots.  Observable
estimators are imported from ``eval_utils.py`` and are called with the same
arguments and formulas as ``eval_new.py``.

The output NPZ contains both the complete bootstrap distributions (keys named
``bootstrap_*``) and the mean/std aliases consumed by the existing scripts in
``inf_research``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Mapping, Sequence

import numpy as np
import torch
import tqdm

from eval_utils import (
    median_of_means,
    median_of_means_X,
    median_of_means_annni_energy,
    median_of_means_annni_structure_factor_z,
    median_of_means_correlation,
    median_of_means_heisenberg,
    median_of_means_j1j2_dimer_proxy,
    median_of_means_j1j2_energy,
    median_of_means_single_pauli,
    median_of_means_two,
    median_of_means_xxz_energy_and_derivative,
    median_of_means_xxz_magnetization,
)


MODEL_CHOICES = ("TFI", "Heisenberg", "xxz", "J1J2", "ANNNI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap eval_gen_mea_data.py JSON snapshots and calculate the "
            "same physical observables as eval_new.py."
        )
    )
    parser.add_argument("--input", required=True, help="Snapshot JSON from eval_gen_mea_data.py.")
    parser.add_argument("--output", required=True, help="Output bootstrap-observable NPZ.")
    parser.add_argument("--predict_model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--bootstrap_size",
        "--M",
        dest="bootstrap_size",
        type=int,
        default=None,
        help="Rows drawn per parameter value. Default: the smallest group size.",
    )
    parser.add_argument(
        "--bootstrap_repetitions",
        "--B",
        dest="bootstrap_repetitions",
        type=int,
        default=1000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=10,
        help="Median-of-means partitions; eval_new.py uses 10.",
    )
    parser.add_argument(
        "--d_values",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Correlation distances, in eval_new.py order.",
    )
    parser.add_argument(
        "--h_length",
        type=int,
        default=None,
        help="Optional check for the number of distinct saved parameter values.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J_xy", type=float, default=1.0)
    parser.add_argument(
        "--annni_scan",
        choices=["h", "kappa"],
        default="h",
        help="Which ANNNI parameter is stored in column zero.",
    )
    parser.add_argument("--annni_fixed_h", type=float, default=0.6)
    parser.add_argument("--annni_fixed_kappa", type=float, default=0.5)
    return parser.parse_args()


def load_snapshot_groups(path: str) -> tuple[np.ndarray, List[np.ndarray], int, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    rows = np.asarray(loaded, dtype=np.float64)
    del loaded

    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("Snapshot JSON must be a non-empty rectangular list of rows.")
    if rows.shape[1] < 3 or (rows.shape[1] - 1) % 2 != 0:
        raise ValueError(
            "Each row must have format [parameter, P1, b1, ..., PN, bN]; "
            "got width {}.".format(rows.shape[1])
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("Snapshot JSON contains NaN or infinite values.")

    num_qubits = (rows.shape[1] - 1) // 2
    measurement_values = rows[:, 1:]
    rounded = np.rint(measurement_values)
    if not np.allclose(measurement_values, rounded, atol=1e-6, rtol=0.0):
        raise ValueError("Measurement-basis and outcome columns must contain integer tokens.")
    measurements = rounded.astype(np.int64, copy=False)
    bases = measurements[:, 0::2]
    outcomes = measurements[:, 1::2]
    if not np.all(np.isin(bases, [2, 3, 4])):
        raise ValueError("Measurement bases must use tokens 2=X, 3=Y, 4=Z.")
    if not np.all(np.isin(outcomes, [0, 1])):
        raise ValueError("Measurement outcomes must be binary tokens 0 or 1.")

    params, inverse, counts = np.unique(rows[:, 0], return_inverse=True, return_counts=True)
    groups = [np.ascontiguousarray(measurements[inverse == i]) for i in range(params.size)]
    return params.astype(np.float64), groups, num_qubits, counts.astype(np.int64)


def as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def build_evaluator(args: argparse.Namespace) -> Callable[[float, torch.Tensor], Dict[str, np.ndarray]]:
    d_values = tuple(int(value) for value in args.d_values)
    num_parts = int(args.num_parts)

    if args.predict_model == "TFI":
        def evaluate(param: float, sqe: torch.Tensor) -> Dict[str, np.ndarray]:
            return {
                "energy": np.asarray(as_float(median_of_means(param, sqe, num_parts))),
                "zz": np.asarray(
                    [as_float(median_of_means_two(d, sqe, num_parts)) for d in d_values]
                ),
                "x_string": np.asarray(
                    [as_float(median_of_means_X(d - 1, sqe, num_parts)) for d in d_values]
                ),
            }
        return evaluate

    if args.predict_model == "Heisenberg":
        def evaluate(param: float, sqe: torch.Tensor) -> Dict[str, np.ndarray]:
            xx = np.asarray([
                as_float(median_of_means_correlation(sqe, d, "X", num_parts))
                for d in d_values
            ])
            yy = np.asarray([
                as_float(median_of_means_correlation(sqe, d, "Y", num_parts))
                for d in d_values
            ])
            zz = np.asarray([
                as_float(median_of_means_correlation(sqe, d, "Z", num_parts))
                for d in d_values
            ])
            return {
                "energy": np.asarray(as_float(median_of_means_heisenberg(param, sqe, num_parts))),
                "corr_xx": xx,
                "corr_yy": yy,
                "corr_zz": zz,
                "corr_spin_dot": 0.25 * (xx + yy + zz),
            }
        return evaluate

    if args.predict_model == "xxz":
        def evaluate(param: float, sqe: torch.Tensor) -> Dict[str, np.ndarray]:
            energy, derivative = median_of_means_xxz_energy_and_derivative(
                args.J_xy, param, sqe, num_parts
            )
            magnetization_z, magnetization_staggered = median_of_means_xxz_magnetization(
                sqe, num_parts
            )
            return {
                "energy": np.asarray(as_float(energy)),
                "energy_derivative": np.asarray(as_float(derivative)),
                "magnetization_z": np.asarray(as_float(magnetization_z)),
                "magnetization_staggered": np.asarray(as_float(magnetization_staggered)),
                "corr_xx": np.asarray(
                    as_float(median_of_means_correlation(sqe, 1, "X", num_parts))
                ),
                "corr_yy": np.asarray(
                    as_float(median_of_means_correlation(sqe, 1, "Y", num_parts))
                ),
                "corr_zz": np.asarray(
                    as_float(median_of_means_correlation(sqe, 1, "Z", num_parts))
                ),
            }
        return evaluate

    if args.predict_model == "J1J2":
        def evaluate(param: float, sqe: torch.Tensor) -> Dict[str, np.ndarray]:
            # eval_gen_mea_data.py labels this scalar alpha.  For the current
            # J1=1 datasets alpha and J2 coincide; for general J1 use J2=alpha*J1.
            j2 = float(param) * float(args.J1)
            xx = np.asarray([
                as_float(median_of_means_correlation(sqe, d, "X", num_parts))
                for d in d_values
            ])
            yy = np.asarray([
                as_float(median_of_means_correlation(sqe, d, "Y", num_parts))
                for d in d_values
            ])
            zz = np.asarray([
                as_float(median_of_means_correlation(sqe, d, "Z", num_parts))
                for d in d_values
            ])
            return {
                "energy": np.asarray(
                    as_float(median_of_means_j1j2_energy(args.J1, j2, sqe, num_parts))
                ),
                "corr_xx": xx,
                "corr_yy": yy,
                "corr_zz": zz,
                # eval_new.py intentionally uses sigma-dot (no factor 1/4) here.
                "corr_spin_dot": xx + yy + zz,
                "dimer_proxy": np.asarray(
                    as_float(median_of_means_j1j2_dimer_proxy(sqe, num_parts))
                ),
            }
        return evaluate

    if args.predict_model == "ANNNI":
        def evaluate(param: float, sqe: torch.Tensor) -> Dict[str, np.ndarray]:
            if args.annni_scan == "h":
                h_value = float(param)
                kappa = float(args.annni_fixed_kappa)
            else:
                h_value = float(args.annni_fixed_h)
                kappa = float(param)
            return {
                "energy": np.asarray(
                    as_float(
                        median_of_means_annni_energy(
                            h_value, sqe, kappa=kappa, J1=args.J1, num_parts=num_parts
                        )
                    )
                ),
                "zz": np.asarray([
                    as_float(median_of_means_correlation(sqe, d, "Z", num_parts))
                    for d in d_values
                ]),
                "x": np.asarray(
                    as_float(median_of_means_single_pauli(sqe, "X", num_parts))
                ),
                "structure_factor_pi": np.asarray(
                    as_float(median_of_means_annni_structure_factor_z(sqe, np.pi, num_parts))
                ),
                "structure_factor_pi_over_2": np.asarray(
                    as_float(
                        median_of_means_annni_structure_factor_z(sqe, np.pi / 2.0, num_parts)
                    )
                ),
            }
        return evaluate

    raise AssertionError("Unhandled model: {}".format(args.predict_model))


def allocate_distributions(
    example: Mapping[str, np.ndarray], repetitions: int, h_length: int
) -> Dict[str, np.ndarray]:
    distributions: Dict[str, np.ndarray] = {}
    for name, value in example.items():
        shape = tuple(np.asarray(value).shape)
        # Scalar observable: [B, H]; component observable: [B, ..., H].
        distributions[name] = np.empty((repetitions,) + shape + (h_length,), dtype=np.float64)
    return distributions


def store_observables(
    distributions: Mapping[str, np.ndarray],
    values: Mapping[str, np.ndarray],
    bootstrap_index: int,
    param_index: int,
) -> None:
    if set(distributions) != set(values):
        raise ValueError("Observable keys changed between evaluations.")
    for name, value in values.items():
        array = np.asarray(value, dtype=np.float64)
        expected = distributions[name].shape[1:-1]
        if array.shape != expected:
            raise ValueError(
                "Observable '{}' changed shape from {} to {}.".format(name, expected, array.shape)
            )
        distributions[name][(bootstrap_index,) + (slice(None),) * array.ndim + (param_index,)] = array


def add_summary_fields(
    output: Dict[str, np.ndarray], distributions: Mapping[str, np.ndarray], confidence: float
) -> None:
    alpha = (1.0 - confidence) / 2.0
    for name, values in distributions.items():
        output["bootstrap_{}".format(name)] = values
        output["{}_mean".format(name)] = np.mean(values, axis=0)
        output["{}_std".format(name)] = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(values[0])
        output["{}_ci_low".format(name)] = np.quantile(values, alpha, axis=0)
        output["{}_ci_high".format(name)] = np.quantile(values, 1.0 - alpha, axis=0)


def add_eval_new_aliases(
    output: Dict[str, np.ndarray],
    distributions: Mapping[str, np.ndarray],
    params: np.ndarray,
    args: argparse.Namespace,
) -> None:
    def mean(name: str) -> np.ndarray:
        return output["{}_mean".format(name)]

    def std(name: str) -> np.ndarray:
        return output["{}_std".format(name)]

    model = args.predict_model
    if model == "TFI":
        output.update({
            "hs_gpt": params,
            "gpt_eval_points": distributions["zz"],
            "gpt_eval_pointsX": distributions["x_string"],
            "gpt_eval_points_energy": distributions["energy"],
            "zz_mean": mean("zz"),
            "zz_std": std("zz"),
            "xs_mean": mean("x_string"),
            "xs_std": std("x_string"),
            "energy_mean": mean("energy"),
            "energy_std": std("energy"),
        })
    elif model == "Heisenberg":
        output.update({
            "Js": params,
            "gpt_eval_energy_raw": distributions["energy"],
            "gpt_eval_corr_XX_raw": distributions["corr_xx"],
            "gpt_eval_corr_YY_raw": distributions["corr_yy"],
            "gpt_eval_corr_ZZ_raw": distributions["corr_zz"],
            "gpt_eval_corr_spin_dot_raw": distributions["corr_spin_dot"],
            "energy_mean": mean("energy"),
            "energy_std": std("energy"),
            "corr_XX_mean": mean("corr_xx"),
            "corr_XX_std": std("corr_xx"),
            "corr_YY_mean": mean("corr_yy"),
            "corr_YY_std": std("corr_yy"),
            "corr_ZZ_mean": mean("corr_zz"),
            "corr_ZZ_std": std("corr_zz"),
            "corr_spin_dot_mean": mean("corr_spin_dot"),
            "corr_spin_dot_std": std("corr_spin_dot"),
        })
    elif model == "xxz":
        output.update({
            "Deltas": params,
            "J_xy": np.asarray(args.J_xy),
            "shadow_eval_energy_raw": distributions["energy"],
            "shadow_eval_energy_derivative_raw": distributions["energy_derivative"],
            "shadow_eval_magnetization_z_raw": distributions["magnetization_z"],
            "shadow_eval_magnetization_s_raw": distributions["magnetization_staggered"],
            "shadow_eval_corr_XX_single_raw": distributions["corr_xx"],
            "shadow_eval_corr_YY_single_raw": distributions["corr_yy"],
            "shadow_eval_corr_ZZ_single_raw": distributions["corr_zz"],
            "energy_mean": mean("energy"),
            "energy_std": std("energy"),
            "energy_derivative_mean": mean("energy_derivative"),
            "energy_derivative_std": std("energy_derivative"),
            "magnetization_z_mean": mean("magnetization_z"),
            "magnetization_z_std": std("magnetization_z"),
            "magnetization_s_mean": mean("magnetization_staggered"),
            "magnetization_s_std": std("magnetization_staggered"),
            "corr_XX_mean_single": mean("corr_xx"),
            "corr_XX_std_single": std("corr_xx"),
            "corr_YY_mean_single": mean("corr_yy"),
            "corr_YY_std_single": std("corr_yy"),
            "corr_ZZ_mean_single": mean("corr_zz"),
            "corr_ZZ_std_single": std("corr_zz"),
        })
    elif model == "J1J2":
        output.update({
            "J2s": params * float(args.J1),
            "alphas": params,
            "J1": np.asarray(args.J1),
            "shadow_eval_energy_raw": distributions["energy"],
            "shadow_eval_corr_XX_raw": distributions["corr_xx"],
            "shadow_eval_corr_YY_raw": distributions["corr_yy"],
            "shadow_eval_corr_ZZ_raw": distributions["corr_zz"],
            "shadow_eval_corr_spin_dot_raw": distributions["corr_spin_dot"],
            "shadow_eval_dimer_proxy_raw": distributions["dimer_proxy"],
            "energy_mean": mean("energy"),
            "energy_std": std("energy"),
            "corr_XX_mean": mean("corr_xx"),
            "corr_XX_std": std("corr_xx"),
            "corr_YY_mean": mean("corr_yy"),
            "corr_YY_std": std("corr_yy"),
            "corr_ZZ_mean": mean("corr_zz"),
            "corr_ZZ_std": std("corr_zz"),
            "corr_spin_dot_mean": mean("corr_spin_dot"),
            "corr_spin_dot_std": std("corr_spin_dot"),
            "dimer_proxy_mean": mean("dimer_proxy"),
            "dimer_proxy_std": std("dimer_proxy"),
        })
    elif model == "ANNNI":
        output.update({
            "hs": params,
            "J1": np.asarray(args.J1),
            "annni_scan": np.asarray(args.annni_scan),
            "kappa": np.asarray(args.annni_fixed_kappa),
            "fixed_h": np.asarray(args.annni_fixed_h),
            "shadow_eval_energy_raw": distributions["energy"],
            "shadow_eval_ZZ_raw": distributions["zz"],
            "shadow_eval_X_raw": distributions["x"],
            "shadow_eval_structure_factor_pi_raw": distributions["structure_factor_pi"],
            "shadow_eval_structure_factor_pi_over_2_raw": distributions["structure_factor_pi_over_2"],
            "energy_mean": mean("energy"),
            "energy_std": std("energy"),
            "zz_mean": mean("zz"),
            "zz_std": std("zz"),
            "x_mean": mean("x"),
            "x_std": std("x"),
            "structure_factor_pi_mean": mean("structure_factor_pi"),
            "structure_factor_pi_std": std("structure_factor_pi"),
            "structure_factor_pi_over_2_mean": mean("structure_factor_pi_over_2"),
            "structure_factor_pi_over_2_std": std("structure_factor_pi_over_2"),
        })


def validate_args(args: argparse.Namespace) -> None:
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap_repetitions must be positive.")
    if args.bootstrap_size is not None and args.bootstrap_size <= 0:
        raise ValueError("--bootstrap_size/--M must be positive.")
    if args.num_parts <= 0:
        raise ValueError("--num_parts must be positive.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between 0 and 1.")
    if not args.d_values or any(value <= 0 for value in args.d_values):
        raise ValueError("--d_values must contain positive integers.")
    if len(set(args.d_values)) != len(args.d_values):
        raise ValueError("--d_values must not contain duplicates.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    params, groups, num_qubits, counts = load_snapshot_groups(args.input)
    h_length = int(params.size)
    if args.h_length is not None and h_length != args.h_length:
        raise ValueError(
            "Expected h_length={}, but JSON contains {} distinct parameter values.".format(
                args.h_length, h_length
            )
        )
    if any(d > num_qubits for d in args.d_values):
        raise ValueError("A requested correlation distance exceeds N={}.".format(num_qubits))

    bootstrap_size = int(np.min(counts)) if args.bootstrap_size is None else args.bootstrap_size
    if bootstrap_size < args.num_parts:
        raise ValueError(
            "M={} must be at least num_parts={} because eval_new.py's median-of-means "
            "estimators require non-empty parts.".format(bootstrap_size, args.num_parts)
        )

    # Include N in the seed so separate finite-size jobs do not accidentally use
    # the same resampling indices when their group sizes happen to match.
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, num_qubits]))
    evaluator = build_evaluator(args)
    distributions: Dict[str, np.ndarray] | None = None

    print("Loaded {} snapshots: N={}, h_length={}, group sizes={}..{}".format(
        int(np.sum(counts)), num_qubits, h_length, int(np.min(counts)), int(np.max(counts))
    ))
    print("Bootstrap: B={}, M={}, rows per replicate={}".format(
        args.bootstrap_repetitions, bootstrap_size, h_length * bootstrap_size
    ))

    progress = tqdm.trange(args.bootstrap_repetitions, desc="Bootstrap observables", unit="rep")
    for bootstrap_index in progress:
        for param_index, (param, source) in enumerate(zip(params, groups)):
            indices = rng.integers(0, source.shape[0], size=bootstrap_size)
            sqe = torch.from_numpy(source[indices])
            values = evaluator(float(param), sqe)
            if distributions is None:
                distributions = allocate_distributions(
                    values, args.bootstrap_repetitions, h_length
                )
            store_observables(distributions, values, bootstrap_index, param_index)

    if distributions is None:
        raise RuntimeError("No bootstrap observables were calculated.")

    output: Dict[str, np.ndarray] = {
        "N": np.asarray(num_qubits),
        "params": params,
        "d_values": np.asarray(args.d_values, dtype=np.int64),
        "predict_model": np.asarray(args.predict_model),
        "source_json": np.asarray(os.path.abspath(args.input)),
        "source_group_counts": counts,
        "bootstrap_repetitions": np.asarray(args.bootstrap_repetitions),
        "bootstrap_size": np.asarray(bootstrap_size),
        "num_parts": np.asarray(args.num_parts),
        "seed": np.asarray(args.seed),
        "confidence": np.asarray(args.confidence),
    }
    add_summary_fields(output, distributions, args.confidence)
    add_eval_new_aliases(output, distributions, params, args)

    parent = os.path.dirname(os.path.abspath(args.output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print("Saved bootstrap observables to: {}".format(args.output))
    print("Bootstrap keys: {}".format(
        ", ".join("bootstrap_{}".format(name) for name in distributions)
    ))


if __name__ == "__main__":
    main()
