from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import analyze_attention_distance_oseq_rope as base

try:
    import compare_attention_kernel_to_exact_correlations as exact_compare
except ImportError:
    exact_compare = None


MASK_TOKEN_ID = -1.0
METRIC_NAMES = list(base.METRIC_NAMES)
KERNEL_FIELDNAMES = [
    "num_qubits",
    "g",
    "step",
    "steps",
    "mask_fraction_before",
    "mask_fraction_after",
    "num_masked_before",
    "num_masked_after",
    "layer",
    "head",
    "query_scope",
    "key_type",
    "delta_site",
    "attention_mass",
    "num_prompts",
    "seed",
    "distance_mode",
    "unmask_schedule",
    "decode_mode",
]


def parse_args() -> argparse.Namespace:
    repo_dshadow = Path(__file__).resolve().parents[1]
    default_output = repo_dshadow / "attention_analysis" / "results" / "oseq_rope_diffusion_step_attention"
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Qdmodel_oseq_rope attention along the iterative masked-denoising "
            "generation trajectory. The model has no explicit time embedding; step is "
            "represented by the prompt state, i.e. which b tokens remain masked."
        )
    )

    parser.add_argument("--model_path", type=str, default="", help="Trained checkpoint path.")
    parser.add_argument("--qubits", nargs="+", default=["6", "8", "10", "12", "14"])
    parser.add_argument("--g_values", nargs="+", default=["0.0", "0.3", "0.5", "0.7", "0.9", "1.0"])
    parser.add_argument("--steps", type=int, default=4, help="Number of iterative unmasking steps to analyze.")
    parser.add_argument("--num_prompts", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--decode_mode", choices=["argmax", "sample"], default="sample")
    parser.add_argument(
        "--unmask_schedule",
        choices=["floor", "ceil"],
        default="floor",
        help=(
            "floor mirrors test_1.generate_oseq_batch; ceil guarantees all initially "
            "masked b tokens are filled by the final step."
        ),
    )
    parser.add_argument(
        "--query_scopes",
        nargs="+",
        default=["masked_b", "all_b"],
        choices=["masked_b", "filled_b", "all_b"],
        help="Which b-query subset to average attention over at each step.",
    )
    parser.add_argument(
        "--key_types",
        nargs="+",
        default=["P", "b"],
        choices=["P", "b", "g", "non_g"],
        help="Key groups for site-kernel rows.",
    )
    parser.add_argument(
        "--distance_mode",
        choices=["pbc_signed", "pbc_unsigned", "open_unsigned"],
        default="pbc_signed",
    )
    parser.add_argument("--pbc_distance", action="store_true", help="Use PBC site distance for scalar distance metrics.")
    parser.add_argument("--record_final", action="store_true", help="Also record one forward pass after the last update.")

    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--layer_num", type=int, default=None)
    parser.add_argument("--head_num", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--max_N", type=int, default=None)
    parser.add_argument("--rope_scaling_type", choices=["none", "linear", "dynamic", "ntk"], default=None)
    parser.add_argument("--rope_scaling_factor", type=float, default=None)
    parser.add_argument("--rope_theta", type=float, default=None)
    parser.add_argument("--checkpoint_non_strict", action="store_true")

    parser.add_argument(
        "--exact_files",
        nargs="+",
        default=None,
        help="Optional exact/eval .npz files used to compute kernel-exact similarity per diffusion step.",
    )
    parser.add_argument(
        "--exact_num_qubits",
        nargs="+",
        type=int,
        default=None,
        help="Optional N values matching --exact_files when N cannot be inferred.",
    )
    parser.add_argument("--correlation", default="ZZ", choices=["ZZ", "XX", "YY", "spin_dot", "Xstring"])
    parser.add_argument("--correlation_key", default="auto")
    parser.add_argument("--param_key", default="auto")
    parser.add_argument("--g_match_mode", choices=["interp", "nearest"], default="interp")
    parser.add_argument("--g_match_tol", type=float, default=1e-6)
    parser.add_argument("--use_abs_correlation", action="store_true")
    parser.add_argument("--min_distance", type=int, default=1)
    parser.add_argument(
        "--exact_attention_level",
        choices=["summary", "per_layer", "per_head"],
        default="per_head",
        help="Which kernel rows to use for optional exact-correlation profile comparisons.",
    )

    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def make_initial_prompt(g_value: float, P_batch: np.ndarray, device: Any) -> Any:
    batch_size, num_qubits = P_batch.shape
    seq_len = 1 + 2 * num_qubits
    prompt = torch.full((batch_size, seq_len, 1), MASK_TOKEN_ID, dtype=torch.float32, device=device)
    prompt[:, 0, 0] = float(g_value)
    prompt[:, 1:seq_len:2, 0] = torch.as_tensor(P_batch, dtype=torch.float32, device=device)
    return prompt


def make_prompt_P_cache(num_qubits: int, num_prompts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(2, 5, size=(num_prompts, num_qubits), dtype=np.int64)


def b_positions_for_length(seq_len: int, device: Any) -> Any:
    return torch.arange(2, seq_len, 2, dtype=torch.long, device=device)


def query_indicator_from_scope(current_masked_b: Any, query_scope: str) -> Any:
    if query_scope == "masked_b":
        return current_masked_b
    if query_scope == "filled_b":
        return ~current_masked_b
    if query_scope == "all_b":
        return torch.ones_like(current_masked_b, dtype=torch.bool)
    raise ValueError(f"Unknown query_scope: {query_scope}")


def sequence_site_index_torch(num_qubits: int, device: Any) -> Any:
    seq_len = 1 + 2 * num_qubits
    site_index = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    site_index[1:seq_len:2] = torch.arange(num_qubits, dtype=torch.long, device=device)
    site_index[2:seq_len:2] = torch.arange(num_qubits, dtype=torch.long, device=device)
    return site_index


def safe_head_average(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def compute_attention_metrics_for_scope(
    attention: Any,
    num_qubits: int,
    current_masked_b: Any,
    query_scope: str,
    pbc_distance: bool,
) -> Dict[str, np.ndarray]:
    attention = attention.to(dtype=torch.float64, device="cpu")
    current_masked_b = current_masked_b.to(device="cpu", dtype=torch.bool)
    batch_size, num_heads, seq_len, _ = attention.shape
    b_pos = torch.arange(2, seq_len, 2, dtype=torch.long)
    q_valid_b = query_indicator_from_scope(current_masked_b, query_scope).to(dtype=torch.float64)
    q_valid = torch.zeros((batch_size, seq_len), dtype=torch.float64)
    q_valid[:, b_pos] = q_valid_b
    denom = q_valid.sum().clamp_min(1.0)

    key_pos = torch.arange(seq_len, dtype=torch.long)
    selected_attention = attention * q_valid[:, None, :, None]
    token_dist = (key_pos[:, None] - key_pos[None, :]).abs().to(torch.float64)
    mean_token_distance = (selected_attention * token_dist[None, None, :, :]).sum(dim=(0, 2, 3)) / denom

    eps = torch.finfo(torch.float64).eps
    entropy_per_query = -(attention.clamp_min(eps) * attention.clamp_min(eps).log()).sum(dim=-1)
    entropy = (entropy_per_query * q_valid[:, None, :]).sum(dim=(0, 2)) / denom
    entropy_norm = entropy / math.log(max(seq_len, 2))

    P_pos = torch.arange(1, seq_len, 2, dtype=torch.long)
    attention_to_g = selected_attention[..., 0].sum(dim=(0, 2)) / denom
    attention_to_P = selected_attention.index_select(dim=-1, index=P_pos).sum(dim=(0, 2, 3)) / denom
    attention_to_b = selected_attention.index_select(dim=-1, index=b_pos).sum(dim=(0, 2, 3)) / denom

    site_index = sequence_site_index_torch(num_qubits, device="cpu")
    valid_key = site_index >= 0
    q_sites = site_index
    raw_diff = (q_sites[:, None] - site_index[None, :]).abs().to(torch.float64)
    if pbc_distance:
        raw_diff = torch.minimum(raw_diff, num_qubits - raw_diff)
    site_dist = torch.zeros((seq_len, seq_len), dtype=torch.float64)
    site_dist[:, valid_key] = raw_diff[:, valid_key]

    non_g_mass = attention[..., valid_key].sum(dim=-1).clamp_min(eps)
    site_distance_per_query = (attention * site_dist[None, None, :, :]).sum(dim=-1) / non_g_mass
    mean_site_distance = (site_distance_per_query * q_valid[:, None, :]).sum(dim=(0, 2)) / denom

    norm_denom = (num_qubits // 2) if pbc_distance else (num_qubits - 1)
    norm_denom = max(int(norm_denom), 1)
    mean_site_distance_norm = mean_site_distance / float(norm_denom)

    same_site = (q_sites[:, None] == site_index[None, :]) & valid_key[None, :]
    near_site = (raw_diff <= 1.0) & valid_key[None, :]
    attention_to_same_site = (
        attention * same_site.to(torch.float64)[None, None, :, :]
    ).sum(dim=-1)
    attention_to_same_site = (attention_to_same_site * q_valid[:, None, :]).sum(dim=(0, 2)) / denom
    attention_to_near_site_1 = (
        attention * near_site.to(torch.float64)[None, None, :, :]
    ).sum(dim=-1)
    attention_to_near_site_1 = (attention_to_near_site_1 * q_valid[:, None, :]).sum(dim=(0, 2)) / denom

    invalid_scope = float(q_valid.sum()) <= 0
    if invalid_scope:
        nan = torch.full((num_heads,), float("nan"), dtype=torch.float64)
        mean_token_distance = nan
        mean_site_distance = nan
        mean_site_distance_norm = nan
        attention_to_g = nan
        attention_to_P = nan
        attention_to_b = nan
        attention_to_same_site = nan
        attention_to_near_site_1 = nan
        entropy = nan
        entropy_norm = nan

    return {
        "mean_token_distance": mean_token_distance.numpy(),
        "mean_token_distance_norm": (mean_token_distance / max(seq_len - 1, 1)).numpy(),
        "mean_site_distance": mean_site_distance.numpy(),
        "mean_site_distance_norm": mean_site_distance_norm.numpy(),
        "attention_to_g": attention_to_g.numpy(),
        "attention_to_P": attention_to_P.numpy(),
        "attention_to_b": attention_to_b.numpy(),
        "attention_to_same_site": attention_to_same_site.numpy(),
        "attention_to_near_site_1": attention_to_near_site_1.numpy(),
        "entropy": entropy.numpy(),
        "entropy_norm": entropy_norm.numpy(),
    }


def displacement_matrix_np(query_sites: np.ndarray, key_sites: np.ndarray, num_qubits: int, distance_mode: str) -> np.ndarray:
    raw = key_sites[None, :] - query_sites[:, None]
    if distance_mode == "open_unsigned":
        return np.abs(raw)
    if distance_mode == "pbc_unsigned":
        abs_raw = np.abs(raw)
        return np.minimum(abs_raw, num_qubits - abs_raw)
    if distance_mode == "pbc_signed":
        wrapped = np.mod(raw, num_qubits)
        wrapped = np.where(wrapped > (num_qubits / 2.0), wrapped - num_qubits, wrapped)
        return wrapped.astype(int)
    raise ValueError(f"Unknown distance_mode: {distance_mode}")


def positions_for_key_type_np(num_qubits: int, key_type: str) -> np.ndarray:
    seq_len = 1 + 2 * num_qubits
    if key_type == "P":
        return np.arange(1, seq_len, 2, dtype=int)
    if key_type == "b":
        return np.arange(2, seq_len, 2, dtype=int)
    if key_type == "non_g":
        return np.arange(1, seq_len, dtype=int)
    if key_type == "g":
        return np.asarray([0], dtype=int)
    raise ValueError(f"Unknown key_type: {key_type}")


def sequence_site_index_np(num_qubits: int) -> np.ndarray:
    seq_len = 1 + 2 * num_qubits
    site_ids = np.full(seq_len, -1, dtype=int)
    site_ids[1:seq_len:2] = np.arange(num_qubits, dtype=int)
    site_ids[2:seq_len:2] = np.arange(num_qubits, dtype=int)
    return site_ids


def compute_kernel_for_scope(
    attention: Any,
    num_qubits: int,
    current_masked_b: Any,
    query_scope: str,
    key_types: Sequence[str],
    distance_mode: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    attn = attention.detach().cpu().numpy().astype(np.float64, copy=False)
    current_masked = current_masked_b.detach().cpu().numpy().astype(bool, copy=False)
    _, num_heads, _, _ = attn.shape
    site_ids = sequence_site_index_np(num_qubits)
    q_pos = np.arange(2, 1 + 2 * num_qubits, 2, dtype=int)
    q_sites = site_ids[q_pos]

    if query_scope == "masked_b":
        query_valid = current_masked.astype(np.float64)
    elif query_scope == "filled_b":
        query_valid = (~current_masked).astype(np.float64)
    elif query_scope == "all_b":
        query_valid = np.ones_like(current_masked, dtype=np.float64)
    else:
        raise ValueError(f"Unknown query_scope: {query_scope}")
    denom = float(query_valid.sum())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if denom <= 0:
        for key_type in key_types:
            out[key_type] = {}
        return out

    attn_q = attn[:, :, q_pos, :]
    for key_type in key_types:
        key_pos = positions_for_key_type_np(num_qubits, key_type)
        if key_type == "g":
            mass = (attn_q[:, :, :, 0] * query_valid[:, None, :]).sum(axis=(0, 2)) / denom
            out.setdefault("g", {})["g"] = mass
            continue

        key_sites = site_ids[key_pos]
        deltas = displacement_matrix_np(q_sites, key_sites, num_qubits, distance_mode)
        for delta in sorted(np.unique(deltas).tolist()):
            mask = (deltas == delta).astype(np.float64)
            per_query = (attn_q[:, :, :, key_pos] * mask[None, None, :, :]).sum(axis=-1)
            mass = (per_query * query_valid[:, None, :]).sum(axis=(0, 2)) / denom
            out.setdefault(key_type, {})[str(int(delta))] = mass
    return out


def metric_row_base(
    num_qubits: int,
    g_value: float,
    step: int,
    steps: int,
    mask_fraction_before: float,
    mask_fraction_after: float,
    num_masked_before: float,
    num_masked_after: float,
    layer: int,
    head: int,
    query_scope: str,
    num_prompts: int,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "num_qubits": int(num_qubits),
        "g": float(g_value),
        "step": int(step),
        "steps": int(steps),
        "mask_fraction_before": float(mask_fraction_before),
        "mask_fraction_after": float(mask_fraction_after),
        "num_masked_before": float(num_masked_before),
        "num_masked_after": float(num_masked_after),
        "layer": int(layer),
        "head": int(head),
        "query_scope": query_scope,
        "num_prompts": int(num_prompts),
        "seed": int(seed),
        "unmask_schedule": args.unmask_schedule,
        "decode_mode": args.decode_mode,
    }


def add_metric_rows_for_step(
    metric_rows: List[Dict[str, Any]],
    attention_maps: Sequence[Any],
    num_qubits: int,
    g_value: float,
    step: int,
    current_masked_before: Any,
    current_masked_after: Any,
    query_scopes: Sequence[str],
    num_prompts: int,
    seed: int,
    args: argparse.Namespace,
) -> None:
    mask_fraction_before = float(current_masked_before.float().mean().item())
    mask_fraction_after = float(current_masked_after.float().mean().item())
    num_masked_before = float(current_masked_before.float().sum(dim=1).mean().item())
    num_masked_after = float(current_masked_after.float().sum(dim=1).mean().item())

    for query_scope in query_scopes:
        per_head_rows: List[Dict[str, Any]] = []
        for layer_idx, attention in enumerate(attention_maps):
            metrics = compute_attention_metrics_for_scope(
                attention=attention,
                num_qubits=num_qubits,
                current_masked_b=current_masked_before,
                query_scope=query_scope,
                pbc_distance=args.pbc_distance,
            )
            num_heads = int(attention.shape[1])
            for head_idx in range(num_heads):
                row = metric_row_base(
                    num_qubits,
                    g_value,
                    step,
                    args.steps,
                    mask_fraction_before,
                    mask_fraction_after,
                    num_masked_before,
                    num_masked_after,
                    layer_idx,
                    head_idx,
                    query_scope,
                    num_prompts,
                    seed,
                    args,
                )
                for metric_name in METRIC_NAMES:
                    row[metric_name] = float(metrics[metric_name][head_idx])
                metric_rows.append(row)
                per_head_rows.append(row)

        for layer_idx in sorted({int(row["layer"]) for row in per_head_rows}):
            source = [row for row in per_head_rows if int(row["layer"]) == layer_idx]
            row = metric_row_base(
                num_qubits,
                g_value,
                step,
                args.steps,
                mask_fraction_before,
                mask_fraction_after,
                num_masked_before,
                num_masked_after,
                layer_idx,
                -1,
                query_scope,
                num_prompts,
                seed,
                args,
            )
            for metric_name in METRIC_NAMES:
                row[metric_name] = safe_head_average(np.asarray([r[metric_name] for r in source], dtype=float))
            metric_rows.append(row)

        row = metric_row_base(
            num_qubits,
            g_value,
            step,
            args.steps,
            mask_fraction_before,
            mask_fraction_after,
            num_masked_before,
            num_masked_after,
            -1,
            -1,
            query_scope,
            num_prompts,
            seed,
            args,
        )
        for metric_name in METRIC_NAMES:
            row[metric_name] = safe_head_average(np.asarray([r[metric_name] for r in per_head_rows], dtype=float))
        metric_rows.append(row)


def add_kernel_rows_for_step(
    kernel_rows: List[Dict[str, Any]],
    attention_maps: Sequence[Any],
    num_qubits: int,
    g_value: float,
    step: int,
    current_masked_before: Any,
    current_masked_after: Any,
    query_scopes: Sequence[str],
    num_prompts: int,
    seed: int,
    args: argparse.Namespace,
) -> None:
    mask_fraction_before = float(current_masked_before.float().mean().item())
    mask_fraction_after = float(current_masked_after.float().mean().item())
    num_masked_before = float(current_masked_before.float().sum(dim=1).mean().item())
    num_masked_after = float(current_masked_after.float().sum(dim=1).mean().item())

    for query_scope in query_scopes:
        real_rows: List[Dict[str, Any]] = []
        for layer_idx, attention in enumerate(attention_maps):
            kernel = compute_kernel_for_scope(
                attention=attention,
                num_qubits=num_qubits,
                current_masked_b=current_masked_before,
                query_scope=query_scope,
                key_types=args.key_types,
                distance_mode=args.distance_mode,
            )
            num_heads = int(attention.shape[1])
            for key_type, delta_map in kernel.items():
                for delta_label, head_values in delta_map.items():
                    for head_idx in range(num_heads):
                        row = metric_row_base(
                            num_qubits,
                            g_value,
                            step,
                            args.steps,
                            mask_fraction_before,
                            mask_fraction_after,
                            num_masked_before,
                            num_masked_after,
                            layer_idx,
                            head_idx,
                            query_scope,
                            num_prompts,
                            seed,
                            args,
                        )
                        row.update(
                            {
                                "key_type": key_type,
                                "delta_site": delta_label,
                                "attention_mass": float(head_values[head_idx]),
                                "distance_mode": args.distance_mode,
                            }
                        )
                        kernel_rows.append(row)
                        real_rows.append(row)

        group_keys = sorted({(r["layer"], r["key_type"], r["delta_site"]) for r in real_rows})
        for layer_idx, key_type, delta_label in group_keys:
            source = [
                row
                for row in real_rows
                if row["layer"] == layer_idx and row["key_type"] == key_type and row["delta_site"] == delta_label
            ]
            row = metric_row_base(
                num_qubits,
                g_value,
                step,
                args.steps,
                mask_fraction_before,
                mask_fraction_after,
                num_masked_before,
                num_masked_after,
                int(layer_idx),
                -1,
                query_scope,
                num_prompts,
                seed,
                args,
            )
            row.update(
                {
                    "key_type": key_type,
                    "delta_site": delta_label,
                    "attention_mass": safe_head_average(np.asarray([r["attention_mass"] for r in source], dtype=float)),
                    "distance_mode": args.distance_mode,
                }
            )
            kernel_rows.append(row)

        global_keys = sorted({(r["key_type"], r["delta_site"]) for r in real_rows})
        for key_type, delta_label in global_keys:
            source = [row for row in real_rows if row["key_type"] == key_type and row["delta_site"] == delta_label]
            row = metric_row_base(
                num_qubits,
                g_value,
                step,
                args.steps,
                mask_fraction_before,
                mask_fraction_after,
                num_masked_before,
                num_masked_after,
                -1,
                -1,
                query_scope,
                num_prompts,
                seed,
                args,
            )
            row.update(
                {
                    "key_type": key_type,
                    "delta_site": delta_label,
                    "attention_mass": safe_head_average(np.asarray([r["attention_mass"] for r in source], dtype=float)),
                    "distance_mode": args.distance_mode,
                }
            )
            kernel_rows.append(row)


def choose_unmask_count(num_qubits: int, steps: int, schedule: str) -> int:
    if schedule == "ceil":
        return max(1, int(math.ceil(float(num_qubits) / max(1, steps))))
    return max(1, int(num_qubits // max(1, steps)))


def run_trajectory_batch(
    model: Any,
    prompt: Any,
    num_qubits: int,
    g_value: float,
    seed: int,
    args: argparse.Namespace,
    metric_rows: List[Dict[str, Any]],
    kernel_rows: List[Dict[str, Any]],
) -> None:
    b_pos = b_positions_for_length(prompt.shape[1], prompt.device)
    unmask_count = choose_unmask_count(num_qubits, args.steps, args.unmask_schedule)
    generator = None
    if args.decode_mode == "sample":
        generator = torch.Generator(device=prompt.device)
        generator.manual_seed(int(seed + 7919 * num_qubits + round(1000 * float(g_value))))

    for step in range(args.steps):
        current_masked = prompt[:, b_pos, 0] == MASK_TOKEN_ID
        if not bool(current_masked.any()):
            break
        with torch.no_grad():
            logits, attention_maps = model(prompt, mask_indices=current_masked, return_attn_weights=True)
        probs = torch.softmax(logits / float(args.temperature), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        entropy_masked = entropy.masked_fill(~current_masked, float("inf"))
        k = min(int(unmask_count), num_qubits)
        idx = torch.topk(entropy_masked, k=k, largest=False, dim=1).indices
        gathered = entropy_masked.gather(1, idx)
        valid = torch.isfinite(gathered)

        prompt_after = prompt.clone()
        if bool(valid.any()):
            p_selected = probs[..., 1].gather(1, idx)
            if args.decode_mode == "sample":
                rand = torch.rand(p_selected.shape, device=prompt.device, generator=generator)
                pred_values = (rand < p_selected).float()
            else:
                pred_values = (p_selected > 0.5).float()
            global_pos = 2 + 2 * idx
            b_idx = torch.arange(prompt.shape[0], device=prompt.device).unsqueeze(1).expand_as(global_pos)
            prompt_after[b_idx[valid], global_pos[valid], 0] = pred_values[valid]

        next_masked = prompt_after[:, b_pos, 0] == MASK_TOKEN_ID
        add_metric_rows_for_step(
            metric_rows,
            attention_maps,
            num_qubits,
            g_value,
            step,
            current_masked,
            next_masked,
            args.query_scopes,
            prompt.shape[0],
            seed,
            args,
        )
        add_kernel_rows_for_step(
            kernel_rows,
            attention_maps,
            num_qubits,
            g_value,
            step,
            current_masked,
            next_masked,
            args.query_scopes,
            prompt.shape[0],
            seed,
            args,
        )
        prompt = prompt_after

    if args.record_final:
        current_masked = prompt[:, b_pos, 0] == MASK_TOKEN_ID
        with torch.no_grad():
            _, attention_maps = model(prompt, mask_indices=current_masked, return_attn_weights=True)
        add_metric_rows_for_step(
            metric_rows,
            attention_maps,
            num_qubits,
            g_value,
            args.steps,
            current_masked,
            current_masked,
            args.query_scopes,
            prompt.shape[0],
            seed,
            args,
        )
        add_kernel_rows_for_step(
            kernel_rows,
            attention_maps,
            num_qubits,
            g_value,
            args.steps,
            current_masked,
            current_masked,
            args.query_scopes,
            prompt.shape[0],
            seed,
            args,
        )


def analyze_condition(
    model: Any,
    num_qubits: int,
    g_value: float,
    P_cache: np.ndarray,
    device: Any,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    metric_rows: List[Dict[str, Any]] = []
    kernel_rows: List[Dict[str, Any]] = []
    num_prompts = int(P_cache.shape[0])
    for start in range(0, num_prompts, args.batch_size):
        stop = min(start + args.batch_size, num_prompts)
        prompt = make_initial_prompt(float(g_value), P_cache[start:stop], device)
        run_trajectory_batch(
            model=model,
            prompt=prompt,
            num_qubits=num_qubits,
            g_value=float(g_value),
            seed=seed + start,
            args=args,
            metric_rows=metric_rows,
            kernel_rows=kernel_rows,
        )
    return metric_rows, kernel_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    key_names = [
        "num_qubits",
        "g",
        "step",
        "steps",
        "layer",
        "head",
        "query_scope",
        "unmask_schedule",
        "decode_mode",
    ]
    value_names = [
        "mask_fraction_before",
        "mask_fraction_after",
        "num_masked_before",
        "num_masked_after",
    ] + METRIC_NAMES
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in key_names)].append(row)

    out: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        weight = np.asarray([float(row.get("num_prompts", 1.0)) for row in group], dtype=float)
        total_weight = float(np.sum(weight))
        merged = {name: value for name, value in zip(key_names, key)}
        merged["num_prompts"] = int(round(total_weight))
        merged["seed"] = group[0].get("seed", "")
        for name in value_names:
            values = np.asarray([float(row.get(name, float("nan"))) for row in group], dtype=float)
            finite = np.isfinite(values)
            if not np.any(finite):
                merged[name] = float("nan")
            else:
                merged[name] = float(np.sum(values[finite] * weight[finite]) / np.sum(weight[finite]))
        out.append(merged)
    return out


def aggregate_kernel_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    key_names = [
        "num_qubits",
        "g",
        "step",
        "steps",
        "layer",
        "head",
        "query_scope",
        "key_type",
        "delta_site",
        "distance_mode",
        "unmask_schedule",
        "decode_mode",
    ]
    value_names = [
        "mask_fraction_before",
        "mask_fraction_after",
        "num_masked_before",
        "num_masked_after",
        "attention_mass",
    ]
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in key_names)].append(row)

    out: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        weight = np.asarray([float(row.get("num_prompts", 1.0)) for row in group], dtype=float)
        total_weight = float(np.sum(weight))
        merged = {name: value for name, value in zip(key_names, key)}
        merged["num_prompts"] = int(round(total_weight))
        merged["seed"] = group[0].get("seed", "")
        for name in value_names:
            values = np.asarray([float(row.get(name, float("nan"))) for row in group], dtype=float)
            finite = np.isfinite(values)
            if not np.any(finite):
                merged[name] = float("nan")
            else:
                merged[name] = float(np.sum(values[finite] * weight[finite]) / np.sum(weight[finite]))
        out.append(merged)
    return out


def metric_fieldnames() -> List[str]:
    return [
        "num_qubits",
        "g",
        "step",
        "steps",
        "mask_fraction_before",
        "mask_fraction_after",
        "num_masked_before",
        "num_masked_after",
        "layer",
        "head",
        "query_scope",
        "num_prompts",
        "seed",
        "unmask_schedule",
        "decode_mode",
    ] + METRIC_NAMES


def level_matches(row: Mapping[str, Any], level: str) -> bool:
    layer = int(row["layer"])
    head = int(row["head"])
    if level == "summary":
        return layer == -1 and head == -1
    if level == "per_layer":
        return layer >= 0 and head == -1
    if level == "per_head":
        return layer >= 0 and head >= 0
    raise ValueError(f"Unknown level: {level}")


def load_exact_by_n(args: argparse.Namespace) -> Dict[int, Any]:
    if not args.exact_files:
        return {}
    if exact_compare is None:
        raise RuntimeError("compare_attention_kernel_to_exact_correlations.py could not be imported.")
    exact_args = argparse.Namespace(
        exact_files=args.exact_files,
        num_qubits=args.exact_num_qubits,
        correlation=args.correlation,
        correlation_key=args.correlation_key,
        param_key=args.param_key,
    )
    return exact_compare.load_exact_files(exact_args)


def aggregate_kernel_abs_distance(
    kernel_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, float, int, int, str, str, int, int], float] = defaultdict(float)
    meta: Dict[Tuple[int, float, int, int, str, str, int, int], Mapping[str, Any]] = {}
    for row in kernel_rows:
        if row.get("key_type") == "g":
            continue
        if row.get("delta_site") in ("", "g"):
            continue
        if not level_matches(row, args.exact_attention_level):
            continue
        distance = abs(int(float(row["delta_site"])))
        if distance < args.min_distance:
            continue
        key = (
            int(row["num_qubits"]),
            float(row["g"]),
            int(row["step"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["query_scope"]),
            str(row["key_type"]),
            distance,
        )
        grouped[key] += float(row["attention_mass"])
        meta[key] = row
    out: List[Dict[str, Any]] = []
    for key, mass in sorted(grouped.items(), key=lambda item: str(item[0])):
        num_qubits, g_value, step, layer, head, query_scope, key_type, distance = key
        source = meta[key]
        out.append(
            {
                "num_qubits": num_qubits,
                "g": g_value,
                "step": step,
                "steps": int(source["steps"]),
                "mask_fraction_before": float(source["mask_fraction_before"]),
                "mask_fraction_after": float(source["mask_fraction_after"]),
                "layer": layer,
                "head": head,
                "query_scope": query_scope,
                "key_type": key_type,
                "distance": distance,
                "attention_mass": float(mass),
            }
        )
    return out


def compute_exact_profile_rows(
    kernel_rows: Sequence[Mapping[str, Any]],
    exact_by_n: Mapping[int, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if not exact_by_n:
        return []
    groups: Dict[Tuple[int, float, int, str, str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in aggregate_kernel_abs_distance(kernel_rows, args):
        groups[
            (
                int(row["num_qubits"]),
                float(row["g"]),
                int(row["step"]),
                str(row["query_scope"]),
                str(row["key_type"]),
                int(row["layer"]),
                int(row["head"]),
            )
        ].append(row)

    exact_lookup_args = argparse.Namespace(
        g_match_mode=args.g_match_mode,
        g_match_tol=args.g_match_tol,
        use_abs_correlation=args.use_abs_correlation,
    )
    out: List[Dict[str, Any]] = []
    for (num_qubits, g_value, step, query_scope, key_type, layer, head), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        if num_qubits not in exact_by_n:
            continue
        rows = sorted(rows, key=lambda row: int(row["distance"]))
        distances: List[int] = []
        attention: List[float] = []
        correlation: List[float] = []
        for row in rows:
            distance = int(row["distance"])
            exact_result = exact_compare.exact_value_at(exact_by_n[num_qubits], distance, g_value, exact_lookup_args)
            if exact_result is None:
                continue
            corr_value, _, _ = exact_result
            distances.append(distance)
            attention.append(float(row["attention_mass"]))
            correlation.append(float(corr_value))
        if len(distances) < 2:
            continue
        d_arr = np.asarray(distances, dtype=float)
        attn_arr = np.asarray(attention, dtype=float)
        corr_signed = np.asarray(correlation, dtype=float)
        corr_mass = np.abs(corr_signed)
        target = corr_mass if args.use_abs_correlation else corr_signed
        attn_profile = exact_compare.normalize_nonnegative(attn_arr)
        corr_profile = exact_compare.normalize_nonnegative(corr_mass)
        attention_length = float(np.sum(d_arr * attn_profile)) if np.all(np.isfinite(attn_profile)) else float("nan")
        correlation_length_proxy = float(np.sum(d_arr * exact_compare.normalize_nonnegative(corr_mass))) if np.sum(corr_mass) > 0 else float("nan")
        out.append(
            {
                "num_qubits": num_qubits,
                "g": g_value,
                "step": step,
                "query_scope": query_scope,
                "key_type": key_type,
                "layer": layer,
                "head": head,
                "num_distances": len(distances),
                "distances": ";".join(str(d) for d in distances),
                "profile_pearson": exact_compare.pearsonr(attn_profile, exact_compare.normalize_nonnegative(np.abs(target))),
                "profile_spearman": exact_compare.spearmanr(attn_profile, exact_compare.normalize_nonnegative(np.abs(target))),
                "profile_cosine": exact_compare.cosine_similarity(attn_profile, corr_profile),
                "attention_length": attention_length,
                "correlation_length_proxy": correlation_length_proxy,
                "attention_profile": ";".join(f"{v:.12g}" for v in attn_profile),
                "correlation_profile": ";".join(f"{v:.12g}" for v in corr_profile),
                "signed_correlation_profile": ";".join(f"{v:.12g}" for v in corr_signed),
            }
        )
    return out


def plot_attention_masses_vs_step(rows: Sequence[Mapping[str, Any]], output_dir: Path, dpi: int) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")
    summary = [row for row in rows if int(row["layer"]) == -1 and int(row["head"]) == -1]
    paths: List[Path] = []
    for num_qubits in sorted({int(row["num_qubits"]) for row in summary}):
        for query_scope in sorted({row["query_scope"] for row in summary}):
            selected = [row for row in summary if int(row["num_qubits"]) == num_qubits and row["query_scope"] == query_scope]
            if not selected:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True)
            metrics = [("attention_to_g", "to g"), ("attention_to_P", "to P"), ("attention_to_b", "to b")]
            for ax, (metric, title) in zip(axes, metrics):
                for g_value in sorted({float(row["g"]) for row in selected}):
                    series = sorted([row for row in selected if float(row["g"]) == g_value], key=lambda row: int(row["step"]))
                    ax.plot([int(row["step"]) for row in series], [float(row[metric]) for row in series], marker="o", label=f"g={g_value:g}")
                ax.set_title(title)
                ax.set_xlabel("denoising step")
                ax.grid(alpha=0.25)
            axes[0].set_ylabel("attention mass")
            axes[-1].legend(frameon=False, fontsize=8, ncol=2)
            fig.suptitle(f"N={num_qubits}, {query_scope} queries")
            fig.tight_layout()
            path = output_dir / f"diffusion_step_attention_mass_N{num_qubits}_{query_scope}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def plot_distance_vs_step(rows: Sequence[Mapping[str, Any]], output_dir: Path, dpi: int) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")
    summary = [row for row in rows if int(row["layer"]) == -1 and int(row["head"]) == -1]
    paths: List[Path] = []
    for num_qubits in sorted({int(row["num_qubits"]) for row in summary}):
        for query_scope in sorted({row["query_scope"] for row in summary}):
            selected = [row for row in summary if int(row["num_qubits"]) == num_qubits and row["query_scope"] == query_scope]
            if not selected:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            for g_value in sorted({float(row["g"]) for row in selected}):
                series = sorted([row for row in selected if float(row["g"]) == g_value], key=lambda row: int(row["step"]))
                ax.plot(
                    [int(row["step"]) for row in series],
                    [float(row["mean_site_distance_norm"]) for row in series],
                    marker="o",
                    label=f"g={g_value:g}",
                )
            ax.set_xlabel("denoising step")
            ax.set_ylabel("mean site distance norm")
            ax.set_title(f"N={num_qubits}, {query_scope} queries")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            fig.tight_layout()
            path = output_dir / f"diffusion_step_mean_site_distance_N{num_qubits}_{query_scope}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def plot_kernel_heatmaps_vs_step(kernel_rows: Sequence[Mapping[str, Any]], output_dir: Path, dpi: int) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")
    summary = [
        row for row in kernel_rows
        if int(row["layer"]) == -1 and int(row["head"]) == -1 and row.get("delta_site") not in ("", "g")
    ]
    paths: List[Path] = []
    for num_qubits in sorted({int(row["num_qubits"]) for row in summary}):
        for query_scope in sorted({row["query_scope"] for row in summary}):
            for key_type in sorted({row["key_type"] for row in summary if row["key_type"] != "g"}):
                for g_value in sorted({float(row["g"]) for row in summary}):
                    selected = [
                        row for row in summary
                        if int(row["num_qubits"]) == num_qubits
                        and row["query_scope"] == query_scope
                        and row["key_type"] == key_type
                        and float(row["g"]) == g_value
                    ]
                    if not selected:
                        continue
                    steps = sorted({int(row["step"]) for row in selected})
                    deltas = sorted({int(float(row["delta_site"])) for row in selected})
                    matrix = np.full((len(deltas), len(steps)), np.nan, dtype=float)
                    lookup = {(int(float(row["delta_site"])), int(row["step"])): float(row["attention_mass"]) for row in selected}
                    for i, delta in enumerate(deltas):
                        for j, step in enumerate(steps):
                            matrix[i, j] = lookup.get((delta, step), np.nan)
                    fig, ax = plt.subplots(figsize=(max(4.8, 0.7 * len(steps) + 2.2), max(4.0, 0.28 * len(deltas) + 1.8)))
                    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
                    ax.set_xticks(np.arange(len(steps)))
                    ax.set_xticklabels([str(step) for step in steps])
                    ax.set_yticks(np.arange(len(deltas)))
                    ax.set_yticklabels([str(delta) for delta in deltas])
                    ax.set_xlabel("denoising step")
                    ax.set_ylabel("site displacement")
                    ax.set_title(f"N={num_qubits}, g={g_value:g}, {query_scope} -> {key_type}")
                    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
                    cbar.set_label("attention mass")
                    fig.tight_layout()
                    path = output_dir / f"diffusion_step_kernel_N{num_qubits}_g{str(g_value).replace('.', 'p')}_{query_scope}_to_{key_type}.png"
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    paths.append(path)
    return paths


def plot_exact_similarity_vs_step(profile_rows: Sequence[Mapping[str, Any]], output_dir: Path, dpi: int) -> List[Path]:
    if not profile_rows:
        return []
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")
    paths: List[Path] = []
    for num_qubits in sorted({int(row["num_qubits"]) for row in profile_rows}):
        for query_scope in sorted({row["query_scope"] for row in profile_rows}):
            for key_type in sorted({row["key_type"] for row in profile_rows}):
                selected = [
                    row for row in profile_rows
                    if int(row["num_qubits"]) == num_qubits
                    and row["query_scope"] == query_scope
                    and row["key_type"] == key_type
                ]
                if not selected:
                    continue
                fig, ax = plt.subplots(figsize=(6.2, 4.0))
                for g_value in sorted({float(row["g"]) for row in selected}):
                    series = []
                    for step in sorted({int(row["step"]) for row in selected if float(row["g"]) == g_value}):
                        vals = [
                            float(row["profile_cosine"])
                            for row in selected
                            if float(row["g"]) == g_value and int(row["step"]) == step and math.isfinite(float(row["profile_cosine"]))
                        ]
                        if vals:
                            series.append((step, float(np.median(vals))))
                    if series:
                        ax.plot([x for x, _ in series], [y for _, y in series], marker="o", label=f"g={g_value:g}")
                ax.set_xlabel("denoising step")
                ax.set_ylabel("median profile cosine")
                ax.set_ylim(0, 1.03)
                ax.set_title(f"N={num_qubits}, {query_scope} -> {key_type}")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False, fontsize=8, ncol=2)
                fig.tight_layout()
                path = output_dir / f"diffusion_step_exact_similarity_N{num_qubits}_{query_scope}_to_{key_type}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                paths.append(path)
    return paths


def write_metadata(path: Path, args: argparse.Namespace, qubits: Sequence[int], g_values: Sequence[float], model_metadata: Mapping[str, Any], row_counts: Mapping[str, int]) -> None:
    payload = {
        "args": vars(args),
        "qubits": list(map(int, qubits)),
        "g_values": list(map(float, g_values)),
        "row_counts": dict(row_counts),
        "model": model_metadata,
        "notes": {
            "step": (
                "Generation step in the iterative masked-denoising loop. Qdmodel_oseq_rope has "
                "no explicit timestep embedding; the step is represented by the prompt state."
            ),
            "mask_fraction_before": "Fraction of b tokens still masked before this step's forward pass.",
            "query_scope=masked_b": "Only b query positions still masked before this step.",
            "query_scope=filled_b": "Only b query positions already filled before this step.",
            "query_scope=all_b": "Every b query position, whether masked or filled.",
            "step=0": "The all-b-masked initial denoising pass, matching the default previous attention analyses.",
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError("PyTorch is required for diffusion-step attention analysis.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.num_prompts <= 0:
        raise ValueError("--num_prompts must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    qubits = base.parse_int_values(args.qubits)
    g_values = base.parse_float_values(args.g_values)
    device = base.resolve_device(args.device)
    torch.manual_seed(args.seed)
    model, config, model_metadata = base.load_model(args, qubits, device)
    print(f"Using device: {device}")
    print(f"Model config: {config}")

    all_metric_rows: List[Dict[str, Any]] = []
    all_kernel_rows: List[Dict[str, Any]] = []
    for num_qubits in qubits:
        cache_seed = int(args.seed + 1009 * num_qubits)
        P_cache = make_prompt_P_cache(num_qubits, args.num_prompts, cache_seed)
        for g_value in tqdm(g_values, desc=f"N={num_qubits}", unit="g"):
            metric_rows, kernel_rows = analyze_condition(
                model=model,
                num_qubits=num_qubits,
                g_value=float(g_value),
                P_cache=P_cache,
                device=device,
                args=args,
                seed=cache_seed,
            )
            all_metric_rows.extend(metric_rows)
            all_kernel_rows.extend(kernel_rows)

    all_metric_rows = aggregate_metric_rows(all_metric_rows)
    all_kernel_rows = aggregate_kernel_rows(all_kernel_rows)

    exact_by_n = load_exact_by_n(args)
    exact_profile_rows = compute_exact_profile_rows(all_kernel_rows, exact_by_n, args) if exact_by_n else []

    if args.dry_run:
        print(f"Dry run completed: {len(all_metric_rows)} metric rows, {len(all_kernel_rows)} kernel rows.")
        if exact_profile_rows:
            print(f"Computed {len(exact_profile_rows)} kernel-exact profile rows.")
        for row in all_metric_rows[: min(5, len(all_metric_rows))]:
            print({key: row[key] for key in ("num_qubits", "g", "step", "layer", "head", "query_scope", "attention_to_g")})
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "diffusion_step_attention_metrics.csv"
    kernel_path = output_dir / "diffusion_step_site_kernel.csv"
    metadata_path = output_dir / "diffusion_step_attention_metadata.json"
    write_csv(metric_path, all_metric_rows, metric_fieldnames())
    write_csv(kernel_path, all_kernel_rows, KERNEL_FIELDNAMES)
    print(f"Wrote diffusion-step attention metrics: {metric_path}")
    print(f"Wrote diffusion-step site kernels: {kernel_path}")

    exact_profile_path: Optional[Path] = None
    if exact_profile_rows:
        exact_profile_path = output_dir / "diffusion_step_kernel_exact_profile_similarity.csv"
        write_csv(
            exact_profile_path,
            exact_profile_rows,
            [
                "num_qubits",
                "g",
                "step",
                "query_scope",
                "key_type",
                "layer",
                "head",
                "num_distances",
                "distances",
                "profile_pearson",
                "profile_spearman",
                "profile_cosine",
                "attention_length",
                "correlation_length_proxy",
                "attention_profile",
                "correlation_profile",
                "signed_correlation_profile",
            ],
        )
        print(f"Wrote diffusion-step kernel/exact profiles: {exact_profile_path}")

    write_metadata(
        metadata_path,
        args,
        qubits,
        g_values,
        model_metadata,
        {
            "metric_rows": len(all_metric_rows),
            "kernel_rows": len(all_kernel_rows),
            "exact_profile_rows": len(exact_profile_rows),
        },
    )
    print(f"Wrote metadata: {metadata_path}")

    if not args.no_plots:
        written: List[Path] = []
        written.extend(plot_attention_masses_vs_step(all_metric_rows, output_dir, args.dpi))
        written.extend(plot_distance_vs_step(all_metric_rows, output_dir, args.dpi))
        written.extend(plot_kernel_heatmaps_vs_step(all_kernel_rows, output_dir, args.dpi))
        written.extend(plot_exact_similarity_vs_step(exact_profile_rows, output_dir, args.dpi))
        for path in written:
            print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
