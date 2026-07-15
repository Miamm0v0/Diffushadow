from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import analyze_attention_distance_oseq_rope as base

torch = base.torch


KEY_TYPES = ["P", "b", "non_g", "g"]


def parse_args() -> argparse.Namespace:
    repo_dshadow = Path(__file__).resolve().parents[1]
    default_output = repo_dshadow / "attention_analysis" / "results" / "oseq_rope_site_kernel"

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Qdmodel_oseq_rope attention into site-displacement kernels, "
            "for example b_i -> P_{i+r} and b_i -> b_{i+r}."
        )
    )
    parser.add_argument("--model_path", type=str, default="", help="Trained checkpoint path.")
    parser.add_argument("--qubits", nargs="+", default=["6", "8", "10", "12", "14"])
    parser.add_argument("--g_values", nargs="+", default=["0.0", "0.2", "0.5", "0.7", "1.0"])
    parser.add_argument(
        "--query_groups",
        nargs="+",
        default=["b"],
        choices=["b", "P", "all", "non_g"],
        help="Query groups. Site kernels use only query tokens with a site id.",
    )
    parser.add_argument(
        "--key_types",
        nargs="+",
        default=["P", "b"],
        choices=KEY_TYPES,
        help="Key groups to aggregate. Use non_g for P+b combined, g for attention to the scalar g token.",
    )
    parser.add_argument(
        "--distance_mode",
        choices=["pbc_signed", "pbc_unsigned", "open_unsigned"],
        default="pbc_signed",
        help="How site displacement r = key_site - query_site is represented.",
    )
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask_mode", choices=["all", "random", "none"], default="all")
    parser.add_argument("--mask_prob", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--layer_num", type=int, default=None)
    parser.add_argument("--head_num", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--max_N", type=int, default=None)
    parser.add_argument("--rope_scaling_type", choices=["none", "linear", "dynamic", "ntk"], default=None)
    parser.add_argument("--rope_scaling_factor", type=float, default=None)
    parser.add_argument("--rope_theta", type=float, default=None)
    parser.add_argument("--checkpoint_non_strict", action="store_true")

    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--plot_key_types",
        nargs="+",
        default=["P", "b"],
        choices=KEY_TYPES,
        help="Key groups to draw as summary heatmaps.",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def sequence_site_index(num_qubits: int) -> np.ndarray:
    seq_len = 1 + 2 * num_qubits
    sites = np.full(seq_len, -1, dtype=int)
    sites[1:seq_len:2] = np.arange(num_qubits, dtype=int)
    sites[2:seq_len:2] = np.arange(num_qubits, dtype=int)
    return sites


def positions_for_query_group(num_qubits: int, query_group: str) -> np.ndarray:
    seq_len = 1 + 2 * num_qubits
    if query_group == "b":
        return np.arange(2, seq_len, 2, dtype=int)
    if query_group == "P":
        return np.arange(1, seq_len, 2, dtype=int)
    if query_group == "non_g":
        return np.arange(1, seq_len, dtype=int)
    if query_group == "all":
        return np.arange(0, seq_len, dtype=int)
    raise ValueError(f"Unknown query_group: {query_group}")


def positions_for_key_type(num_qubits: int, key_type: str) -> np.ndarray:
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


def displacement_matrix(
    query_sites: np.ndarray,
    key_sites: np.ndarray,
    num_qubits: int,
    distance_mode: str,
) -> np.ndarray:
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


def compute_kernel(
    attention: Any,
    num_qubits: int,
    query_group: str,
    key_types: Sequence[str],
    distance_mode: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    # attention: torch tensor [B, H, L, L], detached on CPU by the model.
    attn = attention.detach().cpu().numpy().astype(np.float64, copy=False)
    _, num_heads, _, _ = attn.shape

    site_ids = sequence_site_index(num_qubits)
    q_pos_all = positions_for_query_group(num_qubits, query_group)
    valid_query = site_ids[q_pos_all] >= 0
    q_pos = q_pos_all[valid_query]
    if q_pos.size == 0:
        raise ValueError(f"query_group={query_group!r} does not contain site tokens.")

    q_sites = site_ids[q_pos]
    attn_q = attn[:, :, q_pos, :]
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for key_type in key_types:
        key_pos = positions_for_key_type(num_qubits, key_type)
        if key_type == "g":
            mass = attn_q[:, :, :, 0].mean(axis=(0, 2))
            out.setdefault("g", {})["g"] = mass
            continue

        key_sites = site_ids[key_pos]
        deltas = displacement_matrix(q_sites, key_sites, num_qubits, distance_mode)
        for delta in sorted(np.unique(deltas).tolist()):
            mask = (deltas == delta).astype(np.float64)
            mass = (attn_q[:, :, :, key_pos] * mask[None, None, :, :]).sum(axis=-1).mean(axis=(0, 2))
            out.setdefault(key_type, {})[str(int(delta))] = mass

    # The combined P+b mass is useful for checking normalization and total locality.
    if "non_g" not in out and "non_g" in key_types:
        out["non_g"] = {}
    return out


def update_accumulators(
    accumulators: MutableMapping[Tuple[int, int, str, str], MutableMapping[str, float]],
    attention_maps: Sequence[Any],
    num_qubits: int,
    query_group: str,
    key_types: Sequence[str],
    distance_mode: str,
    batch_weight: int,
) -> None:
    for layer_idx, attention in enumerate(attention_maps):
        kernel = compute_kernel(attention, num_qubits, query_group, key_types, distance_mode)
        num_heads = int(attention.shape[1])
        for key_type, delta_map in kernel.items():
            for delta_label, head_values in delta_map.items():
                for head_idx in range(num_heads):
                    acc = accumulators[(layer_idx, head_idx, key_type, delta_label)]
                    acc["_weight"] += float(batch_weight)
                    acc["attention_mass"] += float(head_values[head_idx]) * float(batch_weight)


def make_summary_row(
    source_rows: Sequence[Mapping[str, Any]],
    num_qubits: int,
    g_value: float,
    layer: int,
    head: int,
    query_group: str,
    key_type: str,
    delta_site: str,
    num_prompts: int,
    seed: int,
    distance_mode: str,
) -> Dict[str, Any]:
    values = np.asarray([float(row["attention_mass"]) for row in source_rows], dtype=float)
    return {
        "num_qubits": int(num_qubits),
        "g": float(g_value),
        "layer": int(layer),
        "head": int(head),
        "query_group": query_group,
        "key_type": key_type,
        "delta_site": delta_site,
        "attention_mass": float(np.nanmean(values)),
        "num_prompts": int(num_prompts),
        "seed": int(seed),
        "distance_mode": distance_mode,
    }


def finalize_rows(
    accumulators: Mapping[Tuple[int, int, str, str], Mapping[str, float]],
    num_qubits: int,
    g_value: float,
    query_group: str,
    num_prompts: int,
    seed: int,
    distance_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    real_rows: Dict[Tuple[int, int, str, str], Dict[str, Any]] = {}

    for layer_idx, head_idx, key_type, delta_site in sorted(accumulators.keys()):
        acc = accumulators[(layer_idx, head_idx, key_type, delta_site)]
        weight = float(acc.get("_weight", 0.0))
        if weight <= 0:
            continue
        row = {
            "num_qubits": int(num_qubits),
            "g": float(g_value),
            "layer": int(layer_idx),
            "head": int(head_idx),
            "query_group": query_group,
            "key_type": key_type,
            "delta_site": delta_site,
            "attention_mass": float(acc["attention_mass"] / weight),
            "num_prompts": int(num_prompts),
            "seed": int(seed),
            "distance_mode": distance_mode,
        }
        rows.append(row)
        real_rows[(layer_idx, head_idx, key_type, delta_site)] = row

    layer_ids = sorted({layer for layer, _, _, _ in real_rows.keys()})
    key_delta_pairs = sorted({(key_type, delta) for _, _, key_type, delta in real_rows.keys()})
    for layer_idx in layer_ids:
        for key_type, delta_site in key_delta_pairs:
            source = [
                row
                for (layer, _, kt, delta), row in real_rows.items()
                if layer == layer_idx and kt == key_type and delta == delta_site
            ]
            if source:
                rows.append(
                    make_summary_row(
                        source,
                        num_qubits,
                        g_value,
                        layer_idx,
                        -1,
                        query_group,
                        key_type,
                        delta_site,
                        num_prompts,
                        seed,
                        distance_mode,
                    )
                )

    for key_type, delta_site in key_delta_pairs:
        source = [
            row
            for (_, _, kt, delta), row in real_rows.items()
            if kt == key_type and delta == delta_site
        ]
        if source:
            rows.append(
                make_summary_row(
                    source,
                    num_qubits,
                    g_value,
                    -1,
                    -1,
                    query_group,
                    key_type,
                    delta_site,
                    num_prompts,
                    seed,
                    distance_mode,
                )
            )
    return rows


def empty_accumulator() -> MutableMapping[str, float]:
    return defaultdict(float)


def analyze_condition(
    model: Any,
    num_qubits: int,
    g_value: float,
    prompt_cache: base.PromptCache,
    query_groups: Sequence[str],
    key_types: Sequence[str],
    distance_mode: str,
    batch_size: int,
    device: Any,
    seed: int,
) -> List[Dict[str, Any]]:
    group_accumulators = {
        group: defaultdict(empty_accumulator) for group in query_groups
    }
    num_prompts = int(prompt_cache.P.shape[0])

    for start in range(0, num_prompts, batch_size):
        stop = min(start + batch_size, num_prompts)
        prompt, mask_indices = base.build_prompt_batch(
            g_value=g_value,
            P_batch=prompt_cache.P[start:stop],
            b_batch=prompt_cache.b[start:stop],
            mask_batch=prompt_cache.mask[start:stop],
            device=device,
        )
        with torch.no_grad():
            _, attention_maps = model(prompt, mask_indices=mask_indices, return_attn_weights=True)

        for query_group in query_groups:
            update_accumulators(
                accumulators=group_accumulators[query_group],
                attention_maps=attention_maps,
                num_qubits=num_qubits,
                query_group=query_group,
                key_types=key_types,
                distance_mode=distance_mode,
                batch_weight=stop - start,
            )

    rows: List[Dict[str, Any]] = []
    for query_group in query_groups:
        rows.extend(
            finalize_rows(
                group_accumulators[query_group],
                num_qubits,
                g_value,
                query_group,
                num_prompts,
                seed,
                distance_mode,
            )
        )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "num_qubits",
        "g",
        "layer",
        "head",
        "query_group",
        "key_type",
        "delta_site",
        "attention_mass",
        "num_prompts",
        "seed",
        "distance_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_delta(delta_site: Any) -> Optional[int]:
    try:
        return int(delta_site)
    except (TypeError, ValueError):
        return None


def plot_summary_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    query_groups: Sequence[str],
    key_types: Sequence[str],
    dpi: int,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")

    summary = [
        row
        for row in rows
        if int(row["layer"]) == -1 and int(row["head"]) == -1
    ]
    paths: List[Path] = []

    for query_group in query_groups:
        for key_type in key_types:
            selected = [
                row
                for row in summary
                if row["query_group"] == query_group
                and row["key_type"] == key_type
                and parse_delta(row["delta_site"]) is not None
            ]
            if not selected:
                continue
            for num_qubits in sorted({int(row["num_qubits"]) for row in selected}):
                n_rows = [row for row in selected if int(row["num_qubits"]) == num_qubits]
                g_values = sorted({float(row["g"]) for row in n_rows})
                deltas = sorted({parse_delta(row["delta_site"]) for row in n_rows if parse_delta(row["delta_site"]) is not None})
                matrix = np.full((len(deltas), len(g_values)), np.nan, dtype=float)
                lookup = {
                    (parse_delta(row["delta_site"]), float(row["g"])): float(row["attention_mass"])
                    for row in n_rows
                }
                for i, delta in enumerate(deltas):
                    for j, g_value in enumerate(g_values):
                        matrix[i, j] = lookup.get((delta, g_value), np.nan)

                fig, ax = plt.subplots(figsize=(max(5.2, 0.9 * len(g_values) + 2), max(4.0, 0.35 * len(deltas) + 1.8)))
                im = ax.imshow(matrix, aspect="auto", cmap="viridis")
                ax.set_xticks(np.arange(len(g_values)))
                ax.set_xticklabels([f"{g:g}" for g in g_values])
                ax.set_yticks(np.arange(len(deltas)))
                ax.set_yticklabels([str(delta) for delta in deltas])
                ax.set_xlabel("g")
                ax.set_ylabel("site displacement")
                ax.set_title(f"N={num_qubits}, {query_group} queries -> {key_type} keys")
                cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
                cbar.set_label("attention mass")
                fig.tight_layout()
                path = output_dir / f"site_kernel_N{num_qubits}_{query_group}_to_{key_type}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                paths.append(path)
    return paths


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    qubits: Sequence[int],
    g_values: Sequence[float],
    model_metadata: Mapping[str, Any],
    row_count: int,
) -> None:
    payload = {
        "args": vars(args),
        "qubits": list(map(int, qubits)),
        "g_values": list(map(float, g_values)),
        "row_count": int(row_count),
        "model": model_metadata,
        "notes": {
            "attention_mass": (
                "For each query token, attention weights are summed over keys with the "
                "given key_type and site displacement, then averaged over prompts and queries."
            ),
            "delta_site": "g rows use delta_site='g'; site-key rows use integer displacements.",
            "layer=-1, head=-1": "Average over every real layer/head row.",
            "head=-1": "Average over heads within that layer.",
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    qubits = base.parse_int_values(args.qubits)
    g_values = base.parse_float_values(args.g_values)
    if args.num_prompts <= 0:
        raise ValueError("--num_prompts must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.mask_mode == "random" and not (0.0 <= args.mask_prob <= 1.0):
        raise ValueError("--mask_prob must be in [0, 1].")
    if torch is None:
        raise RuntimeError("PyTorch is required for attention kernel analysis.")

    device = base.resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"Using device: {device}")
    model, model_config, model_metadata = base.load_model(args, qubits, device)
    print(f"Model config: {model_config}")

    all_rows: List[Dict[str, Any]] = []
    for num_qubits in qubits:
        cache_seed = int(args.seed + 1009 * num_qubits)
        prompt_cache = base.make_prompt_cache(
            num_qubits=num_qubits,
            num_prompts=args.num_prompts,
            seed=cache_seed,
            mask_mode=args.mask_mode,
            mask_prob=args.mask_prob,
        )
        for g_value in base.tqdm(g_values, desc=f"N={num_qubits}", unit="g"):
            all_rows.extend(
                analyze_condition(
                    model=model,
                    num_qubits=num_qubits,
                    g_value=float(g_value),
                    prompt_cache=prompt_cache,
                    query_groups=args.query_groups,
                    key_types=args.key_types,
                    distance_mode=args.distance_mode,
                    batch_size=args.batch_size,
                    device=device,
                    seed=cache_seed,
                )
            )

    if args.dry_run:
        print(f"Dry run completed. Generated {len(all_rows)} kernel rows.")
        for row in all_rows[: min(8, len(all_rows))]:
            print(row)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attention_site_kernel.csv"
    metadata_path = output_dir / "attention_site_kernel_metadata.json"
    write_csv(all_rows, csv_path)
    write_metadata(metadata_path, args, qubits, g_values, model_metadata, len(all_rows))
    print(f"Wrote kernel CSV: {csv_path}")
    print(f"Wrote metadata: {metadata_path}")

    if not args.no_plots:
        for path in plot_summary_heatmaps(
            all_rows,
            output_dir,
            args.query_groups,
            args.plot_key_types,
            args.dpi,
        ):
            print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
