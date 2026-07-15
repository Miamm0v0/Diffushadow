from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # Keep --help usable outside the project runtime environment.
    torch = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Allows --help, --dry_run, and --no_plots without matplotlib.
    plt = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is listed in requirements, but keep CLI robust.
    def tqdm(iterable, **kwargs):
        return iterable


MASK_TOKEN_ID = -1.0
METRIC_NAMES = [
    "mean_token_distance",
    "mean_token_distance_norm",
    "mean_site_distance",
    "mean_site_distance_norm",
    "attention_to_g",
    "attention_to_P",
    "attention_to_b",
    "attention_to_same_site",
    "attention_to_near_site_1",
    "entropy",
    "entropy_norm",
]


@dataclass
class ModelConfig:
    hidden_dim: int
    num_layers: int
    head_count: int
    max_seq_len: int
    max_N: int
    rope_scaling_type: Optional[str]
    rope_scaling_factor: float
    rope_theta: float


@dataclass
class PromptCache:
    P: np.ndarray
    b: np.ndarray
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    repo_dshadow = Path(__file__).resolve().parents[1]
    default_output = repo_dshadow / "attention_analysis" / "results" / "oseq_rope_attention_distance"

    parser = argparse.ArgumentParser(
        description=(
            "Analyze attention distance of Qdmodel_oseq_rope.py across qubit sizes "
            "and g values."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to a trained Qdmodel_oseq_rope checkpoint. Empty means random weights.",
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        default=["6", "8", "10", "12", "14"],
        help="Qubit sizes. Accepts spaces or commas, for example: --qubits 6 8 10.",
    )
    parser.add_argument(
        "--g_values",
        nargs="+",
        default=["0.0", "0.2", "0.5", "0.7", "1.0"],
        help="g values. Accepts spaces or commas, for example: --g_values 0,0.2,0.5.",
    )
    parser.add_argument(
        "--query_groups",
        nargs="+",
        default=["b"],
        choices=["b", "P", "all", "non_g"],
        help="Query token groups used when computing attention distance.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=64,
        help="Number of random P sequences per (N, g). The same P set is reused across g.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Analysis batch size.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for prompt generation.")
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="all",
        choices=["all", "random", "none"],
        help="How to mask b tokens before extracting attention.",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.5,
        help="Mask probability when --mask_mode random.",
    )
    parser.add_argument(
        "--pbc_distance",
        action="store_true",
        help="Use periodic site distance min(|i-j|, N-|i-j|) for site-distance metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Model hidden dimension. If omitted, infer from checkpoint when possible.",
    )
    parser.add_argument(
        "--layer_num",
        type=int,
        default=None,
        help="Number of transformer layers. If omitted, infer from checkpoint when possible.",
    )
    parser.add_argument(
        "--head_num",
        type=int,
        default=None,
        help=(
            "Number of attention heads. If absent, use checkpoint args if available, "
            "otherwise train_oseq.py's oseq_rope default of 8."
        ),
    )
    parser.add_argument("--max_seq_len", type=int, default=None, help="RoPE max sequence length.")
    parser.add_argument("--max_N", type=int, default=None, help="Maximum qubit size for model init.")
    parser.add_argument(
        "--rope_scaling_type",
        type=str,
        default=None,
        choices=["none", "linear", "dynamic", "ntk"],
        help="RoPE scaling type. If omitted, use checkpoint args or none.",
    )
    parser.add_argument(
        "--rope_scaling_factor",
        type=float,
        default=None,
        help="RoPE scaling factor. If omitted, use checkpoint args or 1.0.",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=None,
        help="RoPE theta. If omitted, use checkpoint args or 10000.0.",
    )
    parser.add_argument(
        "--checkpoint_non_strict",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output),
        help="Directory for CSV, metadata JSON, and figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--plot_metric",
        type=str,
        default="mean_site_distance",
        choices=METRIC_NAMES,
        help="Metric used in the main distance plots.",
    )
    parser.add_argument("--no_plots", action="store_true", help="Only write CSV and metadata.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run analysis and print a preview without writing files.",
    )
    return parser.parse_args()


def parse_int_values(values: Sequence[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            parsed.append(int(part))
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def parse_float_values(values: Sequence[str]) -> List[float]:
    parsed: List[float] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            parsed.append(float(part))
    if not parsed:
        raise ValueError("Expected at least one float value.")
    return parsed


def resolve_device(raw_device: str) -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is required for attention analysis. Activate the project environment first.")
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def safe_torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def extract_model_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if looks_like_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, torch.nn.Module):
                return value.state_dict()
            if looks_like_state_dict(value):
                return value
    raise ValueError("Checkpoint does not contain a recognizable model state_dict.")


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return dict(state_dict)
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return dict(state_dict)


def extract_checkpoint_args(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    for key in ("args", "config", "model_args"):
        value = checkpoint.get(key)
        if isinstance(value, argparse.Namespace):
            return vars(value)
        if isinstance(value, dict):
            return value
    return {}


def infer_hidden_dim(state_dict: Optional[Mapping[str, torch.Tensor]]) -> Optional[int]:
    if not state_dict:
        return None
    weight = state_dict.get("g_proj.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    weight = state_dict.get("output_layer.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    return None


def infer_num_layers(state_dict: Optional[Mapping[str, torch.Tensor]]) -> Optional[int]:
    if not state_dict:
        return None
    layer_ids = []
    pattern = re.compile(r"^layers\.(\d+)\.")
    for key in state_dict.keys():
        match = pattern.match(str(key))
        if match:
            layer_ids.append(int(match.group(1)))
    if not layer_ids:
        return None
    return max(layer_ids) + 1


def normalize_rope_scaling_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ("", "none", "off", "false"):
        return None
    if normalized in ("ntk", "dynamic_ntk"):
        return "dynamic"
    return normalized


def first_present(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def resolve_model_config(
    args: argparse.Namespace,
    qubits: Sequence[int],
    state_dict: Optional[Mapping[str, torch.Tensor]],
    checkpoint_args: Mapping[str, Any],
) -> ModelConfig:
    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        hidden_dim = infer_hidden_dim(state_dict)
    if hidden_dim is None:
        hidden_dim = first_present(checkpoint_args, ("hidden_dim",))
    if hidden_dim is None:
        hidden_dim = 128

    num_layers = args.layer_num
    if num_layers is None:
        num_layers = infer_num_layers(state_dict)
    if num_layers is None:
        num_layers = first_present(checkpoint_args, ("layer_num", "num_layers"))
    if num_layers is None:
        num_layers = 3

    head_count = args.head_num
    if head_count is None:
        head_count = first_present(checkpoint_args, ("head_num", "head_count"))
    if head_count is None:
        head_count = 8

    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        max_seq_len = first_present(checkpoint_args, ("max_seq_len",))
    if max_seq_len is None:
        max_seq_len = max(4096, 1 + 2 * max(qubits))

    max_N = args.max_N
    if max_N is None:
        max_N = first_present(checkpoint_args, ("max_N",))
    if max_N is None:
        max_N = max(qubits)

    rope_scaling_type = args.rope_scaling_type
    if rope_scaling_type is None:
        rope_scaling_type = first_present(checkpoint_args, ("rope_scaling_type",))
    rope_scaling_type = normalize_rope_scaling_type(rope_scaling_type)

    rope_scaling_factor = args.rope_scaling_factor
    if rope_scaling_factor is None:
        rope_scaling_factor = first_present(checkpoint_args, ("rope_scaling_factor",))
    if rope_scaling_factor is None:
        rope_scaling_factor = 1.0

    rope_theta = args.rope_theta
    if rope_theta is None:
        rope_theta = first_present(checkpoint_args, ("rope_theta",))
    if rope_theta is None:
        rope_theta = 10000.0

    hidden_dim = int(hidden_dim)
    head_count = int(head_count)
    if hidden_dim % head_count != 0:
        raise ValueError(
            f"hidden_dim ({hidden_dim}) must be divisible by head_num ({head_count})."
        )

    return ModelConfig(
        hidden_dim=hidden_dim,
        num_layers=int(num_layers),
        head_count=head_count,
        max_seq_len=int(max_seq_len),
        max_N=int(max_N),
        rope_scaling_type=rope_scaling_type,
        rope_scaling_factor=float(rope_scaling_factor),
        rope_theta=float(rope_theta),
    )


def build_model(config: ModelConfig) -> torch.nn.Module:
    dshadow_dir = Path(__file__).resolve().parents[1]
    if str(dshadow_dir) not in sys.path:
        sys.path.insert(0, str(dshadow_dir))

    from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope

    return QuantumDiffusionModel_oseq_rope(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        head_count=config.head_count,
        max_seq_len=config.max_seq_len,
        max_N=config.max_N,
        rope_scaling_type=config.rope_scaling_type,
        rope_scaling_factor=config.rope_scaling_factor,
        rope_theta=config.rope_theta,
    )


def load_model(
    args: argparse.Namespace,
    qubits: Sequence[int],
    device: torch.device,
) -> Tuple[torch.nn.Module, ModelConfig, Dict[str, Any]]:
    checkpoint: Any = None
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    checkpoint_args: Dict[str, Any] = {}

    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        checkpoint = safe_torch_load(model_path, map_location="cpu")
        state_dict = strip_module_prefix(extract_model_state_dict(checkpoint))
        checkpoint_args = extract_checkpoint_args(checkpoint)

    config = resolve_model_config(args, qubits, state_dict, checkpoint_args)
    if args.head_num is None and first_present(checkpoint_args, ("head_num", "head_count")) is None:
        print(
            "No head count was found in CLI args or checkpoint metadata; defaulting to "
            "head_num=8. Pass --head_num explicitly if the checkpoint used a different value."
        )
    model = build_model(config)

    if state_dict is not None:
        incompatible = model.load_state_dict(state_dict, strict=not args.checkpoint_non_strict)
        if args.checkpoint_non_strict:
            if incompatible.missing_keys:
                print(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"Unexpected keys: {incompatible.unexpected_keys}")
        print(f"Loaded checkpoint: {args.model_path}")
    else:
        print("No --model_path was supplied; analyzing a randomly initialized model.")

    model.to(device)
    model.eval()
    metadata = {
        "model_config": config.__dict__,
        "checkpoint_args": checkpoint_args,
    }
    return model, config, metadata


def make_prompt_cache(
    num_qubits: int,
    num_prompts: int,
    seed: int,
    mask_mode: str,
    mask_prob: float,
) -> PromptCache:
    rng = np.random.default_rng(seed)
    P = rng.integers(2, 5, size=(num_prompts, num_qubits), dtype=np.int64)
    b = rng.integers(0, 2, size=(num_prompts, num_qubits), dtype=np.int64)

    if mask_mode == "all":
        mask = np.ones((num_prompts, num_qubits), dtype=bool)
    elif mask_mode == "none":
        mask = np.zeros((num_prompts, num_qubits), dtype=bool)
    elif mask_mode == "random":
        mask = rng.random((num_prompts, num_qubits)) < float(mask_prob)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    return PromptCache(P=P, b=b, mask=mask)


def build_prompt_batch(
    g_value: float,
    P_batch: np.ndarray,
    b_batch: np.ndarray,
    mask_batch: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_qubits = P_batch.shape
    seq_len = 1 + 2 * num_qubits
    prompt = torch.full((batch_size, seq_len, 1), MASK_TOKEN_ID, dtype=torch.float32, device=device)
    prompt[:, 0, 0] = float(g_value)

    P_tensor = torch.as_tensor(P_batch, dtype=torch.float32, device=device)
    b_tensor = torch.as_tensor(b_batch, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)

    prompt[:, 1:seq_len:2, 0] = P_tensor
    prompt[:, 2:seq_len:2, 0] = torch.where(
        mask_tensor,
        torch.full_like(b_tensor, MASK_TOKEN_ID),
        b_tensor,
    )
    return prompt, mask_tensor


def query_positions(num_qubits: int, query_group: str) -> torch.Tensor:
    seq_len = 1 + 2 * num_qubits
    if query_group == "b":
        return torch.arange(2, seq_len, 2, dtype=torch.long)
    if query_group == "P":
        return torch.arange(1, seq_len, 2, dtype=torch.long)
    if query_group == "non_g":
        return torch.arange(1, seq_len, dtype=torch.long)
    if query_group == "all":
        return torch.arange(0, seq_len, dtype=torch.long)
    raise ValueError(f"Unknown query_group: {query_group}")


def site_index_for_sequence(num_qubits: int) -> torch.Tensor:
    seq_len = 1 + 2 * num_qubits
    site_index = torch.full((seq_len,), -1, dtype=torch.long)
    site_index[1:seq_len:2] = torch.arange(num_qubits, dtype=torch.long)
    site_index[2:seq_len:2] = torch.arange(num_qubits, dtype=torch.long)
    return site_index


def compute_attention_metrics(
    attention: torch.Tensor,
    num_qubits: int,
    query_group: str,
    pbc_distance: bool,
) -> Dict[str, np.ndarray]:
    # attention: [B, H, L, L], already detached on CPU by Qdmodel_oseq_rope.
    attention = attention.to(dtype=torch.float64, device="cpu")
    batch_size, num_heads, seq_len, _ = attention.shape
    q_pos = query_positions(num_qubits, query_group)
    key_pos = torch.arange(seq_len, dtype=torch.long)

    attn_q = attention[:, :, q_pos, :]
    token_dist = (q_pos[:, None] - key_pos[None, :]).abs().to(torch.float64)
    mean_token_distance = (attn_q * token_dist[None, None, :, :]).sum(dim=-1).mean(dim=(0, 2))

    eps = torch.finfo(torch.float64).eps
    entropy = -(attn_q.clamp_min(eps) * attn_q.clamp_min(eps).log()).sum(dim=-1).mean(dim=(0, 2))
    entropy_norm = entropy / math.log(max(seq_len, 2))

    P_pos = torch.arange(1, seq_len, 2, dtype=torch.long)
    b_pos = torch.arange(2, seq_len, 2, dtype=torch.long)
    attention_to_g = attn_q[..., 0].mean(dim=(0, 2))
    attention_to_P = attn_q.index_select(dim=-1, index=P_pos).sum(dim=-1).mean(dim=(0, 2))
    attention_to_b = attn_q.index_select(dim=-1, index=b_pos).sum(dim=-1).mean(dim=(0, 2))

    site_index = site_index_for_sequence(num_qubits)
    q_sites = site_index[q_pos]
    valid_query = q_sites >= 0
    valid_key = site_index >= 0

    mean_site_distance = torch.full((num_heads,), float("nan"), dtype=torch.float64)
    mean_site_distance_norm = torch.full((num_heads,), float("nan"), dtype=torch.float64)
    attention_to_same_site = torch.full((num_heads,), float("nan"), dtype=torch.float64)
    attention_to_near_site_1 = torch.full((num_heads,), float("nan"), dtype=torch.float64)

    if bool(valid_query.any()):
        valid_q_pos = q_pos[valid_query]
        valid_q_sites = q_sites[valid_query]
        attn_site_q = attention[:, :, valid_q_pos, :]

        raw_diff = (valid_q_sites[:, None] - site_index[None, :]).abs()
        raw_diff = raw_diff.to(torch.float64)
        if pbc_distance:
            raw_diff = torch.minimum(raw_diff, num_qubits - raw_diff)

        site_dist = torch.zeros((valid_q_pos.numel(), seq_len), dtype=torch.float64)
        site_dist[:, valid_key] = raw_diff[:, valid_key]
        non_g_mass = attn_site_q[..., valid_key].sum(dim=-1).clamp_min(eps)
        site_distance_per_query = (
            (attn_site_q * site_dist[None, None, :, :]).sum(dim=-1) / non_g_mass
        )
        mean_site_distance = site_distance_per_query.mean(dim=(0, 2))

        denom = (num_qubits // 2) if pbc_distance else (num_qubits - 1)
        denom = max(int(denom), 1)
        mean_site_distance_norm = mean_site_distance / float(denom)

        same_site = (valid_q_sites[:, None] == site_index[None, :]) & valid_key[None, :]
        near_site = (raw_diff <= 1.0) & valid_key[None, :]
        attention_to_same_site = (
            attn_site_q * same_site.to(torch.float64)[None, None, :, :]
        ).sum(dim=-1).mean(dim=(0, 2))
        attention_to_near_site_1 = (
            attn_site_q * near_site.to(torch.float64)[None, None, :, :]
        ).sum(dim=-1).mean(dim=(0, 2))

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


def empty_metric_accumulator() -> MutableMapping[str, float]:
    return defaultdict(float)


def update_accumulators(
    accumulators: MutableMapping[Tuple[int, int], MutableMapping[str, float]],
    attention_maps: Sequence[torch.Tensor],
    num_qubits: int,
    query_group: str,
    pbc_distance: bool,
    batch_weight: int,
) -> None:
    for layer_idx, attention in enumerate(attention_maps):
        metrics = compute_attention_metrics(
            attention=attention,
            num_qubits=num_qubits,
            query_group=query_group,
            pbc_distance=pbc_distance,
        )
        num_heads = attention.shape[1]
        for head_idx in range(num_heads):
            acc = accumulators[(layer_idx, head_idx)]
            acc["_weight"] += float(batch_weight)
            for metric_name, values in metrics.items():
                value = float(values[head_idx])
                if math.isfinite(value):
                    acc[metric_name] += value * float(batch_weight)


def finalize_rows_for_condition(
    accumulators: Mapping[Tuple[int, int], Mapping[str, float]],
    num_qubits: int,
    g_value: float,
    query_group: str,
    num_prompts: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    per_head_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for layer_idx, head_idx in sorted(accumulators.keys()):
        acc = accumulators[(layer_idx, head_idx)]
        weight = float(acc.get("_weight", 0.0))
        if weight <= 0:
            continue
        row: Dict[str, Any] = {
            "num_qubits": int(num_qubits),
            "g": float(g_value),
            "layer": int(layer_idx),
            "head": int(head_idx),
            "query_group": query_group,
            "num_prompts": int(num_prompts),
            "seed": int(seed),
        }
        for metric_name in METRIC_NAMES:
            row[metric_name] = float(acc.get(metric_name, float("nan")) / weight)
        per_head_rows[(layer_idx, head_idx)] = row
        rows.append(row)

    layer_ids = sorted({layer for layer, _ in per_head_rows.keys()})
    for layer_idx in layer_ids:
        layer_rows = [row for (layer, _), row in per_head_rows.items() if layer == layer_idx]
        rows.append(make_summary_row(layer_rows, num_qubits, g_value, layer_idx, -1, query_group, num_prompts, seed))

    all_head_rows = list(per_head_rows.values())
    if all_head_rows:
        rows.append(make_summary_row(all_head_rows, num_qubits, g_value, -1, -1, query_group, num_prompts, seed))

    return rows


def make_summary_row(
    source_rows: Sequence[Mapping[str, Any]],
    num_qubits: int,
    g_value: float,
    layer: int,
    head: int,
    query_group: str,
    num_prompts: int,
    seed: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "num_qubits": int(num_qubits),
        "g": float(g_value),
        "layer": int(layer),
        "head": int(head),
        "query_group": query_group,
        "num_prompts": int(num_prompts),
        "seed": int(seed),
    }
    for metric_name in METRIC_NAMES:
        values = np.asarray([float(source[metric_name]) for source in source_rows], dtype=float)
        row[metric_name] = float(np.nanmean(values))
    return row


def analyze_condition(
    model: torch.nn.Module,
    num_qubits: int,
    g_value: float,
    prompt_cache: PromptCache,
    query_groups: Sequence[str],
    batch_size: int,
    device: torch.device,
    pbc_distance: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    group_accumulators: Dict[str, MutableMapping[Tuple[int, int], MutableMapping[str, float]]] = {
        group: defaultdict(empty_metric_accumulator) for group in query_groups
    }

    num_prompts = int(prompt_cache.P.shape[0])
    for start in range(0, num_prompts, batch_size):
        stop = min(start + batch_size, num_prompts)
        prompt, mask_indices = build_prompt_batch(
            g_value=g_value,
            P_batch=prompt_cache.P[start:stop],
            b_batch=prompt_cache.b[start:stop],
            mask_batch=prompt_cache.mask[start:stop],
            device=device,
        )
        with torch.no_grad():
            _, attention_maps = model(prompt, mask_indices=mask_indices, return_attn_weights=True)
        batch_weight = stop - start
        for query_group in query_groups:
            update_accumulators(
                accumulators=group_accumulators[query_group],
                attention_maps=attention_maps,
                num_qubits=num_qubits,
                query_group=query_group,
                pbc_distance=pbc_distance,
                batch_weight=batch_weight,
            )

    rows: List[Dict[str, Any]] = []
    for query_group in query_groups:
        rows.extend(
            finalize_rows_for_condition(
                accumulators=group_accumulators[query_group],
                num_qubits=num_qubits,
                g_value=g_value,
                query_group=query_group,
                num_prompts=num_prompts,
                seed=seed,
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
        "num_prompts",
        "seed",
    ] + METRIC_NAMES
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sorted_unique(rows: Sequence[Mapping[str, Any]], key: str) -> List[Any]:
    return sorted({row[key] for row in rows})


def summary_rows(rows: Sequence[Mapping[str, Any]], query_group: str) -> List[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["query_group"] == query_group and int(row["layer"]) == -1 and int(row["head"]) == -1
    ]


def plot_metric_vs_g(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    query_group: str,
    metric_name: str,
    dpi: int,
) -> Optional[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")
    selected = summary_rows(rows, query_group)
    if not selected:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for num_qubits in sorted_unique(selected, "num_qubits"):
        print(f"Plotting {metric_name} vs g for N={num_qubits} ({query_group} queries)...")
        series = sorted(
            [row for row in selected if row["num_qubits"] == num_qubits],
            key=lambda row: float(row["g"]),
        )
        ax.plot(
            [float(row["g"]) for row in series],
            [float(row[metric_name] / num_qubits) for row in series],
            marker="o",
            linewidth=1.8,
            label=f"N={num_qubits}",
        )
    ax.set_xlabel("g")
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.set_title(f"{metric_name.replace('_', ' ')} vs g ({query_group} queries)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.2, 1), loc='upper right')
    fig.tight_layout()
    path = output_dir / f"{metric_name}_vs_g_{query_group}_queries.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attention_masses(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    query_group: str,
    dpi: int,
) -> Optional[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")
    selected = summary_rows(rows, query_group)
    if not selected:
        return None

    mass_metrics = ["attention_to_g", "attention_to_P", "attention_to_b"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, metric_name in zip(axes, mass_metrics):
        for num_qubits in sorted_unique(selected, "num_qubits"):
            series = sorted(
                [row for row in selected if row["num_qubits"] == num_qubits],
                key=lambda row: float(row["g"]),
            )
            ax.plot(
                [float(row["g"]) for row in series],
                [float(row[metric_name]) for row in series],
                marker="o",
                linewidth=1.5,
                label=f"N={num_qubits}",
            )
        ax.set_title(metric_name.replace("attention_to_", "to "))
        ax.set_xlabel("g")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("attention mass")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Attention mass by key type ({query_group} queries)", y=1.04)
    fig.tight_layout()
    path = output_dir / f"attention_mass_vs_g_{query_group}_queries.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_layer_head_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    query_group: str,
    metric_name: str,
    dpi: int,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it or pass --no_plots.")
    real_head_rows = [
        row
        for row in rows
        if row["query_group"] == query_group and int(row["layer"]) >= 0 and int(row["head"]) >= 0
    ]
    paths: List[Path] = []
    if not real_head_rows:
        return paths

    for num_qubits in sorted_unique(real_head_rows, "num_qubits"):
        n_rows = [row for row in real_head_rows if row["num_qubits"] == num_qubits]
        g_values = sorted_unique(n_rows, "g")
        layer_head = sorted({(int(row["layer"]), int(row["head"])) for row in n_rows})
        matrix = np.full((len(layer_head), len(g_values)), np.nan, dtype=float)
        lookup = {
            (int(row["layer"]), int(row["head"]), float(row["g"])): float(row[metric_name])
            for row in n_rows
        }
        for i, (layer_idx, head_idx) in enumerate(layer_head):
            for j, g_value in enumerate(g_values):
                matrix[i, j] = lookup.get((layer_idx, head_idx, float(g_value)), np.nan)

        fig_height = max(4.0, 0.32 * len(layer_head) + 1.8)
        fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(g_values) + 2.5), fig_height))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(g_values)))
        ax.set_xticklabels([f"{g:g}" for g in g_values])
        ax.set_yticks(np.arange(len(layer_head)))
        ax.set_yticklabels([f"L{layer}H{head}" for layer, head in layer_head], fontsize=8)
        ax.set_xlabel("g")
        ax.set_ylabel("layer/head")
        ax.set_title(f"N={num_qubits}, {metric_name.replace('_', ' ')} ({query_group} queries)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label(metric_name.replace("_", " "))
        fig.tight_layout()
        path = output_dir / f"{metric_name}_layer_head_heatmap_N{num_qubits}_{query_group}_queries.png"
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
        "metric_notes": {
            "mean_token_distance": "sum_j attention(query,key_j) * abs(token_i-token_j)",
            "mean_site_distance": (
                "Attention-weighted qubit-site distance over P/b keys only; attention to g is "
                "excluded and the remaining mass is renormalized per query."
            ),
            "layer=-1, head=-1": "Average over every real layer/head row.",
            "head=-1": "Average over heads within that layer.",
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    qubits = parse_int_values(args.qubits)
    g_values = parse_float_values(args.g_values)
    if args.num_prompts <= 0:
        raise ValueError("--num_prompts must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.mask_mode == "random" and not (0.0 <= args.mask_prob <= 1.0):
        raise ValueError("--mask_prob must be in [0, 1].")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"Using device: {device}")
    model, config, model_metadata = load_model(args, qubits, device)
    print(f"Model config: {config}")

    all_rows: List[Dict[str, Any]] = []
    for num_qubits in qubits:
        cache_seed = int(args.seed + 1009 * num_qubits)
        prompt_cache = make_prompt_cache(
            num_qubits=num_qubits,
            num_prompts=args.num_prompts,
            seed=cache_seed,
            mask_mode=args.mask_mode,
            mask_prob=args.mask_prob,
        )
        progress = tqdm(g_values, desc=f"N={num_qubits}", unit="g")
        for g_value in progress:
            rows = analyze_condition(
                model=model,
                num_qubits=num_qubits,
                g_value=float(g_value),
                prompt_cache=prompt_cache,
                query_groups=args.query_groups,
                batch_size=args.batch_size,
                device=device,
                pbc_distance=args.pbc_distance,
                seed=cache_seed,
            )
            all_rows.extend(rows)

    if args.dry_run:
        print(f"Dry run completed. Generated {len(all_rows)} metric rows.")
        for row in all_rows[: min(5, len(all_rows))]:
            preview = {key: row[key] for key in ("num_qubits", "g", "layer", "head", "query_group")}
            preview[args.plot_metric] = row[args.plot_metric]
            print(preview)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "attention_distance_metrics.csv"
    metadata_path = output_dir / "attention_distance_metadata.json"
    write_csv(all_rows, csv_path)
    write_metadata(metadata_path, args, qubits, g_values, model_metadata, len(all_rows))
    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote metadata: {metadata_path}")

    if not args.no_plots:
        written: List[Path] = []
        for query_group in args.query_groups:
            path = plot_metric_vs_g(
                rows=all_rows,
                output_dir=output_dir,
                query_group=query_group,
                metric_name=args.plot_metric,
                dpi=args.dpi,
            )
            if path is not None:
                written.append(path)
            path = plot_attention_masses(
                rows=all_rows,
                output_dir=output_dir,
                query_group=query_group,
                dpi=args.dpi,
            )
            if path is not None:
                written.append(path)
            written.extend(
                plot_layer_head_heatmaps(
                    rows=all_rows,
                    output_dir=output_dir,
                    query_group=query_group,
                    metric_name=args.plot_metric,
                    dpi=args.dpi,
                )
            )
        for path in written:
            print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
