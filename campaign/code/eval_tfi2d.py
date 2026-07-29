#!/usr/bin/env python3
"""
Cross-scale evaluation of Diffushadow on the 2D TFI torus.

Loads an oseq_rope checkpoint trained on small 2D lattices, generates
classical-shadow snapshots at a (possibly larger, unseen) lattice size,
estimates site-averaged ZZ correlations for displacements (0,1), (1,0),
(0,2), and compares against the exact cache written by
generate_tfi2d_dataset.py.

Estimator conventions match eval_utils.py: local factor 3 per matched
Pauli-Z (factor 9 for pairs), site-averaged, median-of-means with K
parts.  Generation reuses the entropy-guided iterative denoiser
(generate_oseq_batch) verbatim from test_1.py.

Usage:
  python tfi2d/eval_tfi2d.py \
      --model_path checkpoints2d/tfi2d_oseq_rope.pth \
      --lx 4 --ly 6 \
      --exact_npz data2d/tfi2d_4x6_exact.npz \
      --sample_size 20000 --diffusion_steps 4 \
      --out_npz campaign/tfi2d_eval_4x6.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope  # noqa: E402

MASK_TOKEN_ID = -1.0


# --- copied verbatim from test_1.py (generate_oseq_batch) so that the 2D
# --- evaluation uses the exact generation procedure of the 1D experiments.
def generate_oseq_batch(model, prompt, steps, mask_id=MASK_TOKEN_ID,
                        temperature=1.0, use_sampling=False):
    model.eval()
    prompt = prompt.clone()
    B, L, _ = prompt.shape
    b_positions = torch.arange(2, L, 2, device=prompt.device)
    N = b_positions.numel()

    is_masked = (prompt.squeeze(-1) == mask_id)
    total_masked = is_masked[:, b_positions].sum(dim=1)
    num_unmask_per_step = torch.clamp(total_masked.max() // max(1, steps), min=1).item()

    for _ in range(steps):
        is_still_masked = (prompt.squeeze(-1) == mask_id)
        current_masked_b = is_still_masked[:, b_positions]
        if not current_masked_b.any():
            break
        with torch.no_grad():
            logits = model(prompt, mask_indices=current_masked_b)
        probs = torch.softmax(logits / temperature, dim=-1)
        p_b1 = probs[..., 1]
        eps = 1e-12
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
        entropy_masked = entropy.masked_fill(~current_masked_b, float("inf"))
        k = min(num_unmask_per_step, N)
        idx = torch.topk(entropy_masked, k=k, largest=False, dim=1).indices
        gathered_entropy = entropy_masked.gather(1, idx)
        valid = torch.isfinite(gathered_entropy)
        if not valid.any():
            break
        p_selected = p_b1.gather(1, idx)
        if use_sampling:
            pred_values = torch.bernoulli(p_selected).float()
        else:
            pred_values = (p_selected > 0.5).float()
        global_pos = 2 + 2 * idx
        b_idx = torch.arange(B, device=prompt.device).unsqueeze(1).expand_as(global_pos)
        b_idx = b_idx[valid]
        g_idx = global_pos[valid]
        v = pred_values[valid]
        prompt[b_idx, g_idx, 0] = v
    return prompt
# --- end copy


def torus_pairs(lx: int, ly: int):
    h1, v1, h2 = [], [], []
    for r in range(ly):
        for c in range(lx):
            s = r * lx + c
            h1.append((s, r * lx + (c + 1) % lx))
            v1.append((s, ((r + 1) % ly) * lx + c))
            h2.append((s, r * lx + (c + 2) % lx))
    return h1, v1, h2


def shadow_zz(P: np.ndarray, b: np.ndarray, pairs) -> np.ndarray:
    """Per-snapshot site-averaged shadow estimate of <Z_i Z_j>.
    P: [M,N] tokens 2/3/4;  b: [M,N] bits.  Returns [M]."""
    zhat = 3.0 * (2.0 * b - 1.0) * (P == 4)
    acc = np.zeros(P.shape[0])
    for i, j in pairs:
        acc += zhat[:, i] * zhat[:, j]
    return acc / len(pairs)


def median_of_means(per_snapshot: np.ndarray, k: int = 10) -> float:
    parts = np.array_split(per_snapshot, k)
    return float(np.median([p.mean() for p in parts]))


def load_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    raise ValueError(f"unrecognized checkpoint format: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--lx", type=int, default=4)
    ap.add_argument("--ly", type=int, required=True)
    ap.add_argument("--exact_npz", type=str, default="",
                    help="exact/DMRG reference npz; optional — beyond-ED sizes "
                         "may pass --gs instead")
    ap.add_argument("--gs", type=float, nargs="*", default=None,
                    help="g values to generate at (required if no --exact_npz)")
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--sample_size", type=int, default=20000)
    ap.add_argument("--diffusion_steps", type=int, default=4)
    ap.add_argument("--gen_batch_size", type=int, default=1024)
    ap.add_argument("--mom_parts", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--layer_num", type=int, default=4)
    ap.add_argument("--head_num", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump_json", type=str, default="",
                    help="also dump generated snapshots as training-format rows "
                         "[g,P1,b1,...] (for recursive scale extrapolation)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    n = args.lx * args.ly
    L = 1 + 2 * n
    if args.exact_npz:
        exact = np.load(args.exact_npz)
        gs = np.asarray(exact["gs"], dtype=float)
    else:
        if not args.gs:
            raise SystemExit("need --exact_npz or --gs")
        exact = None
        gs = np.asarray(args.gs, dtype=float)
    print(f"eval 2D TFI {args.lx}x{args.ly} (N={n}) on {device}; "
          f"{len(gs)} g-values x {args.sample_size} snapshots", flush=True)

    model = QuantumDiffusionModel_oseq_rope(
        hidden_dim=args.hidden_dim, num_layers=args.layer_num,
        head_count=args.head_num, max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
    ).to(device)
    model.load_state_dict(load_state_dict(args.model_path))
    model.eval()

    h1, v1, h2 = torus_pairs(args.lx, args.ly)
    est = {"h1": [], "v1": [], "h2": []}
    r_positions = torch.arange(1, L, 2, device=device)
    dump_rows = []

    for g in gs:
        P_all, b_all = [], []
        remaining = args.sample_size
        while remaining > 0:
            B = min(args.gen_batch_size, remaining)
            prompt = torch.full((B, L, 1), MASK_TOKEN_ID, device=device)
            prompt[:, 0, 0] = float(g)
            rand_P = torch.randint(2, 5, (B, n), device=device).float()
            prompt[:, r_positions, 0] = rand_P
            completed = generate_oseq_batch(
                model=model, prompt=prompt, steps=args.diffusion_steps,
                temperature=1.0, use_sampling=True)
            b_vals = (completed[:, torch.arange(2, L, 2, device=device), 0] > 0.5).float()
            P_all.append(rand_P.cpu().numpy())
            b_all.append(b_vals.cpu().numpy())
            remaining -= B
        P = np.concatenate(P_all).astype(np.int8)
        b = np.concatenate(b_all).astype(np.int8)
        if args.dump_json:
            for t in range(P.shape[0]):
                row = [float(g)]
                for i in range(n):
                    row += [int(P[t, i]), int(b[t, i])]
                dump_rows.append(row)

        for key, pairs in (("h1", h1), ("v1", v1), ("h2", h2)):
            est[key].append(median_of_means(shadow_zz(P, b, pairs), args.mom_parts))
        if exact is not None:
            print(f"  g={g:.3f}  gen h1={est['h1'][-1]:+.4f} (exact {exact['corr_h1'][len(est['h1'])-1]:+.4f})"
                  f"  v1={est['v1'][-1]:+.4f} (exact {exact['corr_v1'][len(est['v1'])-1]:+.4f})"
                  f"  h2={est['h2'][-1]:+.4f} (exact {exact['corr_h2'][len(est['h2'])-1]:+.4f})",
                  flush=True)
        else:
            print(f"  g={g:.3f}  gen h1={est['h1'][-1]:+.4f}  v1={est['v1'][-1]:+.4f}"
                  f"  h2={est['h2'][-1]:+.4f}  (no reference)", flush=True)

    gen_h1, gen_v1, gen_h2 = (np.array(est[k]) for k in ("h1", "v1", "h2"))
    save = dict(gs=gs, gen_h1=gen_h1, gen_v1=gen_v1, gen_h2=gen_h2,
                lx=args.lx, ly=args.ly, num_qubits=n,
                sample_size=args.sample_size, diffusion_steps=args.diffusion_steps)
    if exact is not None:
        mse = {k: float(np.mean((np.array(est[k]) - np.asarray(exact[f"corr_{k}"])) ** 2))
               for k in ("h1", "v1", "h2")}
        print(f"MSE  h1={mse['h1']:.3e}  v1={mse['v1']:.3e}  h2={mse['h2']:.3e}")
        save.update(exact_h1=exact["corr_h1"], exact_v1=exact["corr_v1"],
                    exact_h2=exact["corr_h2"],
                    mse_h1=mse["h1"], mse_v1=mse["v1"], mse_h2=mse["h2"])
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **save)
    print(f"saved: {args.out_npz}")
    if args.dump_json:
        import json
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dump_rows, open(args.dump_json, "w"))
        print(f"dumped {len(dump_rows)} generated snapshots -> {args.dump_json}")


if __name__ == "__main__":
    main()
