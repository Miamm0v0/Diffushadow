#!/usr/bin/env python3
"""Avenue #3: evaluate a J1-J2-cylinder-trained checkpoint at width W.
Generates Pauli-6 shadows, estimates site-averaged spin-dot correlations on
ring / axial / diagonal bonds (factor 9 per matched pair, median-of-means),
compares against the DMRG reference npz from j1j2_2d_dataset.py.

Usage:
  python eval_j1j2_2d.py --model_path ckpt.pth --W 5 --L 8 \
      --ref_npz data_j2d/j1j2_5x8_ref.npz --out_npz results_j2d/eval_5x8.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope  # noqa: E402
from eval_tfi2d import generate_oseq_batch, load_state_dict, MASK_TOKEN_ID, \
    median_of_means  # noqa: E402
from j1j2_2d_dataset import bonds  # noqa: E402


def shadow_spin_dot(P, b, pairs):
    """Per-snapshot site-averaged shadow estimate of <sigma_i.sigma_j>:
    sum_alpha 9 * s_i s_j * 1[P_i = P_j = alpha]."""
    s = 2.0 * b - 1.0
    acc = np.zeros(P.shape[0])
    for i, j in pairs:
        match = (P[:, i] == P[:, j]).astype(np.float64)
        acc += 9.0 * s[:, i] * s[:, j] * match
    return acc / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--W', type=int, required=True)
    ap.add_argument('--L', type=int, default=8)
    ap.add_argument('--ref_npz', required=True)
    ap.add_argument('--out_npz', required=True)
    ap.add_argument('--sample_size', type=int, default=20000)
    ap.add_argument('--gen_batch_size', type=int, default=1024)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--layer_num', type=int, default=4)
    ap.add_argument('--head_num', type=int, default=8)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    W, L = args.W, args.L
    n = W * L
    Ltok = 1 + 2 * n
    ref = np.load(args.ref_npz)
    j2s = np.asarray(ref['j2s'], float)

    nn, nnn = bonds(W, L)
    ring = [(i, j) for (i, j) in nn if abs(i - j) < W]
    axial = [(i, j) for (i, j) in nn if abs(i - j) >= W]

    model = QuantumDiffusionModel_oseq_rope(
        hidden_dim=args.hidden_dim, num_layers=args.layer_num,
        head_count=args.head_num).to(device)
    model.load_state_dict(load_state_dict(args.model_path))
    model.eval()

    r_positions = torch.arange(1, Ltok, 2, device=device)
    b_positions = torch.arange(2, Ltok, 2, device=device)
    est = {'sd_ring': [], 'sd_axial': [], 'sd_diag': []}
    for j2 in j2s:
        P_l, b_l = [], []
        remaining = args.sample_size
        while remaining > 0:
            B = min(args.gen_batch_size, remaining)
            prompt = torch.full((B, Ltok, 1), MASK_TOKEN_ID, device=device)
            prompt[:, 0, 0] = float(j2)
            rand_P = torch.randint(2, 5, (B, n), device=device).float()
            prompt[:, r_positions, 0] = rand_P
            done = generate_oseq_batch(model=model, prompt=prompt, steps=n,
                                       temperature=1.0, use_sampling=True)
            P_l.append(rand_P.cpu().numpy())
            b_l.append((done[:, b_positions, 0] > 0.5).cpu().numpy())
            remaining -= B
        P = np.concatenate(P_l).astype(np.int8)
        b = np.concatenate(b_l).astype(np.float64)
        for key, pairs in (('sd_ring', ring), ('sd_axial', axial),
                           ('sd_diag', nnn)):
            est[key].append(median_of_means(shadow_spin_dot(P, b, pairs), 10))
        print(f"  j2={j2:.2f} ring={est['sd_ring'][-1]:+.4f} "
              f"(ref {float(ref['sd_ring'][list(j2s).index(j2)]):+.4f})  "
              f"diag={est['sd_diag'][-1]:+.4f} "
              f"(ref {float(ref['sd_diag'][list(j2s).index(j2)]):+.4f})",
              flush=True)

    save = dict(j2s=j2s, W=W, L=L, sample_size=args.sample_size)
    for k in est:
        save[f"gen_{k}"] = np.array(est[k])
        save[f"ref_{k}"] = np.asarray(ref[k])
        save[f"mse_{k}"] = float(np.mean((np.array(est[k]) - np.asarray(ref[k])) ** 2))
    print("MSE", {k: f"{save[f'mse_{k}']:.3e}" for k in est})
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **save)
    print(f"saved {args.out_npz}")


if __name__ == '__main__':
    main()
