#!/usr/bin/env python3
"""R1: evaluate a Rydberg-trained oseq_rope checkpoint.

Generates occupation snapshots (all basis tokens fixed to Z) with the same
entropy-guided sequential denoiser as the paper, then estimates the density
profile and the connected density-density structure factor S(q), compared
against the DMRG reference npz from rydberg_dataset.py.

Usage:
  python tfi2d/eval_rydberg.py --model_path checkpoints_ryd/ryd_oseq.pth \
      --N 64 --ref_npz data_ryd/ryd_rb1.2_N64_ref.npz \
      --out_npz results_ryd/ryd_eval_N64.npz --sample_size 20000
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
from eval_tfi2d import generate_oseq_batch, load_state_dict, MASK_TOKEN_ID  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--ref_npz', required=True)
    ap.add_argument('--out_npz', required=True)
    ap.add_argument('--sample_size', type=int, default=20000)
    ap.add_argument('--gen_batch_size', type=int, default=2048)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--layer_num', type=int, default=4)
    ap.add_argument('--head_num', type=int, default=8)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n, L = args.N, 1 + 2 * args.N
    ref = np.load(args.ref_npz)
    deltas = np.asarray(ref['deltas'], dtype=float)
    qs = np.asarray(ref['qs'], dtype=float)

    model = QuantumDiffusionModel_oseq_rope(
        hidden_dim=args.hidden_dim, num_layers=args.layer_num,
        head_count=args.head_num).to(device)
    model.load_state_dict(load_state_dict(args.model_path))
    model.eval()

    r_positions = torch.arange(1, L, 2, device=device)
    b_positions = torch.arange(2, L, 2, device=device)
    gen_density, gen_Sq = [], []
    for d in deltas:
        b_all = []
        remaining = args.sample_size
        while remaining > 0:
            B = min(args.gen_batch_size, remaining)
            prompt = torch.full((B, L, 1), MASK_TOKEN_ID, device=device)
            prompt[:, 0, 0] = float(d)
            prompt[:, r_positions, 0] = 4.0  # occupation basis = all Z
            completed = generate_oseq_batch(model=model, prompt=prompt,
                                            steps=n, temperature=1.0,
                                            use_sampling=True)
            b_all.append((completed[:, b_positions, 0] > 0.5).cpu().numpy())
            remaining -= B
        # token convention: b stores the TeNPy SpinHalfSite basis index, whose
        # index 0 is 'up' (occupied); occupation n = 1 - b.
        b = 1.0 - np.concatenate(b_all).astype(np.float64)  # [M, N] occupations
        dens = b.mean(axis=0)
        conn = (b.T @ b) / b.shape[0] - np.outer(dens, dens)
        idx = np.arange(n)
        S = np.array([np.real(np.exp(1j * q * (idx[:, None] - idx[None, :]))
                              * conn).sum() / n for q in qs])
        gen_density.append(dens)
        gen_Sq.append(S)
        qpk_g = qs[np.argmax(S[1:]) + 1]
        qpk_r = qs[np.argmax(np.asarray(ref['Sq'])[list(deltas).index(d)][1:]) + 1]
        print(f"  d={d:+.3f} <n>={dens.mean():.4f} "
              f"(ref {np.asarray(ref['density'])[list(deltas).index(d)].mean():.4f})"
              f"  q_peak={qpk_g:.3f} (ref {qpk_r:.3f})", flush=True)

    gen_density = np.array(gen_density)
    gen_Sq = np.array(gen_Sq)
    mse_n = float(np.mean((gen_density.mean(1) - np.asarray(ref['density']).mean(1)) ** 2))
    mse_S = float(np.mean((gen_Sq - np.asarray(ref['Sq'])) ** 2))
    print(f"MSE density={mse_n:.3e}  S(q)={mse_S:.3e}")
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, deltas=deltas, qs=qs,
             gen_density=gen_density, gen_Sq=gen_Sq,
             ref_density=ref['density'], ref_Sq=ref['Sq'],
             mse_density=mse_n, mse_Sq=mse_S, N=n)
    print(f"saved {args.out_npz}")


if __name__ == '__main__':
    main()
