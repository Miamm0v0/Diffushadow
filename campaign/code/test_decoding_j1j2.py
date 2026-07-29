#!/usr/bin/env python3
"""Decisive test for manuscript Fig. 3h,i: is the decay of the generated
J1-J2 nearest-neighbour correlation with N caused by the number of
denoising steps?

Generates from the student's own J1-J2 checkpoint at several N, using
(a) few-step decoding (the README default) and (b) sequential decoding
(one site per step), and compares <sigma_i.sigma_{i+1}> with the exact
value.  Exact reference at J2=0 (Bethe): -1.7726 (Pauli convention).

Usage:
  python test_decoding_j1j2.py --model <ckpt.pth> --layers 8 --hidden 256 \
      --Ns 24 48 72 --j2 0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve()
for _p in (_here.parent, _here.parent.parent, _here.parent.parent.parent):
    sys.path.insert(0, str(_p))
from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope  # noqa: E402
from eval_tfi2d import generate_oseq_batch, load_state_dict, MASK_TOKEN_ID  # noqa: E402

BETHE_J2_0 = -1.7726   # 4*(1/4 - ln2), Pauli convention


def spin_dot_r1(P, b, n):
    """Shadow estimate of <sigma_i.sigma_{i+1}>, site-averaged, PBC."""
    s = 2.0 * b - 1.0
    acc = np.zeros(P.shape[0])
    for i in range(n):
        j = (i + 1) % n
        acc += 9.0 * s[:, i] * s[:, j] * (P[:, i] == P[:, j])
    return acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--Ns', type=int, nargs='+', default=[24, 48, 72])
    ap.add_argument('--j2', type=float, default=0.0)
    ap.add_argument('--samples', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--rope_scaling_type', default=None,
                    help='none|linear|dynamic  (test-time RoPE extrapolation)')
    ap.add_argument('--rope_scaling_factor', type=float, default=1.0)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rst = None if args.rope_scaling_type in (None, 'none') else args.rope_scaling_type
    model = QuantumDiffusionModel_oseq_rope(
        hidden_dim=args.hidden, num_layers=args.layers,
        head_count=args.heads, rope_scaling_type=rst,
        rope_scaling_factor=args.rope_scaling_factor).to(dev)
    model.load_state_dict(load_state_dict(args.model))
    model.eval()

    print(f"J2={args.j2}  exact <s.s>(r=1) = {BETHE_J2_0:+.4f} (Bethe, J2=0)")
    print(f"{'N':>5} {'steps=2':>12} {'steps=N':>12}   (Pauli convention)")
    for n in args.Ns:
        L = 1 + 2 * n
        out = {}
        for label, steps in (("few", 2), ("seq", n)):
            P_all, b_all = [], []
            remaining = args.samples
            while remaining > 0:
                B = min(args.batch, remaining)
                prompt = torch.full((B, L, 1), MASK_TOKEN_ID, device=dev)
                prompt[:, 0, 0] = float(args.j2)
                rp = torch.arange(1, L, 2, device=dev)
                randP = torch.randint(2, 5, (B, n), device=dev).float()
                prompt[:, rp, 0] = randP
                done = generate_oseq_batch(model=model, prompt=prompt, steps=steps,
                                           temperature=1.0, use_sampling=True)
                bp = torch.arange(2, L, 2, device=dev)
                P_all.append(randP.cpu().numpy())
                b_all.append((done[:, bp, 0] > 0.5).cpu().numpy())
                remaining -= B
            P = np.concatenate(P_all).astype(np.int8)
            b = np.concatenate(b_all).astype(np.float64)
            out[label] = float(np.median([x.mean() for x in
                                          np.array_split(spin_dot_r1(P, b, n), 10)]))
        print(f"{n:>5} {out['few']:>12.4f} {out['seq']:>12.4f}", flush=True)


if __name__ == '__main__':
    main()
