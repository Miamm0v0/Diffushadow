#!/usr/bin/env python3
"""Measurement inpainting demonstration — the capability with no QMC/DMRG
analogue.  Take exact ED classical shadows of the 4x4 TFI torus, hide the
outcomes of half the sites, let the model complete them, and test whether
the completions carry the correct correlations WITH the observed sites
(cross-mask ZZ), compared against exact values.

Two mask patterns:
  prefix      : second half of the site order hidden (a fixed-order AR model
                could also condition on this)
  interleaved : every other site hidden (impossible for fixed-order AR;
                trivial for masked diffusion)
A model that ignores the conditioning gives cross-mask ZZ -> uncorrelated;
correct inpainting reproduces the exact ZZ across observed/hidden pairs.

Usage:
  python tfi2d/inpaint_demo.py --model_path checkpoints2d/tfi2d_oseq_rope_final.pth \
      --gs 0.2 0.5 0.75 0.9 --num_test 2000 --out_npz campaign/inpaint_demo.npz
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
from generate_tfi2d_dataset import ground_state_tfi2d, sample_shadows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--lx', type=int, default=4)
    ap.add_argument('--ly', type=int, default=4)
    ap.add_argument('--gs', type=float, nargs='+', default=[0.2, 0.5, 0.75, 0.9])
    ap.add_argument('--num_test', type=int, default=2000)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--layer_num', type=int, default=4)
    ap.add_argument('--head_num', type=int, default=8)
    ap.add_argument('--out_npz', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = args.lx * args.ly
    L = 1 + 2 * n
    model = QuantumDiffusionModel_oseq_rope(
        hidden_dim=args.hidden_dim, num_layers=args.layer_num,
        head_count=args.head_num).to(device)
    model.load_state_dict(load_state_dict(args.model_path))
    model.eval()

    masks = {
        'prefix': np.arange(n) >= n // 2,          # hidden sites
        'interleaved': (np.arange(n) % 2) == 1,
    }
    rng = np.random.default_rng(11)
    out = {}
    for g in args.gs:
        _, psi = ground_state_tfi2d(args.lx, args.ly, g)
        shadows = sample_shadows(psi, n, args.num_test, rng)   # [M, 2n]
        P = shadows[:, 0::2].astype(np.int64)
        b = shadows[:, 1::2].astype(np.int64)
        # exact ZZ (full ensemble) for cross-pairs
        zhat_full = 3.0 * (2.0 * b - 1.0) * (P == 4)
        for mname, hidden in masks.items():
            obs = ~hidden
            # build prompts: bases known everywhere, outcomes masked on hidden
            prompt = torch.full((args.num_test, L, 1), MASK_TOKEN_ID)
            for t in range(args.num_test):
                prompt[t, 0, 0] = float(g)
            prompt[:, 1::2, 0] = torch.from_numpy(P.astype(np.float32))
            bb = torch.from_numpy(b.astype(np.float32))
            bb[:, hidden] = MASK_TOKEN_ID
            prompt[:, 2::2, 0] = bb
            prompt = prompt.to(device)
            done = generate_oseq_batch(model=model, prompt=prompt,
                                       steps=int(hidden.sum()),
                                       temperature=1.0, use_sampling=True)
            b_done = (done[:, torch.arange(2, L, 2), 0] > 0.5).cpu().numpy().astype(np.int64)
            zhat_done = 3.0 * (2.0 * b_done - 1.0) * (P == 4)
            # cross-mask nearest-neighbour ZZ pairs (one obs, one hidden)
            pairs = []
            for r in range(args.ly):
                for c in range(args.lx):
                    s = r * args.lx + c
                    for t2 in (r * args.lx + (c + 1) % args.lx,
                               ((r + 1) % args.ly) * args.lx + c):
                        if obs[s] and hidden[t2]:
                            pairs.append((s, t2))
            zz_exact = np.mean([np.mean(zhat_full[:, i] * zhat_full[:, j])
                                for i, j in pairs])
            zz_inp = np.mean([np.mean(zhat_full[:, i] * zhat_done[:, j])
                              for i, j in pairs])
            # baseline: completions with NO conditioning (fresh unconditional
            # samples in place of the hidden sites)
            prompt2 = torch.full((args.num_test, L, 1), MASK_TOKEN_ID, device=device)
            prompt2[:, 0, 0] = float(g)
            prompt2[:, 1 + 2 * torch.arange(n, device=device), 0] = \
                torch.from_numpy(P.astype(np.float32)).to(device)
            done2 = generate_oseq_batch(model=model, prompt=prompt2, steps=n,
                                        temperature=1.0, use_sampling=True)
            b_un = (done2[:, torch.arange(2, L, 2), 0] > 0.5).cpu().numpy().astype(np.int64)
            zhat_un = 3.0 * (2.0 * b_un - 1.0) * (P == 4)
            zz_uncond = np.mean([np.mean(zhat_full[:, i] * zhat_un[:, j])
                                 for i, j in pairs])
            out[f"g{g}_{mname}"] = (zz_exact, zz_inp, zz_uncond)
            print(f"g={g:.2f} {mname:12s}: exact={zz_exact:+.4f}  "
                  f"inpainted={zz_inp:+.4f}  unconditional={zz_uncond:+.4f}",
                  flush=True)

    np.savez(args.out_npz,
             **{k: np.array(v) for k, v in out.items()},
             gs=np.array(args.gs))
    print(f"saved {args.out_npz}")


if __name__ == '__main__':
    main()
