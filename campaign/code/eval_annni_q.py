#!/usr/bin/env python3
"""Avenue #2: q-resolved structure factor from an ANNNI-trained Diffushadow
checkpoint.  Generates Pauli-6 shadows (1D oseq layout), estimates the full
site-averaged C_ZZ(r), the structure factor S(q), and the peak wavevector —
the observables that locate floating-phase/commensurate boundaries.

Usage:
  python tfi2d/eval_annni_q.py --model_path ckpt.pth --N 96 \
      --hs 0.05 ... --out_npz results_annni/annni_q_N96.npz \
      [--ref_npz dmrg_refs/annni06_dmrg_ref_N96.npz] --sample_size 10000
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
    ap.add_argument('--hs', type=float, nargs='+', required=True)
    ap.add_argument('--out_npz', required=True)
    ap.add_argument('--ref_npz', default='')
    ap.add_argument('--sample_size', type=int, default=10000)
    ap.add_argument('--gen_batch_size', type=int, default=1024)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--layer_num', type=int, default=4)
    ap.add_argument('--head_num', type=int, default=8)
    ap.add_argument('--dump_json', type=str, default='',
                    help='dump generated snapshots as training rows (recursion)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n, L = args.N, 1 + 2 * args.N
    model = QuantumDiffusionModel_oseq_rope(
        hidden_dim=args.hidden_dim, num_layers=args.layer_num,
        head_count=args.head_num).to(device)
    model.load_state_dict(load_state_dict(args.model_path))
    model.eval()

    qs = np.linspace(0, np.pi, 97)
    r_positions = torch.arange(1, L, 2, device=device)
    b_positions = torch.arange(2, L, 2, device=device)
    C_all, S_all, qpk = [], [], []
    dump_rows = []
    for h in args.hs:
        P_l, b_l = [], []
        remaining = args.sample_size
        while remaining > 0:
            B = min(args.gen_batch_size, remaining)
            prompt = torch.full((B, L, 1), MASK_TOKEN_ID, device=device)
            prompt[:, 0, 0] = float(h)
            rand_P = torch.randint(2, 5, (B, n), device=device).float()
            prompt[:, r_positions, 0] = rand_P
            done = generate_oseq_batch(model=model, prompt=prompt, steps=n,
                                       temperature=1.0, use_sampling=True)
            P_l.append(rand_P.cpu().numpy())
            b_l.append((done[:, b_positions, 0] > 0.5).cpu().numpy())
            remaining -= B
        P = np.concatenate(P_l).astype(np.int8)
        b = np.concatenate(b_l).astype(np.int8)
        if args.dump_json:
            for t in range(P.shape[0]):
                row = [float(h)]
                for i in range(n):
                    row += [int(P[t, i]), int(b[t, i])]
                dump_rows.append(row)
        zhat = 3.0 * (2.0 * b - 1.0) * (P == 4)            # [M, N]
        # site-averaged C_ZZ(r) for all r (PBC), unbiased across snapshots
        M = zhat.shape[0]
        C = np.empty(n)
        C[0] = 1.0
        for r in range(1, n):
            prod = zhat * np.roll(zhat, -r, axis=1)
            C[r] = prod.mean() * 1.0
        S = np.array([np.real(np.sum(np.exp(1j * q * np.arange(n)) * C))
                      for q in qs])
        C_all.append(C); S_all.append(S)
        qpk.append(qs[int(np.argmax(S[1:])) + 1])
        print(f"  h={h:.3f} q_peak={qpk[-1]:.4f} S(pi)={S[-1]:+.3f} "
              f"S(pi/2)={S[len(qs)//2]:+.3f}", flush=True)

    save = dict(hs=np.array(args.hs), qs=qs, C_r=np.array(C_all),
                Sq=np.array(S_all), q_peak=np.array(qpk), N=n)
    if args.ref_npz:
        ref = np.load(args.ref_npz)
        save['ref_params'] = ref['params']
        save['ref_C_r'] = ref['C_r']
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **save)
    print(f"saved {args.out_npz}")
    if args.dump_json:
        import json
        Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dump_rows, open(args.dump_json, 'w'))
        print(f"dumped {len(dump_rows)} snapshots -> {args.dump_json}")


if __name__ == '__main__':
    main()
