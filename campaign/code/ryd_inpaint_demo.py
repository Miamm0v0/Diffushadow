#!/usr/bin/env python3
"""Atom-loss inpainting demo on the Rydberg chain (R1 cut B, Z3 regime).

Real atom-array experiments lose a fraction of atoms per shot (trap loss,
detection error); the affected sites yield no outcome.  This demo takes
fresh DMRG-sampled occupation snapshots, discards a random fraction f of
sites per snapshot (different sites each shot, as in a real device), lets
the model complete them conditioned on the surviving sites, and tests
whether the completions restore (i) the density profile, (ii) the
connected structure factor S(q) with its Z3 peak at q = 2pi/3, and
(iii) the connected density-density correlation across observed/lost
nearest-neighbour pairs.

Baseline "unconditional": hidden sites filled from fresh unconditioned
model samples — what you would get by ignoring the surviving atoms.

Usage:
  python tfi2d/ryd_inpaint_demo.py --model_path checkpoints_ryd/ryd_rb2.35_oseq_final.pth \
      --test_json data_ryd/ryd_rb2.35_N32_testinp.json \
      --ref_npz data_ryd/ryd_rb2.35_N32_testinp_ref.npz \
      --loss_fracs 0.1 0.3 --out_npz results_ryd/ryd_inpaint.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Qdmodel_oseq_rope import QuantumDiffusionModel_oseq_rope  # noqa: E402
from eval_tfi2d import generate_oseq_batch, load_state_dict, MASK_TOKEN_ID  # noqa: E402


def structure_factor(occ, qs):
    """Connected density-density S(q) from an occupation ensemble [M, N]."""
    dens = occ.mean(axis=0)
    conn = (occ.T @ occ) / occ.shape[0] - np.outer(dens, dens)
    n = occ.shape[1]
    idx = np.arange(n)
    return np.array([np.real(np.exp(1j * q * (idx[:, None] - idx[None, :]))
                             * conn).sum() / n for q in qs])


def complete(model, device, delta, b_tok, hidden, batch):
    """Model-complete masked occupation tokens.  b_tok [M,N] int tokens,
    hidden [M,N] bool (True = lost atom).  Returns completed tokens."""
    M, n = b_tok.shape
    L = 1 + 2 * n
    out = np.empty_like(b_tok)
    k = int(hidden[0].sum())
    for s0 in range(0, M, batch):
        s1 = min(s0 + batch, M)
        B = s1 - s0
        prompt = torch.full((B, L, 1), float(MASK_TOKEN_ID))
        prompt[:, 0, 0] = float(delta)
        prompt[:, 1::2, 0] = 4.0
        bb = torch.from_numpy(b_tok[s0:s1].astype(np.float32))
        bb[torch.from_numpy(hidden[s0:s1])] = MASK_TOKEN_ID
        prompt[:, 2::2, 0] = bb
        done = generate_oseq_batch(model=model, prompt=prompt.to(device),
                                   steps=k, temperature=1.0, use_sampling=True)
        out[s0:s1] = (done[:, torch.arange(2, L, 2), 0] > 0.5).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--test_json', required=True)
    ap.add_argument('--ref_npz', required=True)
    ap.add_argument('--loss_fracs', type=float, nargs='+', default=[0.1, 0.3])
    ap.add_argument('--deltas', type=float, nargs='+', default=None,
                    help='restrict to these delta values (default: all in json)')
    ap.add_argument('--max_per_delta', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--layer_num', type=int, default=4)
    ap.add_argument('--head_num', type=int, default=8)
    ap.add_argument('--out_npz', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ref = np.load(args.ref_npz)
    qs = np.asarray(ref['qs'], dtype=float)
    ref_d = list(np.asarray(ref['deltas'], dtype=float))

    rows = json.load(open(args.test_json))
    n = int(ref['N']) if 'N' in ref else len(rows[0]) - 1
    oseq = (len(rows[0]) == 1 + 2 * n)   # tokenized [d,P1,b1,...] vs raw [d,b1..bN]
    by_delta = {}
    for row in rows:
        d = float(row[0])
        if args.deltas and not any(abs(d - x) < 1e-9 for x in args.deltas):
            continue
        bits = row[2::2] if oseq else row[1:]
        if len(by_delta.get(d, ())) < args.max_per_delta:
            by_delta.setdefault(d, []).append(bits)

    model = QuantumDiffusionModel_oseq_rope(
        hidden_dim=args.hidden_dim, num_layers=args.layer_num,
        head_count=args.head_num).to(device)
    model.load_state_dict(load_state_dict(args.model_path))
    model.eval()

    rng = np.random.default_rng(11)
    out = {'qs': qs, 'deltas': np.array(sorted(by_delta)),
           'loss_fracs': np.array(args.loss_fracs)}
    for delta in sorted(by_delta):
        b_tok = np.array(by_delta[delta], dtype=np.int64)      # TeNPy index
        occ_exact = 1.0 - b_tok                                # n = 1 - b
        M = b_tok.shape[0]
        ri = ref_d.index(delta)
        out[f'd{delta}_ref_density'] = np.asarray(ref['density'])[ri]
        out[f'd{delta}_ref_Sq'] = np.asarray(ref['Sq'])[ri]
        out[f'd{delta}_exact_Sq'] = structure_factor(occ_exact, qs)

        # unconditional ensemble (shared across loss fractions)
        b_un = complete(model, device, delta,
                        np.zeros_like(b_tok),
                        np.ones((M, n), dtype=bool), args.batch)
        occ_un = 1.0 - b_un

        for f in args.loss_fracs:
            k = max(1, round(f * n))
            hidden = np.zeros((M, n), dtype=bool)
            for t in range(M):
                hidden[t, rng.choice(n, size=k, replace=False)] = True
            b_inp = complete(model, device, delta, b_tok, hidden, args.batch)
            occ_inp = 1.0 - b_inp
            # splice-uncond baseline: hidden sites taken from occ_un
            occ_spl = np.where(hidden, occ_un, occ_exact)

            # cross-partition connected nn correlation (obs i, lost i+1)
            def cross_corr(occ_h):
                num, den = 0.0, 0
                for i in range(n - 1):
                    sel = (~hidden[:, i]) & hidden[:, i + 1]
                    if sel.sum() < 50:
                        continue
                    a, c = occ_exact[sel, i], occ_h[sel, i + 1]
                    num += np.mean(a * c) - a.mean() * c.mean()
                    den += 1
                return num / max(den, 1)

            res = dict(
                density=occ_inp.mean(axis=0),
                Sq=structure_factor(occ_inp, qs),
                Sq_spl=structure_factor(occ_spl, qs),
                cross_exact=cross_corr(occ_exact),
                cross_inp=cross_corr(occ_inp),
                cross_un=cross_corr(occ_un),
            )
            for key, v in res.items():
                out[f'd{delta}_f{f}_{key}'] = v
            pk = lambda S: qs[np.argmax(S[1:]) + 1]
            print(f"d={delta:+.2f} f={f:.2f}: "
                  f"dens MSE={np.mean((res['density'] - out[f'd{delta}_ref_density'])**2):.2e}  "
                  f"qpk inp={pk(res['Sq']):.3f} spl={pk(res['Sq_spl']):.3f} "
                  f"ref={pk(out[f'd{delta}_ref_Sq']):.3f}  "
                  f"cross exact={res['cross_exact']:+.4f} inp={res['cross_inp']:+.4f} "
                  f"un={res['cross_un']:+.4f}", flush=True)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **out)
    print(f"saved {args.out_npz}")


if __name__ == '__main__':
    main()
