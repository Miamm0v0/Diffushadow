#!/usr/bin/env python3
"""R1: Rydberg-chain dataset generator (experiment-format data).

Model (standard blockade-regime chain, open boundaries as in experiments):

    H/Omega = (1/2) sum_i sigma^x_i  -  (Delta/Omega) sum_i n_i
              + sum_{i<j} (R_b/a)^6 / (j-i)^6  n_i n_j,     n = (sigma^z+1)/2

vdW tail truncated at range --max-range (default 5; (5)^-6 ~ 6e-5).

Produces:
  * training rows [delta_over_omega, b_1, ..., b_N]  (Z-basis occupation
    snapshots from MPS perfect sampling — the SAME data format a Rydberg
    machine produces; NO basis tokens)
  * per-size reference npz: deltas, E0, density, density-density S(q) on a
    q-grid (for validation and boundary mapping)

Usage (node 109, tenpy env):
  python tfi2d/rydberg_dataset.py --N 16 --rb 1.2 \
      --deltas -1.0:4.0:21 --samples 3000 \
      --json-out data_ryd/ryd_rb1.2_N16_train.json \
      --ref-out data_ryd/ryd_rb1.2_N16_ref.npz --workers 20
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def parse_grid(spec: str) -> np.ndarray:
    lo, hi, n = spec.split(':')
    return np.linspace(float(lo), float(hi), int(n))


def solve_point(task):
    N, rb, delta, chi, max_range, samples, seed = task
    from tenpy.algorithms import dmrg
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    class RydbergChain(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)

        def init_terms(self, mp):
            d = mp.get('delta', 0.0)
            rb6 = mp.get('rb', 1.2) ** 6
            rng = mp.get('max_range', 5)
            # n = Sz + 1/2 (sigma^z = 2 Sz);  H in units of Omega
            self.add_onsite(1.0, 0, 'Sx')            # (1/2) sigma^x = Sx
            self.add_onsite(-d, 0, 'Sz')             # -delta*n -> -delta*Sz + const
            for r in range(1, rng + 1):
                v = rb6 / r ** 6
                self.add_coupling(v, 0, 'Sz', 0, 'Sz', r)
                self.add_onsite(v, 0, 'Sz')          # cross terms n_i n_j
                # constant terms dropped (shift E only; we report shifted E)

    mp = dict(lattice='Chain', L=N, bc_x='open', bc_MPS='finite',
              delta=float(delta), rb=float(rb), max_range=int(max_range))
    model = RydbergChain(mp)
    psi = MPS.from_product_state(model.lat.mps_sites(), ['down'] * N, bc='finite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
        'max_E_err': 1e-9, 'max_sweeps': 40,
    })
    E, psi = eng.run()
    psi.canonical_form()  # restore strict normalization (sample_measurements
    # enforces norm_tol, and post-DMRG truncation leaves a small deficit)

    dens = np.asarray(psi.expectation_value('Sz')) + 0.5
    C = np.asarray(psi.correlation_function('Sz', 'Sz'))  # <Sz_i Sz_j>
    nn = C + 0.5 * (dens[:, None] - 0.5) + 0.5 * (dens[None, :] - 0.5) + 0.25
    # connected density-density structure factor on q grid
    qs = np.linspace(0, np.pi, 61)
    conn = nn - dens[:, None] * dens[None, :]
    idx = np.arange(N)
    S = np.array([np.real(np.exp(1j * q * (idx[:, None] - idx[None, :])) * conn).sum() / N
                  for q in qs])

    snaps = None
    if samples > 0:
        rng_ = np.random.default_rng(seed)
        snaps = np.empty((samples, N), dtype=np.int8)
        for t in range(samples):
            sigmas, _ = psi.sample_measurements(rng=rng_)
            # SpinHalfSite local basis: index 0 = down (n=0), 1 = up (n=1)
            snaps[t] = np.asarray(sigmas, dtype=np.int8)
    return float(delta), float(E), dens, qs, S, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--rb', type=float, default=1.2)
    ap.add_argument('--deltas', type=str, required=True, help='lo:hi:n')
    ap.add_argument('--samples', type=int, default=3000)
    ap.add_argument('--chi', type=int, default=128)
    ap.add_argument('--max-range', type=int, default=5)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json-out', type=str, default='')
    ap.add_argument('--ref-out', type=str, required=True)
    args = ap.parse_args()

    deltas = parse_grid(args.deltas)
    tasks = [(args.N, args.rb, d, args.chi, args.max_range, args.samples,
              args.seed + 977 * k) for k, d in enumerate(deltas)]
    print(f"Rydberg chain N={args.N} Rb/a={args.rb}: {len(deltas)} deltas, "
          f"{args.samples} snapshots each", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(solve_point, tasks))
    results.sort(key=lambda r: r[0])

    ds = np.array([r[0] for r in results])
    E = np.array([r[1] for r in results])
    dens = np.array([r[2] for r in results])
    qs = results[0][3]
    S = np.array([r[4] for r in results])
    Path(args.ref_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.ref_out, deltas=ds, E0=E, density=dens, qs=qs, Sq=S,
             rb=args.rb, N=args.N, chi=args.chi, max_range=args.max_range)
    print(f"saved refs: {args.ref_out}")
    for d, e, dn, _, s, _ in results:
        qpk = qs[np.argmax(s[1:]) + 1]
        print(f"  d={d:+.3f} E={e:+.4f} <n>={dn.mean():.4f} q_peak={qpk:.3f}")

    if args.samples > 0 and args.json_out:
        rows = []
        for d, _, _, _, _, snaps in results:
            for t in range(snaps.shape[0]):
                rows.append([float(d)] + [int(v) for v in snaps[t]])
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.json_out, 'w'))
        print(f"saved training json: {args.json_out} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
