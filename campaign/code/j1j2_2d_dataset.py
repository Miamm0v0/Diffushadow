#!/usr/bin/env python3
"""Avenue #3: 2D J1-J2 Heisenberg on W x L cylinders — DMRG ground states,
random-Pauli classical-shadow sampling from the MPS, and reference
observables.  This is the training-data factory for the frustrated
width-extrapolation program (QMC sign-blocked, ED size-blocked).

Model (Pauli convention, matching the 1D manuscript):
  H = J1 sum_<ij> sigma_i.sigma_j + J2 sum_<<ij>> sigma_i.sigma_j
  <ij>  : NN bonds of the square lattice (ring + axial)
  <<ij>>: diagonal next-nearest bonds
  Cylinder: periodic around the width W (ring), open along the length L.
  Site order: s = x*W + y  (x = 0..L-1 column, y = 0..W-1 ring position).

Shadow sampling: for each snapshot draw a random Pauli basis per site,
apply the corresponding single-site rotation to a copy of the MPS
(U = |up><+P| + |down><-P|, so Sz-sampling == measuring P), then perfect-
sample.  Token encoding identical to the 1D generator: P in {2:X,3:Y,4:Z},
b = 1 for outcome +1.

Usage:
  python j1j2_2d_dataset.py --W 4 --L 8 --j2s 0.2 0.4 0.5 0.6 0.8 \
      --samples 3000 --chi 800 --workers 5 \
      --json-out data_j2d/j1j2_4x8_train.json --ref-out data_j2d/j1j2_4x8_ref.npz
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def bonds(W, L):
    nn, nnn = [], []
    for x in range(L):
        for y in range(W):
            s = x * W + y
            nn.append((s, x * W + (y + 1) % W))          # ring
            if x + 1 < L:
                nn.append((s, (x + 1) * W + y))          # axial
                nnn.append((s, (x + 1) * W + (y + 1) % W))
                nnn.append((s, (x + 1) * W + (y - 1) % W))
    return nn, nnn


def solve_point(task):
    W, L, j2, chi, samples, seed = task
    import tenpy.linalg.np_conserved as npc
    from tenpy.algorithms import dmrg
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    n = W * L
    nn, nnn = bonds(W, L)

    class J1J2Cyl(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)

        def init_terms(self, mp):
            for blist, J in ((nn, 1.0), (nnn, mp.get('j2', 0.0))):
                if abs(J) < 1e-15:
                    continue
                for (i, j) in blist:
                    a, b = (i, j) if i < j else (j, i)
                    self.add_coupling_term(4.0 * J, a, b, 'Sz', 'Sz')
                    self.add_coupling_term(2.0 * J, a, b, 'Sp', 'Sm',
                                           plus_hc=True)

    mp = dict(lattice='Chain', L=n, bc_MPS='finite', j2=float(j2))
    model = J1J2Cyl(mp)
    init = (['up', 'down'] * n)[:n]
    psi = MPS.from_product_state(model.lat.mps_sites(), init, bc='finite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-9},
        'max_E_err': 1e-8, 'max_sweeps': 30,
    })
    E, psi = eng.run()
    psi.canonical_form()

    # reference observables: site-averaged spin-dot for the three bond types
    Czz = 4.0 * np.asarray(psi.correlation_function('Sz', 'Sz'))
    Cpm = np.asarray(psi.correlation_function('Sp', 'Sm'))
    Cmp = np.asarray(psi.correlation_function('Sm', 'Sp'))
    Cdot = Czz + 2.0 * np.real(Cpm + Cmp)
    ring = [(i, j) for (i, j) in nn if abs(i - j) < W]
    axial = [(i, j) for (i, j) in nn if abs(i - j) >= W]
    out = {
        'E': float(E),
        'sd_ring': float(np.mean([Cdot[i, j] for i, j in ring])),
        'sd_axial': float(np.mean([Cdot[i, j] for i, j in axial])),
        'sd_diag': float(np.mean([Cdot[i, j] for i, j in nnn])),
    }

    snaps = None
    if samples > 0:
        # basis-rotation gates as npc arrays (legs p, p*)
        site = psi.sites[0]
        s2 = 2.0 ** -0.5
        U_X = npc.Array.from_ndarray(
            np.array([[s2, s2], [s2, -s2]], dtype=complex),
            [site.leg, site.leg.conj()], labels=['p', 'p*'])
        U_Y = npc.Array.from_ndarray(
            np.array([[s2, -1j * s2], [s2, 1j * s2]], dtype=complex),
            [site.leg, site.leg.conj()], labels=['p', 'p*'])
        rng = np.random.default_rng(seed)
        snaps = np.empty((samples, 2 * n), dtype=np.int8)
        for t in range(samples):
            bases = rng.integers(2, 5, size=n)          # 2:X 3:Y 4:Z
            psi2 = psi.copy()
            for i in range(n):
                if bases[i] == 2:
                    psi2.apply_local_op(i, U_X)
                elif bases[i] == 3:
                    psi2.apply_local_op(i, U_Y)
            sigmas, _ = psi2.sample_measurements(rng=rng)
            # SpinHalfSite index 0 = 'up' = +1 outcome -> b token 1
            for i in range(n):
                snaps[t, 2 * i] = bases[i]
                snaps[t, 2 * i + 1] = 1 if int(sigmas[i]) == 0 else 0
            if (t + 1) % 500 == 0:
                print(f"    [W{W} j2={j2}] {t + 1}/{samples} snapshots",
                      flush=True)
    return float(j2), out, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, required=True)
    ap.add_argument('--L', type=int, default=8)
    ap.add_argument('--j2s', type=float, nargs='+', required=True)
    ap.add_argument('--samples', type=int, default=3000)
    ap.add_argument('--chi', type=int, default=800)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json-out', type=str, default='')
    ap.add_argument('--ref-out', type=str, required=True)
    args = ap.parse_args()

    tasks = [(args.W, args.L, float(j2), args.chi, args.samples,
              args.seed + 811 * k) for k, j2 in enumerate(args.j2s)]
    print(f"J1-J2 cylinder {args.W}x{args.L} (N={args.W * args.L}), "
          f"{len(tasks)} J2 points, chi={args.chi}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(solve_point, tasks))
    results.sort(key=lambda r: r[0])

    j2s = np.array([r[0] for r in results])
    save = dict(j2s=j2s, W=args.W, L=args.L, chi=args.chi,
                E=np.array([r[1]['E'] for r in results]))
    for k in ('sd_ring', 'sd_axial', 'sd_diag'):
        save[k] = np.array([r[1][k] for r in results])
    Path(args.ref_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.ref_out, **save)
    print(f"saved refs: {args.ref_out}")
    for j2, o, _ in results:
        print(f"  j2={j2:.2f} E/N={o['E'] / (args.W * args.L):+.5f} "
              f"ring={o['sd_ring']:+.4f} axial={o['sd_axial']:+.4f} "
              f"diag={o['sd_diag']:+.4f}")

    if args.samples > 0 and args.json_out:
        rows = []
        for j2, _, snaps in results:
            for t in range(snaps.shape[0]):
                rows.append([float(j2)] + [int(v) for v in snaps[t]])
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.json_out, 'w'))
        print(f"saved training json: {args.json_out} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
