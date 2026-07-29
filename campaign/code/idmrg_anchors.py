#!/usr/bin/env python3
"""WS1: iDMRG thermodynamic-limit anchors for TFI / ANNNI / J1J2 (1D, Pauli
conventions as in the manuscript).  These are the true N->infinity values the
paper's 1/N extrapolations must land on.

Per parameter point the model is solved at two bond dimensions (default
128, 256) so finite-entanglement drift is visible in the output.

Outputs one npz per model: params, and per-chi observable arrays.
  tfi  : ZZ(r) for r in --distances, X_mean, e0
  annni: C(r) profile to R, SF_pi, SF_pi_2 (connected), X_mean, e0
  j1j2 : spin_dot(1even/1odd/2), dimer order |sd1e-sd1o|, e0

Run:
  python idmrg_anchors.py --model tfi   --params 0.0:1.0:11 --out idmrg_tfi.npz
  python idmrg_anchors.py --model annni --kappa 0.4 --params 0.0:2.0:21 --out idmrg_annni.npz
  python idmrg_anchors.py --model j1j2  --params 0.0:1.0:11 --out idmrg_j1j2.npz
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RMAX = 200


def parse_grid(spec):
    lo, hi, n = spec.split(':')
    return np.linspace(float(lo), float(hi), int(n))


def idmrg_point(task):
    model_name, p, kappa, chi = task
    from tenpy.algorithms import dmrg
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    class M(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)

        def init_terms(self, mp):
            if model_name == 'tfi':
                g = mp.get('p', 0.5)
                self.add_onsite(-2.0 * g, 0, 'Sx')
                self.add_coupling(-4.0 * (1.0 - g), 0, 'Sz', 0, 'Sz', 1)
            elif model_name == 'annni':
                h = mp.get('p', 0.0)
                kap = mp.get('kappa', 0.4)
                self.add_coupling(-4.0, 0, 'Sz', 0, 'Sz', 1)
                self.add_coupling(4.0 * kap, 0, 'Sz', 0, 'Sz', 2)
                self.add_onsite(-2.0 * h, 0, 'Sx')
            else:
                j2 = mp.get('p', 0.0)
                for dx, J in ((1, 1.0), (2, j2)):
                    if abs(J) < 1e-15:
                        continue
                    self.add_coupling(4.0 * J, 0, 'Sz', 0, 'Sz', dx)
                    self.add_coupling(2.0 * J, 0, 'Sp', 0, 'Sm', dx, plus_hc=True)

    mp = dict(lattice='Chain', L=2, bc_MPS='infinite', p=float(p),
              kappa=float(kappa))
    model = M(mp)
    init = ['up', 'down'] if model_name == 'j1j2' else ['up', 'up']
    psi = MPS.from_product_state(model.lat.mps_sites(), init, bc='infinite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-12},
        'max_E_err': 1e-10, 'max_sweeps': 200,
    })
    e0, psi = eng.run()          # energy per site (infinite)

    out = {'e0': float(np.mean(e0)) if np.ndim(e0) else float(e0)}
    if model_name == 'j1j2':
        # dimerized phase breaks translation: report both bond parities
        Czz = psi.correlation_function('Sz', 'Sz', sites1=[0, 1], sites2=range(0, 4))
        Cpm = psi.correlation_function('Sp', 'Sm', sites1=[0, 1], sites2=range(0, 4))
        Cmp = psi.correlation_function('Sm', 'Sp', sites1=[0, 1], sites2=range(0, 4))

        def sdot(i, j):
            return float(4.0 * Czz[i, j] + 2.0 * np.real(Cpm[i, j] + Cmp[i, j]))
        sd1e, sd1o = sdot(0, 1), sdot(1, 2)
        sd2 = 0.5 * (sdot(0, 2) + sdot(1, 3))
        out.update(sd1_even=sd1e, sd1_odd=sd1o, sd1_mean=0.5 * (sd1e + sd1o),
                   sd2=sd2, dimer_order=abs(sd1e - sd1o))
    else:
        mz = 2.0 * np.asarray(psi.expectation_value('Sz'))
        mx = 2.0 * np.asarray(psi.expectation_value('Sx'))
        C = 4.0 * np.asarray(
            psi.correlation_function('Sz', 'Sz', sites1=[0], sites2=range(0, RMAX))
        )[0]
        conn = C - mz[0] * mz[np.arange(RMAX) % 2]
        out.update(X_mean=float(np.mean(mx)), mz=float(np.mean(np.abs(mz))))
        if model_name == 'tfi':
            out['ZZ'] = [float(C[r]) for r in (1, 2, 3, 10, 20)]
        else:
            rs = np.arange(RMAX)
            out['C_r'] = C.tolist()
            out['SF_pi'] = float(np.real(np.sum(np.exp(1j * np.pi * rs) * conn)))
            out['SF_pi_2'] = float(np.real(np.sum(np.exp(1j * np.pi / 2 * rs) * conn)))
    return p, chi, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['tfi', 'annni', 'j1j2'], required=True)
    ap.add_argument('--params', type=str, required=True)
    ap.add_argument('--kappa', type=float, default=0.4)
    ap.add_argument('--chis', type=int, nargs='+', default=[128, 256])
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '2')
    ps = parse_grid(args.params)
    tasks = [(args.model, float(p), args.kappa, chi)
             for p in ps for chi in args.chis]
    print(f"iDMRG {args.model}: {len(tasks)} points", flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for p, chi, out in pool.map(idmrg_point, tasks):
            res[(p, chi)] = out
            print(f"  p={p:.3f} chi={chi} e0={out['e0']:+.8f}", flush=True)

    save = {'params': ps, 'chis': np.array(args.chis), 'model': args.model,
            'kappa': args.kappa}
    scalar_keys = sorted({k for v in res.values() for k in v
                          if k not in ('C_r', 'ZZ')})
    for chi in args.chis:
        for k in scalar_keys:
            save[f"{k}_chi{chi}"] = np.array(
                [res[(float(p), chi)].get(k, np.nan) for p in ps], dtype=float)
        if args.model == 'annni':
            save[f"C_r_chi{chi}"] = np.array(
                [res[(float(p), chi)]['C_r'] for p in ps])
        if args.model == 'tfi':
            save[f"ZZ_chi{chi}"] = np.array(
                [res[(float(p), chi)]['ZZ'] for p in ps]).T
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
