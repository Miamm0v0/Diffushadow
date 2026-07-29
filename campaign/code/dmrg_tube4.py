#!/usr/bin/env python3
"""2D TFI on the width-4 torus/tube: DMRG references beyond ED, and the
infinite-tube (Ly -> infinity) iDMRG anchor for thermodynamic-limit checks.

The finite mode builds EXACTLY the same Hamiltonian as
generate_tfi2d_dataset.py (row-major flattening, torus bonds via
torus_bonds), so observables are directly comparable to generated data.

Modes:
  finite   : 4 x Ly torus at given sizes (e.g. Ly = 8 10 12), site-averaged
             ZZ for displacements (0,1),(1,0),(0,2)  -> tube_dmrg_ref_4xLY.npz
  infinite : 4 x inf tube (iDMRG, MPS unit cell = 2*Lx sites to allow
             AFM-free ground states), same observables -> tube_idmrg.npz

Run (node 109):
  python tfi2d/dmrg_tube4.py finite  --lys 8 10 12 --gs 0.1 ... --chi 512 --workers 20
  python tfi2d/dmrg_tube4.py infinite --gs 0.1 ... --chis 128 256 --workers 10
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_tfi2d_dataset import torus_bonds  # noqa: E402

LX = int(os.environ.get('TUBE_LX', 4))


def finite_point(task):
    ly, g, chi = task
    from tenpy.algorithms import dmrg
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    n, h1, v1, h2 = torus_bonds(LX, ly)

    class TubeTFI(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)

        def init_terms(self, mp):
            gg = mp.get('g', 0.5)
            self.add_onsite(-2.0 * gg, 0, 'Sx')
            for (i, j) in h1 + v1:
                a, b = (i, j) if i < j else (j, i)
                self.add_coupling_term(-4.0 * (1.0 - gg), a, b, 'Sz', 'Sz')

    mp = dict(lattice='Chain', L=n, bc_MPS='finite', g=float(g))
    model = TubeTFI(mp)
    psi = MPS.from_product_state(model.lat.mps_sites(), ['up'] * n, bc='finite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
        'max_E_err': 1e-9, 'max_sweeps': 40,
    })
    E, psi = eng.run()
    C = 4.0 * np.asarray(psi.correlation_function('Sz', 'Sz'))
    out = {}
    for key, pairs in (('h1', h1), ('v1', v1), ('h2', h2)):
        out[key] = float(np.mean([C[i, j] for i, j in pairs]))
    return ly, float(g), float(E), out


def infinite_point(task):
    g, chi = task
    from tenpy.algorithms import dmrg
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    cell = 2 * LX  # two rings per MPS unit cell

    class TubeTFIInf(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)

        def init_terms(self, mp):
            gg = mp.get('g', 0.5)
            self.add_onsite(-2.0 * gg, 0, 'Sx')
            J = -4.0 * (1.0 - gg)
            for s in range(cell):
                ring, col = divmod(s, LX)
                # ring bond
                jr = ring * LX + (col + 1) % LX
                a, b = sorted((s, jr))
                self.add_coupling_term(J, a, b, 'Sz', 'Sz')
                # axial bond to next ring (crosses unit cell for ring==1)
                ja = s + LX
                self.add_coupling_term(J, s, ja, 'Sz', 'Sz')

    mp = dict(lattice='Chain', L=cell, bc_MPS='infinite', g=float(g))
    model = TubeTFIInf(mp)
    psi = MPS.from_product_state(model.lat.mps_sites(), ['up'] * cell,
                                 bc='infinite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-12},
        'max_E_err': 1e-10, 'max_sweeps': 200,
    })
    e0, psi = eng.run()
    # site-averaged bond correlations from the unit cell
    C = 4.0 * np.asarray(psi.correlation_function(
        'Sz', 'Sz', sites1=range(LX), sites2=range(3 * LX)))
    h1 = float(np.mean([C[i, (i // LX) * LX + ((i % LX) + 1) % LX]
                        for i in range(LX)]))
    v1 = float(np.mean([C[i, i + LX] for i in range(LX)]))
    h2 = float(np.mean([C[i, (i // LX) * LX + ((i % LX) + 2) % LX]
                        for i in range(LX)]))
    e = float(np.mean(e0)) if np.ndim(e0) else float(e0)
    return g, chi, e, {'h1': h1, 'v1': v1, 'h2': h2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['finite', 'infinite'])
    ap.add_argument('--lys', type=int, nargs='+', default=[8, 10, 12])
    ap.add_argument('--gs', type=float, nargs='+', required=True)
    ap.add_argument('--chi', type=int, default=512)
    ap.add_argument('--chis', type=int, nargs='+', default=[128, 256])
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--out-dir', type=str, default='dmrg_refs')
    args = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == 'finite':
        tasks = [(ly, g, args.chi) for ly in args.lys for g in args.gs]
        print(f"finite tube: {len(tasks)} points", flush=True)
        res = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for ly, g, E, o in pool.map(finite_point, tasks):
                res[(ly, g)] = (E, o)
                print(f"  4x{ly} g={g:.3f} E/N={E / (LX * ly):+.6f} "
                      f"h1={o['h1']:+.5f} v1={o['v1']:+.5f}", flush=True)
        for ly in args.lys:
            gs = np.array(sorted(g for (l, g) in res if l == ly))
            np.savez(out / f"tube_dmrg_ref_4x{ly}.npz", gs=gs,
                     E0=np.array([res[(ly, g)][0] for g in gs]),
                     corr_h1=np.array([res[(ly, g)][1]['h1'] for g in gs]),
                     corr_v1=np.array([res[(ly, g)][1]['v1'] for g in gs]),
                     corr_h2=np.array([res[(ly, g)][1]['h2'] for g in gs]),
                     lx=LX, ly=ly, num_qubits=LX * ly, chi=args.chi)
            print(f"saved {out}/tube_dmrg_ref_4x{ly}.npz")
    else:
        tasks = [(float(g), chi) for g in args.gs for chi in args.chis]
        print(f"infinite tube: {len(tasks)} points", flush=True)
        res = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for g, chi, e, o in pool.map(infinite_point, tasks):
                res[(g, chi)] = (e, o)
                print(f"  g={g:.3f} chi={chi} e0={e:+.8f} h1={o['h1']:+.5f}",
                      flush=True)
        gs = np.array(sorted({g for (g, _) in res}))
        save = {'gs': gs, 'chis': np.array(args.chis)}
        for chi in args.chis:
            for k in ('h1', 'v1', 'h2'):
                save[f"corr_{k}_chi{chi}"] = np.array(
                    [res[(g, chi)][1][k] for g in gs])
            save[f"e0_chi{chi}"] = np.array([res[(g, chi)][0] for g in gs])
        np.savez(out / "tube_idmrg.npz", **save)
        print(f"saved {out}/tube_idmrg.npz")


if __name__ == '__main__':
    main()
