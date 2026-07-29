#!/usr/bin/env python3
"""WS1: TeNPy DMRG reference values for the 1D TFI chain (PBC), at the
system sizes reached by the recursive scale extrapolation (N = 32..108),
where ED is impossible.  Convention matches the manuscript:

    H(g) = -(1-g) sum_i sigma^z_i sigma^z_{i+1} - g sum_i sigma^x_i   (PBC)

Outputs one npz per size: gs, E0, X_mean, ZZ[r_index, g_index]
(site-averaged <sigma^z_i sigma^z_{i+r}>, periodic).

  --selftest : N=12 DMRG vs dense ED (must agree to ~1e-8) before any
               production run is trusted.

Run (node 109):
  /data/diffushadow_2d/tenpy_env/bin/python tfi2d/dmrg_validate_tfi.py \
      --sizes 32 48 64 72 96 108 --hs-from <student npz with h_values_true> \
      --distances 1 2 3 10 20 --chi 256 --workers 40 --out-dir dmrg_refs
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def dmrg_point(task):
    N, g, distances, chi = task
    # imports inside worker (tenpy is heavy; keeps failures per-task)
    from tenpy.algorithms import dmrg
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    class TFIPauli(CouplingMPOModel):
        def init_sites(self, model_params):
            return SpinHalfSite(conserve=None)

        def init_terms(self, model_params):
            gg = model_params.get('g', 0.5)
            # sigma^z = 2 Sz, sigma^x = 2 Sx
            self.add_onsite(-2.0 * gg, 0, 'Sx')
            self.add_coupling(-4.0 * (1.0 - gg), 0, 'Sz', 0, 'Sz', 1)

    mp = dict(lattice='Chain', L=N, bc_x='periodic', bc_MPS='finite', g=g)
    model = TFIPauli(mp)
    psi = MPS.from_product_state(model.lat.mps_sites(), ['up'] * N, bc='finite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
        'max_E_err': 1e-9,
        'max_sweeps': 40,
    })
    E, psi = eng.run()
    C = psi.correlation_function('Sz', 'Sz')  # N x N of <Sz_i Sz_j>
    zz = []
    for r in distances:
        vals = [4.0 * C[i, (i + r) % N] for i in range(N)]
        zz.append(float(np.mean(vals)))
    x = float(np.mean(2.0 * psi.expectation_value('Sx')))
    return N, g, float(E), x, zz


def selftest():
    """N=12 dense ED vs DMRG for a few g."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    N = 12
    dim = 2 ** N
    idx = np.arange(dim, dtype=np.int64)
    zbits = [1 - 2 * ((idx >> (N - 1 - s)) & 1).astype(np.float64) for s in range(N)]

    def ed(g):
        diag = np.zeros(dim)
        for i in range(N):
            diag += -(1 - g) * zbits[i] * zbits[(i + 1) % N]

        def mv(v):
            out = diag * v
            for s in range(N):
                out -= g * v.reshape(-1, 2, 1 << (N - 1 - s))[:, ::-1, :].reshape(-1)
            return out

        op = spla.LinearOperator((dim, dim), matvec=mv)
        val, vec = spla.eigsh(op, k=1, which='SA',
                              v0=np.full(dim, dim ** -0.5), maxiter=5000)
        psi = vec[:, 0]
        p = psi * psi
        zz1 = np.mean([np.dot(p, zbits[i] * zbits[(i + 1) % N]) for i in range(N)])
        return float(val[0]), float(zz1)

    ok = True
    for g in (0.3, 0.5, 0.9):
        e_ed, zz_ed = ed(g)
        _, _, e_dm, _, zz_dm = dmrg_point((N, g, [1], 128))
        de, dz = abs(e_ed - e_dm), abs(zz_ed - zz_dm[0])
        ok &= de < 1e-7 and dz < 1e-7
        print(f"  selftest g={g}: |dE|={de:.2e} |dZZ1|={dz:.2e}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+', default=[32, 48, 64, 72, 96])
    ap.add_argument('--hs', type=float, nargs='*', default=None)
    ap.add_argument('--hs-from', type=str, default='',
                    help='npz with h_values_true to copy the g grid from')
    ap.add_argument('--distances', type=int, nargs='+', default=[1, 2, 3, 10, 20])
    ap.add_argument('--chi', type=int, default=256)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--out-dir', type=str, default='dmrg_refs')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    if args.hs_from:
        hs = np.asarray(np.load(args.hs_from, allow_pickle=True)['h_values_true'],
                        dtype=float).ravel()
    elif args.hs:
        hs = np.asarray(args.hs, dtype=float)
    else:
        raise SystemExit('need --hs or --hs-from')

    os.environ.setdefault('OMP_NUM_THREADS', '2')
    tasks = [(N, float(g), args.distances, args.chi) for N in args.sizes for g in hs]
    print(f"{len(tasks)} DMRG points ({len(args.sizes)} sizes x {len(hs)} g), "
          f"chi={args.chi}, workers={args.workers}", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for N, g, E, x, zz in pool.map(dmrg_point, tasks):
            results[(N, g)] = (E, x, zz)
            print(f"  N={N} g={g:.4f} E/N={E / N:+.6f} ZZ1={zz[0]:+.5f}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for N in args.sizes:
        gs = np.array(sorted(g for (n, g) in results if n == N))
        E = np.array([results[(N, g)][0] for g in gs])
        X = np.array([results[(N, g)][1] for g in gs])
        ZZ = np.array([results[(N, g)][2] for g in gs]).T  # [n_dist, n_g]
        np.savez(out / f"tfi_dmrg_ref_N{N}.npz", gs=gs, E0=E, X_mean=X,
                 ZZ=ZZ, distances=np.array(args.distances), chi=args.chi)
        print(f"saved {out}/tfi_dmrg_ref_N{N}.npz")


if __name__ == '__main__':
    main()
