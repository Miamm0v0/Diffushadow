#!/usr/bin/env python3
"""WS1/WS2: TeNPy DMRG references for the 1D J1-J2 and ANNNI chains (PBC),
Pauli conventions exactly as in the manuscript:

  J1J2 : H = J1 sum (sig.sig)_{i,i+1} + J2 sum (sig.sig)_{i,i+2}
  ANNNI: H = -J1 sum sz sz_{i+1} + kappa*J1 sum sz sz_{i+2} - h sum sx

Outputs per size:
  j1j2 : params(=J2), E0, spin_dot[r,pk] for r=1,2,3, dimer_proxy
  annni: params(=h),  E0, ZZ[r,pk] r=1,2,3, X_mean, SF_pi, SF_pi_2

--selftest runs N=12 DMRG vs matrix-free ED for both models.

Run:
  python dmrg_validate_1d.py --model j1j2 --sizes 32 48 64 72 96 \
      --params 0.0:1.0:11 --chi 512 --workers 30 --out-dir dmrg_refs
  python dmrg_validate_1d.py --model annni --kappa 0.4 --sizes 32 48 64 72 96 \
      --params 0.0:2.0:21 --chi 256 --workers 30 --out-dir dmrg_refs
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def parse_grid(spec):
    if ':' in spec:
        lo, hi, n = spec.split(':')
        return np.linspace(float(lo), float(hi), int(n))
    return np.array([float(spec)])


def dmrg_point(task):
    model_name, N, p, kappa, chi = task
    from tenpy.algorithms import dmrg
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite

    class M(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)

        def init_terms(self, mp):
            if model_name == 'j1j2':
                j2 = mp.get('p', 0.0)
                for dx, J in ((1, 1.0), (2, j2)):
                    if abs(J) < 1e-15:
                        continue
                    self.add_coupling(4.0 * J, 0, 'Sz', 0, 'Sz', dx)
                    self.add_coupling(2.0 * J, 0, 'Sp', 0, 'Sm', dx, plus_hc=True)
            else:  # annni
                h = mp.get('p', 0.0)
                kap = mp.get('kappa', 0.4)
                self.add_coupling(-4.0, 0, 'Sz', 0, 'Sz', 1)
                self.add_coupling(4.0 * kap, 0, 'Sz', 0, 'Sz', 2)
                self.add_onsite(-2.0 * h, 0, 'Sx')

    mp = dict(lattice='Chain', L=N, bc_x='periodic', bc_MPS='finite',
              p=float(p), kappa=float(kappa))
    model = M(mp)
    init = (['up', 'down'] * N)[:N] if model_name == 'j1j2' else ['up'] * N
    psi = MPS.from_product_state(model.lat.mps_sites(), init, bc='finite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
        'max_E_err': 1e-9, 'max_sweeps': 40,
    })
    E, psi = eng.run()

    Czz = 4.0 * np.asarray(psi.correlation_function('Sz', 'Sz'))
    out = {}
    if model_name == 'j1j2':
        Cpm = np.asarray(psi.correlation_function('Sp', 'Sm'))
        Cmp = np.asarray(psi.correlation_function('Sm', 'Sp'))
        Cdot = Czz + 2.0 * np.real(Cpm + Cmp)
        sd = [float(np.mean([Cdot[i, (i + r) % N] for i in range(N)]))
              for r in (1, 2, 3)]
        out['spin_dot'] = sd
        out['dimer_proxy'] = sd[0] - sd[1]
    else:
        zz = [float(np.mean([Czz[i, (i + r) % N] for i in range(N)]))
              for r in (1, 2, 3)]
        cbar = np.array([np.mean([Czz[i, (i + r) % N] for i in range(N)])
                         for r in range(N)])
        out['ZZ'] = zz
        out['C_r'] = cbar.tolist()  # full site-averaged profile: disorder-line
        # (Peschel-Emery) analysis needs the oscillation onset in C(r)
        out['SF_pi'] = float(np.real(np.sum(np.exp(1j * np.pi * np.arange(N)) * cbar)))
        out['SF_pi_2'] = float(np.real(np.sum(np.exp(1j * np.pi / 2 * np.arange(N)) * cbar)))
        out['X_mean'] = float(np.mean(2.0 * psi.expectation_value('Sx')))
    return N, float(p), float(E), out


def selftest():
    import scipy.sparse.linalg as spla
    N = 12
    dim = 2 ** N
    idx = np.arange(dim, dtype=np.int64)
    z = [1 - 2 * ((idx >> (N - 1 - s)) & 1).astype(np.float64) for s in range(N)]

    def flip2(v, i, j):
        # reshape-free double flip via index xor
        return v[idx ^ ((1 << (N - 1 - i)) | (1 << (N - 1 - j)))]

    def ed_j1j2(j2):
        diag = np.zeros(dim)
        for dist, J in ((1, 1.0), (2, j2)):
            for i in range(N):
                diag += J * z[i] * z[(i + dist) % N]

        def mv(v):
            out = diag * v
            for dist, J in ((1, 1.0), (2, j2)):
                for i in range(N):
                    j = (i + dist) % N
                    out += J * (1.0 - z[i] * z[j]) * flip2(v, i, j)
            return out
        val, vec = spla.eigsh(spla.LinearOperator((dim, dim), matvec=mv),
                              k=1, which='SA', maxiter=8000)
        psi = vec[:, 0]
        p = psi * psi
        sd1 = np.mean([np.dot(p, z[i] * z[(i + 1) % N]) for i in range(N)])
        # full spin-dot via <H_nn>: easier to just compare energy + zz part
        return float(val[0]), float(sd1)

    _, _, E_dm, out = dmrg_point(('j1j2', N, 0.5, 0.0, 256))
    E_ed, _ = ed_j1j2(0.5)
    print(f"  j1j2 J2=0.5: E_ed={E_ed:.8f} E_dmrg={E_dm:.8f} |dE|={abs(E_ed-E_dm):.2e}")
    print(f"    spin_dot(1)={out['spin_dot'][0]:+.6f} (MG analytic -1.5, finite-N ~ -1.51)")
    ok = abs(E_ed - E_dm) < 1e-6 and abs(E_ed / N + 1.5) < 1e-9

    def ed_annni(h, kap):
        diag = np.zeros(dim)
        for i in range(N):
            diag += -z[i] * z[(i + 1) % N] + kap * z[i] * z[(i + 2) % N]

        def mv(v):
            out = diag * v
            for s in range(N):
                out -= h * v.reshape(-1, 2, 1 << (N - 1 - s))[:, ::-1, :].reshape(-1)
            return out
        val, _ = spla.eigsh(spla.LinearOperator((dim, dim), matvec=mv),
                            k=1, which='SA', maxiter=8000)
        return float(val[0])

    _, _, E_dm2, out2 = dmrg_point(('annni', N, 0.6, 0.4, 256))
    E_ed2 = ed_annni(0.6, 0.4)
    print(f"  annni h=0.6 k=0.4: E_ed={E_ed2:.8f} E_dmrg={E_dm2:.8f} "
          f"|dE|={abs(E_ed2-E_dm2):.2e}  SF_pi={out2['SF_pi']:.4f}")
    ok &= abs(E_ed2 - E_dm2) < 1e-6
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['j1j2', 'annni'], required=False)
    ap.add_argument('--sizes', type=int, nargs='+', default=[32, 48, 64, 72, 96])
    ap.add_argument('--params', type=str, default='0.0:1.0:11')
    ap.add_argument('--kappa', type=float, default=0.4)
    ap.add_argument('--chi', type=int, default=256)
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--out-dir', type=str, default='dmrg_refs')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not args.model:
        raise SystemExit('--model required')

    os.environ.setdefault('OMP_NUM_THREADS', '2')
    ps = parse_grid(args.params)
    tasks = [(args.model, N, float(p), args.kappa, args.chi)
             for N in args.sizes for p in ps]
    print(f"{args.model}: {len(tasks)} DMRG points, chi={args.chi}, "
          f"workers={args.workers}", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for N, p, E, out in pool.map(dmrg_point, tasks):
            results[(N, p)] = (E, out)
            key = 'spin_dot' if args.model == 'j1j2' else 'ZZ'
            print(f"  N={N} p={p:.3f} E/N={E / N:+.6f} {key}1={out[key][0]:+.5f}",
                  flush=True)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for N in args.sizes:
        grid = np.array(sorted(p for (n, p) in results if n == N))
        E = np.array([results[(N, p)][0] for p in grid])
        save = dict(params=grid, E0=E, N=N, chi=args.chi, model=args.model)
        if args.model == 'j1j2':
            save['spin_dot'] = np.array(
                [results[(N, p)][1]['spin_dot'] for p in grid]).T
            save['dimer_proxy'] = np.array(
                [results[(N, p)][1]['dimer_proxy'] for p in grid])
        else:
            save['kappa'] = args.kappa
            for k in ('SF_pi', 'SF_pi_2', 'X_mean'):
                save[k] = np.array([results[(N, p)][1][k] for p in grid])
            save['ZZ'] = np.array([results[(N, p)][1]['ZZ'] for p in grid]).T
            save['C_r'] = np.array([results[(N, p)][1]['C_r'] for p in grid])
        fn = outdir / f"{args.model}_dmrg_ref_N{N}.npz"
        np.savez(fn, **save)
        print(f"saved {fn}")


if __name__ == '__main__':
    main()
