#!/usr/bin/env python3
"""Exact Bayes bound for atom-loss completion on the Rydberg chain.

For every shot, condition the DMRG ground state on the atoms that survived and
read the exact conditional marginal at each lost site:

    p_j = <psi| P_obs  n_j  P_obs |psi> / <psi| P_obs |psi>

No method can beat  E[max(p, 1-p)]  at reconstructing a lost atom, or
E[p^2 + (1-p)^2]  when sampling, given the same information.  Both are
properties of the state, not of any algorithm.

Validation: with no sites observed the conditional marginals must reproduce the
DMRG density profile; the script refuses to write results unless they do.

Usage (on the cluster, tenpy env):
  python tfi2d/ryd_bayes_bound.py --masks results_ryd/ryd_paired_sample.npz \
      --out results_ryd/ryd_bayes_bound.npz --chi_bound 48
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def ground_state(n, rb, delta, chi, max_range):
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
            self.add_onsite(1.0, 0, 'Sx')
            self.add_onsite(-d, 0, 'Sz')
            for r in range(1, rng + 1):
                v = rb6 / r ** 6
                self.add_coupling(v, 0, 'Sz', 0, 'Sz', r)
                self.add_onsite(v, 0, 'Sz')

    model = RydbergChain(dict(lattice='Chain', L=n, bc_x='open', bc_MPS='finite',
                              delta=float(delta), rb=float(rb), max_range=int(max_range)))
    psi = MPS.from_product_state(model.lat.mps_sites(), ['down'] * n, bc='finite')
    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'mixer': True, 'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
        'max_E_err': 1e-9, 'max_sweeps': 40})
    eng.run()
    psi.canonical_form()
    return psi


def mps_tensors(psi, chi_bound):
    """Right-canonical tensors as numpy arrays [vL, p, vR], optionally compressed."""
    if chi_bound and max(psi.chi) > chi_bound:
        psi.compress_svd({'trunc_params': {'chi_max': chi_bound, 'svd_min': 1e-10}})
        psi.canonical_form()
    return [psi.get_B(i, form='B').to_ndarray() for i in range(psi.L)]


def conditional_marginals(T, occ, hidden):
    """Exact p(n_j = 1 | observed atoms) for every hidden site, batched over shots.

    T: list of [vL, p, vR]; occ: [S, N] occupations; hidden: [S, N] bool.
    Physical index convention: p = 0 is occupied (n = 1), p = 1 is empty.
    """
    S, N = occ.shape
    allow = np.empty((S, N, 2))                      # which physical values each site may take
    allow[:, :, 0] = np.where(hidden, 1.0, (occ == 1).astype(float))
    allow[:, :, 1] = np.where(hidden, 1.0, (occ == 0).astype(float))

    def sweep(order):
        envs, E = {}, np.ones((S, 1, 1))
        for i in order:
            envs[i] = E
            A = T[i]
            nxt = 0.0
            for p in (0, 1):
                w = allow[:, i, p]
                if not w.any():
                    continue
                Ap = A[:, p, :]
                if order[0] == 0:
                    nxt = nxt + w[:, None, None] * np.einsum('sab,ax,by->sxy', E, Ap, Ap.conj())
                else:
                    nxt = nxt + w[:, None, None] * np.einsum('sxy,ax,by->sab', E, Ap, Ap.conj())
            E = nxt
        return envs, E

    L, total = sweep(list(range(N)))
    R, _ = sweep(list(range(N - 1, -1, -1)))
    out = np.full((S, N), np.nan)
    for j in range(N):
        A = T[j]
        num = np.einsum('sab,ax,by,sxy->s', L[j], A[:, 0, :], A[:, 0, :].conj(), R[j])
        den = num + np.einsum('sab,ax,by,sxy->s', L[j], A[:, 1, :], A[:, 1, :].conj(), R[j])
        with np.errstate(invalid='ignore', divide='ignore'):
            out[:, j] = np.real(num) / np.real(den)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks', required=True, help='npz with gal_exact / gal_hidden per delta')
    ap.add_argument('--out', required=True)
    ap.add_argument('--N', type=int, default=32)
    ap.add_argument('--rb', type=float, default=2.35)
    ap.add_argument('--chi', type=int, default=128)
    ap.add_argument('--chi_bound', type=int, default=48)
    ap.add_argument('--max_range', type=int, default=5)
    ap.add_argument('--shots', type=int, default=1000)
    args = ap.parse_args()

    src = np.load(args.masks)
    deltas = [float(x) for x in src['deltas']]
    fracs = [float(f) for f in src['loss_fracs']]
    out = {'deltas': np.array(deltas), 'loss_fracs': np.array(fracs)}

    for dl in deltas:
        psi = ground_state(args.N, args.rb, dl, args.chi, args.max_range)
        dens_dmrg = np.asarray(psi.expectation_value('Sz')) + 0.5
        T = mps_tensors(psi, args.chi_bound)

        # validation: observing nothing must reproduce the DMRG density profile
        none_hidden = np.ones((4, args.N), dtype=bool)
        chk = conditional_marginals(T, np.zeros((4, args.N)), none_hidden)[0]
        err = float(np.max(np.abs(chk - dens_dmrg)))
        print(f"delta={dl:+.2f}  chi={max(psi.chi)}  validation |p - n_DMRG|max = {err:.2e}", flush=True)
        if err > 5e-3:
            raise SystemExit("validation FAILED — conditional marginals do not reproduce the density")

        for f in fracs:
            occ = np.asarray(src[f'd{dl}_f{f}_gal_exact'])[:args.shots]
            hid = np.asarray(src[f'd{dl}_f{f}_gal_hidden'])[:args.shots].astype(bool)
            p = conditional_marginals(T, occ, hid)
            out[f'd{dl}_f{f}_p'] = p
            out[f'd{dl}_f{f}_hidden'] = hid
            out[f'd{dl}_f{f}_exact'] = occ
            ph = p[hid]
            print(f"   f={f}: reconstruction ceiling {100*np.mean(np.maximum(ph, 1-ph)):.2f}%  "
                  f"sampling ceiling {100*np.mean(ph**2 + (1-ph)**2):.2f}%", flush=True)
        out[f'd{dl}_density'] = dens_dmrg

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print("saved", args.out)


if __name__ == '__main__':
    main()
