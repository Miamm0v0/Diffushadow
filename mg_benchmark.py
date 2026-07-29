#!/usr/bin/env python3
r"""
Majumdar-Ghosh exact-dimer benchmark for Diffushadow (J1-J2 chain).

WHY THIS EXISTS
---------------
At the Majumdar-Ghosh (MG) point J2/J1 = 1/2, the ground state of the 1D
J1-J2 Heisenberg chain (periodic, even N) is *exactly* the nearest-neighbour
singlet-dimer product state -- known in closed form, for every N. That gives a
nontrivial, size-independent, analytic ground-truth for the exact same
spin-dot observable the manuscript already reports, and it is the frustrated
analogue of the exact-TFI benchmark. Passing it is strong evidence that
Diffushadow reproduces *frustration* physics, not just smooth correlations.

Conventions match eval_cal_exact.py:
  H = J1 * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    + J2 * sum_i (X_i X_{i+2} + Y_i Y_{i+2} + Z_i Z_{i+2})            (PBC)
  spin_dot(r) = < X_i X_{i+r} + Y_i Y_{i+r} + Z_i Z_{i+r} >, site-averaged,
                with Pauli operators (sigma), NOT spin-1/2 S.

ANALYTIC MG VALUES (Pauli convention, translationally averaged, J1=1, J2=0.5):
  <sigma_i . sigma_{i+1}> = -3/2      (half the bonds are singlets: sigma.sigma=-3)
  <sigma_i . sigma_{i+r}> =  0   for r >= 2
  E0 / N                  = -3/2
  dimer_proxy = sd(1) - sd(2) = -3/2  (its maximal magnitude over J2)

USAGE
-----
  # 1) self-check: exact ED reproduces the analytic MG state
  python mg_benchmark.py --num_qubits 12

  # 2) grade a Diffushadow run (npz written by eval_new.py, --predict_model J1J2)
  python mg_benchmark.py --num_qubits 12 --gen_npz results/j1j2_eval_N12.npz

Only needs numpy + scipy (already in requirements.txt).
"""
from __future__ import annotations

import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MG_J2_OVER_J1 = 0.5

# analytic MG references (Pauli convention), independent of N (even N, PBC)
ANALYTIC_SPIN_DOT = {1: -1.5, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
ANALYTIC_ENERGY_PER_SITE = -1.5
ANALYTIC_DIMER_PROXY = -1.5  # sd(1) - sd(2)

_I = sp.identity(2, format="csr", dtype=np.float64)
_X = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
_Y_IM = sp.csr_matrix(np.array([[0.0, -1.0], [1.0, 0.0]]))  # Y = i * this; YY = -(this (x) this)
_Z = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))


def _op_at(n, sites_ops):
    """Kronecker product over n qubits; sites_ops maps site->2x2 sparse op."""
    out = None
    for q in range(n):
        m = sites_ops.get(q, _I)
        out = m if out is None else sp.kron(out, m, format="csr")
    return out


def _two_site(n, i, j, a, b):
    return _op_at(n, {i: a, j: b})


def pure_dimer_covering(n):
    r"""Exact MG eigenstate: product of singlets on bonds (0,1),(2,3),...,(n-2,n-1).

    Singlet = (|01> - |10>)/sqrt(2). This single covering is an exact ground
    state at the MG point and yields the clean analytic correlations
    <sigma.sigma>(1) = -3/2, <sigma.sigma>(r>=2) = 0 with NO finite-size
    correction -- it is the right object to state as the analytic anchor.
    """
    singlet = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0)  # basis |00>,|01>,|10>,|11>
    psi = None
    for _ in range(n // 2):
        psi = singlet if psi is None else np.kron(psi, singlet)
    return psi


def spin_dot_of_state(n, psi, max_distance=5):
    """Site-averaged <sigma_i . sigma_{i+r}> for an arbitrary state vector."""
    psi = psi / np.linalg.norm(psi)
    sd = {}
    for r in range(1, max_distance + 1):
        acc = 0.0
        for i in range(n):
            j = (i + r) % n
            for a in (_X, _Z):
                op = _two_site(n, i, j, a, a)
                acc += float(psi @ (op @ psi))
            yy = -_two_site(n, i, j, _Y_IM, _Y_IM)
            acc += float(psi @ (yy @ psi))
        sd[r] = acc / n
    return sd


def build_j1j2(n, j1, j2):
    """Sparse J1-J2 Hamiltonian, PBC, Pauli convention (matches eval_cal_exact)."""
    dim = 2 ** n
    h = sp.csr_matrix((dim, dim), dtype=np.float64)
    for dist, coup in ((1, j1), (2, j2)):
        for i in range(n):
            j = (i + dist) % n
            xx = _two_site(n, i, j, _X, _X)
            yy = -_two_site(n, i, j, _Y_IM, _Y_IM)  # Y_i Y_j = -(iY')_i (iY')_j... => real, sign fixed
            zz = _two_site(n, i, j, _Z, _Z)
            h = h + coup * (xx + yy + zz)
    return h.tocsr()


def exact_spin_dot(n, j1=1.0, j2=MG_J2_OVER_J1, max_distance=5):
    """Ground-state site-averaged <sigma_i . sigma_{i+r}> for r=1..max_distance."""
    h = build_j1j2(n, j1, j2)
    # MG point is 2-fold degenerate (PBC, even N); averaged correlations are
    # identical for either dimer covering, so k=2 + pick lowest is safe.
    k = 2 if abs(j2 / j1 - MG_J2_OVER_J1) < 1e-9 else 1
    vals, vecs = spla.eigsh(h, k=max(k, 1), which="SA")
    order = np.argsort(vals)
    e0 = float(vals[order[0]])
    psi = vecs[:, order[0]]
    psi = psi / np.linalg.norm(psi)

    return e0, e0 / n, spin_dot_of_state(n, psi, max_distance)


def _load_generated(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)
    j2s = None
    for cand in ("J2s", "J2_values", "params", "gs"):
        if cand in keys:
            j2s = np.asarray(d[cand], dtype=float)
            break
    if j2s is None:
        raise KeyError(f"no J2 axis in {npz_path}; keys={sorted(keys)}")
    sd_key = next((c for c in ("corr_spin_dot_mean", "correlations_spin_dot",
                               "spin_dot_mean") if c in keys), None)
    if sd_key is None:
        raise KeyError(f"no spin_dot array in {npz_path}; keys={sorted(keys)}")
    sd = np.asarray(d[sd_key], dtype=float)
    # orient so axis-1 is the J2 axis
    if sd.ndim == 2 and sd.shape[1] != len(j2s) and sd.shape[0] == len(j2s):
        sd = sd.T
    return j2s, sd, d


def _fmt(x):
    return f"{x:+.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_qubits", type=int, default=12)
    ap.add_argument("--J1", type=float, default=1.0)
    ap.add_argument("--max_distance", type=int, default=5)
    ap.add_argument("--gen_npz", type=str, default=None,
                    help="Diffushadow J1J2 results npz (from eval_new.py)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="pass/fail |error| threshold on spin_dot")
    args = ap.parse_args()

    N = args.num_qubits
    if N % 2:
        raise SystemExit("MG dimer state requires even N (periodic ring).")

    print(f"\n=== Majumdar-Ghosh benchmark  (N={N}, J1={args.J1}, J2={args.J1*MG_J2_OVER_J1}) ===")
    e0, e_per_site, sd = exact_spin_dot(N, args.J1, args.J1 * MG_J2_OVER_J1, args.max_distance)
    sd_pure = spin_dot_of_state(N, pure_dimer_covering(N), args.max_distance)

    print("\n[1] MG signatures (this is the finite-N ground truth to grade the model against)")
    # Energy is the definitive, EXACT MG signature -- holds at any even N.
    de = abs(e_per_site - ANALYTIC_ENERGY_PER_SITE)
    print(f"    E0/N = {_fmt(e_per_site)}  (analytic MG = {_fmt(ANALYTIC_ENERGY_PER_SITE)}, "
          f"|dE|={de:.2e})  -> {'EXACT MATCH' if de < 1e-6 else 'MISMATCH'}")
    print("    spin_dot  sigma_i.sigma_{i+r}:")
    print("    r   ED(finite-N)  pure-covering  N->inf analytic")
    ok_ops = True
    for r in range(1, args.max_distance + 1):
        an = ANALYTIC_SPIN_DOT.get(r, 0.0)
        ok_ops &= abs(sd_pure[r] - an) < 1e-6      # validates operator code exactly
        print(f"    {r}  {_fmt(sd[r])}       {_fmt(sd_pure[r])}        {_fmt(an)}")
    fs = 3.0 * (0.5 ** (N / 2))  # size of the O(2^-N/2) degeneracy correction
    print(f"    (finite-N ED deviates from the analytic anchor by O(2^-N/2) ~ {fs:.3f}; "
          f"pure single covering is exact -3/2, 0.)")
    print(f"    -> energy exact + operators validated: "
          f"{'PASS' if (de < 1e-6 and ok_ops) else 'FAIL'}")

    if not args.gen_npz:
        print("\n(no --gen_npz given; skipping model grading)")
        print("Anchor for the manuscript: at J2/J1=0.5 the model MUST give "
              "sd(1)~-1.5 and sd(r>=2)~0.\n")
        return

    print("\n[2] Diffushadow-generated vs exact at the MG point")
    # NOTE: the PBC ground state at the MG point is 2-fold degenerate, and a
    # Lanczos vector is an ARBITRARY superposition of the two dimer coverings;
    # its site-averaged correlations contain run-dependent interference terms.
    # The stable, physically meaningful reference is the ANALYTIC anchor
    # (equivalently, the symmetrized mixture of the two coverings), so the
    # pass/fail grade below is taken against the analytic values.
    j2s, gsd, _ = _load_generated(args.gen_npz)
    idx = int(np.argmin(np.abs(j2s - args.J1 * MG_J2_OVER_J1)))
    print(f"    matched generated J2={j2s[idx]:.4f} (target {args.J1*MG_J2_OVER_J1})")
    n_dist = gsd.shape[0]
    ok_gen = True
    print("    r   generated  ED(arb. mix)  analytic   |gen-analytic|")
    for r in range(1, min(args.max_distance, n_dist) + 1):
        g = float(gsd[r - 1, idx])
        ref_ex = sd[r]
        ref_an = ANALYTIC_SPIN_DOT.get(r, 0.0)
        err = abs(g - ref_an)
        ok_gen &= err < args.tol
        print(f"    {r}  {_fmt(g)}   {_fmt(ref_ex)}   {_fmt(ref_an)}   {err:.3f}")
    # dimer signature check: sd(1) strongly negative, sd(2) near zero
    sig = float(gsd[0, idx]) < -1.0 and abs(float(gsd[1, idx])) < 0.3
    print(f"    dimer signature [sd(1)<-1.0 and |sd(2)|<0.3]: {'YES' if sig else 'NO'}")
    print(f"    -> generation matches MG within tol={args.tol}: "
          f"{'PASS' if ok_gen else 'FAIL'}\n")


if __name__ == "__main__":
    main()
