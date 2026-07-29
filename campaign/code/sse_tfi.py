#!/usr/bin/env python3
"""Compact SSE QMC for ferromagnetic transverse-field Ising models
(Sandvik, PRE 68, 056701 (2003) scheme), on an arbitrary bond list —
covers the 1D chain and our 2D tori/tubes via the same flattening as the
rest of the pipeline.

  H = -J sum_bonds Z_i Z_j - h sum_i X_i     (J > 0, h > 0; Pauli operators)

Purpose: independent, error-barred cross-check of DMRG/ED references and of
generated observables (diagonal sector: <Z_i Z_j>, E).  Sign-free.
NOTE: uniform-ferromagnetic bonds only (TFI-type); the mixed-sign ANNNI
case needs modified cluster rules and is deliberately NOT implemented.

Always run --selftest first: it gates against ED at N=12 and refuses to
be trusted otherwise.

Usage:
  python sse_tfi.py --selftest
  python sse_tfi.py --geometry chain --N 96 --g 0.3 --beta 192 \
      --sweeps 20000 --bins 20
  python sse_tfi.py --geometry torus --lx 4 --ly 12 --g 0.75 --beta 96 ...
"""
from __future__ import annotations

import argparse

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    raise SystemExit("needs numba: pip install numba")


# operator encoding in the string: -1 = identity;
# 2*b   = diagonal bond op on bond b
# even codes >= 2*NB never used for sites; site ops:
# 2*NB + 2*i     = diagonal site op on site i   (constant h)
# 2*NB + 2*i + 1 = off-diagonal site flip on site i
@njit(cache=True)
def _sweep(spins, opstring, bond_i, bond_j, J, h, beta, rng_state):
    n = spins.shape[0]
    nb = bond_i.shape[0]
    M = opstring.shape[0]
    # count non-identity
    nops = 0
    for p in range(M):
        if opstring[p] != -1:
            nops += 1
    # ---- diagonal update ----
    add_w = beta * (2.0 * J * nb + h * n)   # total diagonal insertion weight
    for p in range(M):
        op = opstring[p]
        if op == -1:
            # try insert: choose bond-diag (weight 2J each) or site-diag (h each)
            r = np.random.random() * (2.0 * J * nb + h * n)
            if r < 2.0 * J * nb:
                b = int(r / (2.0 * J))
                if b >= nb:
                    b = nb - 1
                if spins[bond_i[b]] == spins[bond_j[b]]:
                    # weight 2J (constant shift makes antiparallel weight 0)
                    if np.random.random() * (M - nops) < add_w:
                        opstring[p] = 2 * b
                        nops += 1
            else:
                i = int((r - 2.0 * J * nb) / h)
                if i >= n:
                    i = n - 1
                if np.random.random() * (M - nops) < add_w:
                    opstring[p] = 2 * nb + 2 * i
                    nops += 1
        elif op >= 2 * nb and (op - 2 * nb) % 2 == 1:
            # off-diagonal site op: propagate
            spins[(op - 2 * nb) // 2] *= -1
        else:
            # diagonal op: try remove
            if np.random.random() * add_w < (M - nops + 1):
                opstring[p] = -1
                nops -= 1
    # ---- cluster update (Sandvik's TFI clusters) ----
    # build linked vertex list: each op has 2 legs (site ops) or 4 legs (bond ops)
    # For compactness use the simpler "multibranch" flip: site ops delimit
    # cluster segments on each site line; bond ops weld the two site lines.
    # We implement it via union-find over (site, segment) pieces.
    nseg = np.zeros(n, dtype=np.int64)
    # first pass: count segments per site (segments delimited by site ops)
    for p in range(M):
        op = opstring[p]
        if op >= 2 * nb:
            nseg[(op - 2 * nb) // 2] += 1
    seg_off = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        seg_off[i + 1] = seg_off[i] + nseg[i] + 1
    tot = seg_off[n]
    parent = np.arange(tot, dtype=np.int64)

    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cur = np.zeros(n, dtype=np.int64)   # current segment index per site
    # weld: bond ops join current segments of their two sites; site ops advance
    for p in range(M):
        op = opstring[p]
        if op == -1:
            continue
        if op < 2 * nb:
            b = op // 2
            a = find(seg_off[bond_i[b]] + cur[bond_i[b]], parent)
            c = find(seg_off[bond_j[b]] + cur[bond_j[b]], parent)
            parent[a] = c
        else:
            i = (op - 2 * nb) // 2
            cur[i] += 1
    # periodic closure in imaginary time: last segment == first segment
    for i in range(n):
        a = find(seg_off[i] + cur[i], parent)
        c = find(seg_off[i], parent)
        parent[a] = c
    # flip each cluster with p=1/2; a flip toggles spins of segment 0 pieces
    # and toggles diag<->offdiag of the site ops at flipped-segment boundaries.
    flip = np.zeros(tot, dtype=np.int8)
    for x in range(tot):
        if parent[x] == x:
            flip[x] = 1 if np.random.random() < 0.5 else 0
    cur[:] = 0
    for p in range(M):
        op = opstring[p]
        if op >= 2 * nb and op != -1:
            i = (op - 2 * nb) // 2
            f1 = flip[find(seg_off[i] + cur[i], parent)]
            f2 = flip[find(seg_off[i] + cur[i] + 1, parent)] if cur[i] + 1 <= nseg[i] \
                else flip[find(seg_off[i], parent)]
            if f1 != f2:
                opstring[p] = op ^ 1     # toggle diagonal <-> off-diagonal
            cur[i] += 1
    for i in range(n):
        if flip[find(seg_off[i], parent)] == 1:
            spins[i] *= -1
    return nops


def run_sse(n, bonds, J, h, beta, sweeps, therm, bins, seed, pairs_list):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    spins = np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8)
    bond_i = np.array([b[0] for b in bonds], dtype=np.int64)
    bond_j = np.array([b[1] for b in bonds], dtype=np.int64)
    M = max(64, int(1.3 * beta * (2 * J * len(bonds) + h * n)))
    opstring = np.full(M, -1, dtype=np.int64)

    for _ in range(therm):
        nops = _sweep(spins, opstring, bond_i, bond_j, J, h, beta, 0)
        # grow string if needed
        if nops > 0.8 * opstring.shape[0]:
            extra = np.full(opstring.shape[0] // 2, -1, dtype=np.int64)
            opstring = np.concatenate([opstring, extra])

    per_bin = max(1, sweeps // bins)
    E_bins, C_bins = [], []
    for _b in range(bins):
        e_acc = 0.0
        c_acc = np.zeros(len(pairs_list))
        for _s in range(per_bin):
            nops = _sweep(spins, opstring, bond_i, bond_j, J, h, beta, 0)
            # energy: E = -<nops>/beta + const shift (J*nb + h*n)
            e_acc += -nops / beta + J * len(bonds) + h * n
            for k, pl in enumerate(pairs_list):
                acc = 0.0
                for (i, j) in pl:
                    acc += spins[i] * spins[j]
                c_acc[k] += acc / len(pl)
        E_bins.append(e_acc / per_bin)
        C_bins.append(c_acc / per_bin)
    E_bins = np.array(E_bins)
    C_bins = np.array(C_bins)
    return (E_bins.mean(), E_bins.std() / np.sqrt(bins),
            C_bins.mean(axis=0), C_bins.std(axis=0) / np.sqrt(bins))


def selftest():
    import scipy.sparse.linalg as spla
    N = 12
    g = 0.3
    J, h = (1 - g), g
    dim = 2 ** N
    idx = np.arange(dim, dtype=np.int64)
    z = [1 - 2 * ((idx >> (N - 1 - s)) & 1).astype(np.float64) for s in range(N)]
    diag = np.zeros(dim)
    for i in range(N):
        diag += -J * z[i] * z[(i + 1) % N]

    def mv(v):
        out = diag * v
        for s in range(N):
            out -= h * v.reshape(-1, 2, 1 << (N - 1 - s))[:, ::-1, :].reshape(-1)
        return out
    val, vec = spla.eigsh(spla.LinearOperator((dim, dim), matvec=mv), k=1,
                          which='SA', maxiter=8000)
    psi = vec[:, 0]
    p = psi * psi
    zz_ed = np.mean([np.dot(p, z[i] * z[(i + 1) % N]) for i in range(N)])
    e_ed = float(val[0])

    bonds = [(i, (i + 1) % N) for i in range(N)]
    pairs = [bonds]
    E, dE, C, dC = run_sse(N, bonds, J, h, beta=48.0, sweeps=6000, therm=2000,
                           bins=12, seed=7, pairs_list=pairs)
    print(f"  ED : E={e_ed:.6f}  ZZ1={zz_ed:.6f}")
    print(f"  SSE: E={E:.4f}({dE:.4f})  ZZ1={C[0]:.4f}({dC[0]:.4f})")
    ok = abs(E - e_ed) < max(4 * dE, 0.05) and abs(C[0] - zz_ed) < max(4 * dC[0], 0.02)
    print("SSE SELFTEST", "PASS" if ok else "FAIL — DO NOT USE RESULTS")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--geometry', choices=['chain', 'torus'], default='chain')
    ap.add_argument('--N', type=int, default=32)
    ap.add_argument('--lx', type=int, default=4)
    ap.add_argument('--ly', type=int, default=8)
    ap.add_argument('--g', type=float, required=False, default=0.3)
    ap.add_argument('--beta', type=float, default=64.0)
    ap.add_argument('--sweeps', type=int, default=20000)
    ap.add_argument('--therm', type=int, default=4000)
    ap.add_argument('--bins', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    J, h = (1 - args.g), args.g
    if args.geometry == 'chain':
        n = args.N
        bonds = [(i, (i + 1) % n) for i in range(n)]
        pairs = [bonds,
                 [(i, (i + 2) % n) for i in range(n)],
                 [(i, (i + 3) % n) for i in range(n)]]
        labels = ['ZZ(1)', 'ZZ(2)', 'ZZ(3)']
    else:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate_tfi2d_dataset import torus_bonds
        n, h1, v1, h2 = torus_bonds(args.lx, args.ly)
        bonds = h1 + v1
        pairs = [h1, v1, h2]
        labels = ['h1', 'v1', 'h2']

    E, dE, C, dC = run_sse(n, bonds, J, h, args.beta, args.sweeps, args.therm,
                           args.bins, args.seed, pairs)
    print(f"N={n} g={args.g} beta={args.beta}: E/N = {E / n:.6f} ({dE / n:.6f})")
    for lab, c, dc in zip(labels, C, dC):
        print(f"  {lab} = {c:+.5f} ({dc:.5f})")


if __name__ == '__main__':
    main()
