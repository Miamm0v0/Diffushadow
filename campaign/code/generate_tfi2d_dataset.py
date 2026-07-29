#!/usr/bin/env python3
"""
2D transverse-field Ising (TFI) dataset generator for Diffushadow.

Geometry: Lx x Ly square lattice on a torus (PBC both directions),
row-major site ordering  s = row*Lx + col.  With fixed width Lx and
growing Ly, a vertical bond is always Lx sites away in the ordering,
i.e. a constant token offset of 2*Lx in the oseq layout -- the
transferable invariant that lets the 1D-trained architecture be reused
in 2D without modification.

Hamiltonian (paper convention, interpolation parameter g in [0,1]):

    H(g) = -(1-g) * sum_<ij> Z_i Z_j  -  g * sum_i X_i        (torus)

For the infinite square lattice the critical point (h/J)_c ~= 3.04438
maps to g_c = 3.04438/4.04438 ~= 0.7527.

Outputs (same conventions as generate_j1j2_annni_dataset.py):
  * training JSON rows [g, P1, b1, ..., PN, bN], P in {2:X,3:Y,4:Z}, b in {0,1}
  * exact-cache npz per size: gs, E0, e0_per_site, corr_h1, corr_v1, corr_h2
    (site-averaged <Z_i Z_j> for displacements (0,1), (1,0), (0,2))

Ground states come from matrix-free Lanczos (scipy LinearOperator):
diagonal ZZ part precomputed once per lattice, X part via bit-flip
reshapes.  24 qubits (4x6) needs ~3 GB per worker.

Usage:
  # training data (shadows + exact cache)
  python tfi2d/generate_tfi2d_dataset.py --lx 4 --ly 3 \
      --gs 0.05 0.125 0.2 0.275 0.35 0.425 0.5 0.575 0.65 0.725 0.8 0.875 0.95 \
      --samples-per-g 3000 --json-out data2d/tfi2d_4x3_train.json \
      --exact-out data2d/tfi2d_4x3_exact.npz --workers 13

  # exact cache only (validation sizes)
  python tfi2d/generate_tfi2d_dataset.py --lx 4 --ly 6 \
      --gs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.9 \
      --samples-per-g 0 --exact-out data2d/tfi2d_4x6_exact.npz --workers 5
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

SQRT2 = np.sqrt(2.0)
# single-qubit basis-change matrices acting on the measured qubit,
# identical to the 1D generator (H for X, Ydag for Y, identity for Z)
H_MEASURE = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / SQRT2
# eigenvectors of Y: |y+> = (|0> + i|1>)/sqrt2 listed first (outcome bit 1
# convention matches generate_quantum_measure_output_pure: row 0 = outcome 1)
Y_DAG = np.array([[1.0, -1.0j], [1.0, 1.0j]], dtype=np.complex128) / SQRT2


def torus_bonds(lx: int, ly: int):
    """Nearest-neighbour bonds and the r=2 horizontal pairs on an Lx x Ly torus."""
    if lx < 3 or ly < 3:
        raise ValueError("need Lx,Ly >= 3 on a torus (avoid doubled wrap bonds)")
    n = lx * ly
    h1, v1, h2 = [], [], []
    for r in range(ly):
        for c in range(lx):
            s = r * lx + c
            h1.append((s, r * lx + (c + 1) % lx))
            v1.append((s, ((r + 1) % ly) * lx + c))
            h2.append((s, r * lx + (c + 2) % lx))
    return n, h1, v1, h2


def z_arrays(n: int) -> np.ndarray:
    """z[s, idx] = +/-1 eigenvalue of Z_s.  Site 0 is the MOST significant bit
    (kron ordering, matching the 1D generator's sequential-collapse sampler)."""
    idx = np.arange(1 << n, dtype=np.int64)
    z = np.empty((n, 1 << n), dtype=np.int8)
    for s in range(n):
        shift = n - 1 - s
        z[s] = 1 - 2 * ((idx >> shift) & 1).astype(np.int8)
    return z


def zz_bond_sum(z: np.ndarray, bonds) -> np.ndarray:
    """sum over bonds of z_i * z_j, as float64 (g-independent)."""
    acc = np.zeros(z.shape[1], dtype=np.float64)
    for i, j in bonds:
        acc += (z[i].astype(np.int16) * z[j]).astype(np.float64)
    return acc


def ground_state_tfi2d(lx: int, ly: int, g: float, tol: float = 1e-10):
    """Matrix-free Lanczos ground state.  Returns (E0, psi[float64])."""
    n, h1, v1, _ = torus_bonds(lx, ly)
    dim = 1 << n
    z = z_arrays(n)
    diag = -(1.0 - g) * zz_bond_sum(z, h1 + v1)
    del z

    def matvec(psi):
        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        out = diag * psi
        for s in range(n):
            shift = n - 1 - s
            arr = psi.reshape(-1, 2, 1 << shift)
            out -= g * arr[:, ::-1, :].reshape(-1)
        return out

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    # ferromagnet+paramagnet-friendly start: uniform positive vector
    v0 = np.full(dim, 1.0 / np.sqrt(dim))
    vals, vecs = eigsh(op, k=1, which="SA", tol=tol, v0=v0, maxiter=10000)
    psi = vecs[:, 0]
    return float(vals[0]), psi / np.linalg.norm(psi)


def exact_zz(psi: np.ndarray, n: int, pairs) -> float:
    """Site-averaged <Z_i Z_j> over the given displacement pairs."""
    p = psi * psi
    idx = np.arange(p.size, dtype=np.int64)
    acc = 0.0
    for i, j in pairs:
        zi = 1 - 2 * ((idx >> (n - 1 - i)) & 1).astype(np.float64)
        zj = 1 - 2 * ((idx >> (n - 1 - j)) & 1).astype(np.float64)
        acc += float(np.dot(p, zi * zj))
    return acc / len(pairs)


def sample_shadows(psi: np.ndarray, n: int, num_samples: int, rng: np.random.Generator):
    """Sequential-collapse classical shadows; same algorithm and encoding as
    generate_quantum_measure_output_pure in the 1D generator.
    Returns int8 array [num_samples, 2n] = [P1,b1,...,PN,bN]."""
    out = np.empty((num_samples, 2 * n), dtype=np.int8)
    psi_c = psi.astype(np.complex128)
    for t in range(num_samples):
        work = psi_c.copy()
        for q in range(n):
            pauli = int(rng.integers(2, 5))  # 2:X 3:Y 4:Z
            mat = work.reshape(2, -1)
            if pauli == 2:
                mat = H_MEASURE @ mat
            elif pauli == 3:
                mat = Y_DAG @ mat
            p_up = min(max(float(np.vdot(mat[0], mat[0]).real), 0.0), 1.0)
            b = 1 if rng.random() < p_up else 0
            branch = mat[0] if b == 1 else mat[1]
            nrm = np.linalg.norm(branch)
            work = branch / (nrm + 1e-30)
            out[t, 2 * q] = pauli
            out[t, 2 * q + 1] = b
    return out


def _solve_one(task):
    lx, ly, g, samples, seed = task
    n, h1, v1, h2 = torus_bonds(lx, ly)
    e0, psi = ground_state_tfi2d(lx, ly, g)
    corr = (exact_zz(psi, n, h1), exact_zz(psi, n, v1), exact_zz(psi, n, h2))
    shadows = None
    if samples > 0:
        rng = np.random.default_rng(seed)
        shadows = sample_shadows(psi, n, samples, rng)
    print(f"[g={g:.3f}] E0/N={e0 / n:+.6f}  Ch1={corr[0]:+.4f} "
          f"Cv1={corr[1]:+.4f} Ch2={corr[2]:+.4f}"
          + ("" if shadows is None else f"  ({samples} shadows)"), flush=True)
    return g, e0, corr, shadows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lx", type=int, default=4)
    ap.add_argument("--ly", type=int, required=True)
    ap.add_argument("--gs", type=float, nargs="+", required=True)
    ap.add_argument("--samples-per-g", type=int, default=3000)
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--exact-out", type=str, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = args.lx * args.ly
    print(f"2D TFI torus {args.lx}x{args.ly}  (N={n} qubits), "
          f"{len(args.gs)} g-values, {args.samples_per_g} shadows/g", flush=True)
    if args.samples_per_g > 0 and not args.json_out:
        raise SystemExit("--json-out required when --samples-per-g > 0")

    tasks = [(args.lx, args.ly, g, args.samples_per_g, args.seed + 1000 * k)
             for k, g in enumerate(args.gs)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_solve_one, tasks))
    else:
        results = [_solve_one(t) for t in tasks]
    results.sort(key=lambda r: r[0])

    gs = np.array([r[0] for r in results])
    e0 = np.array([r[1] for r in results])
    ch1 = np.array([r[2][0] for r in results])
    cv1 = np.array([r[2][1] for r in results])
    ch2 = np.array([r[2][2] for r in results])

    Path(args.exact_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.exact_out, gs=gs, E0=e0, e0_per_site=e0 / n,
             corr_h1=ch1, corr_v1=cv1, corr_h2=ch2,
             lx=args.lx, ly=args.ly, num_qubits=n)
    print(f"saved exact cache: {args.exact_out}")

    if args.samples_per_g > 0:
        rows = []
        for g, _, _, shadows in results:
            for t in range(shadows.shape[0]):
                rows.append([float(g)] + [int(v) for v in shadows[t]])
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print(f"saved training json: {args.json_out}  ({len(rows)} rows, "
              f"width {1 + 2 * n})")


if __name__ == "__main__":
    main()
