#!/usr/bin/env python3
"""WS1: compare the student's recursively-generated TFI observables
(zz_infinite_fit_values.csv) against quasi-exact DMRG references at the
same sizes (campaign/dmrg_refs/tfi_dmrg_ref_N*.npz).

Produces: figures + a numbers summary for the manuscript placeholders.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paths import FIGURES as OUT, find
ROOT = Path(__file__).resolve().parent.parent
REF = None  # resolved per-file via find()
OUT.mkdir(parents=True, exist_ok=True)

SIZES = [32, 36, 48, 64, 72, 96]
DIST_IDX = {1: 0, 2: 1, 3: 2, 10: 3, 20: 4}

# ---- load DMRG refs ----
refs = {}
for N in SIZES:
    d = np.load(find(f"tfi_dmrg_ref_N{N}.npz"))
    refs[N] = {"gs": d["gs"], "ZZ": d["ZZ"], "dist": list(d["distances"])}

# ---- load student CSV ----
rows = list(csv.DictReader(open(find("student_zz_run1.csv"))))
student = {}  # (d, g) -> {N: zz}
for r in rows:
    dd = int(r["distance"])
    g = round(float(r["g_value"]), 3)
    qs = [int(x) for x in r["qubits"].split(";")]
    zz = [float(x) for x in r["zz_values"].split(";")]
    student[(dd, g)] = dict(zip(qs, zz))
    student.setdefault(("inf", dd), {})[g] = float(r["zz_infinite"])

dists = sorted({d for (d, g) in student if d != "inf" and isinstance(d, int)})
gs_csv = sorted({g for (d, g) in student if isinstance(d, int) and d == dists[0]})
print(f"student CSV: distances {dists}, g grid {gs_csv}")
print(f"DMRG refs:   distances {refs[32]['dist']}, g grid {list(refs[32]['gs'])}")

# ---- per-size comparison ----
summary = []
fig, axes = plt.subplots(1, len(dists), figsize=(4.2 * len(dists), 3.5), sharey=True)
axes = np.atleast_1d(axes)
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(SIZES)))
for ax, dd in zip(axes, dists):
    for ci, N in enumerate(SIZES):
        gs_ref = refs[N]["gs"]
        zz_ref = refs[N]["ZZ"][DIST_IDX[dd]]
        errs, gplot = [], []
        for g in gs_csv:
            if N not in student.get((dd, g), {}):
                continue
            gi = int(np.argmin(np.abs(gs_ref - g)))
            if abs(gs_ref[gi] - g) > 1e-6:
                continue
            e = student[(dd, g)][N] - zz_ref[gi]
            errs.append(e)
            gplot.append(g)
            summary.append((dd, N, g, e))
        ax.plot(gplot, np.abs(errs), "o-", ms=4, color=colors[ci],
                label=f"N={N}" if ax is axes[0] else None)
    ax.set_yscale("log")
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$g$")
    ax.set_title(f"|generated $-$ DMRG|,  r={dd}")
axes[0].set_ylabel("absolute deviation")
axes[0].legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig6_dmrg_validation_error.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig6_dmrg_validation_error.png", dpi=300, bbox_inches="tight")

# ---- overlay figure: curves at largest sizes ----
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), sharey=True)
for ax, N in zip(axes, (48, 72, 96)):
    gs_ref = refs[N]["gs"]
    for dd, color in zip(dists, ("#1f77b4", "#d62728", "#2ca02c")):
        ax.plot(gs_ref, refs[N]["ZZ"][DIST_IDX[dd]], "-", color=color, lw=1.4,
                label=f"DMRG r={dd}" if ax is axes[0] else None)
        gg = [g for g in gs_csv if N in student.get((dd, g), {})]
        ax.plot(gg, [student[(dd, g)][N] for g in gg], "o", ms=5, mfc="none",
                color=color, label=f"generated r={dd}" if ax is axes[0] else None)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$g$")
    ax.set_title(f"N = {N} (recursion; ED impossible)", fontsize=11)
axes[0].set_ylabel(r"$\langle Z_i Z_{i+r} \rangle$")
axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig7_dmrg_validation_curves.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig7_dmrg_validation_curves.png", dpi=300, bbox_inches="tight")

# ---- numbers for the manuscript ----
arr = np.array([(dd, N, g, e) for dd, N, g, e in summary])
abse = np.abs(arr[:, 3])
crit = np.abs(arr[:, 2] - 0.5) <= 0.101
print("\n=== WS1 validation summary (all sizes 32..96, r=1..3) ===")
print(f"points compared      : {len(arr)}")
print(f"mean |dev|  (all)    : {abse.mean():.4f}")
print(f"median |dev| (all)   : {np.median(abse):.4f}")
print(f"mean |dev| off-crit  : {abse[~crit].mean():.4f}")
print(f"mean |dev| near-crit : {abse[crit].mean():.4f}  (g in [0.4,0.6])")
print(f"max  |dev|           : {abse.max():.4f} at (r,N,g)="
      f"{tuple(arr[np.argmax(abse), :3])}")
for N in SIZES:
    m = arr[:, 1] == N
    print(f"  N={N:3d}: mean {abse[m].mean():.4f}  max {abse[m].max():.4f}")
# r=1 only (headline number)
m1 = arr[:, 0] == 1
print(f"r=1 only: mean {abse[m1].mean():.4f}, max {abse[m1].max():.4f}")
