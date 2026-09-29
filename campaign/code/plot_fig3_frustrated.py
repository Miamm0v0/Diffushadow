#!/usr/bin/env python3
"""Draft manuscript Fig. 3: thermodynamic-limit estimates for the frustrated
J1-J2 chain and the ANNNI chain (kappa = 0.4), graded against iDMRG.

Top row: Diffushadow N -> infinity estimates (student's bootstrap fits,
inf_research/bootstrap_*_extrapolated_vs_g_plots_run*/bootstrap_infinite.csv)
with 95% bootstrap intervals, over exact iDMRG curves (chi = 256).
Bottom row: residual (Diffushadow - exact).

ANNNI reference: standard two-sided connected structure factor,
S(q) = 2*SF_stored - (1 - m_z^2), because idmrg_anchors.py stored the
one-sided sum. Points within 0.1 of the FM-PM transition (h_c ~ 0.25, where
interpolating the iDMRG grid is not meaningful) are drawn hollow and not
graded.

Output: campaign/figures/fig3_frustrated/FIG3_frustrated_draft.{png,pdf}
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "campaign/refs/idmrg"
OUT = ROOT / "campaign/figures/fig3_frustrated"
CSV_J1J2 = ROOT / "inf_research/bootstrap_j1j2_extrapolated_vs_g_plots_run1/bootstrap_infinite.csv"
CSV_PI = ROOT / "inf_research/bootstrap_annni_extrapolated_vs_g_plots_run1/bootstrap_infinite.csv"
CSV_PI2 = ROOT / "inf_research/bootstrap_annni_extrapolated_vs_g_plots_run2/bootstrap_infinite.csv"

# palette (dataviz reference instance, light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MODEL = "#2a78d6"            # categorical slot 1
BAND = "#cde2fb"             # blue ramp step 100
H_C = 0.25                   # ANNNI FM-PM transition from iDMRG m_z (between 0.2 and 0.3)
J2_C, J2_MG = 0.2411, 0.5    # Okamoto-Nomura dimerization, Majumdar-Ghosh point


def read_csv(path, want=None):
    rows = list(csv.DictReader(open(path)))
    if want is not None:
        rows = [r for r in rows if r["component"].strip().endswith(want)]
    rows.sort(key=lambda r: float(r["parameter"]))
    p = np.array([float(r["parameter"]) for r in rows])
    v = np.array([float(r["fit_of_finite_mean"]) for r in rows])
    lo = np.array([float(r["ci_low"]) for r in rows])
    hi = np.array([float(r["ci_high"]) for r in rows])
    sizes = rows[0]["qubits"].replace(";", ", ")
    return p, v, lo, hi, sizes


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=2.5)


def main():
    jj = np.load(REFS / "idmrg_j1j2.npz")
    an = np.load(REFS / "idmrg_annni.npz")
    PJ, PA = np.asarray(jj["params"], float), np.asarray(an["params"], float)
    mz = np.asarray(an["mz_chi256"], float)
    ref = {
        "r1": (PJ, np.asarray(jj["sd1_mean_chi256"], float)),
        "r2": (PJ, np.asarray(jj["sd2_chi256"], float)),
        "pi": (PA, 2 * np.asarray(an["SF_pi_chi256"], float) - (1 - mz ** 2)),
        "pi2": (PA, 2 * np.asarray(an["SF_pi_2_chi256"], float) - (1 - mz ** 2)),
    }
    panels = [
        ("r1", read_csv(CSV_J1J2, "1"), r"$J_2/J_1$", r"$\langle\boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_{i+1}\rangle$", r"$J_1$-$J_2$, $r=1$", None),
        ("r2", read_csv(CSV_J1J2, "2"), r"$J_2/J_1$", r"$\langle\boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_{i+2}\rangle$", r"$J_1$-$J_2$, $r=2$", None),
        ("pi", read_csv(CSV_PI), r"$h$", r"$S_Z(\pi)$", r"ANNNI ($\kappa=0.4$)", H_C),
        ("pi2", read_csv(CSV_PI2), r"$h$", r"$S_Z(\pi/2)$", r"ANNNI ($\kappa=0.4$)", H_C),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), height_ratios=[1.6, 1.0],
                             sharex="col", facecolor=SURFACE)
    letters = "abcdefgh"
    for j, (key, (p, v, lo, hi, sizes), xl, yl, title, crit) in enumerate(panels):
        Pg, R = ref[key]
        ex = np.interp(p, Pg, R)
        graded = np.ones_like(p, bool) if crit is None else np.abs(p - crit) >= 0.1

        top, bot = axes[0, j], axes[1, j]
        for ax in (top, bot):
            style(ax)
            if crit is not None:
                ax.axvspan(0, crit, color=GRID, alpha=0.55, lw=0)
            else:
                ax.axvline(J2_C, color=AXIS, lw=0.8)
                ax.axvline(J2_MG, color=AXIS, lw=0.8)

        top.plot(Pg, R, color=INK, lw=1.4, label="exact (iDMRG)", zorder=3)
        top.fill_between(p, lo, hi, color=BAND, lw=0, zorder=1, label="95% bootstrap interval")
        top.plot(p[graded], v[graded], "o", ms=3.6, color=MODEL, mec=SURFACE, mew=0.6,
                 zorder=4, label=r"Diffushadow, $N\to\infty$")
        if (~graded).any():
            top.plot(p[~graded], v[~graded], "o", ms=3.6, mfc=SURFACE, mec=MODEL, mew=0.9, zorder=4)
        top.set_ylabel(yl, fontsize=7.5, color=INK)
        top.set_title(title, fontsize=7.5, color=INK2, pad=3)
        top.text(-0.24, 1.04, letters[j], transform=top.transAxes, fontsize=9, weight="bold", color=INK)

        if crit is not None:
            top.text(crit / 2, 0.96, "FM", transform=top.get_xaxis_transform(), ha="center",
                     va="top", fontsize=6.5, color=INK2)
        else:
            ly, va = (0.97, "top") if j == 0 else (0.03, "bottom")   # keep labels off the curve
            top.text(J2_MG + 0.02, ly, "MG", transform=top.get_xaxis_transform(), ha="left",
                     va=va, fontsize=6.5, color=INK2)
            top.text(J2_C - 0.02, ly, r"$J_{2c}$", transform=top.get_xaxis_transform(), ha="right",
                     va=va, fontsize=6.5, color=INK2)

        res = v - ex
        bot.axhline(0, color=INK2, lw=0.7)
        bot.vlines(p[graded], (lo - ex)[graded], (hi - ex)[graded], color=MODEL, lw=0.8, alpha=0.6)
        bot.plot(p[graded], res[graded], "o", ms=3.2, color=MODEL, mec=SURFACE, mew=0.5)
        if (~graded).any():
            bot.plot(p[~graded], res[~graded], "o", ms=3.2, mfc=SURFACE, mec=MODEL, mew=0.8)
        bot.set_xlabel(xl, fontsize=7.5, color=INK)
        if j == 0:
            bot.set_ylabel("Diffushadow $-$ exact", fontsize=7.5, color=INK)
        mad = np.mean(np.abs(res[graded]))
        n_lo, n_hi = sizes.split(", ")[0], sizes.split(", ")[-1]
        bot.set_title(f"|dev| {mad:.3f}, N {n_lo}–{n_hi}", fontsize=6,
                      color=INK2, pad=2, loc="right")
        bot.text(-0.24, 1.04, letters[j + 4], transform=bot.transAxes, fontsize=9, weight="bold", color=INK)

    h, l = axes[0, 0].get_legend_handles_labels()
    order = [l.index("exact (iDMRG)"), l.index(r"Diffushadow, $N\to\infty$"), l.index("95% bootstrap interval")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.2, h_pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"FIG3_frustrated_draft.{ext}", dpi=300, facecolor=SURFACE)
    print("saved", OUT / "FIG3_frustrated_draft.png")


if __name__ == "__main__":
    main()
