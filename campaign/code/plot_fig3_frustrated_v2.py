#!/usr/bin/env python3
"""Manuscript Fig. 3 (v2): thermodynamic-limit estimates for the frustrated
J1-J2 chain and the ANNNI chain (kappa = 0.4), with the physics of both
models illustrated underneath.

Top row (a-d): Diffushadow N -> infinity estimates (student's bootstrap fits,
inf_research/bootstrap_*_extrapolated_vs_g_plots_run*/bootstrap_infinite.csv)
with 95% bootstrap intervals, over exact iDMRG curves (chi = 256).

Bottom row: (e) J1-J2 chain and its frustrated bonds, (f) its phase diagram
along J2/J1, (g) the ANNNI chain, (h) its phase diagram along h. The phase
axes share the range of the data panels directly above them.

ANNNI reference: standard two-sided connected structure factor,
S(q) = 2*SF_stored - (1 - m_z^2), because idmrg_anchors.py stored the
one-sided sum. Points within 0.1 of the FM-PM transition are hollow and
are not graded.

Output: campaign/figures/fig3_frustrated/FIG3_frustrated_v2.{png,pdf}
(the v1 draft FIG3_frustrated_draft.* is left untouched)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from figstyle import (SURFACE, INK, INK2, MUTED, GRID, AXIS, BLUE, ORANGE, AQUA, BAND,
                      setup, style, letter, save)

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "campaign/refs/idmrg"
OUT = ROOT / "campaign/figures/fig3_frustrated"
CSV_J1J2 = ROOT / "inf_research/bootstrap_j1j2_extrapolated_vs_g_plots_run1/bootstrap_infinite.csv"
CSV_PI = ROOT / "inf_research/bootstrap_annni_extrapolated_vs_g_plots_run1/bootstrap_infinite.csv"
CSV_PI2 = ROOT / "inf_research/bootstrap_annni_extrapolated_vs_g_plots_run2/bootstrap_infinite.csv"

H_C = 0.25                    # ANNNI FM-PM transition (iDMRG m_z: ordered at 0.2, not at 0.3)
H_PE = 0.225                  # Peschel-Emery disorder line, h = 1/(4 kappa) - kappa at kappa = 0.4
J2_C, J2_MG, J2_LIF = 0.2411, 0.5, 0.5241   # Okamoto-Nomura, Majumdar-Ghosh, Lifshitz point
UP, DOWN = "↑", "↓"


def read_csv(path, want=None):
    rows = list(csv.DictReader(open(path)))
    if want is not None:
        rows = [r for r in rows if r["component"].strip().endswith(want)]
    rows.sort(key=lambda r: float(r["parameter"]))
    p = np.array([float(r["parameter"]) for r in rows])
    v = np.array([float(r["fit_of_finite_mean"]) for r in rows])
    lo = np.array([float(r["ci_low"]) for r in rows])
    hi = np.array([float(r["ci_high"]) for r in rows])
    return p, v, lo, hi


def arc(ax, x1, x2, y, rad, color, lw=1.0):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-", color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def spins(ax, xs, y, updown, color=INK, length=0.34, lw=1.1):
    for x, up in zip(xs, updown):
        dy = length if up else -length
        ax.annotate("", xy=(x, y + dy), xytext=(x, y - dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=6))


def chain_panel(ax, kind):
    """kind: 'j1j2' (antiferromagnetic, frustrated) or 'annni' (ferromagnetic + NNN + field)."""
    n = 7
    xs = np.arange(n)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-1.5, 1.35)
    ax.axis("off")
    if kind == "j1j2":
        spins(ax, xs, 0.0, [i % 2 == 0 for i in range(n)])
        for i in range(n - 1):
            arc(ax, i, i + 1, 0.42, -0.55, BLUE)
        for i in range(n - 2):
            arc(ax, i, i + 2, -0.42, 0.42, ORANGE)
        ax.text(0.5, 1.10, r"$J_1 > 0$", color=BLUE, fontsize=6.8, ha="center")
        ax.text(1.0, -1.30, r"$J_2 > 0$  frustrates", color=ORANGE, fontsize=6.8, ha="center")
    else:
        spins(ax, xs, 0.0, [True] * n)
        for i in range(n - 1):
            arc(ax, i, i + 1, 0.42, -0.55, BLUE)
        for i in range(n - 2):
            arc(ax, i, i + 2, -0.42, 0.42, ORANGE)
        ax.text(0.5, 1.10, r"$-J_1$ aligns", color=BLUE, fontsize=6.8, ha="center")
        ax.text(1.6, -1.30, r"$+\kappa J_1$ at $r=2$ frustrates", color=ORANGE, fontsize=6.8,
                ha="center")
        ax.annotate("", xy=(n - 0.35, 1.02), xytext=(n - 1.5, 1.02),
                    arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.1, mutation_scale=6))
        ax.text(n - 0.9, 1.16, r"field $h$", color=AQUA, fontsize=6.8, ha="center")


def phase_panel(ax, kind):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=2.5)
    if kind == "j1j2":
        ax.set_xlim(0, 1)
        ax.axvspan(0, J2_C, color=GRID, alpha=0.75, lw=0)
        ax.axvspan(J2_C, 1, color=BAND, alpha=0.6, lw=0)
        for x in (J2_C, J2_MG):
            ax.axvline(x, color=AXIS, lw=0.8)
        ax.text(J2_C / 2, 0.52, "gapless", ha="center", va="center", rotation=90,
                fontsize=6.4, color=INK)
        ax.text(0.63, 0.86, "dimerized (gapped)", ha="center", va="center", fontsize=6.6, color=INK)
        ax.text(J2_C / 2, 0.16, UP + DOWN, ha="center", va="center", fontsize=6.5, color=INK2)
        ax.text(0.63, 0.58, f"({UP}{DOWN})({UP}{DOWN})", ha="center", va="center",
                fontsize=7.5, color=INK2)
        ax.text(J2_C + 0.015, 0.06, r"$J_{2c}$", ha="left", va="bottom", fontsize=6.2, color=INK2)
        ax.text(J2_MG + 0.015, 0.06, "MG", ha="left", va="bottom", fontsize=6.2, color=INK2)
        ax.annotate("incommensurate", xy=(J2_LIF, 0.30), xytext=(0.99, 0.30), ha="right",
                    va="center", fontsize=6.2, color=INK2,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=5))
        ax.set_xlabel(r"$J_2/J_1$")
    else:
        ax.set_xlim(0, 2)
        ax.axvspan(0, H_C, color=GRID, alpha=0.75, lw=0)
        ax.axvspan(H_C, 2, color=BAND, alpha=0.6, lw=0)
        ax.axvline(H_C, color=AXIS, lw=0.8)
        ax.text(H_C / 2, 0.52, "FM", ha="center", va="center", rotation=90,
                fontsize=6.4, color=INK)
        ax.text(1.25, 0.86, "paramagnetic", ha="center", va="center", fontsize=6.6, color=INK)
        ax.text(H_C / 2, 0.16, UP * 2, ha="center", va="center", fontsize=6.5, color=INK2)
        ax.text(1.25, 0.58, "\u2192\u2192\u2192", ha="center", va="center", fontsize=7.5, color=INK2)
        ax.annotate("Peschel-Emery\nline, " + r"$h=0.225$", xy=(H_PE, 0.30),
                    xytext=(0.46, 0.30), ha="left", va="center", fontsize=6.0, color=INK2,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7, mutation_scale=5))
        ax.text(H_C + 0.04, 0.62, r"$h_c\approx0.25$", ha="left", va="center",
                fontsize=6.2, color=INK2)
        ax.set_xlabel(r"$h$")


def main():
    setup()
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
        ("r1", read_csv(CSV_J1J2, "1"), r"$J_2/J_1$",
         r"$\langle\boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_{i+1}\rangle$", r"$J_1$-$J_2$, $r=1$", None),
        ("r2", read_csv(CSV_J1J2, "2"), r"$J_2/J_1$",
         r"$\langle\boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_{i+2}\rangle$", r"$J_1$-$J_2$, $r=2$", None),
        ("pi", read_csv(CSV_PI), r"$h$", r"$S_Z(\pi)$", r"ANNNI ($\kappa=0.4$)", H_C),
        ("pi2", read_csv(CSV_PI2), r"$h$", r"$S_Z(\pi/2)$", r"ANNNI ($\kappa=0.4$)", H_C),
    ]

    fig = plt.figure(figsize=(7.2, 4.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.05], hspace=0.46, wspace=0.42,
                            left=0.075, right=0.985, top=0.90, bottom=0.10)
    tops = [fig.add_subplot(grid[0, j]) for j in range(4)]

    for j, (key, (p, v, lo, hi), xl, yl, title, crit) in enumerate(panels):
        Pg, R = ref[key]
        ex = np.interp(p, Pg, R)
        graded = np.ones_like(p, bool) if crit is None else np.abs(p - crit) >= 0.1
        ax = tops[j]
        style(ax)
        if crit is not None:
            ax.axvspan(0, crit, color=GRID, alpha=0.55, lw=0)
        else:
            ax.axvline(J2_C, color=AXIS, lw=0.8)
            ax.axvline(J2_MG, color=AXIS, lw=0.8)
        ax.plot(Pg, R, color=INK, lw=1.4, label="exact (iDMRG)", zorder=3)
        ax.fill_between(p, lo, hi, color=BAND, lw=0, zorder=1, label="95% bootstrap interval")
        ax.plot(p[graded], v[graded], "o", ms=3.6, color=BLUE, mec=SURFACE, mew=0.6, zorder=4,
                label=r"Diffushadow, $N\to\infty$")
        if (~graded).any():
            ax.plot(p[~graded], v[~graded], "o", ms=3.6, mfc=SURFACE, mec=BLUE, mew=0.9, zorder=4)
        ax.set_ylabel(yl, color=INK)
        ax.set_xlabel(xl)
        ax.set_title(title, color=INK2, pad=3)
        mad = np.mean(np.abs((v - ex)[graded]))
        ty, tva = (0.95, "top") if j == 1 else (0.05, "bottom")
        ax.text(0.97, ty, f"mean |dev| {mad:.3f}", transform=ax.transAxes, ha="right",
                va=tva, fontsize=6, color=INK2)
        if crit is not None:
            ax.text(crit / 2, 0.96, "FM", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=6.5, color=INK2)
        else:
            ly, va = (0.97, "top") if j == 0 else (0.03, "bottom")
            ax.text(J2_MG + 0.02, ly, "MG", transform=ax.get_xaxis_transform(), ha="left",
                    va=va, fontsize=6.5, color=INK2)
            ax.text(J2_C - 0.02, ly, r"$J_{2c}$", transform=ax.get_xaxis_transform(), ha="right",
                    va=va, fontsize=6.5, color=INK2)
        letter(ax, "abcd"[j])

    bots = [fig.add_subplot(grid[1, j]) for j in range(4)]
    chain_panel(bots[0], "j1j2")
    phase_panel(bots[1], "j1j2")
    chain_panel(bots[2], "annni")
    phase_panel(bots[3], "annni")
    for ax, ttl in zip(bots, (r"$J_1$-$J_2$ model", r"$J_1$-$J_2$ phases",
                              "ANNNI model", "ANNNI phases")):
        ax.set_title(ttl, color=INK2, pad=3, fontsize=7)
    for ax, s in zip(bots, "efgh"):
        letter(ax, s, x=-0.10 if s in "eg" else -0.22)

    h, l = tops[0].get_legend_handles_labels()
    order = [l.index("exact (iDMRG)"), l.index(r"Diffushadow, $N\to\infty$"),
             l.index("95% bootstrap interval")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    save(fig, OUT, "FIG3_frustrated_v2")


if __name__ == "__main__":
    main()
