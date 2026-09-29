#!/usr/bin/env python3
"""Manuscript Fig. 3 (v5): thermodynamic-limit inference in frustrated spin
chains, with the two models and their phase diagrams illustrated underneath.

Nature-style figure conventions: 180 mm double-column width, Helvetica/Arial,
5-7 pt type, hairline rules, bold lower-case panel letters, no panel titles.

a-d  N -> infinity estimates (student's bootstrap fits) with 95% intervals,
     over exact iDMRG curves (chi = 256).
e    J1-J2 chain: competing J1 and J2 bonds.
f    J1-J2 phase diagram on the axis of a and b.
g    ANNNI chain: ferromagnetic J1, frustrating kappa*J1 at r = 2, field h.
h    ANNNI phase diagram on the axis of c and d.

Output: campaign/figures/fig3_frustrated/FIG3_frustrated_v5.{png,pdf}
(v1 FIG3_frustrated_draft.* and v2 FIG3_frustrated_v2.* are left untouched)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "campaign/refs/idmrg"
OUT = ROOT / "campaign/figures/fig3_frustrated"
CSV_J1J2 = ROOT / "inf_research/bootstrap_j1j2_extrapolated_vs_g_plots_run1/bootstrap_infinite.csv"
CSV_PI = ROOT / "inf_research/bootstrap_annni_extrapolated_vs_g_plots_run1/bootstrap_infinite.csv"
CSV_PI2 = ROOT / "inf_research/bootstrap_annni_extrapolated_vs_g_plots_run2/bootstrap_infinite.csv"

SURFACE, INK, INK2, MUTED = "#ffffff", "#000000", "#4d4d4d", "#8c8c8c"
GRID, RULE = "#e6e5e0", "#b3b3b3"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
BAND, FILL_A, FILL_B = "#cde2fb", "#eceae4", "#dceafb"

H_C, H_PE = 0.25, 0.225
J2_C, J2_MG, J2_LIF = 0.2411, 0.5, 0.5241
UP, DOWN = "↑", "↓"
FONTS = ["Arial", "Helvetica", "DejaVu Sans"]


def read_csv(path, want=None):
    rows = list(csv.DictReader(open(path)))
    if want is not None:
        rows = [r for r in rows if r["component"].strip().endswith(want)]
    rows.sort(key=lambda r: float(r["parameter"]))
    g = lambda k: np.array([float(r[k]) for r in rows])
    return g("parameter"), g("fit_of_finite_mean"), g("ci_low"), g("ci_high")


def axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK2, labelsize=6, width=0.5, length=2.2, pad=1.8)


def bare(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(colors=INK2, labelsize=6, width=0.5, length=2.2, pad=1.8)


def letter(ax, s, x=-0.17, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, weight="bold", color=INK,
            ha="left", va="bottom")


def arc(ax, x1, x2, y, rad, color):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-", color=color, lw=0.7,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def chain(ax, kind):
    n = 7
    ax.set_xlim(-0.9, n - 0.1)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    up = [i % 2 == 0 for i in range(n)] if kind == "j1j2" else [True] * n
    for x, u in zip(range(n), up):
        dy = 0.32 if u else -0.32
        ax.annotate("", xy=(x, dy), xytext=(x, -dy),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, shrinkA=0, shrinkB=0,
                                    mutation_scale=4.5))
    for i in range(n - 1):
        arc(ax, i, i + 1, 0.40, -0.55, BLUE)
    for i in range(n - 2):
        arc(ax, i, i + 2, -0.40, 0.42, ORANGE)
    if kind == "j1j2":
        ax.text(3.6, 1.16, r"$J_1>0$", color=BLUE, fontsize=6, ha="center")
        ax.text(3.0, -1.34, r"$J_2>0$, frustrating", color=ORANGE, fontsize=6, ha="center")
        ax.text(-0.9, 1.16, r"$J_1$-$J_2$ chain", color=INK2, fontsize=6, ha="left")
    else:
        ax.text(3.2, 1.16, r"$-J_1$", color=BLUE, fontsize=6, ha="center")
        ax.text(3.0, -1.34, r"$+\kappa J_1$ at $r=2$, frustrating", color=ORANGE, fontsize=6, ha="center")
        ax.annotate("", xy=(n - 0.25, 0.95), xytext=(n - 1.35, 0.95),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.7, mutation_scale=4.5))
        ax.text(n - 0.8, 1.16, r"$h$", color=GREEN, fontsize=6, ha="center")
        ax.text(-0.9, 1.16, "ANNNI chain", color=INK2, fontsize=6, ha="left")


def phase_bar(ax, xmax, split, fills=(FILL_A, FILL_B)):
    """Horizontal phase bar spanning the axis; returns the bar's y extent."""
    y0, y1 = 0.30, 0.56
    ax.add_patch(Rectangle((0, y0), split, y1 - y0, facecolor=fills[0], lw=0,
                           transform=ax.get_xaxis_transform(), clip_on=False))
    ax.add_patch(Rectangle((split, y0), xmax - split, y1 - y0, facecolor=fills[1], lw=0,
                           transform=ax.get_xaxis_transform(), clip_on=False))
    ax.plot([split, split], [y0, y1], color=RULE, lw=0.6,
            transform=ax.get_xaxis_transform(), clip_on=False)
    return y0, y1


def main():
    plt.rcParams.update({"font.family": FONTS, "mathtext.fontset": "dejavusans",
                         "font.size": 6, "axes.labelsize": 6.5, "axes.labelcolor": INK,
                         "legend.fontsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42})

    jj = np.load(REFS / "idmrg_j1j2.npz")
    an = np.load(REFS / "idmrg_annni.npz")
    PJ, PA = np.asarray(jj["params"], float), np.asarray(an["params"], float)
    mz = np.asarray(an["mz_chi256"], float)
    ref = {"r1": (PJ, np.asarray(jj["sd1_mean_chi256"], float)),
           "r2": (PJ, np.asarray(jj["sd2_chi256"], float)),
           "pi": (PA, 2 * np.asarray(an["SF_pi_chi256"], float) - (1 - mz ** 2)),
           "pi2": (PA, 2 * np.asarray(an["SF_pi_2_chi256"], float) - (1 - mz ** 2))}
    d_r1, d_r2 = read_csv(CSV_J1J2, "1"), read_csv(CSV_J1J2, "2")
    panels = [("r1", d_r1, r"$J_2/J_1$", r"$\langle\boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_{i+1}\rangle$", None),
              ("r2", d_r2, r"$J_2/J_1$", r"$\langle\boldsymbol{\sigma}_i\cdot\boldsymbol{\sigma}_{i+2}\rangle$", None),
              ("pi", read_csv(CSV_PI), r"$h$", r"$S_Z(\pi)$", H_C),
              ("pi2", read_csv(CSV_PI2), r"$h$", r"$S_Z(\pi/2)$", H_C)]

    fig = plt.figure(figsize=(7.087, 4.33), facecolor=SURFACE)   # 180 x 110 mm
    grid = fig.add_gridspec(2, 4, height_ratios=[1.6, 1.0], hspace=0.45, wspace=0.40,
                            left=0.068, right=0.99, top=0.92, bottom=0.105)
    tops = [fig.add_subplot(grid[0, j]) for j in range(4)]

    for j, (key, (p, v, lo, hi), xl, yl, crit) in enumerate(panels):
        Pg, R = ref[key]
        ex = np.interp(p, Pg, R)
        graded = np.ones_like(p, bool) if crit is None else np.abs(p - crit) >= 0.1
        ax = tops[j]
        axes_style(ax)
        if crit is not None:
            ax.axvspan(0, crit, color=FILL_A, lw=0, zorder=0)
            ax.text(crit / 2, 0.96, "FM", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=6, color=INK2)
        else:
            for xv, lab in ((J2_C, r"$J_{2c}$"), (J2_MG, "MG")):
                ax.axvline(xv, color=RULE, lw=0.5)
                ly, va = (0.97, "top") if j == 0 else (0.03, "bottom")
                ax.text(xv + (0.015 if lab == "MG" else -0.015), ly, lab, ha="left" if lab == "MG" else "right",
                        va=va, transform=ax.get_xaxis_transform(), fontsize=6, color=INK2)
        ax.plot(Pg, R, color=INK, lw=1.0, zorder=3, label="exact (iDMRG)")
        ax.fill_between(p, lo, hi, color=BAND, lw=0, zorder=1, label="95% interval")
        ax.plot(p[graded], v[graded], "o", ms=2.8, color=BLUE, mec=SURFACE, mew=0.4, zorder=4,
                label=r"Diffushadow, $N\to\infty$")
        if (~graded).any():
            ax.plot(p[~graded], v[~graded], "o", ms=2.8, mfc=SURFACE, mec=BLUE, mew=0.7, zorder=4)
        ax.set_xlabel(xl, labelpad=1.5)
        ax.set_ylabel(yl, labelpad=1.5)
        ty, tva = (0.95, "top") if j == 1 else (0.05, "bottom")
        ax.text(0.97, ty, f"mean deviation {np.mean(np.abs((v - ex)[graded])):.3f}",
                transform=ax.transAxes, ha="right", va=tva, fontsize=5.5, color=INK2)
        letter(ax, "abcd"[j])

    bots = [fig.add_subplot(grid[1, j]) for j in range(4)]
    chain(bots[0], "j1j2")
    chain(bots[2], "annni")

    # ---- f: J1-J2 phase diagram + Majumdar-Ghosh locator ----
    f = bots[1]
    bare(f)
    f.set_xlim(0, 1)
    f.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    y0, y1 = phase_bar(f, 1.0, J2_C)
    f.text(J2_C / 2, (y0 + y1) / 2, "gapless", ha="center", va="center", fontsize=5.5, color=INK)
    f.text((J2_C + 1) / 2, (y0 + y1) / 2, "dimerized", ha="center", va="center", fontsize=5.5, color=INK)
    f.text(J2_C, y0 - 0.05, r"$J_{2c}=0.241$", ha="center", va="top", fontsize=5.5, color=INK2)
    f.annotate("incommensurate", xy=(J2_LIF, 0.14), xytext=(1.0, 0.14), ha="right", va="center",
               fontsize=5.5, color=INK2,
               arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.5, mutation_scale=4))
    f.set_xlabel(r"$J_2/J_1$", labelpad=1.5)

    f.plot([J2_MG], [y1], marker="v", ms=2.4, color=INK, clip_on=False,
           transform=f.get_xaxis_transform())
    f.text(J2_MG, y1 + 0.05, "MG", ha="center", va="bottom", fontsize=5.5, color=INK2,
           transform=f.get_xaxis_transform())

    # ---- h: ANNNI phase diagram ----
    h = bots[3]
    bare(h)
    h.set_xlim(0, 2)
    h.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    y0, y1 = phase_bar(h, 2.0, H_C)
    h.text(H_C / 2, y1 + 0.06, "FM", ha="center", va="bottom", fontsize=5.5, color=INK)
    h.text((H_C + 2) / 2, (y0 + y1) / 2, "paramagnetic", ha="center", va="center", fontsize=5.5, color=INK)
    h.text(H_C + 0.06, y0 - 0.05, r"$h_c\approx0.25$", ha="left", va="top", fontsize=5.5, color=INK2)
    h.annotate("Peschel-Emery line\n" + r"$h=1/4\kappa-\kappa$", xy=(H_PE, 0.74), xytext=(0.56, 0.74),
               ha="left", va="center", fontsize=5.5, color=INK2,
               arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.5, mutation_scale=4))
    h.text(1.125, 0.14, UP + DOWN + " " + UP + DOWN, ha="center", va="center", fontsize=5.5,
           color=SURFACE)   # spacer keeps the baseline consistent with f
    h.set_xlabel(r"$h$", labelpad=1.5)

    for ax, s in zip(bots, "efgh"):
        letter(ax, s, x=-0.06 if s in "eg" else -0.17)

    hs, ls = tops[0].get_legend_handles_labels()
    order = [ls.index("exact (iDMRG)"), ls.index(r"Diffushadow, $N\to\infty$"), ls.index("95% interval")]
    fig.legend([hs[i] for i in order], [ls[i] for i in order], loc="upper center", ncol=3,
               frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.005), handlelength=1.6,
               columnspacing=1.6, handletextpad=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"FIG3_frustrated_v5.{ext}", dpi=600, facecolor=SURFACE)
    print(f"saved {OUT}/FIG3_frustrated_v5.png")


if __name__ == "__main__":
    main()
