"""Shared style for the manuscript figure drafts (Figs. 3-5).

Palette: dataviz reference instance, light mode. Categorical slots 1-3
validate all-pairs; SIZE_RAMP is an ordinal blue ramp (step 250 -> 700)
used for system size, so a given size keeps its colour across panels.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAND = "#cde2fb"
SIZE_RAMP = {16: "#86b6ef", 24: "#5598e7", 32: "#2a78d6", 48: "#1c5cab", 64: "#0d366b"}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                         "axes.labelcolor": INK, "legend.fontsize": 6.2})


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


def letter(ax, s, x=-0.22, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, weight="bold", color=INK)


def save(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
    print("saved", outdir / f"{name}.png")
