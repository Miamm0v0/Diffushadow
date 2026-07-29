#!/usr/bin/env python3
"""Paper figure: 2D TFI cross-scale generation (same-size 4x4, unseen 4x5, 4x6).

Exact ED = lines; Diffushadow-generated shadow estimates = markers.
Panels share the g axis; observables are site-averaged ZZ correlations for
displacements (0,1) horizontal, (1,0) vertical, (0,2) next-nearest horizontal.

Usage:
  python tfi2d/plot_tfi2d.py --results campaign/tfi2d_eval_4x4_seq.npz \
      campaign/tfi2d_eval_4x5_seq.npz campaign/tfi2d_eval_4x6_seq.npz \
      --labels "4x4 (16q, trained size)" "4x5 (20q, unseen)" "4x6 (24q, unseen)" \
      --out campaign/figures/fig_tfi2d_crossscale.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = [
    ("h1", r"$\langle Z_i Z_{i+\hat{x}} \rangle$", "#1f77b4", "o"),
    ("v1", r"$\langle Z_i Z_{i+\hat{y}} \rangle$", "#d62728", "s"),
    ("h2", r"$\langle Z_i Z_{i+2\hat{x}} \rangle$", "#2ca02c", "^"),
]
G_C_2D = 3.04438 / 4.04438  # (h/J)_c=3.04438 -> g_c under H=-(1-g)ZZ-gX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.results) == len(args.labels)

    fig, axes = plt.subplots(1, len(args.results), figsize=(4.0 * len(args.results), 3.4),
                             sharey=True, sharex=True)
    if len(args.results) == 1:
        axes = [axes]

    for ax, path, label in zip(axes, args.results, args.labels):
        d = np.load(path)
        gs = d["gs"]
        gd = np.linspace(gs.min(), gs.max(), 200)
        for key, name, color, marker in SERIES:
            # smooth exact curve via dense interp of the exact grid points
            ax.plot(gs, d[f"exact_{key}"], "-", color=color, lw=1.4,
                    label=f"exact {name}" if ax is axes[0] else None)
            ax.plot(gs, d[f"gen_{key}"], marker, color=color, ms=5, mfc="none",
                    label=f"generated {name}" if ax is axes[0] else None)
        ax.axvline(G_C_2D, color="gray", ls=":", lw=1)
        ax.text(G_C_2D + 0.01, 0.92, r"$g_c^{2\mathrm{D}}$", color="gray", fontsize=9)
        ax.set_xlabel(r"$g$")
        ax.set_title(label, fontsize=11)
        mse = np.mean([float(d[f"mse_{k}"]) for k, *_ in SERIES])
        ax.text(0.03, 0.06, f"mean MSE = {mse:.1e}", transform=ax.transAxes, fontsize=9)
    axes[0].set_ylabel(r"$\langle Z_i Z_j \rangle$")
    axes[0].legend(fontsize=8, loc="center left", frameon=False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
