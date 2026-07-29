#!/usr/bin/env python3
"""Full figure set for the 2D TFI cross-scale proof-of-concept.

Reads the npz files in campaign/ (produced on THK node 109 by
tfi2d/generate_tfi2d_dataset.py, train_oseq.py, tfi2d/eval_tfi2d.py)
and writes five figures into campaign/figures/.

Run:  .venv2d/bin/python tfi2d/make_figures_2d.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paths import FIGURES as OUT, find
ROOT = Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)

SIZES = [(4, "4×4 (16q, trained size)"), (5, "4×5 (20q, unseen)"), (6, "4×6 (24q, unseen)")]
SERIES = [
    ("h1", r"$\langle Z_i Z_{i+\hat{x}} \rangle$", "#1f77b4", "o"),
    ("v1", r"$\langle Z_i Z_{i+\hat{y}} \rangle$", "#d62728", "s"),
    ("h2", r"$\langle Z_i Z_{i+2\hat{x}} \rangle$", "#2ca02c", "^"),
]
G_C = 3.04438 / 4.04438

SEQ = {ly: np.load(find(f"tfi2d_eval_4x{ly}_seq.npz")) for ly, _ in SIZES}
FEW = {ly: np.load(find(f"tfi2d_eval_4x{ly}.npz")) for ly, _ in SIZES}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


# ---- Fig 1: main cross-scale comparison --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True, sharex=True)
for ax, (ly, label) in zip(axes, SIZES):
    d = SEQ[ly]
    for key, name, color, marker in SERIES:
        ax.plot(d["gs"], d[f"exact_{key}"], "-", color=color, lw=1.4,
                label=f"exact {name}" if ax is axes[0] else None)
        ax.plot(d["gs"], d[f"gen_{key}"], marker, color=color, ms=5, mfc="none",
                label=f"generated {name}" if ax is axes[0] else None)
    ax.axvline(G_C, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$g$")
    ax.set_title(label, fontsize=11)
    mse = np.mean([float(d[f"mse_{k}"]) for k, *_ in SERIES])
    ax.text(0.03, 0.06, f"mean MSE = {mse:.1e}", transform=ax.transAxes, fontsize=9)
axes[0].set_ylabel(r"$\langle Z_i Z_j \rangle$")
axes[0].legend(fontsize=8, loc="center left", frameon=False)
fig.tight_layout()
save(fig, "fig1_crossscale_main")

# ---- Fig 2: parity plot (generated vs exact, all sizes/observables) ----
fig, ax = plt.subplots(figsize=(4.2, 4.2))
size_colors = {4: "#7f7f7f", 5: "#ff7f0e", 6: "#9467bd"}
for ly, label in SIZES:
    d = SEQ[ly]
    ex = np.concatenate([d[f"exact_{k}"] for k, *_ in SERIES])
    ge = np.concatenate([d[f"gen_{k}"] for k, *_ in SERIES])
    ax.plot(ex, ge, "o", ms=4, mfc="none", color=size_colors[ly], label=label)
lim = [-0.05, 1.05]
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("exact ED " + r"$\langle Z_i Z_j\rangle$")
ax.set_ylabel("generated " + r"$\langle Z_i Z_j\rangle$")
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig2_parity")

# ---- Fig 3: absolute error vs g ---------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for ly, label in SIZES:
    d = SEQ[ly]
    err = np.mean([np.abs(d[f"gen_{k}"] - d[f"exact_{k}"]) for k, *_ in SERIES], axis=0)
    ax.semilogy(d["gs"], err, "o-", ms=4, color=size_colors[ly], label=label)
ax.axvline(G_C, color="gray", ls=":", lw=1)
ax.text(G_C + 0.005, ax.get_ylim()[1] * 0.5, r"$g_c^{2\mathrm{D}}$", color="gray", fontsize=9)
ax.set_xlabel(r"$g$")
ax.set_ylabel(r"mean $|$generated $-$ exact$|$")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
save(fig, "fig3_error_vs_g")

# ---- Fig 4: decoding-granularity ablation ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
d_seq, d_few = SEQ[4], FEW[4]
ax = axes[0]
ax.plot(d_seq["gs"], d_seq["exact_h1"], "k-", lw=1.4, label="exact ED")
ax.plot(d_few["gs"], d_few["gen_h1"], "x--", color="#d62728", ms=6,
        label="parallel decoding (steps = 4)")
ax.plot(d_seq["gs"], d_seq["gen_h1"], "o", color="#1f77b4", ms=5, mfc="none",
        label="sequential decoding (steps = N)")
ax.axvline(G_C, color="gray", ls=":", lw=1)
ax.set_xlabel(r"$g$"); ax.set_ylabel(r"$\langle Z_i Z_{i+\hat{x}}\rangle$")
ax.set_title("4×4, trained size", fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
w = 0.35
xs = np.arange(3)
mse_few = [np.mean([float(FEW[ly][f"mse_{k}"]) for k, *_ in SERIES]) for ly, _ in SIZES]
mse_seq = [np.mean([float(SEQ[ly][f"mse_{k}"]) for k, *_ in SERIES]) for ly, _ in SIZES]
ax.bar(xs - w / 2, mse_few, w, color="#d62728", label="parallel (steps = 4)")
ax.bar(xs + w / 2, mse_seq, w, color="#1f77b4", label="sequential (steps = N)")
ax.set_yscale("log")
ax.set_xticks(xs, ["4×4", "4×5", "4×6"])
ax.set_ylabel("mean MSE")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
save(fig, "fig4_decoding_ablation")

# ---- Fig 5: error scaling with size + training loss --------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
ns = [16, 20, 24]
for key, name, color, marker in SERIES:
    ax.semilogy(ns, [float(SEQ[ly][f"mse_{key}"]) for ly, _ in SIZES],
                marker + "-", color=color, label=name)
ax.set_xticks(ns, ["16\n(trained)", "20\n(unseen)", "24\n(unseen)"])
ax.set_xlabel("qubits"); ax.set_ylabel("MSE")
ax.legend(fontsize=8, frameon=False)
ax.set_title("cross-scale error growth", fontsize=11)
ax = axes[1]
log = json.load(open(find("tfi2d_loss.json")))[-1]
ax.plot(np.arange(1, len(log["epoch_losses"]) + 1), log["epoch_losses"], "-", color="#1f77b4")
ax.set_xlabel("epoch"); ax.set_ylabel("training loss (masked CE)")
ax.set_title("training convergence", fontsize=11)
fig.tight_layout()
save(fig, "fig5_scaling_and_loss")
print("all figures written to", OUT)
