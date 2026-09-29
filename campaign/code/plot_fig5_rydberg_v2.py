#!/usr/bin/env python3
"""Draft manuscript Fig. 5, v2 (compact 2x2): completing partially observed
Rydberg snapshots.  Panels a, b, d of the full figure (plot_fig5_rydberg.py),
relabelled a, b, c:

  a  single shots, delta = 5.0, 30% loss: measured vs completed (full width)
  b  structure factor, delta = 5.0, 30% loss
  c  per-site reconstruction vs the exact Bayes limit, delta = 0.2-6.0

The full five-panel version is unchanged.  Original header follows.


Held-out DMRG snapshots of the N = 32 Rydberg chain (Rb/a = 2.35). A random
fraction of atoms per shot is discarded ("lost") and completed by
Diffushadow, conditioned on the surviving atoms.

a  single shots at delta = 5.0 (Z3 phase), 30% loss: measured vs completed
b  structure factor at delta = 5.0, 30% loss
c  statistics gained by completing rather than discarding incomplete shots
   (k-site observable, delta = 5.0, 30% loss).  Blockade violations are now
   reported in the text/supplementary instead of plotted.

d  per-site reconstruction accuracy on the lost atoms (most-likely decoding),
   against the exact Bayes limit from conditioning the DMRG state on the
   surviving atoms of each shot (campaign/code/ryd_bayes_bound.py)
e  distribution of the excited-atom number per shot, delta = 5.0, 30% loss

Panels a, b, c and e use posterior samples (right for statistics); panel d uses
most-likely decoding (right for reconstructing one shot).  Both runs share
identical hidden masks.

b and e share one control ("model fill, unconditioned"): snapshots generated
from the model with no observed atoms, spliced into the lost sites of each
shot.  They carry the right density and obey the blockade internally, but know
nothing about the surviving atoms of that particular shot, so they isolate the
effect of conditioning.

Output: campaign/figures/fig5_rydberg/FIG5_rydberg_draft.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from figstyle import SURFACE, INK, INK2, MUTED, AXIS, GRID, BLUE, setup, style, letter, save

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "campaign/results_1d/rydberg"
OUT = ROOT / "campaign/figures/fig5_rydberg"
EXC, GND = "#3a3935", "#ebeae4"
CTRL = "unconditioned"   # same control in b and c: model fill with no observed atoms
LOSS_COL = {0.05: "#5598e7", 0.3: "#1c5cab"}   # ordinal blue: light = 5%, dark = 30%
Z3 = 5.0
N_SHOTS = 12


def tag(d):
    return f"d{float(d)}"


def gallery(rec, f, d):
    return tuple(np.asarray(rec[f"{tag(d)}_f{f}_gal_{k}"]) for k in ("exact", "hidden", "inp"))


def draw_shots(ax, exact, hidden, completed):
    n_shot, n_site = exact.shape
    pad = 3
    for block, arr in enumerate((exact, completed)):
        x0 = block * (n_site + pad)
        for s in range(n_shot):
            y = n_shot - 1 - s
            for i in range(n_site):
                lost = hidden[s, i]
                if block == 0 and lost:
                    ax.add_patch(Rectangle((x0 + i, y), 0.9, 0.9, facecolor=SURFACE,
                                           hatch="//////", edgecolor=MUTED, lw=0.3))
                else:
                    ax.add_patch(Rectangle((x0 + i, y), 0.9, 0.9, lw=0.8 if (block and lost) else 0,
                                           facecolor=EXC if arr[s, i] > 0.5 else GND,
                                           edgecolor=BLUE if (block and lost) else "none"))
        ax.text(x0 + n_site / 2, n_shot + 0.6, ("measured, 30% of atoms lost" if block == 0
                                                else "completed by Diffushadow"),
                ha="center", va="bottom", fontsize=6.8, color=INK2)
    # manual axes: sites along x (1-based), shot number down y
    for block in (0, 1):
        x0 = block * (n_site + pad)
        for site in (1, 8, 16, 24, n_site):
            xc = x0 + site - 0.55
            ax.plot([xc, xc], [-0.45, -0.15], color=AXIS, lw=0.6, clip_on=False)
            ax.text(xc, -0.75, str(site), ha="center", va="top", fontsize=5.6, color=INK2)
        ax.text(x0 + n_site / 2, -1.55, "lattice site", ha="center", va="top",
                fontsize=6.5, color=INK)
    for s in range(n_shot):
        ax.text(-0.7, n_shot - 1 - s + 0.45, str(s + 1), ha="right", va="center",
                fontsize=5.6, color=INK2)
    ax.set_xlim(-0.5, 2 * n_site + pad + 0.5)
    ax.set_ylim(-3.6, n_shot + 2.2)
    ax.axis("off")
    keys = [(Rectangle((0, 0), 1, 1, facecolor=EXC), "Rydberg"),
            (Rectangle((0, 0), 1, 1, facecolor=GND), "ground"),
            (Rectangle((0, 0), 1, 1, facecolor=SURFACE, hatch="//////", edgecolor=MUTED, lw=0.3), "lost"),
            (Rectangle((0, 0), 1, 1, facecolor=GND, edgecolor=BLUE, lw=0.9), "completed site")]
    ax.legend(*zip(*keys), loc="lower center", ncol=4, frameon=False, fontsize=6.2,
              bbox_to_anchor=(0.5, -0.14), handlelength=1.2)
    ax.text(-3.8, n_shot / 2, "shot number", rotation=90, ha="center", va="center",
            fontsize=6.5, color=INK)


def boundary_rate(occ, hidden):
    straddle = hidden[:, :-1] ^ hidden[:, 1:]
    return float((occ[:, :-1] * occ[:, 1:])[straddle].mean())


def main():
    setup()
    samp = np.load(R / "ryd_paired_sample.npz")     # posterior samples: a, b, c, e
    arg = np.load(R / "ryd_paired_argmax.npz")      # most-likely decoding: d
    rec = {0.05: samp, 0.3: samp}                   # identical hidden masks in both runs
    inp = samp
    deltas = [float(d) for d in np.asarray(samp["deltas"])]
    labels = [f"{d:.1f}" for d in deltas]

    fig = plt.figure(figsize=(7.2, 5.4), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.0], hspace=0.42, wspace=0.26,
                            left=0.075, right=0.985, top=0.94, bottom=0.085)
    a = fig.add_subplot(grid[0, :])
    b, d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    # the full figure's panels c and e are still computed (their numbers are
    # printed) but drawn on a scratch figure that is never saved
    scratch = plt.figure()
    c, e = scratch.add_subplot(121), scratch.add_subplot(122)
    for ax in (b, c, d, e):
        style(ax)

    # a: example shots
    ex, hid, comp = gallery(rec[0.3], 0.3, Z3)
    draw_shots(a, ex[:N_SHOTS], hid[:N_SHOTS].astype(bool), comp[:N_SHOTS])
    a.text(0.02, 1.07, r"single shots, $\delta = 5.0$ ($Z_3$ phase)", transform=a.transAxes,
           fontsize=7.5, color=INK2, va="bottom")

    # b: structure factor
    # same ensemble as panel c when the re-run with per-shot control is present
    src = rec[0.3] if f"{tag(Z3)}_f0.3_Sq_spl" in rec[0.3].files else inp
    qs = np.asarray(src["qs"], float)
    b.plot(qs, src[f"{tag(Z3)}_ref_Sq"], color=INK, lw=1.4, label="exact")
    b.plot(qs, src[f"{tag(Z3)}_f0.3_Sq"], color=BLUE, lw=1.2, label="completed")
    b.plot(qs, src[f"{tag(Z3)}_f0.3_Sq_spl"], color=MUTED, lw=1.0, label=CTRL)
    b.axvline(2 * np.pi / 3, color=AXIS, lw=0.8)
    b.text(2 * np.pi / 3 + 0.22, 0.62, r"$q = 2\pi/3$", transform=b.get_xaxis_transform(),
           va="center", fontsize=6.5, color=INK2)
    b.set_xlabel(r"$q$")
    b.set_ylabel(r"$S(q)$")
    b.set_title(r"structure factor, $\delta = 5.0$, 30% loss", color=INK2, pad=3)
    b.legend(frameon=False, loc="upper left", fontsize=5.8)

    # c: statistics gained by completing instead of discarding incomplete shots.
    # Observable: occupation on k sites spaced by the Z3 period. Discarding any
    # shot that lost one of those sites keeps only (1-f)^k of them.
    KS = np.arange(2, 9)
    gex, ghid, gcom = gallery(rec[0.3], 0.3, Z3)
    ghid = ghid.astype(bool)
    gctl = np.asarray(rec[0.3][f"{tag(Z3)}_f0.3_gal_spl"])
    M, N = gex.shape
    ev, sv, se, cv, ce, ns, uv = [], [], [], [], [], [], []
    for k in KS:
        offs = [3 * j for j in range(k)]
        e_, s_, c_, n_, u_ = [], [], [], [], []
        for start in range(N - offs[-1]):
            cols = [start + o for o in offs]
            pe = np.prod(gex[:, cols], axis=1)
            pc = np.prod(gcom[:, cols], axis=1)
            pu = np.prod(gctl[:, cols], axis=1)
            keep = ~ghid[:, cols].any(axis=1)
            e_.append(pe.mean()); c_.append(pc.mean()); n_.append(keep.sum())
            u_.append(pu.mean())
            s_.append(pe[keep].mean() if keep.sum() else np.nan)
        e0, s0, c0, n0 = np.mean(e_), np.nanmean(s_), np.mean(c_), np.mean(n_)
        ev.append(e0); sv.append(s0); cv.append(c0); ns.append(n0); uv.append(np.mean(u_))
        se.append(np.sqrt(max(s0 * (1 - s0), 1e-12) / max(n0, 1)))
        ce.append(np.sqrt(max(c0 * (1 - c0), 1e-12) / M))
    c.plot(KS, ev, color=INK, lw=1.4, zorder=4, label="exact")
    c.errorbar(KS, sv, yerr=se, fmt="s", ms=3.2, mfc=SURFACE, mec=INK, mew=0.9, color=INK,
               elinewidth=0.9, capsize=1.5, label="discard incomplete")
    c.errorbar(KS, cv, yerr=ce, fmt="o", ms=3.2, color=BLUE, mec=SURFACE, mew=0.4,
               elinewidth=0.9, capsize=1.5, label="completed (all shots)")
    c.plot(KS, uv, "^-", ms=3.0, color=MUTED, mec=SURFACE, mew=0.4, lw=0.9, label=CTRL)
    c.set_xlabel(r"sites in the observable, $k$")
    c.set_ylabel(r"$\langle n_i n_{i+3} \cdots \rangle$")
    c.set_title(r"statistics from lossy shots, $\delta = 5.0$", color=INK2, pad=3)
    h_, l_ = c.get_legend_handles_labels()      # same leading order as b and e
    order = [l_.index("exact"), l_.index("completed (all shots)"),
             l_.index(CTRL), l_.index("discard incomplete")]
    c.legend([h_[i] for i in order], [l_[i] for i in order], frameon=False,
             loc="lower left", fontsize=5.6, handlelength=1.4, borderpad=0.1, labelspacing=0.3)

    # blockade: reported in the text / supplementary, not plotted
    rates = {"exact": [], "completed": [], CTRL: []}
    for dl in deltas:
        ex_, hid_, comp_ = gallery(rec[0.3], 0.3, dl)
        hid_ = hid_.astype(bool)
        ctrl = np.asarray(rec[0.3][f"{tag(dl)}_f0.3_gal_spl"])
        for key, occ in (("exact", ex_), ("completed", comp_), (CTRL, ctrl)):
            rates[key].append(100 * boundary_rate(occ, hid_))

    # d: per-shot exactness
    # per-site accuracy on the lost atoms vs the exact limit, 30% loss.  Two runs:
    # the original grid (1000 shots) and an extended grid (500 shots) reaching
    # beyond the training range (delta <= 5.0).  Masks are seeded per
    # (delta, loss), so model, baseline and bound score identical lost sites.
    persite = lambda ex, hid, a: 100 * (ex[hid] == a[hid]).mean()
    sources = [(arg, np.load(R / "ryd_bayes_bound.npz"))]
    if (R / "ryd_ext_argmax.npz").exists() and (R / "ryd_ext_bound.npz").exists():
        sources.append((np.load(R / "ryd_ext_argmax.npz"), np.load(R / "ryd_ext_bound.npz")))
    pts = {}
    for run, bd in sources:
        for dl in [float(x) for x in bd["deltas"]]:
            ex, hid, comp = gallery(run, 0.3, dl)
            hid = hid.astype(bool)
            pj = np.asarray(bd[f"{tag(dl)}_f0.3_p"])[np.asarray(bd[f"{tag(dl)}_f0.3_hidden"]).astype(bool)]
            dens = np.asarray(run[f"{tag(dl)}_ref_density"], float)
            guess = np.tile((dens > 0.5).astype(float), (ex.shape[0], 1))
            pts[dl] = (persite(ex, hid, comp), 100 * np.mean(np.maximum(pj, 1 - pj)),
                       persite(ex, hid, guess))
    dd = sorted(pts)
    acc, ceil, tacc = (np.array([pts[x][k] for x in dd]) for k in range(3))
    span = d.axvspan(5.0, max(dd) + 0.3, color=GRID, alpha=0.6, lw=0, zorder=0)
    l_lim, = d.plot(dd, ceil, color=INK, lw=1.4, zorder=4)
    l_mod, = d.plot(dd, acc, "-o", color=LOSS_COL[0.3], lw=1.0, ms=3.2, mec=SURFACE, mew=0.4, zorder=5)
    l_tri, = d.plot(dd, tacc, "-^", color=MUTED, lw=0.9, ms=3.0, mec=SURFACE, mew=0.4, zorder=3)
    d.legend([l_lim, l_mod, l_tri, span], ["exact limit", "completed", "site-wise guess",
                                           "beyond training range"],
             frameon=False, loc="lower center", bbox_to_anchor=(0.46, 0.0), fontsize=5.6,
             handlelength=1.4, borderpad=0.1, labelspacing=0.3)
    d.set_xlim(-0.1, max(dd) + 0.3)
    print("panel d:", {round(x, 1): tuple(round(v, 1) for v in pts[x]) for x in dd})
    d.set_ylim(84, 100.5)
    d.set_xticks(range(0, 7))
    d.set_xlabel(r"detuning $\delta/\Omega$")
    d.set_ylabel("lost atoms predicted correctly (%)")
    d.set_title("reconstruction, 30% loss", color=INK2, pad=3)

    # e: excited-atom number distribution
    ex, hid, comp = gallery(rec[0.3], 0.3, Z3)
    ctrl = np.asarray(rec[0.3][f"{tag(Z3)}_f0.3_gal_spl"])
    ne, nc, nu = ex.sum(1), comp.sum(1), ctrl.sum(1)
    lo = min(ne.min(), nc.min(), nu.min()) - 0.5
    hi = max(ne.max(), nc.max(), nu.max()) + 1.5
    edges = np.arange(lo, hi)
    e.hist(nu, bins=edges, density=True, histtype="step", color=MUTED, lw=1.0, label=CTRL)
    e.hist(ne, bins=edges, density=True, histtype="step", color=INK, lw=1.3, label="exact")
    e.hist(nc, bins=edges, density=True, histtype="step", color=BLUE, lw=1.1, label="completed")
    e.set_xlabel("excited atoms per shot")
    e.set_ylabel("probability")
    e.set_title(r"counting statistics, $\delta = 5.0$", color=INK2, pad=3)
    h_, l_ = e.get_legend_handles_labels()          # drawn grey-first; show exact, completed, control
    order = [l_.index("exact"), l_.index("completed"), l_.index(CTRL)]
    e.legend([h_[i] for i in order], [l_[i] for i in order],
             frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.95), fontsize=5.8)

    letter(a, "a", x=-0.03, y=1.07)
    letter(b, "b", x=-0.13)
    letter(d, "c", x=-0.13)
    plt.close(scratch)
    save(fig, OUT, "FIG5_rydberg_v2")
    print("blockade (%):", {k: np.round(v, 2).tolist() for k, v in rates.items()})
    print(f"counting std delta=5, 30%: exact {ne.std():.3f} completed {nc.std():.3f}")


if __name__ == "__main__":
    main()
