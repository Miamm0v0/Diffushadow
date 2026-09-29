# Draft Fig. 5 — completing partially observed Rydberg snapshots

- Script: `campaign/code/plot_fig5_rydberg.py`
- Data (regenerated 2026-09-17 with **deterministic per-(delta, loss) masks**, so the
  two decodings are scored on *identical* hidden sites):
  - `results_1d/rydberg/ryd_paired_sample.npz` — posterior samples (panels a, b, c, e)
  - `results_1d/rydberg/ryd_paired_argmax.npz` — most-likely decoding (panel d)
  - `results_1d/rydberg/ryd_bayes_bound.npz`, `ryd_ext_*.npz` — exact limit and extended grid (panel d)
- System: N = 32 atom chain, open ends, Rb/a = 2.35; held-out DMRG shots (seed 4242),
  1000 per detuning; model `checkpoints_ryd/ryd_rb2.35_oseq_final.pth`, trained on N = 16, 24, 32.

**Two decodings, deliberately.** Sampling from the posterior is right for
statistics (b, c, e) and necessarily loses accuracy contests; most-likely
decoding is right for reconstructing one shot (d). Both are reported.

**The control, shared by b, c and e, labelled "unconditioned".** Model samples
drawn from delta alone, with no observed atoms, spliced into the lost sites.
Right density, blockade-obeying inside their own patch, but blind to the
surviving atoms of that shot.

**What "exact" means.** In a, c, d and e it is the same held-out shots *before*
any atoms were discarded. In b the black curve is the DMRG state itself.
"discard incomplete" in c is also real data (surviving atoms only, fewer shots),
which is why it is black rather than grey.

**Figure text policy.** No annotations inside the axes; numbers live here and in
the caption. Legend order is exact, completed, unconditioned in b, c and e.

**Panels.**
- **a** Twelve shots at delta = 5.0, 30% of atoms lost: measured vs completed.
  Axes: lattice site 1-32, shot number 1-12.
- **b** S(q) at delta = 5.0, 30% loss. Peak 0.539 exact,
  0.536 completed, 0.309 unconditioned, all at q = 2pi/3.
- **c** Statistics gained by completing rather than discarding (delta = 5.0, 30% loss).
  k-site occupation spaced by the Z3 period; discarding keeps only (1-f)^k of shots.
  - k=2: exact 0.297, completed 0.295, unconditioned 0.271;
    usable shots 467; precision gain 1.5x.
  - k=8: exact 0.064, completed 0.062, unconditioned 0.026;
    usable shots 28; precision gain 6.1x.
- **d** Per-site reconstruction on the lost atoms (most-likely decoding, 30% loss)
  against the **exact Bayes limit**, over delta = 0.2-6.0. Shaded: beyond the
  training range (the model saw delta <= 5.0 only).
  Grid: 0.2 / 1.8 / 3.4 / 5.0 from the paired runs (1000 shots) plus
  4.0 / 4.4 / 4.8 / 5.2 / 5.6 / 6.0 from `ryd_ext_{sample,argmax,bound}.npz`
  (500 shots; same per-(delta, loss) mask seeding).

  | delta | 0.2 | 1.8 | 3.4 | 4.0 | 4.4 | 4.8 | 5.0 | 5.2 | 5.6 | 6.0 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | model | 87.1 | 92.6 | 99.0 | 99.1 | 99.4 | 97.0 | 97.1 | 97.2 | 97.7 | 97.8 |
  | exact limit | 87.3 | 92.9 | 98.8 | 99.1 | 99.3 | 97.1 | 97.2 | 97.2 | 97.7 | 97.7 |
  | site-wise guess | 87.1 | 90.9 | 98.9 | 98.9 | 99.4 | 86.0 | 86.9 | 87.6 | 88.4 | 89.8 |

  - The model sits on the exact limit at all ten detunings, **including the
    three beyond its training range**.
  - The gap over the site-wise guess switches on between delta = 4.4 and 4.8,
    where an extra atom enters and delocalises (<n> 0.307 -> 0.333; bond
    dimension 53 -> 100). Below that, the commensurate crystal is frozen and
    the limit itself equals the guess.
  - Per-shot numbers (every lost atom correct), original grid: 5% loss
    74.6 / 88.7 / 98.4 / 96.8 %, 30% loss 22.6 / 50.6 / 90.3 / 81.5 %,
    site-wise guess at 30% 22.6 / 45.8 / 89.7 / 30.2 %.
- **e** Excited atoms per shot, delta = 5.0, 30% loss. Std: exact 0.521,
  completed 0.538, unconditioned 1.168 (mean 10.68 vs exact 10.72).

**Blockade (text / supplementary, not plotted).** Forbidden nearest-neighbour
pairs straddling the surviving/completed boundary, 30% loss, ~13 800 seam pairs
per detuning:
  - delta = 0.2: exact 0, completed 1 (0.007%), unconditioned 225 (1.63%)
  - delta = 1.8: exact 0, completed 0 (0.000%), unconditioned 312 (2.26%)
  - delta = 3.4: exact 0, completed 1 (0.007%), unconditioned 15 (0.11%)
  - delta = 5.0: exact 0, completed 2 (0.014%), unconditioned 508 (3.68%)

  Completion is not *exactly* zero on every realization: 4 violating pairs in
  ~55 000 (0.007-0.014%) in this run, against 1060 (0.11-3.68%) for the
  unconditioned control. Quote it as "below 0.02%", not "zero".

**Phases at Rb/a = 2.35** (from the DMRG reference): disordered below
delta ~ 1.2; Z3 order sets in ~1.4; frozen crystal (sites pinned 0.00/0.96)
through ~4.6; at delta ~ 4.7 an extra atom enters and delocalises, so at
delta = 5.0 the occupied sublattice is ambiguous (0.35/0.62). Panel a's
"Z3 phase" label is about the period; that point is the mobile-defect state.

**Exact Bayes limit (panel d).** `campaign/code/ryd_bayes_bound.py` conditions the
DMRG ground state on the atoms that survived in each shot and reads the exact
conditional marginal p_j at every lost site. No method can beat
E[max(p, 1-p)] at reconstruction, or E[p^2 + (1-p)^2] when sampling, given the
same information — it is a property of the state, not of any algorithm.
Validation: with nothing observed the marginals reproduce the DMRG density
profile to 1e-16 (the script refuses to write results otherwise). Bond
dimensions: chi = 26-100; contraction batched over shots.

Per-site accuracy on the lost atoms, delta = 0.2 / 1.8 / 3.4 / 5.0:

| quantity | 30% loss | 5% loss |
|---|---|---|
| site-wise guess (baseline) | 87.3 / 91.2 / 98.7 / 86.7 | 87.3 / 91.5 / 98.7 / 87.4 |
| model, most-likely decoding | 87.1 / 92.6 / 99.0 / 97.1 | 86.8 / 94.1 / 99.2 / 98.3 |
| **exact limit** | **87.3 / 92.9 / 98.8 / 97.2** | **87.4 / 93.4 / 98.9 / 98.4** |
| model, sampled | 79.5 / 88.5 / 97.8 / 95.8 | 79.1 / 90.3 / 98.2 / 96.8 |
| exact limit for sampling | 79.6 / 88.7 / 97.7 / 95.9 | 80.4 / 89.7 / 97.9 / 97.3 |

The model sits on the limit in both decoding modes. Where it appears to gain
nothing over the trivial guess (delta = 0.2 and 3.4) the limit itself is level
with the guess: there is no information to extract, because the state is
effectively deterministic given delta. Where information exists it captures
81% of it at delta = 1.8 and 99% at delta = 5.0.

Statistical note: at delta = 3.4 the measured accuracy (99.0%) sits a
little above the limit (98.8%). The limit is an expectation; a finite
sample can exceed it. The standard error here is 0.10%, so the gap is
~2 sigma and not evidence of anything.

Contiguous gaps do not help: at delta = 3.4 the headroom over the trivial guess
is +0.2 points for isolated sites and 0.0 for gaps of 3, 6 or 9 sites.

**Commensurability control (caption / supplementary).** The delta >= 4.8 gap
exists because N = 32 is not divisible by 3: the eleventh atom has no canonical
position. On a commensurate chain (N = 33, delta = 5.0) the headroom over the
site-wise guess is +0.07 points (99.35% guess, 99.42% limit), against +10.5 at
N = 32. So the advantage appears when the array cannot host a commensurate
crystal — common in experiments (finite arrays, imperfect filling), but a
property of the array, not of the Z3 phase. `campaign/code/ryd_commensurate_check.py`.

**Framing.** The bound is computed from DMRG, so panel d shows *optimality from
measurement data alone*, not an advantage over DMRG. The model never sees the
state or the Hamiltonian and still reaches what full knowledge of the state
allows.

**Wording.** "Conditional completion that preserves physical constraints and
statistics", not "recovery of lost atoms". Same-size claim only. Cite the APS
Open Science paper (Li & Tang, 2026).

---

## v2 (2026-09-29) — `FIG5_rydberg_v2.{png,pdf}` — compact 2x2

Script: `campaign/code/plot_fig5_rydberg_v2.py`. The five-panel draft
(`FIG5_rydberg_draft.*`, `plot_fig5_rydberg.py`) is unchanged; v2 reuses its
panel code, so the numbers are identical.

Layout: panel a spans the top row; b and c share the bottom row.

| v2 | from the full figure | content |
|---|---|---|
| a | a | twelve shots, delta = 5.0, 30% loss: measured vs completed |
| b | b | structure factor, delta = 5.0, 30% loss |
| c | d | per-site reconstruction vs the exact limit, delta = 0.2-6.0 |

Dropped from the main figure: the statistics-gain panel (old c: 1.5x -> 6.1x
precision on k-site observables) and the counting statistics (old e: spread
0.52 exact / 0.54 completed / 1.17 unconditioned). Both remain available for
Extended Data or the text; their numbers are listed above.
