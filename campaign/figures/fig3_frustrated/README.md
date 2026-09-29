# Draft Fig. 3 — frustrated chains (J1-J2, ANNNI κ=0.4)

- Script: `campaign/code/plot_fig3_frustrated.py`
- Data: the student's bootstrap fits, commit 4d3614d
  (`inf_research/bootstrap_*_extrapolated_vs_g_plots_run*/bootstrap_infinite.csv`).
- References: `refs/idmrg/idmrg_j1j2.npz` and `idmrg_annni.npz` (iDMRG, chi=256).

**Panels.** a–d: N→∞ estimates with the 95% bootstrap interval, on the exact
iDMRG curves. e–h: residuals (estimate − exact).

**Numbers.**
- J1-J2 r=1: mean |dev| 0.045. The worst point is J2=0: −1.681 vs −1.7726.
- J1-J2 r=2: mean |dev| 0.041, max 0.091.
- J1-J2 coverage: the exact value is inside the interval at only 3/15 points
  per distance, because the bootstrap interval is statistical only.
- ANNNI S(π): mean |dev| 0.011; S(π/2): 0.010. The exact value is inside the
  interval at 14/14 points away from h_c ≈ 0.25.
- Hollow points lie within 0.1 of h_c and are not graded.

**ANNNI reference.** Standard two-sided connected structure factor,
S(q) = 2·SF_stored − (1 − m_z²). The stored iDMRG value is a one-sided sum.

**Open.**
- A systematic error band for J1-J2.
- ANNNI fits use N ≤ 40; the N=64/72 files are unused (ask the student why).
- A provenance table: checkpoints, training sizes, recursion rounds, decoding steps.

---

## v2 (2026-09-16) — `FIG3_frustrated_v2.{png,pdf}`

Script: `campaign/code/plot_fig3_frustrated_v2.py`. The v1 draft
(`FIG3_frustrated_draft.*`, with residual panels e–h) is unchanged.

**a–d** are identical to v1: N→∞ estimates with the 95% bootstrap interval on
the exact iDMRG curves. The residual row is replaced by physics illustrations,
and each panel now carries its mean deviation as a note.

**e** J1-J2 chain: antiferromagnetic J1 (blue, nearest neighbour) and J2
(orange, next-nearest). J2 frustrates the alternating order. The staggered
arrows are schematic — the Heisenberg chain has no true Néel order.

**f** J1-J2 phases along the same axis as a and b: gapless up to
J2c = 0.2411 (Okamoto–Nomura), spontaneously dimerized and gapped above it,
with the Majumdar–Ghosh point (exact dimer product state) at J2/J1 = 0.5 and
incommensurate real-space correlations beyond the Lifshitz point 0.5241.

**g** ANNNI chain: ferromagnetic J1 (blue), frustrating κJ1 at distance 2
(orange), transverse field h (green).

**h** ANNNI phases along the same axis as c and d: ferromagnetic below
h_c ≈ 0.25, paramagnetic above. The Peschel–Emery disorder line sits at
h = 1/(4κ) − κ = 0.225 for κ = 0.4, just inside the ordered phase.

h_c is bracketed by the iDMRG grid only (ordered at h = 0.2, disordered at
h = 0.3); refine it if the value is quoted in the text.

**Draft caption.** Thermodynamic-limit inference in frustrated spin chains.
**a–d**, Correlations and structure factors extrapolated to N → ∞ from
Diffushadow-generated measurements (markers, with 95% bootstrap intervals),
compared with exact iDMRG (lines). Open markers lie within 0.1 of the ANNNI
transition, where interpolating the reference grid is not meaningful, and are
excluded from the quoted deviations. **e–h**, The two models and their phase
diagrams, on the parameter axes of the panels above.

---

## v3 (2026-09-16) — `FIG3_frustrated_v3.{png,pdf}` — Nature-style

Script: `campaign/code/plot_fig3_frustrated_v3.py`. v1 and v2 are unchanged.

Format follows the Nature figure guide: 180 mm double-column width (110 mm
tall), Helvetica/Arial at 5.5–6.5 pt, bold lower-case panel letters at 8 pt,
hairline rules (0.5 pt), no panel titles or legend boxes, 600 dpi PNG and a
vector PDF with embedded Type 42 fonts.

**e, g** Model schematics: nearest-neighbour bond (blue), frustrating
next-nearest bond (orange), transverse field (green, ANNNI only).

**f** J1-J2 phase diagram on the axis of panels a, b, plus the **Majumdar-Ghosh
locator** (inset).

**h** ANNNI phase diagram on the axis of panels c, d.

### Can the MG point be probed? Yes — and v3 shows it

MG is **not** a phase transition: it is the point *inside* the dimerized phase
where the ground state is exactly a product of nearest-neighbour singlets. In
that state sites i and i+2 always belong to different singlets, so

    <sigma_i . sigma_{i+2}> = 0   identically at J2/J1 = 0.5.

The zero of the r = 2 correlation therefore locates MG. Checks:

- cubic root of the exact iDMRG curve: **0.5000** (method is exact);
- cubic root of the generated N->infinity curve: **0.5116 +- 0.0017**
  (bootstrap-propagated), i.e. 2.3% high.

The error is systematic, not statistical. Note that a *uniform* amplitude bias
cancels in a zero crossing, so this locator is insensitive to the model's
overall ~5% under-correlation; the residual shift comes from the J2-dependence
of that bias. This makes the MG locator a fair "physics probe" demonstration:
the model finds a known exact point of the phase diagram to ~2% without being
told where it is.

**The actual transition at J2c = 0.2411 is not probeable this way.** It is
BKT-type: the gap opens exponentially slowly and two-point correlations are
smooth through it (visible in a and b). Probing it would need the dimer
(four-spin) correlator <(S_i.S_{i+1})(S_{i+r}.S_{i+r+1})> or level
spectroscopy. The four-spin correlator is computable from the same shadow
snapshots, but the shadow variance grows as 3^k with the number of sites k in
the operator (81x for k = 4), so it needs far more snapshots — a possible
addition, not a free one.

**Draft caption (v3).** Thermodynamic-limit inference in frustrated spin
chains. **a-d**, Correlations and structure factors extrapolated to
N -> infinity from Diffushadow-generated measurements (markers, 95% bootstrap
intervals) against exact iDMRG (lines); open markers lie within 0.1 of the
ANNNI transition and are excluded from the quoted deviations. **e**, The
J1-J2 chain: antiferromagnetic J1 with frustrating next-nearest J2; arrows are
schematic. **f**, Its phase diagram: gapless below J2c = 0.241, dimerized
above, with incommensurate correlations beyond 0.524. Inset: the
Majumdar-Ghosh point, where <sigma_i.sigma_{i+2}> vanishes identically,
located from the generated data at J2/J1 = 0.512 +- 0.002 against the exact
0.500. **g**, The ANNNI chain: ferromagnetic J1, frustrating kappa*J1 at
r = 2, transverse field h. **h**, Its phase diagram at kappa = 0.4, with the
Peschel-Emery disorder line inside the ordered phase.

---

## v4 (2026-09-16) — `FIG3_frustrated_v4.{png,pdf}` — MG marked on the data

Script: `campaign/code/plot_fig3_frustrated_v4.py`. v1–v3 unchanged.

Same Nature format as v3. The only change is where the Majumdar-Ghosh result
lives:

- **panel b** now carries it, because it is a *data* result: a hairline at
  zero, a marker at the crossing of the generated curve, and the number
  `0.512 +- 0.002 (exact 0.500)` on a leader.
- **panel f** keeps only an "MG" tick on the phase bar, so the same fact is
  not stated twice.

A zoomed inset inside b was tried and rejected: at 43 mm panel width it
collides with the curve and with the J2c / MG tick labels, and it duplicates
what the zero line already shows. If a zoom is wanted later, the better home
is an Extended Data panel at full size.

**Caption addition for b.** "The next-nearest-neighbour correlation vanishes
identically in the Majumdar-Ghosh state; the zero of the generated curve
locates that point at J2/J1 = 0.512 +- 0.002, against the exact value 0.500."

---

## v5 (2026-09-16) — `FIG3_frustrated_v5.{png,pdf}` — RECOMMENDED

Script: `campaign/code/plot_fig3_frustrated_v5.py`. Earlier versions kept.

Same Nature format as v3/v4, with the Majumdar-Ghosh locator **removed**: no
zero line, no crossing marker, no quoted number. MG survives only as a
labelled landmark of the phase diagram (a vertical rule in a and b, a tick in
f), which is standard for the J1-J2 chain and makes no claim.

Fonts: Arial first, then Helvetica. Helvetica on macOS triggers a
"1 extra bytes in post.stringData array" warning when matplotlib embeds it in
the PDF; Arial embeds cleanly. Both are acceptable to Nature.

The MG analysis itself is preserved in this README (v3 section) and in
`plot_fig3_frustrated_v4.py` if it is ever wanted for Extended Data:
generated zero crossing 0.512 +- 0.002 against the exact 0.500. Note it is a
special point *inside* the dimerized phase, not a transition, and the true
transition at J2c = 0.241 is invisible to two-point correlations.

**Version map.** v1 residual panels; v2 illustrated bottom row; v3 Nature
format + MG inset in f; v4 MG marked on the data in b; **v5 = v3/v4 formatting
without the MG locator — use this one.**
