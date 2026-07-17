# Reference additions for the Diffushadow manuscript — action list

**For:** manuscript follow-up (citation/novelty-positioning pass)
**Files:** the BibTeX entries are in [`missing_refs.bib`](missing_refs.bib) (this folder).
**Task in one line:** merge `missing_refs.bib` into `sn-bibliography.bib`, then add the
citations and the differentiation text described below at the four insertion points.

These references were surfaced by the RAClaw_KG literature graphs
(J1-J2: 176 papers, ANNNI: 109 papers) and reviewed by the supervising check
of the manuscript. None of them are currently cited anywhere in
`sn-article.tex` (verified by grep). The biggest gap is item 1 — a specialist
referee on the J1-J2 chain will notice it immediately.

---

## 1. J1-J2 generative/transformer prior art (HIGHEST PRIORITY)

**Cite:** `viteritti2023transformer` (PRL 130, 236401), `viteritti2022accuracy`
(SciPost Phys. 12, 166), `rende2025foundation` (Nat. Commun. 2025),
`rahaman2023machine` (J. Phys.: Condens. Matter 2023).

**Where (two places):**
1. Introduction, the paragraph contrasting autoregressive and diffusion
   approaches (currently around the `sharir2020deep, hibat2020recurrent,
   yao2024shadowgpt` citations).
2. Results, opening of "Generalization to frustrated and competing-interaction
   spin chains".

**Why:** the Viteritti/Rende/Becca line applies transformer networks to the
*same 1D J1-J2 Hamiltonian* we use. Not citing direct prior work on our own
benchmark model is the largest novelty-positioning gap in the current draft.

**How to differentiate (this is the important part — don't just add the
citations, add the distinction).** Their goal is *variational ground-state
optimization*: the transformer is a wavefunction ansatz whose parameters are
optimized against the Hamiltonian (VMC), and the Hamiltonian must be known
exactly. Our goal is *measurement-distribution generation*: Diffushadow learns
the distribution of classical-shadow outcomes directly from measurement data
(no Hamiltonian in the loss, no variational principle), and its purpose is
cross-scale generation and finite-size extrapolation from small-system data.
Suggested LaTeX for the Introduction (adapt freely):

```latex
Transformer-based variational wave functions have recently achieved
state-of-the-art accuracy for frustrated spin systems, including the
one-dimensional $J_1$--$J_2$ chain studied
here~\cite{viteritti2022accuracy,viteritti2023transformer,rende2025foundation}.
These approaches optimize a neural ansatz against a known Hamiltonian
within variational Monte Carlo. Diffushadow addresses a complementary
problem: it learns the distribution of randomized measurement outcomes
directly from experimentally accessible classical-shadow data, without
access to the Hamiltonian in its training objective, and uses the learned
distribution for cross-scale measurement generation and thermodynamic-limit
inference.
```

Also cite `rahaman2023machine` in the J1-J2 results section as ML-based
phase-transition work on the same chain.

## 2. J1-J2 physics benchmark

**Cite:** `okamoto1992fluid` (Phys. Lett. A 169, 433).

**Where:** Methods, right after the J1-J2 Hamiltonian (Eq. for
$H_{J_1\text{-}J_2}$), and optionally near the thermodynamic-limit discussion
of the $J_1$-$J_2$ results.

**Why:** locates the gapless-to-dimerized transition at
$J_2^c/J_1 \approx 0.2411$ — the canonical reference point any reviewer will
expect when a J1-J2 scan is presented. It also motivates the Majumdar-Ghosh
exact-state benchmark at $J_2/J_1 = 0.5$ (see `mg_benchmark.py` in the repo
root; separate task).

## 3. ANNNI + classical shadows + ML (recent, 2024-2026)

**Cite:** `morais2026distinguishing`, `ho2026unsupervised`, `ito2026learning`.

**Where:** Introduction (recent ML-on-measurement-data context) and/or the
ANNNI results discussion.

**Why:** these combine classical shadows with machine learning for phase
identification on ANNNI-type systems — the "recent probe" referees will expect
us to know about.

**CAUTION — verify before citing:** these entries came from automated
OpenAlex records. `ho2026unsupervised` has an unusual DOI format and
`ito2026learning` is arXiv-only. Please check each DOI/arXiv ID resolves and
the metadata (journal, year, author list) is correct before merging.

## 4. ANNNI finite-size scaling benchmarks

**Cite:** `beccaria2007floating` (PRB 76, 094410), `nagy2011exploring` (NJP 2011).

**Where:** wherever finite-size scaling toward the thermodynamic limit is
claimed for the ANNNI model (Results, extrapolation paragraphs; Methods if a
finite-size-scaling subsection is added).

**Why:** canonical scaling references for the transverse ANNNI chain; they
also make clear that our scan at $\kappa = 0.4$ sits *below* the multicritical
point $\kappa = 0.5$ (no floating/antiphase physics in our parameter range —
worth one clarifying sentence in the ANNNI results text, and note that the
value $\kappa=0.4$ is currently not stated anywhere in the manuscript; add it
to Methods).

---

## Related positioning note (baselines, from the same review)

`yao2024shadowgpt` (ShadowGPT), our main baseline, is **arXiv-only** (not
peer-reviewed as of July 2026) — keep citing it as the closest prior work, but
the published anchors for positioning are:

* Carrasquilla et al., *Nat. Mach. Intell.* **1**, 155 (2019) — already in the
  bib as `carrasquilla2019reconstructing`; the founding published work on
  autoregressive generative models of measurement distributions. Should be
  cited in the autoregressive-vs-diffusion paragraph, not only in the general
  intro list.
* Cho & Kim, *Nat. Commun.* **15**, 7552 (2024) — already in the bib as
  `cho2024machine`; the published bar for classical-shadow data + ML +
  scalability (up to 44 qubits on IBM hardware). Engage it explicitly in the
  Discussion when claiming scalability.
* Wang, Weber, Izaac, Lin, arXiv:2211.16943 — conditional generative
  transformer over shadow data (closest methodological cousin; also
  unpublished). Consider adding to the bib and differentiating: they predict
  properties from a conditional model; we generate measurement configurations
  and extrapolate across system size.

## Checklist

- [ ] Verify DOIs/metadata of the three 2026 ANNNI entries (item 3 caution).
- [ ] Append `missing_refs.bib` entries to `sn-bibliography.bib`.
- [ ] Introduction: add differentiation paragraph (item 1) + Carrasquilla in
      the autoregressive-vs-diffusion paragraph.
- [ ] J1-J2 results section: cite Viteritti/Rende line + `rahaman2023machine`.
- [ ] Methods (J1-J2 Hamiltonian): cite `okamoto1992fluid`.
- [ ] ANNNI results: cite items 3 & 4; state $\kappa=0.4$ explicitly and note
      it is below the multicritical $\kappa=0.5$.
- [ ] Recompile and check no duplicate bib keys / BibTeX warnings.
- [ ] While in the bib: delete unused template entries `bib1`–`bib13`, and
      either cite or remove `ibarra2025autoregressive`, `medvidovic2024neural`,
      `zhang2023transformer`.
