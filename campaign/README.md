# Diffushadow manuscript campaign — data, code and figures

Everything produced during the July 2026 manuscript-improvement campaign
(validation, new benchmarks, and exploratory studies), reorganised into one
tree. Compute ran on the THK cluster; this folder holds the analysed
results, not the raw generated snapshots (those live on node 109 under
`/data/diffushadow_2d/`).

## Layout

```
campaign/
├── code/            all pipeline + analysis scripts (see below)
├── figures/         every figure produced (PDF + PNG)
├── reports/         the multi-section status PDF
├── refs/            reference calculations ("ground truth")
│   ├── dmrg_1d/       TFI & ANNNI chains, N = 32–96
│   ├── dmrg_2d/       4×L tori (the 2D tube)
│   ├── idmrg/         infinite-system (N→∞) anchors
│   └── qmc_sse/       stochastic series expansion QMC, with error bars
├── results_1d/      model outputs for 1D systems
│   ├── tfi/           incl. student_zz_run1.csv (recursion extrapolations)
│   ├── annni/         κ=0.6 floating-phase study
│   ├── j1j2/          frustration scan + exact N=12 profile
│   └── rydberg/       cut A (R_b/a=1.2) and cut B (2.35)
└── results_2d/      model outputs for 2D systems
    ├── tfi_tube/      4×L recursion ladder, 16 → 80 qubits
    ├── tfi_lxl/       L×L width-generalisation pilot
    └── j1j2_cyl/      frustrated cylinders (in progress)
```

Scripts locate data **by filename**, via `code/paths.py` — so moving files
between subfolders will not break them. Use `find("some_file.npz")`.

## Code

**Data generation (reference + training data)**
| script | purpose |
|---|---|
| `generate_tfi2d_dataset.py` | 2D TFI ground states (matrix-free Lanczos) + classical shadows |
| `j1j2_2d_dataset.py` | 2D J1-J2 cylinders: DMRG + **shadow sampling from an MPS** |
| `rydberg_dataset.py` | Rydberg chains (van der Waals), occupation snapshots |
| `braket_r2.py` | real-device AHS program for QuEra Aquila (+ free local emulator) |

**Reference calculations**
| script | purpose |
|---|---|
| `dmrg_validate_tfi.py` / `dmrg_validate_1d.py` | 1D DMRG references (TFI / J1-J2 / ANNNI) |
| `dmrg_tube4.py` | 2D tube: finite DMRG and infinite-system iDMRG |
| `idmrg_anchors.py` | N→∞ anchors for all three 1D models |
| `sse_tfi.py` | **SSE quantum Monte Carlo** (independent of DMRG; has `--selftest`) |

**Evaluation**
| script | purpose |
|---|---|
| `eval_tfi2d.py` | 2D TFI generation + correlation estimation |
| `eval_j1j2_2d.py` | frustrated cylinder evaluation |
| `eval_rydberg.py` | density + structure factor from occupation snapshots |
| `eval_annni_q.py` | q-resolved structure factor (incommensurate physics) |
| `causal_baseline.py` | the autoregressive "twin" control model |
| `inpaint_demo.py` | measurement inpainting (arbitrary-mask conditioning) |
| `mg_benchmark.py` | Majumdar-Ghosh exact-state benchmark (repo root) |

**Analysis / figures**: `make_figures_2d.py`, `analyze_dmrg_validation.py`,
`plot_tfi2d.py`, `build_summary_pdf.py`, `paths.py`

Every reference script has a self-test gate (`--selftest`) that checks it
against exact diagonalisation before production runs. **Run it first.**

## Key figures

| figure | content |
|---|---|
| `fig1_crossscale_main` | 2D TFI cross-scale: trained 4×4 → unseen 4×5, 4×6 |
| `fig4_decoding_ablation` | denoising-step count is physics-critical |
| `fig6/7_dmrg_validation*` | 1D recursion vs DMRG, N = 32–96 |
| `fig9_ar_twin_comparison` | diffusion vs autoregressive twin (2×2 control grid) |
| `fig14_tube_thermolimit` | 2D thermodynamic limit vs exact iDMRG |
| `fig16_sse_crossval` | three-way check: QMC vs DMRG vs generated |
| `fig19_ladder64_qmc` | 64-qubit rung graded by QMC |
| `fig21_annni06_verdict` | recursion fails to track incommensurate order |
| **`fig22_FIG2_UPGRADE_PROTOTYPE`** | **what manuscript Fig. 2d/e/g should become** |

## Findings that change the manuscript

1. **Denoising steps must equal N.** Coarse decoding samples sites
   conditionally independently and destroys long-range correlations
   (30–300× MSE). Currently unreported in Methods; the README's example
   value of 2 would not reproduce the published results.
2. **Recursion is validated off-critical, biased near criticality.**
   Mean deviation 0.017 (N ≤ 96) away from g_c, rising to 0.167 near it.
   Thermodynamic-limit claims must be restricted accordingly.
3. **Extrapolations need error budgets, not just fit errors.** The
   statistical fit uncertainty (±0.005) underestimates the true deviation
   from the exact N→∞ answer (0.05) by ~10×.
4. **Recursion has a depth ceiling of ~3–4 rounds.** Beyond that, error
   grows ~0.02/round *and retraining does not repair it* — the bias is
   baked into the self-generated data. Fresh exact/QMC anchors are
   required at intermediate rungs.
5. **Modulated (incommensurate) order is not preserved** under recursion,
   unlike ferromagnetic/Néel order (`fig21`).
6. **Training sizes must be commensurate** with any candidate modulation
   (a period-4 phase cannot be learned from N = 10 or 14).
7. **Measurement inpainting works** — arbitrary-mask conditional
   completion, with no analogue in DMRG/QMC and impossible for fixed-order
   autoregressive models.

## Reproducibility issues found in the main repo

- `eval_new.py` imports a module (`numericalmodel`) that is not in the repo.
- `eval_new.py` writes to a hard-coded path in a former user's home
  directory; it crashes for anyone else (after results are saved).
- Evaluation `.npz` files store `h_values_true = [nan]` — the real
  parameter grids survive only in the CSV outputs.

## Manuscript text

Drop-in LaTeX blocks are in `../manuscript/0716/`:
`tfi2d_section.tex`, `reframing_ws1.tex`, `ws2_physics_text.tex`,
`ws3_architecture_text.tex`, plus `REFERENCES_TODO.md` and
`missing_refs.bib`.
