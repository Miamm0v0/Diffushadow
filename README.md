# Diffushadow oseq-rope Experiments

This directory contains the runnable code used for the Diffushadow
`oseq_rope` experiments.  The main goal is to train a diffusion model on
small quantum systems and evaluate whether the learned sampler generalizes to
larger qubit numbers.

The supported public workflow in this folder is:

1. Generate training measurement data.
2. Train `Qdmodel_oseq_rope.py` with `train_oseq.py`.
3. Evaluate the trained model with `eval_new.py`.
4. Compare against exact diagonalization with `eval_cal_exact.py`.
5. Plot observables and finite-size extrapolations.

Some scripts still contain historical options for older model variants
(`oseq`, `nseq`, `nseq_rope`, `oseq_rope_sharepos`, `oseq_rope_gnn`,
`likeshadow`, etc.).  Those were kept for record keeping and ablation tests.
The intended model for this release is `--model_type oseq_rope`.

## Main Files

- `Qdmodel_oseq_rope.py`  
  RoPE-based ordered-sequence diffusion model.  This is the model used in the
  reported experiments.

- `train_oseq.py`  
  Training entry point.  Use it with `--model_type oseq_rope`.

- `eval_new.py`  
  Evaluation and sample-generation entry point for TFI, J1J2, and ANNNI
  observables.

- `eval_cal_exact.py`  
  Exact diagonalization / sparse exact cache generation for comparison.

- `generate_j1j2_annni_dataset.py`  
  Dataset generator for J1J2 and ANNNI training data in the format consumed by
  `train_oseq.py`.

- `draw_fig_properties.py`  
  Plot model-generated observables against exact values.

- `inf_research/`  
  Finite-size extrapolation plotting scripts.

- `attn_analysis/`  
  Attention-analysis scripts used for inspecting learned attention kernels.

## Environment

The code expects a Python environment with:

- Python 3.10 or newer
- PyTorch
- NumPy
- SciPy
- Matplotlib
- tqdm

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy matplotlib tqdm
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy scipy matplotlib tqdm
```

CUDA is recommended for training and evaluation with large sample counts.

## Data Format

Training samples use the ordered-sequence format:

```text
[param, P1, b1, P2, b2, ..., PN, bN]
```

where:

- `param` is the physical scan parameter, such as `J2/J1` for J1J2 or
  transverse field `h` for ANNNI.
- `Pi` is the measurement basis token.
- `bi` is the measurement bit.

This is the format expected by `train_oseq.py` when using `--model_type
oseq_rope`.

## Generate Training Data

J1J2 example:

```bash
python generate_j1j2_annni_dataset.py j1j2 \
  --num-qubits 10 \
  --params 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --samples-per-param 800 \
  --json-out data/j1j2_train_10q.json \
  --pt-out data/j1j2_train_10q.pt
```

ANNNI example scanning transverse field `h` at fixed `kappa`:

```bash
python generate_j1j2_annni_dataset.py annni \
  --num-qubits 10 \
  --scan h \
  --kappa 0.4 \
  --params 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 \
  --samples-per-param 800 \
  --json-out data/annni_kappa0.4_train_10q.json \
  --pt-out data/annni_kappa0.4_train_10q.pt
```

For production runs, use a denser parameter grid and more samples per
parameter.

## Train `oseq_rope`

Single-size training:

```bash
python train_oseq.py \
  --model_type oseq_rope \
  --data_path data/j1j2_train_10q.json \
  --qubits 10 \
  --hidden_dim 128 \
  --layer_num 4 \
  --head_num 8 \
  --epochs 100 \
  --batch_size 256 \
  --model_save_path checkpoints/j1j2_10q_oseq_rope.pth \
  --loss_save_path logs/j1j2_10q_loss.json
```

Multi-size training:

```bash
python train_oseq.py \
  --model_type oseq_rope \
  --data_path data/j1j2_train_8q.json data/j1j2_train_10q.json \
  --qubits 8 10 \
  --hidden_dim 128 \
  --layer_num 4 \
  --head_num 8 \
  --epochs 100 \
  --batch_size 256 \
  --model_save_path checkpoints/j1j2_multi_oseq_rope.pth \
  --loss_save_path logs/j1j2_multi_loss.json
```

When evaluating at larger system sizes, keep the RoPE settings consistent
between training and evaluation:

```bash
--rope_scaling_type none --rope_scaling_factor 1.0 --rope_theta 10000
```

Use different values only if the training run used those same settings.

## Exact Reference Cache

Generate exact J1J2 values:

```bash
python eval_cal_exact.py \
  --predict_model J1J2 \
  --num_qubits 16 \
  --J1 1.0
```

Generate exact ANNNI values:

```bash
python eval_cal_exact.py \
  --predict_model ANNNI \
  --num_qubits 16 \
  --annni_kappa 0.4 \
  --J1 1.0
```

The cache files are saved under `eval_exact_cache/` by default.

## Evaluate a Trained Model

J1J2 example:

```bash
python eval_new.py \
  --model_type oseq_rope \
  --input_sequence oseq \
  --predict_model J1J2 \
  --num_qubits 16 \
  --h_length 41 \
  --model_path checkpoints/j1j2_10q_oseq_rope.pth \
  --save_data_path results/j1j2_eval_N16.npz \
  --exact_value \
  --sample_size_per_h 10000 \
  --repeat_times 6 \
  --diffusion_steps 2
```

ANNNI example:

```bash
python eval_new.py \
  --model_type oseq_rope \
  --input_sequence oseq \
  --predict_model ANNNI \
  --annni_kappa 0.4 \
  --num_qubits 16 \
  --h_length 41 \
  --model_path checkpoints/annni_10q_oseq_rope.pth \
  --save_data_path results/annni_kappa0.4_eval_N16.npz \
  --exact_value \
  --sample_size_per_h 10000 \
  --repeat_times 6 \
  --diffusion_steps 2
```

Important output keys include:

- J1J2: `J2s`, `corr_ZZ_mean`, `corr_XX_mean`, `corr_YY_mean`,
  `corr_spin_dot_mean`, `dimer_proxy_mean`, `energy_mean`.
- ANNNI: `hs`, `zz_mean`, `x_mean`, `structure_factor_pi_mean`,
  `structure_factor_pi_over_2_mean`, `energy_mean`.

## Plot Observables

J1J2 spin-dot correlation:

```bash
python draw_fig_properties.py \
  --predict_model J1J2 \
  --plot_type spin_dot \
  --distance 1 \
  --files results/j1j2_eval_N10.npz results/j1j2_eval_N16.npz \
  --labels N10 N16 \
  --output_dir figures
```

ANNNI structure factor:

```bash
python draw_fig_properties.py \
  --predict_model ANNNI \
  --plot_type sf_pi \
  --files results/annni_eval_N10.npz results/annni_eval_N16.npz \
  --labels N10 N16 \
  --output_dir figures
```

If an evaluation file stores many parameter values but only a subset should be
shown, use:

```bash
--num_h_values 21
```

This selects evenly spaced saved parameter points over the full range.

## Finite-Size Extrapolation

The `inf_research/` scripts plot observables as functions of inverse system
size.  Typical scripts are:

- `inf_research/plot_zz_inverse_qubit.py`
- `inf_research/plot_j1j2_spin_dot_inverse_qubit.py`
- `inf_research/plot_j1j2_spin_dot_inverse_qubit_g_indices.py`
- `inf_research/plot_sf_inverse_qubit.py`
- `inf_research/plot_sf_inverse_g_indices.py`

Use the `*_g_indices.py` versions when the paper figure should focus on a
selected set of parameter indices rather than the full grid.

## Supported Models and Legacy Code

For the GitHub release, the maintained model is:

```bash
--model_type oseq_rope
```

Other model names may still appear in command-line options or imports because
they were used during testing and ablations.  They are not the recommended
reproduction path for the experiments described here.

If this directory is published as a standalone repository, make sure any
legacy imports left in `eval_new.py` are either included or removed.  The
examples above use the `oseq_rope` path only.

## Suggested Repository Layout

```text
Diffushadow_run/
  Qdmodel_oseq_rope.py
  train_oseq.py
  eval_new.py
  eval_cal_exact.py
  eval_utils.py
  generate_j1j2_annni_dataset.py
  draw_fig_properties.py
  inf_research/
  attn_analysis/
  data/          # generated locally, usually not committed
  checkpoints/   # generated locally, usually not committed
  results/       # generated locally, usually not committed
  figures/       # generated locally, commit only selected paper figures
```

Large generated files such as `.pt`, `.pth`, `.npz`, and full result folders
are usually better kept outside git or tracked with a separate artifact
storage mechanism.
