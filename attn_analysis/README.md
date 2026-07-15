# Attention analysis

Utilities in this folder analyze attention behavior for diffusion shadow models.

The project pipeline is:

1. `train_oseq.py` trains a masked discrete diffusion model on classical-shadow
   rows in oseq layout: `[g, P1, b1, P2, b2, ..., PN, bN]`.
2. `Qdmodel_oseq_rope.py` embeds the scalar coupling `g`, random Pauli bases
   `P`, masked measurement bits `b`, then applies RoPE transformer blocks.
3. `eval_new.py` denoises masked `b` tokens, samples many shadow rows, and
   estimates observables such as ZZ correlations and energy.
4. `plot_zz_vs_inverse_qubit.py` fits finite-size ZZ values versus `1/N`.
5. `plot_zz_extrapolated_vs_g.py` plots the extrapolated `N -> infinity`
   ZZ curves against `g`.

## Qdmodel oseq RoPE attention distance

`analyze_attention_distance_oseq_rope.py` compares the attention span of
`Qdmodel_oseq_rope.py` across qubit sizes and `g` values.

Example:

```powershell
python DshadowGPT\attention_analysis\analyze_attention_distance_oseq_rope.py `
  --model_path DshadowGPT\model_train\your_model.pth `
  --qubits 6 8 10 12 14 `
  --g_values 0 0.2 0.5 0.7 1.0 `
  --head_num 8 `
  --layer_num 6 `
  --hidden_dim 256 `
  --num_P_sequences 128 `
  --rope_scaling_type dynamic `
  --rope_scaling_factor 4.0 `
  --output_dir /data/fyp26lyh/Diffushadow2/attn_analysis_1
  

```

The script writes:

- `attention_distance_metrics.csv`: per-layer, per-head metrics plus summary rows.
- `attention_distance_metadata.json`: run settings and metric definitions.
- `*.png`: summary curves and layer/head heatmaps.

Default behavior masks every `b` token and computes attention distance from `b`
queries. This matches the model's usual prediction target.

Important columns:

- `mean_token_distance`: attention-weighted distance in sequence-token index space.
- `mean_site_distance`: attention-weighted distance in qubit-site space over `P/b`
  keys. Attention to `g` is tracked separately by `attention_to_g`.
- `layer=-1, head=-1`: average over all real layer/head rows.
- `head=-1`: average over heads for a single layer.

## Site-displacement attention kernel

`analyze_attention_site_kernel_oseq_rope.py` asks a more physical question:
from a query site `i`, how much attention goes to `P` or `b` tokens at site
offset `r`?

Example:

```powershell
python DshadowGPT\attention_analysis\analyze_attention_site_kernel_oseq_rope.py `
  --model_path DshadowGPT\model_train\your_model.pth `
  --qubits 6 8 10 12 14 `
  --g_values 0 0.2 0.5 0.7 1.0 `
  --head_num 8 `
  --layer_num 4 `
  --key_types P b g `
  --num_prompts 128
```

This writes `attention_site_kernel.csv` plus heatmaps of the averaged kernel
`b_i -> P_{i+r}` and `b_i -> b_{i+r}`. It is useful for seeing whether heads
become local, long-range, or asymmetric as `g` changes.

## Attention vs ZZ finite-size scaling

`relate_attention_to_zz_scaling.py` joins attention metrics with the fit CSV
from `plot_zz_vs_inverse_qubit.py` or `plot_zz_extrapolated_vs_g.py`.

Example:

```powershell
python DshadowGPT\attention_analysis\relate_attention_to_zz_scaling.py `
  --attention_csv DshadowGPT\attention_analysis\results\oseq_rope_attention_distance\attention_distance_metrics.csv `
  --fit_csv zz_vs_inverse_qubit_plots\zz_infinite_fit_values.csv `
  --attention_level per_head `
  --aggregate_qubits
```

The output ranks which attention metrics and heads correlate with:

- extrapolated `ZZ(N -> infinity)`;
- finite-size slope magnitude `abs_slope`;
- fit quality `r_squared`.

This is a good first pass at asking whether the model's internal attention
geometry tracks the physical scaling behavior rather than only memorizing token
statistics.
