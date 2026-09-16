# `plot_figs` 绘图脚本使用说明

本目录包含 TFI、J1J2 和 ANNNI 的 bootstrap 有限尺寸外推与绘图脚本。所有示例都假定当前目录是仓库根目录 `Diffushadow_run`。

## 1. 环境与输入文件

至少需要：

```bash
pip install numpy matplotlib
```

为了让 `plot_tfi_zz_inverse_g_indices.py` 同时找到本目录中的 `bootstrap_inf.py` 和上一级目录中的 `plot_zz_inverse_qubit.py`，建议运行脚本前设置一次 `PYTHONPATH`。

Bash：

```bash
export PYTHONPATH="$PWD/inf_research:$PWD/inf_research/plot_figs${PYTHONPATH:+:$PYTHONPATH}"
```

PowerShell：

```powershell
$env:PYTHONPATH = "$PWD\inf_research;$PWD\inf_research\plot_figs"
```

本文的多行命令使用 Bash 的反斜杠 `\` 续行。在 PowerShell 中可以把命令写成一行，或者把每行末尾的 `\` 换成反引号 `` ` ``。

### Bootstrap NPZ 的主要字段

这些脚本通常读取 `bootstrap_snapshot_observables.py` 产生的、每个系统尺寸一个的 NPZ 文件：

- `N`：系统尺寸；如果没有，脚本会尝试从文件名推断，也可以在命令行显式提供。
- `params`、`hs_gpt`、`hs`、`J2s` 等：扫描参数。
- `d_values`：相关函数所对应的距离。
- TFI：`bootstrap_zz`，通常形状为 `[B, D, H]`。
- J1J2：`bootstrap_corr_spin_dot`，通常形状为 `[B, D, H]`。
- ANNNI：`bootstrap_structure_factor_pi` 和 `bootstrap_structure_factor_pi_over_2`，通常形状为 `[B, H]`。

其中 `B` 是 bootstrap 次数，`D` 是距离数，`H` 是扫描参数点数。

可以先检查 NPZ 中保存的键和参数网格：

```bash
python -c "import numpy as np; d=np.load('inf_research/eval_results_TFI/Diffushadow_boot_16q.npz'); print(d.files); print(d['params'])"
```

如果参数键不是 `params`，把最后的 `d['params']` 改为实际键名。

## 2. 脚本用途速查

| 脚本 | 用途 | 是否直接运行 |
|---|---|---|
| `bootstrap_inf.py` | 对每个 bootstrap replicate 做关于 `1/N` 的拟合，生成热力学极限值和置信区间 | 是 |
| `plot_bootstrap_inf.py` | 根据 `bootstrap_inf.py` 输出的 CSV，画热力学极限物理量随参数变化的图 | 是 |
| `plot_tfi_zz_inverse_g_indices.py` | 画选定 TFI 参数 index 下的 ZZ 随 `1/N` 变化、外推点和 CI | 是 |
| `plot_j1j2_spin_dot_inverse_qubit_g_indices.py` | 画选定 J1J2 参数 index 下的 spin-dot 随 `1/N` 变化 | 是 |
| `plot_annni_sf_inverse_g_indices.py` | 画选定 ANNNI 参数 index 下的结构因子随 `1/N` 变化 | 是 |
| `plot_bootstrap_inverse_g_indices.py` | J1J2/ANNNI 两个入口共用的实现 | 否 |

## 3. 通用热力学极限计算：`bootstrap_inf.py`

这个脚本读取多个系统尺寸的 bootstrap NPZ。对于每个 bootstrap replicate，它把物理量拟合为 `1/N` 的多项式，并取 `1/N=0` 的截距作为热力学极限。所有截距的分位数构成置信区间。

### TFI ZZ 示例

```bash
python inf_research/plot_figs/bootstrap_inf.py \
  --files \
    inf_research/eval_results_TFI/Diffushadow_boot_10q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_12q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_16q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_20q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_24q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_32q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_36q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_64q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_72q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_96q.npz \
  --observable_keys bootstrap_zz \
  --distance 1 \
  --fit_degree 1 \
  --confidence 0.95 \
  --pairing independent \
  --param_label g \
  --plot \
  --output_dir inf_research/bootstrap_tfi_zz_run1
```

主要输出：

- `bootstrap_infinite.csv`：每个参数点的有限尺寸均值拟合值、bootstrap mean/median/std 和 CI。
- `bootstrap_infinite.npz`：同一批结果的 NPZ 版本。
- `bootstrap_infinite_*.png`：使用 `--plot` 时生成的热力学极限曲线。

### J1J2 spin-dot 示例

```bash
python inf_research/plot_figs/bootstrap_inf.py \
  --files \
    inf_research/eval_results_J1J2/Diffushadow2_boot_16q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_20q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_24q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_32q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_40q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_64q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_72q2.npz \
  --observable_keys bootstrap_corr_spin_dot \
  --distance 1 2 \
  --fit_degree 1 \
  --confidence 0.95 \
  --param_label J2 \
  --output_dir inf_research/bootstrap_j1j2_run1
```

### ANNNI 结构因子示例

结构因子没有距离维度，因此不需要 `--distance`：

```bash
python inf_research/plot_figs/bootstrap_inf.py \
  --files \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_16q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_20q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_24q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_32q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_40q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_64q1.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_72q1.npz \
  --observable_keys \
    bootstrap_structure_factor_pi \
    bootstrap_structure_factor_pi_over_2 \
  --fit_degree 1 \
  --confidence 0.95 \
  --param_label h \
  --output_dir inf_research/bootstrap_annni_run1
```

### 常用参数

- `--parameter_values` 或 `--g_values`：只计算指定的参数值。
- `--distance`、`--distances` 或 `--components`：选择距离/分量。
- `--fit_degree 1`：关于 `1/N` 线性拟合；改为 `2` 即二次拟合。
- `--confidence 0.95`：95% 分位数置信区间。
- `--B 500`：只使用前 500 个已保存的 bootstrap replicate。
- `--pairing independent`：不同尺寸的 replicate 随机配对，默认且通常推荐。
- `--pairing index`：保留各 NPZ 中 replicate 的原始 index 配对。
- `--num_qubits ...`：当 NPZ 没有 `N` 且文件名无法推断尺寸时，按 `--files` 顺序显式提供尺寸。

## 4. 绘制热力学极限随参数变化：`plot_bootstrap_inf.py`

该脚本读取上一节生成的 `bootstrap_infinite.csv`，不再重新拟合。默认中心曲线使用 `fit_of_finite_mean`，图例为 `Diffushadow N→∞ estimate`。

### TFI ZZ 与置信带

```bash
python inf_research/plot_figs/plot_bootstrap_inf.py \
  --bootstrap_csv inf_research/bootstrap_tfi_zz_run1/bootstrap_infinite.csv \
  --observable_keys bootstrap_zz \
  --components 1 \
  --center fit_of_finite_mean \
  --ci_style band \
  --param_label g \
  --output_dir inf_research/bootstrap_tfi_zz_run1/plots
```

如果还要叠加 `plot_tfi_exact_zz_inverse_qubit.py` 生成的精确解 CSV：

```bash
python inf_research/plot_figs/plot_bootstrap_inf.py \
  --bootstrap_csv inf_research/bootstrap_tfi_zz_run1/bootstrap_infinite.csv \
  --exact_csv path/to/tfi_exact_zz_infinite_fit_errors.csv \
  --observable_keys bootstrap_zz \
  --components 1 \
  --exact_series thermodynamic \
  --ci_style both \
  --param_label g \
  --output_dir inf_research/bootstrap_tfi_zz_run1/plots_with_exact
```

可选中心值：

- `--center fit_of_finite_mean`：对各尺寸的有限尺寸均值进行拟合得到的截距，默认。
- `--center bootstrap_mean`：bootstrap 截距分布的均值。
- `--center bootstrap_median`：bootstrap 截距分布的中位数。

可选 CI 样式：`band`、`errorbar`、`both`、`none`。

## 5. TFI：选定 g index 的 ZZ–`1/N` 图

入口是 `plot_tfi_zz_inverse_g_indices.py`。它支持两条互斥输入路径：

- `--bootstrap_files`：绘制有限尺寸 percentile CI 和 `N→∞` CI。
- `--files`：读取普通 eval NPZ 的 mean/std，只进行均值拟合。

`--g-index` 使用从 0 开始的数组下标。运行前应先检查 NPZ 的参数数组，确认 index 对应的实际 g 值。

### Bootstrap 输入

```bash
python inf_research/plot_figs/plot_tfi_zz_inverse_g_indices.py \
  --bootstrap_files \
    inf_research/eval_results_TFI/Diffushadow_boot_10q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_12q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_16q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_20q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_24q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_32q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_36q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_64q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_72q.npz \
    inf_research/eval_results_TFI/Diffushadow_boot_96q.npz \
  --g-index 4 10 \
  --distances 1 \
  --fit_degree 1 \
  --confidence 0.95 \
  --output_dir inf_research/tfi_zz_inverse_g_indices
```

常用显示选项：

- `--no_annotate_infinite`：隐藏星号旁边的外推值/CI 文字，星号和 CI 仍保留。
- `--no_errorbar`：不画有限尺寸和外推点的误差条。
- `--annotate_n`：在每个有限尺寸点旁标出 `N`。
- `--no_y_break`：禁用自动断轴。
- `--fig_width`、`--fig_height`、`--dpi`：调整图尺寸和清晰度。

### 普通 eval 输入

把 `--bootstrap_files` 换成 `--files`；如果 NPZ 或文件名中没有系统尺寸，同时提供 `--num_qubits`：

```bash
python inf_research/plot_figs/plot_tfi_zz_inverse_g_indices.py \
  --files result_N10.npz result_N12.npz result_N16.npz result_N20.npz \
  --num_qubits 10 12 16 20 \
  --g-index 4 10 \
  --distances 1 \
  --output_dir inf_research/tfi_zz_inverse_eval
```

## 6. J1J2：选定 J2 index 的 spin-dot–`1/N` 图

请运行包装入口 `plot_j1j2_spin_dot_inverse_qubit_g_indices.py`，不要直接运行共享后端。

```bash
python inf_research/plot_figs/plot_j1j2_spin_dot_inverse_qubit_g_indices.py \
  --bootstrap_files \
    inf_research/eval_results_J1J2/Diffushadow2_boot_16q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_20q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_24q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_32q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_40q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_64q2.npz \
    inf_research/eval_results_J1J2/Diffushadow2_boot_72q2.npz \
  --g_indices 4 10 \
  --index_base 0 \
  --distances 1 2 \
  --fit_degree 1 \
  --confidence 0.95 \
  --output_dir inf_research/j1j2_spin_dot_inverse_g_indices
```

这个入口要求 `--g_indices` 恰好给出两个不同的参数 index。默认 `--index_base 0`；如果希望命令行 index 从 1 开始，可以使用 `--index_base 1`。

普通 eval NPZ 同样使用 `--files`，并可通过 `--spin_dot_key` 和 `--spin_dot_std_key` 显式指定数组键。

## 7. ANNNI：选定 h index 的结构因子–`1/N` 图

请运行包装入口 `plot_annni_sf_inverse_g_indices.py`：

```bash
python inf_research/plot_figs/plot_annni_sf_inverse_g_indices.py \
  --bootstrap_files \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_16q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_20q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_24q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_32q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_40q2.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_64q1.npz \
    inf_research/eval_results_ANNNI/Diffushadow2_boot_72q1.npz \
  --g_indices 4 10 \
  --index_base 0 \
  --q_values pi pi_over_2 \
  --fit_degree 1 \
  --confidence 0.95 \
  --output_dir inf_research/annni_sf_inverse_g_indices
```

`--q_values` 可以是：

- `pi`：绘制 `S_Z(π)`。
- `pi_over_2`：绘制 `S_Z(π/2)`。
- 同时给出二者：分别保存两张图，并额外保存一张 combined 图。

这个入口也要求 `--g_indices` 恰好给出两个不同的 index。这里参数的物理名称是 `h`，只是为了与三个模型共用接口，参数选项仍命名为 `--g_indices`。

## 8. 共享后端：`plot_bootstrap_inverse_g_indices.py`

该文件实现 J1J2 和 ANNNI 的读取、拟合、置信区间、绘图和 CSV 输出，但文件本身没有命令行主入口，因此不要执行：

```bash
python inf_research/plot_figs/plot_bootstrap_inverse_g_indices.py
```

应根据模型执行：

```bash
python inf_research/plot_figs/plot_j1j2_spin_dot_inverse_qubit_g_indices.py ...
```

或：

```bash
python inf_research/plot_figs/plot_annni_sf_inverse_g_indices.py ...
```

## 9. 常见问题

### 无法推断系统尺寸 N

如果 NPZ 中没有 `N`，文件名也不符合 `N16`、`16q`、`qubits_16` 等格式，请显式传入与文件一一对应的尺寸：

```bash
--bootstrap_num_qubits 16 20 24 32 40 64 72
```

普通 eval 输入使用 `--num_qubits`。

### 不同 N 的参数 index 对应了不同参数值

脚本会检查相同 index 在每个 N 的 NPZ 中是否对应同一参数值。如果报参数不一致，不要简单增大 `--match_tol`；应先检查各 NPZ 的参数网格是否真的一致。

### 拟合至少需要多少个 N

`--fit_degree k` 至少需要 `k+1` 个可用系统尺寸。实际分析建议使用更多尺寸，并比较一次和二次拟合的稳定性。

### `independent` 和 `index` 如何选择

不同 N 的 snapshot bootstrap 通常是独立产生的，因此默认使用 `--pairing independent`。只有在不同 N 的第 b 个 replicate 确实具有明确的一一对应关系时，才使用 `--pairing index`。

### 只想隐藏外推数值文字

对三个 inverse-size 入口使用：

```bash
--no_annotate_infinite
```

这不会删除 `1/N=0` 处的外推星号和置信区间。
