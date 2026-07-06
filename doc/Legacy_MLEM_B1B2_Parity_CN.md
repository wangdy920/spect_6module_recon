# Legacy CUDA MLEM 与 B1/B2 compact 一致性验证说明

生成时间：2026-07-06

## 0. 默认矩阵与签收结论（Release 口径）

| 项 | 说明 |
|----|------|
| **默认系统矩阵** | **260601**（`-chunkCSRDir test_data/6module/260601`）；主线 Release 与验收均以此为准 |
| **默认投影** | `test_data/sys_260601_recon_ref/PSC_6module_24pin_10MBq_600s_test_1.bin`（B1/B2 parity 共用） |
| **B1（260601）plain 50 iter** | **已签收**：vs `sys_260601_recon_ref/recon_iter050.img`，`rmse/ref_mean ≈ 1.06×10⁻⁵` |
| **B2（260525）** | 归档 `legacy_ref/recon_iter050.img` **来源暂无法确认**；本阶段**不以 B2 作为 release hard gate** |

Release 构建与 GPU 目标见 `doc/RELEASE_20260706_CN.md`。

## 1. 目的

在统一投影与相同重建参数（50 iter plain MLEM；可选 Butterworth）下，分别做 **B1/B1**、**B2/B2**（不做 B1↔B2 交叉）。

| 路径 | 主线 | 默认优先级 |
|------|------|------------|
| **B1** | 260601 `-chunkCSRDir` | **Release 默认 / 已签收** |
| **B2** | 260525 `-loadCSR` + `-compactMeta` | Legacy/reference-only；对照待补 |

## 2. 投影与归档 reference

### 2.1 投影

```text
test_data/sys_260601_recon_ref/PSC_6module_24pin_10MBq_600s_test_1.bin
```

### 2.2 B1 归档 reference（260601，已确认）

| 项 | 值 |
|----|-----|
| 输出 | `test_data/sys_260601_recon_ref/recon_iter050.img` |
| Legacy 原始命令 | `./spect_recon --matrix-dir .../matrix_PSC_6modules_24pin_YSO_855x855_DOI_PEN --proj .../PSC_6module_24pin_10MBq_600s_test_1.bin --out recon_output --iters 50 --gpus 2 --save-every 5` |
| 重建 | 纯 MLEM 50 iter |

### 2.3 B2 归档 reference（260525，来源未确认）

`test_data/legacy_ref/recon_iter050.img` 对应 **260525** lineage；与统一 `test_1.bin` 联用时总量不匹配，**暂不作为签收基线**。待 projection/矩阵来源核实后再启用 B2/B2 gate。

## 3. 主线 SPECT_recon（默认 260601）

```powershell
staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe `
  -df test_data\sys_260601_recon_ref\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -chunkCSRDir test_data\6module\260601 `
  -matrixLayout compact-symmetric -sparseBackend cuda `
  -alg mlem -subsets 1 -it 50 `
  -fout recon_out\b1_plain_it50\recon
```

B2（非默认，待 reference 确认后用于对照）：

```powershell
staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe `
  -df test_data\sys_260601_recon_ref\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -loadCSR test_data\6module\260525\matrix_6module.csrb `
  -compactMeta test_data\6module\260525\metadata.bin `
  -matrixLayout compact-symmetric -sparseBackend cuda `
  -alg mlem -subsets 1 -it 50 `
  -dim 108,108,80 -vox 0.3 -ang 360 -detN 48,48 `
  -fout recon_out\b2_plain_it50\recon
```

## 4. Legacy 编译（对照用，4070 + 5060/5070）

```powershell
nvcc -O3 -std=c++14 `
  -gencode arch=compute_89,code=sm_89 `
  -gencode arch=compute_120,code=sm_120 `
  6module_bak_src\spect_recon_mlem.cu `
  -o staging\SPECT_recon_rc_dev\bin\spect_recon_mlem_legacy.exe `
  -lcusparse -lcufft
```

Butterworth：`--butter-cutoff 0.35 --butter-order 2 --butter-mode every`（对齐主线 `-butter 0.35,2`）。

## 5. 一键 parity 脚本

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_6module_legacy_b1b2_parity.ps1 `
  -SpectExe staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe `
  -Iterations 50 -SkipCompile -SkipB2
```

`-SkipB2`：本 release 阶段仅跑 B1 签收；去掉后可连同 B2 对照（无 hard gate）。

关键 metrics：`metrics/b1_plain_vs_archived_260601_img.json`

## 6. 对比指标

`scripts/compare_compact_legacy.py`；默认 `rmse_over_reference_mean < 0.02`，`sum_rel_diff < 0.005`。

## 7. 2026-07-06 实测摘要

| 对比 | 结论 |
|------|------|
| **B1 vs `sys_260601_recon_ref/recon_iter050.img`** | **通过**（Release 签收） |
| B2 vs fresh legacy / `legacy_ref` + test_1 | 未过；B2 暂不 gate |

Codex 复核（2026-07-06）：`scripts/run_6module_legacy_b1b2_parity.ps1 -SpectExe build\Release\bin\Release\SPECT_recon.exe -Iterations 50 -SkipCompile -SkipB2 -SkipButter -SkipLegacy -OutputRoot recon_out\legacy_b1b2_parity_it50_codex_20260706` 通过。指标：`rmse_over_reference_mean=1.062910420816022e-05`，`sum_rel_diff=1.8900022603998347e-09`，`relative_l2=9.518835301935594e-06`；metrics 文件：`recon_out/legacy_b1b2_parity_it50_codex_20260706/metrics/b1_plain_vs_archived_260601_img.json`。

## 8. 相关文件

- `doc/RELEASE_20260706_CN.md`
- `scripts/run_6module_legacy_b1b2_parity.ps1`
- `test_data/sys_260601_recon_ref/`
- `test_data/6module/260601/`
