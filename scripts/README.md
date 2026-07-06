# 脚本入口（与重构流程集成）

本目录提供对外脚本入口，用于把仓库内的辅助脚本（Python/PowerShell/MATLAB）融入到“构建 → 测试 → 验证/分析”的常用流程中。

## 1) L-curve（β 扫描）

- 脚本：`scripts/run_l_curve.ps1`（封装 `utils/run_l_curve.py`）
- 依赖：Python + `numpy`, `matplotlib`（见 `scripts/requirements.txt`）
- 可选自检：`python scripts/l_curve_smoke_test.py`（只校验参数文件解析与 `[METRIC]` 正则）

示例（推荐用 `-BaseArg` 逐 token 传参，避免 Windows 路径/空格问题）：

```powershell
.\scripts\run_l_curve.ps1 `
  -BaseArg @(
    "-dfList", ".\\gatedata\\MDP_4h\\data_append_Bed1.mat,.\\gatedata\\MDP_4h\\data_append_Bed2.mat,.\\gatedata\\MDP_4h\\data_append_Bed3.mat,.\\gatedata\\MDP_4h\\data_append_Bed4.mat",
    "-dim", "100,100,100",
    "-vox", "0.3,0.3,0.3",
    "-ang", "360",
    "-detN", "40,40",
    "-it", "5:5",
    "-th", "16",
    "-loadCSR", ".\\gatedata\\matrix_TDA_1mm_N100_vox030_cw.csrb",
    "-mapCSR", "on",
    "-butter", "0.35,2",
    "-bedOverlap", "16,16,16,16",
    "-cropEnd", "16,16",
    "-fout", "recon_out\\lcurve_run"
  ) `
  -RegType TVIsotropic `
  -Betas "1e-4:1e-1:8" `
  -Output "recon_out\\l_curve_results.png"
```

## 2) 四床位数据 smoke 回归（`SPECT_rectify_S_0105`）

- 脚本：`scripts/run_4bed_smoke.ps1`
- 目标：在同一套输入上快速验证 `MLEM / BSREM` 主流程均可运行并产出结果
- 默认输入：
  - `gatedata\SPECT_rectify_S_0105\data_append_bed{1..4}.mat`
  - `gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb`

示例：

```powershell
.\scripts\run_4bed_smoke.ps1 -Threads 8 -Iters 1 -Subsets 5
```

可选：带背景项回归（同一背景文件复用到 4 个床位）：

```powershell
.\scripts\run_4bed_smoke.ps1 -UseBackground
```

## 3) BSREM `kappa/soft-mask` 参数扫面（四床位）

- 脚本：`scripts/run_bsrem_kappa_sweep.ps1`
- 目标：在固定四床位口径下扫面 `kappa clip` 与 `soft-mask (band/lambda)`，输出可复现的推荐默认值
- 辅助脚本：`scripts/eval_mhd_stats.py`（读取 `.mhd/.raw`，计算 inside/outside 统计、NU4 风格 CV 代理和多床位 seam score）

示例：

```powershell
.\scripts\run_bsrem_kappa_sweep.ps1 `
  -CoarseIters 6 `
  -ConfirmIters 20 `
  -Threads 16 `
  -RegSpec "RDP,1e-4,2.0,0.01" `
  -PriorMaskMhd ".\recon_out\perf_check_fix\multibed_mlem_it20.mhd" `
  -MaskThreshold 0.02
```

输出：

- `recon_out\bsrem_kappa_sweep\coarse_results.csv`
- `recon_out\bsrem_kappa_sweep\confirm_results.csv`
- `recon_out\bsrem_kappa_sweep\SUMMARY.md`

补充参数（可选）：

- `-BaseNz`：每床位 Z 尺寸（默认 `100`，用于 seam 位置推导）
- `-SeamWindow`：seam 梯度窗口（默认 `2`）
- `-EdgeCoverageAware`：按“相对 baseline seam 比值”进行推荐筛选（默认开）
- `-SeamTolerance`：相对 baseline seam 可接受增幅（默认 `0.20`，即 `<=1.2x`）

`scripts/eval_mhd_stats.py` 额外支持：

- `--ref-image`：计算 `rmse/nrmse/mae`（对参考体）
- `--seam-z-list` + `--seam-window`：计算多床位 stitching 连续性指标（`seam_score`）

## 4) CUDA SpMV 基准（CPU vs GPU 精度/速度/显存）

- 脚本：`scripts/run_cuda_spmv_benchmark.ps1`
- 源码：`tools/cuda_spmv_benchmark.cu`
- 目的：以 `.csrb` 系统矩阵对比 CPU 与 CUDA cuSPARSE 的 `A*x` / `A^T*x` 结果，输出：
  - 精度：`max_abs / rmse / l2_rel`
  - 速度：`cpu_ms_avg / gpu_ms_avg / speedup / GFLOPS`
  - 内存：`host_bytes / gpu_alloc_bytes / gpu_mem_delta_bytes`

示例：

```powershell
.\scripts\run_cuda_spmv_benchmark.ps1 `
  -MatrixPath ".\gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb" `
  -CpuThreads 16 `
  -CpuIters 5 `
  -GpuIters 50 `
  -WarmupIters 5 `
  -MaxRows 80000 `
  -Op both `
  -Rebuild
```

说明：

- 默认 `-MaxRows 80000`（对 16GB 显存设备更稳妥）。
- 默认 `-Op both`，同时评估前向 `A*x` 与反向 `A^T*x`；可改为 `spmv` 或 `spmvt`。
- 脚本默认启用 OOM 自动回退（逐步降低 `maxRows` 重试）；可用 `-NoAutoFallback` 关闭。

输出：

- `recon_out\cuda_spmv_bench\cuda_spmv_*.json`
- `recon_out\cuda_spmv_bench\cuda_spmv_*.log`

## 5) 6module chunk 系统矩阵转 CSRB

- 工具：`spect_convert_6module_sysmat_to_csrb`
- 源码：`tools/convert_6module_sysmat_to_csrb.cpp`
- 输入：`6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v2.cu` 生成的 `metadata.bin + chunk_*.bin`
- 输出：当前 SPECT-recon 可用的 `.csrb` 单文件

示例：

```powershell
cmake --build build_cuda_scaffold --target spect_convert_6module_sysmat_to_csrb
.\build_cuda_scaffold\bin\spect_convert_6module_sysmat_to_csrb.exe `
  --matrix-dir ".\matrix_PSC_6modules_24pin_YSO_3X5_PEN_DOI_V2" `
  --out ".\gatedata\matrix_6module.csrb"
```

说明：

- 转换器采用三遍流式 I/O：第一遍只读每行 `nnz` 并生成 `row_ptr`，第二遍写 `col_idx`，第三遍写 `val`；内存主要是 `row_ptr`，不会把整套 `col_idx/val` 读入内存。
- 若 `metadata.bin` 显示 45/90/Z 对称压缩，输出是 direct compact CSRB，不会展开 full-angle/full-Z 对称；标准 SPECT-recon 输入应使用行数匹配的 compact 投影，或重新生成无对称压缩的系统矩阵。
- 当前 CSRB 的 `row_ptr` 为 `uint32`，因此总 `nnz` 不能超过 `2^32-1`。

## 6) 6module compact-symmetric 真实数据验收

- 脚本：`scripts/run_6module_compact_acceptance.ps1`
- 目标：按 `doc/CompactSymmetricBackend_Execution_Target_CN.md` 的固定 6module 单床位命令运行 compact-symmetric smoke，生成 QA 三视图/JSON，并在旧 CUDA reference 存在时自动做数值对齐。
- 辅助脚本：
  - `scripts/qa_compact_cylinder.py`：输出中心 XY/XZ/YZ 三视图和基础 QA 指标。
  - `scripts/compare_compact_legacy.py`：对比新 compact 后端输出与旧 CUDA MLEM 输出。

## 6b) B1/B2 与 Legacy MLEM 50 iter 一致性（2026-07-06）

- 文档：`doc/Legacy_MLEM_B1B2_Parity_CN.md`
- 脚本：`scripts/run_6module_legacy_b1b2_parity.ps1`
- 对照程序：`6module_bak_src/spect_recon_mlem.cu` → `build_legacy_6module/spect_recon_mlem_legacy.exe`
- **对比口径**：B1/B1、B2/B2；默认投影 `test_data/sys_260601_recon_ref/PSC_6module_24pin_10MBq_600s_test_1.bin`（B1/B2 共用）
- 归档 reference：B1 → `sys_260601_recon_ref/recon_iter050.img`；B2 → `legacy_ref/recon_iter050.img`（260525；与 test_1 投影联用需见文档 §2.3）
- 案例：`plain` + `butter`（legacy 用 `--butter-mode every`）
- B1/B1 需 `-Matrix260601LegacyDir` 指向含 `chunk_*.bin` 的 260601 等价目录（legacy 不读 `.csrb`）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_6module_legacy_b1b2_parity.ps1 `
  -SpectExe staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe -Iterations 50
```

默认输入：

```text
projection:   gatedata\6module\PSC_6module_24pin_10MBq_600s_1.bin
compact CSRB: gatedata\6module\matrix_6module.csrb
legacy ref:   recon_out\legacy_6module_compact_mlem_psf15_butter\recon_iter001.raw
```

完整 smoke：

```powershell
.\scripts\run_6module_compact_acceptance.ps1
```

只基于已有 candidate 输出重跑 QA/对齐：

```powershell
.\scripts\run_6module_compact_acceptance.ps1 -SkipRecon
```

当前旧 CUDA 对齐默认阈值：

```text
rmse / reference_mean <= 0.02
sum_rel_diff          <= 0.005
```

说明：当前 1 iter 圆柱输出存在 ring-like 边界增强，旧 CUDA reference 在同一 QA 口径下也不满足 `inside disk > outside annulus`。因此 single-bed 首轮验收应同时看图像有限性/连续性/无明显 folding 错误，以及新后端对旧 CUDA 的数值对齐。

## 7) CUDA OSEM 对比（优先多床位 full-subset CPU vs GPU 时间/内存）

- 脚本：`scripts/run_cuda_osem_compare.ps1`
- 源码：`tools/cuda_osem_compare.cpp`
- 目的：以单床位或多床位 full-subset OSEM 口径对比 CPU 与 CUDA 的一次或多次迭代，输出：
  - 时间：`cpu.ms_avg / gpu.ms_avg / speedup`
  - 精度：最终图像 `max_abs / rmse / l2_rel`
  - 内存：进程 working set 与 GPU persistent memory delta

示例：

```powershell
.\scripts\run_cuda_osem_compare.ps1 `
  -MatrixPath ".\gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb" `
  -Threads 16 `
  -Iterations 1 `
  -Subsets 1 `
  -WarmupRuns 1 `
  -Repeats 2 `
  -MaxRows 20000 `
  -Rebuild
```

说明：

- 当前 compare 工具优先覆盖“无 PSF、无 CTAC”的 full-subset OSEM 路径，既可单床位，也可多床位协同。
- `-Subsets 1` 是当前主使用口径；单床位和多床位都会优先复用整矩阵 warm-path。
- 多床位请额外提供 `-BedCount`，并在需要 overlap 时提供 `-BedOverlaps` 与 `-BaseDim`。
- `-Subsets > 1` 仍然保留，但目前是次优路线，尤其在系统矩阵场景下通常比 full-subset 更慢。
- 当前已验证的多床位样例：`-BedCount 2 -BedOverlaps 16 -BaseDim 100,100,100 -Subsets 1 -MaxRows 20000`，结果约 `CPU 276.1 ms / GPU 27.3 ms / 10.10x`。
- 额外测得的 full-subset 多床位样例：`-BedCount 3 -BedOverlaps 16,16 -BaseDim 100,100,100 -Subsets 1 -MaxRows 20000` 约为 `CPU 1270.5 ms / GPU 38.4 ms / 33.07x`；`-BedCount 4 -BedOverlaps 16,16,16 -BaseDim 100,100,100 -Subsets 1 -MaxRows 20000` 约为 `CPU 958.7 ms / GPU 278.8 ms / 3.44x`。4 床位仍保持数值一致，但当前性能没有沿 3 床位继续线性扩展。

多床位示例：

```powershell
.\scripts\run_cuda_osem_compare.ps1 `
  -MatrixPath ".\gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb" `
  -Threads 16 `
  -Iterations 1 `
  -Subsets 1 `
  -BedCount 2 `
  -BedOverlaps "16" `
  -BaseDim "100,100,100" `
  -MaxRows 20000
```

输出：

- `recon_out\cuda_osem_compare\cuda_osem_*.json`
- `recon_out\cuda_osem_compare\cuda_osem_*.log`

## 8) 原始 vs 重构：对比构建/运行

对比测试脚本位于仓库根目录（文档：`docs/Testing/COMPARISON_TEST_GUIDE.md`）：

- `build_comparison.ps1`：分别构建 `build_original/` 与 `build_refactored/`
- `run_comparison_test.ps1`：用相同数据跑两套可执行并对比输出
- `compare_refactor.py`：使用 SimpleITK 做像素级差异统计（需要额外 Python 依赖）

## 9) MATLAB 工具

`utils/*.m`：PSF map/phantom 相关辅助脚本（依赖 MATLAB/Octave 环境）。
