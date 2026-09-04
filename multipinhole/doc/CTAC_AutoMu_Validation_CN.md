# CTAC / auto-mu 圆柱主验收口径

## 结论

A2 / B1 / B5 的主验收使用均匀圆柱、slab、overlap profile 和 mu-map 匹配表；Derenzo 只做二级图像质量 sanity，不作为 CTAC 首签指标。

原因：CTAC 和 auto-mu 首先要证明衰减因子、几何方向、床位拼接和分割边界正确。Derenzo 的 rod 可见性同时受 PSF、正则、滤波、噪声和迭代次数影响，容易掩盖 CTAC 错误。

## 主 gate

| 阶段 | 主仿体 | 主指标 |
| --- | --- | --- |
| A2 compact single-bed CTAC | 单床均匀水圆柱 / slab | zero-mu parity、径向 profile、角度依赖、`-ctacRotSign`、compact baking 趋势 |
| B1 compact CTAC multi-bed | 跨 2-3 bed overlap 的长均匀圆柱 | Z seam、overlap profile、每床 table/cache 不串床 |
| B5 auto mu-map | 均匀水圆柱 + cold cavity / shell | body Dice/volume、外部泄漏、内部孔洞填充、auto-mu CTAC vs true-mu CTAC 输出差异 |

## 当前调试入口

基于 260601 release reference 的 FOV 满圆柱：

```powershell
python scripts\qa_ctac_automu_cylinder.py `
  --activity test_data\sys_260601_recon_ref\recon_iter050.img `
  --out-dir recon_out\ctac_automu_cylinder_debug_260601\maps
```

默认 `--threshold-frac 0.05` 保持 auto-mu 运行口径；默认 `--measure-threshold-frac 0.35` 只用于从带 FOV 外噪声的 `recon_iter050.img` 拟合 true cylinder。260601 reference 在该测量阈值下约为直径 30 mm、高度 21 mm。

输出：

- `*_activity.mhd/.raw`：把 headerless `.img` 转为主程序可读 MHD。
- `*_true_mu.mhd/.raw`：由测得圆柱中心、直径、高度拟合出的理想 water mu-map。
- `*_auto_mu.mhd/.raw`：按 auto-mu 分割口径生成的 water mu-map。
- `*_auto_mu_cylmasked.mhd/.raw`：把测得圆柱作为 FOV mask 后的 auto-mu，用于和主程序 `-cyl/-cylc -ctacAutoMu` 输出对齐。
- `*_metrics.json`：直径、高度、Dice、false positive / false negative、mu-map 差异。
- `*_radial_profile.csv`、`*_z_profile.csv`：径向和 Z 向剖面表。
- `*_views.png`：调试视图。

端到端 true-mu vs auto-mu CTAC debug：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ctac_automu_cylinder_debug.ps1 `
  -SpectExe build\Release\bin\Release\SPECT_recon.exe `
  -IterationList 20,50,100
```

该脚本先生成 true/auto mu-map，再分别运行：

- `-ctac -attMap <true_mu.mhd>`
- `-ctac -ctacAutoMu <activity.mhd>`

两个 run 都会传入由 reference 测得的 `-cyl/-cylc`，只用于调试时约束 FOV 外噪声；最后用 `scripts/compare_recon_metrics.py` 比较 auto-mu CTAC 与 true-mu CTAC 输出差异。

`-Iterations 1` 只用于快速 smoke，不能作为 CTAC/auto-mu 输出差异的主验收。主对比至少记录 `20/50/100` iter；若要观察长期漂移，可继续追加更高 iteration。

## 判定

本入口是调试 gate，不是最终生产签收。可接受的首轮结论形式：

- `keep`：mu-map Dice 高、外部泄漏和内部漏分割低，auto-mu CTAC 与 true-mu CTAC 输出差异可解释。
- `blocked`：参考圆柱本身 FOV 噪声或边界截断导致 true cylinder 拟合不可信。
- `reject`：`-ctacRotSign`、seam、table/cache 或 auto-mu 分割错误被 profile 明确抓到。

## 2026-07-06 本地调试基线

输入：

- projection：`test_data/sys_260601_recon_ref/PSC_6module_24pin_10MBq_600s_test_1.bin`
- reference activity：`test_data/sys_260601_recon_ref/recon_iter050.img`
- matrix：`test_data/6module/260601`
- 快速 smoke 输出目录：`recon_out/ctac_automu_cylinder_debug_260601`
- 长迭代输出目录：`recon_out/ctac_automu_cylinder_debug_260601_long`

圆柱测量：

| 指标 | 数值 |
| --- | --- |
| true-cylinder 测量阈值 | `0.35 * P99.9 = 0.0431215` |
| auto-mu 默认阈值 | `0.05 * P99.9 = 0.00616022` |
| 测得中心（0-based voxel） | `(53.3844, 53.6627, 39.5)` |
| 测得直径 / 高度 | `30.0029 mm / 21.0 mm` |

mu-map 匹配：

| 对比 | Dice | FP/true | FN/true | 结论 |
| --- | ---: | ---: | ---: | --- |
| default auto-mu vs true cylinder | `0.746119` | `0.680532` | `3.64e-06` | FOV 外噪声导致过分割，不作为 keep |
| cylinder-masked auto-mu vs true cylinder | `0.999993` | `0` | `1.45e-05` | keep，用于 debug CTAC 输出对照 |

端到端 CTAC：

| 对比 | 指标 | 数值 |
| --- | --- | ---: |
| C++ auto-mu vs Python cylinder-masked auto-mu | relative L2 | `1.686e-08` |

长迭代 auto-mu CTAC vs true-mu CTAC：

| Iter | NCC | relative L2 | RMSE / reference positive mean | sum relative difference |
| ---: | ---: | ---: | ---: | ---: |
| 20 | `0.999999963` | `3.472e-05` | `3.500e-05` | `2.750e-06` |
| 50 | `0.999999941` | `6.061e-05` | `6.157e-05` | `3.009e-06` |
| 100 | `0.999999901` | `1.065e-04` | `1.097e-04` | `3.164e-06` |

快速 1-iter smoke 曾测得 auto-mu CTAC vs true-mu CTAC `relative L2=3.873e-06`、`RMSE/reference positive mean=3.897e-06`、`sum_relative_difference=1.083e-06`；该结果只证明入口和 mask 对齐可用，不作为主 gate。

Decision：`keep` for debug-gated cylinder-masked auto-mu；`non-blocker` for default 0.05 oversegmentation because this dataset has known FOV-outside noise and the debug wrapper now records the failure explicitly.

## 2026-07-08 compact fused CTAC 回归

改动：compact chunk float CTAC 默认不再把 attenuation factor bake 进
CSR values；forward、backward、sensitivity 的 CUDA nnz loop 直接查：

```text
effective[t] = value[t] * att[fullAngle(row) * nvox + transformedVoxel(col)] / 65535
```

旧 baked path 仅保留为显式 debug/reference：设置
`SPECT_COMPACT_CTAC_BAKED_REFERENCE=1` 才会启用。

代数 gate：

- `spect_cuda_scaffold_tests.exe --gtest_filter=CudaScaffoldTest.CompactAttenBakedSpmvMatchesCpuWhenCudaAvailable --gtest_brief=1`
- 默认 fused：passed。
- `SPECT_COMPACT_CTAC_BAKED_REFERENCE=1` baked reference：passed。
- 覆盖 forward、backward、sensitivity、adjoint dot-product、unit attenuation
  zero-mu parity，以及 Sym90Z `symId=0..3` 和 `zMirror=false/true`。

端到端圆柱 gate 输出目录：

- `recon_out/ctac_automu_cylinder_debug_260601_fused_ctac_20260708`

auto-mu CTAC vs true-mu CTAC：

| Iter | NCC | relative L2 | RMSE / reference positive mean | sum relative difference |
| ---: | ---: | ---: | ---: | ---: |
| 20 | `0.999999963095` | `3.472e-05` | `3.500e-05` | `2.750e-06` |
| 50 | `0.999999941147` | `6.061e-05` | `6.157e-05` | `3.009e-06` |
| 100 | `0.999999900582` | `1.065e-04` | `1.097e-04` | `3.164e-06` |

日志口径：

- 每次 run 默认输出 `Run configuration` 块，包含 projection、matrix、
  output、iterations/dim/voxel、CTAC/PSF/Butterworth/reg 参数。
- CTAC path 明确打印：`compact CTAC: attenuation factors fused in native CUDA forward/backward/sensitivity kernels.`

20 iter true-mu performance gate（复用 denom cache，`-vb 2`，GPU precheck
约 20% util / 1012 MiB used）：

| mode | rep | outer wall | internal full command wall | compact forward-ratio-backward |
| --- | ---: | ---: | ---: | ---: |
| fused default | 1 | `41.343 s` | `39.968 s` | `18.402 s` |
| fused default | 2 | `40.926 s` | `39.524 s` | `18.073 s` |
| baked reference | 1 | terminated | `>300 s` lower bound, stopped at iter 12/20 | not completed |

Performance summary：

- `recon_out/ctac_automu_cylinder_debug_260601_fused_ctac_20260708/perf_baked_vs_fused_it20/summary.json`

Decision：`keep` fused CTAC as default. Baked reference remains explicit
debug/parity only; it is not viable as default on this 260601 CTAC gate.

## 2026-09-04 因子分解 CTAC（fused 核与 baked 参考路径删除）

改动：衰减因子不再在 nnz 循环里查表。`h_ij = a_ij · att[fullAngle(i,k), M_k(j)]`
只依赖行的全角度与未旋转体素，所以两种矩阵格式都按 compact 角度逐个处理：
分支批输入先乘上该分支全角度的因子切片（CSRB `compact_branch_batch_atten_scale`，
CSRQ `csrq_pack_branch8_atten`），反投散射结果在收口时乘同一因子
（`compact_transform_backward_add_batch_atten`），SpMV 核就是无 CTAC 时的核
（CSRB 用 packed-K 行核 + cuSPARSE 转置，按角度的行子区间）。每床只暂存 chunk
触及的 60 个全角度切片（0.38 GB，双缓冲、页锁定源表、与计算重叠），不再每床驻留
2.3 GB 整表。逐窗能量指数 `att^(μ_w(E_k)/μ_w(E_map))` 在同一处施加
（`-ctacWindowEnergies`）。删除：`compact_bake_atten_values`、
`compact_atten_fused_spmv/spmvt`、`compactAttenSpmv[T]FloatCUDA`、
`compactDeviceRatioAttenSpmvTAccumulateFloatCUDA`、`CompactAttenBakeParams`、
`SPECT_COMPACT_CTAC_BAKED_REFERENCE` 与 `dAttenTable_` 逐床驻留表，以及
CudaChunkCache 里只剩自引用的 prepared/attenuated 主机 chunk 缓存。

Gate：

- 单元测试 `CsrqCompactMatrixTest.CtacFactorisationMatchesCsrbAndIsAdjoint`：
  CSRB 与 CSRQ 都对照独立的主机 oracle（逐 nnz 模型、双精度），灵敏度 / 分子 /
  前投 / 两床（主机输入与常驻批）relL2 ≤ 1e-5，伴随恒等式 ≤ 1e-5。
- 真实数据（MDP 小鼠 Tc-99m，Bed2–3，`prepare_ct_mu_map.py` 真实 CT μ 图）：新
  CSRB 路径 vs 旧 fused 路径 MLEM it5/10/15/20 relL2 8.9e-8 / 1.3e-7 / 1.0e-7 /
  **1.2e-7**，分母在新路径重算；CSRQ 同一对照 1.4e-7。
- 时间（两床）：CSRB CTAC 44 s/iter → **2.35 s/iter**（核 fwd 10.0 + bwd 15.0 +
  收口 1.4 s / 20 迭代，与无 CTAC 的 CSRB 相当）；CSRQ CTAC 0.60 s/iter。
- CSRB 生产守护（Derenzo MLEM it50，无 CTAC）relL2 1.15e-7 vs fp32 参考，分母重算
  vs 缓存 1.6e-8，同 873 936 非零。

日志口径：CTAC path 现在打印
`compact CTAC: per-bed attenuation factors applied in the image domain per compact angle (branch-batch scale before the forward, transform-add after the backward); the SpMV kernels are the plain ones.`
