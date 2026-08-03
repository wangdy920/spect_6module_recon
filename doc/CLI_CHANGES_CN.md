# SPECT_recon CLI 变更速查

本文只记录容易误用的废弃入口和替代写法；完整参数以 `USAGE_CN.md` 与
发布包 `PARAMETERS_CN.md` 为准。

## 必须改的调用

| 旧写法 | 当前写法 / 行为 |
| --- | --- |
| `-alg mlem|bsrem|admm` | 专用报错；改用 `-opti mlem|bsrem` |
| `-opti admm` | `Unknown optimisation algorithm`；ADMM 已退场 |
| `-subsets <N>` | 专用报错；compact 布局恒单 subset（MLEM 式全量更新） |
| `-indexMat` / `-weightMat` | `Unknown argument`；矩阵唯一入口 `-chunkCSRDir` |
| `-loadCSR` / `-mapCSR` / `-saveCSR` | `Unknown argument`；单文件 CSRB 入口已退场 |
| `-compactMeta` / `-matrixLayout` | `Unknown argument`；布局恒 compact-symmetric |
| `-sparseBackend cpu|cuda` / `-spBackend ...` | 专用报错；稀疏后端固定 CUDA |
| `-v <0\|1\|2>`（旧 verbosity 别名） | 改用 `-vb <0\|1\|2>`；**`-v` / `--version` 现为输出版本号** |
| `-butterBackend fftw|cufft|auto` / `-bwBackend` | 专用报错；Butterworth 后端固定 CUDA/cuFFT |
| `-psf <sigma>` | `-psf <fwhm_mm[,N]>`，默认 `1.5,4` |
| `-psfProj` / `-conv` | 专用报错；公开 PSF 只保留 image-domain Operator |
| `-psfMode PostSmooth|ProjDomain|DistanceDependent` | `-psfMode None|Operator` |
| `-ctacModel per-voxel` | `-ctacModel line-integral` 或省略 |
| `-prior ...` / anatomical prior 参数 | 专用报错；公开正则只保留 TV/RDP |
| `-reg NLM|Tikhonov|LogCosh|MRP|Bowsher|PLS...` | `-reg TV,...` 或 `-reg RDP,...` |
| `-regPower` / `-regSigma` / `-regKernel` 等 | 专用报错；细调入口随非公开正则一并退场 |
| `-joe <n>` | 专用报错；Joseph-like 平滑退场，改用 `-psf` / `-butter` / `-smooth` |
| `-Aprune <eps>` / `-Astream` / `-ApruneMask` | 专用报错；compact chunk CSRB 在生成阶段完成剪枝，读取恒为逐 chunk 流式/mmap |
| `-bsremSoftBand <vox>` | 专用报错；soft-mask 过渡带从未实现（mask 为硬二值加权），外权重用 `-bsremSoftLambda` |

## INI 死键

以下键已从 `config/spect_config.ini` 移除（未知 INI 键被静默忽略）：

- `[SystemMatrix] matrix_layout`、`map_csr`（布局/读取方式固定）
- `[SystemMatrix] prune_epsilon`、`stream_build`、`prune_mask`（剪枝在矩阵生成阶段完成）
- `[Reconstruction] subsets`（恒单 subset）
- `[MultiBed] overlap_in_recon`（joint 联合重建退场，仅 stitch）
- `[Reconstruction] joseph_passes`（Joseph-like 平滑退场）
- `[Reconstruction] bsrem_soft_mask_band_vox`、`algo_bsrem.conf` 的 `[BSREM] soft_mask_band`（过渡带从未实现）
- ADMM 相关键（算法退场）
- `algo_*.conf` 中 `[Algorithm] subsets`

## 多床接缝（stitch 唯一模式）

| 参数 | 当前语义 |
| --- | --- |
| `[MultiBed] crop_method=center66_cosine10` | 默认：每床只保留中心 66 层，以固定 56 层步进排列，并在 10 层重叠区做归一化 cosine 融合 |
| `[MultiBed] crop_method=sigmoid` | 保留的对照：在完整床位 overlap 上使用归一化 `sigmoid(k=12)` 权重融合 |
| `[MultiBed] crop_method=direct` | 保留 hard-cut 作为控制/回退方法 |
| `[MultiBed] seam_postprocess` | 请求接缝后处理（默认 1）；仅在多床 weighted 输出 overlap 生效时运行 |

各床按零 overlap 独立重建，融合只发生在输出保存前。中心方案要求每个实际接缝满足 `bed_overlap = 单床 Nz - 56`。

## 推荐最小命令

```powershell
.\bin\SPECT_recon.exe `
  -df D:\data\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -chunkCSRDir .\bin\matrix\260601 `
  -opti mlem `
  -it 50:5 `
  -o recon_output
```

如果发布包内 `bin/matrix/` 只有一个包含 `metadata.bin` 的矩阵目录，
`-chunkCSRDir` 也可以省略。
