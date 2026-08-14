# SPECT-recon Capability Matrix

> 对齐日期：2026-08-14。以 `src/cli/ArgumentParser.cpp`、`src/pipeline/MainProgram.cpp`、`src/pipeline/VolumeOutput.cpp` 和 compact CUDA 路径为唯一事实来源。

## 证据入口

| 区域 | 代码 |
|---|---|
| CLI 校验 | `src/cli/ArgumentParser.cpp`（`registerHandlers` / `validateArgs`） |
| compact 几何 | `src/io/CompactSymmetryMetadata.cpp`、`ReconDataLoader.cpp::applyCompactMetadataGeometry` |
| 主流程 | `src/pipeline/MainProgram.cpp` |
| compact CUDA | `src/sparse/cuda/CompactSymmetricProjectorCUDA.cpp` |
| 回归 | `tests/`、本地 `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/current_production_parity_metrics.json`、`reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html` |

## 耐久决策

| 决策 | 结论 |
|---|---|
| 唯一矩阵输入 | **`-chunkCSRDir`**（或 `bin/matrix` 唯一目录自动发现）；v4 `metadata.bin`(88B/96B) + `chunk_*.csrb`，SYM_90Z compact-symmetric |
| 唯一布局 | compact-symmetric；standard 布局与单文件 CSRB/MAT 矩阵入口已整体退场 |
| 算法 | **MLEM / BSREM**；subset=1 语义（全量更新），无 `-subsets` 参数；OSEM/ADMM 退场 |
| 公开正则 | 仅 **TV / RDP**；可选 registered CT 边权，无 CT 时 activity-only |
| 公开 PSF | image-domain **`operator` / `sieve` / `none`**；`-psf` 为 FWHM mm；`-psfKernel adjoint|castor` 选择真伴随 Gaussian 或严格 CASToR erf/Z-norm/forward-reuse 语义；发布默认 `sieve + castor + 1.3,3` |
| 稀疏 / 滤波 | 固定 **CUDA-first / cuFFT**；CLI smooth/Butterworth 为显式 `in_recon`；发布默认在保存副本上独立施加 CASToR Smooth 0.30 mm/3σ 与 2× zero-pad Butterworth 5 cpcm/order 8 |
| CTAC 模型 | **`att[angle, voxel]` fused** line-integral；factor 生成仅 CUDA（16-angle 分批）；AutoMuMap（`-ctacAutoMu` 从早期迭代体生成水 mu-map） |
| 多床位 | 唯一 canonical stitch：`Bed1=min physical z`、连续 Bed1..Bed5 子集、`z_bed_flip=1`、中心 `step+10` 层（260601 为 66）、prototype robust gain-fit、`step` 层步进（260601 为 56）、10 层 cosine 融合；固定多床 ROI `36×66 mm`、输出 XY `100×100`。`-attMap` local-Z 与 `-regCT` storage-Z 分离，`z_out_flip` 仅作用于保存输出 |
| 校正权重 | 坏探测器 mask（`-badDetectorFile`）、normFactor 倒数灵敏度（`-normFactorFile`）、detZ 行权重（`-detZRowWeightsFile`，仅 sensitivity） |
| GPU | **单 GPU**（`-cudaDevice` / `SPECT_CUDA_DEVICE`）；branch/bed 默认合批 SpMM，保留 `SPECT_COMPACT_BRANCH_BATCH=0` 性能基线；不规划多 GPU |
| 输出 | 保存未滤波 `<prefix>_it<N>`，以及从同一主输出独立派生的 `_smooth` / `_butter` MHD/RAW；发布默认对各副本后置非负钳制，manifest 记录参数与六文件清单；退出码 `0` 成功/help/dumpConf、`1` CLI/配置错误、`2` runtime |

## 能力表

| 路径 | 特性 | 实现 | 生产签收 | 备注 |
|---|---|---:|---:|---|
| compact 260601 | MLEM CUDA | 是 | **是（B1 gate）** | subset=1 全量更新 |
| compact 260601 | BSREM + TV/RDP | 是 | 路径/parity；beta 标定未完 | 可选 CT 边权 |
| compact 260601 | `operator`/`sieve` PSF | 是 | 是 | XY 各向同性 |
| compact 260601 | Butterworth | 是 | 是 | cuFFT；仅显式 `in_recon`，与 `sieve` 互斥 |
| compact 260601 | CTAC 单床 | 是 | 圆柱 true/auto gate | fused 默认 |
| compact 260601 | multi-bed CTAC | 是 | 数值 parity 通过 | 每床 fused factors |
| compact 260601 | 多床 canonical stitch | 是 | **是（五组数据）** | 两组 EP、两组 mouse 与单床 Derenzo 的对象/数值/输出端点全部通过 |
| compact 260601 | 16 GB GPU、50 次 MLEM 性能 | 是 | 单床通过；四床有界 | 三次中位数：Derenzo 104.59 s，EPguan 189.67 s；64 GB 主存 + NVMe 矩阵口径 |

## CTAC 边界（简）

- 生产：`effective = value * att[fullAngle, voxel] / 65535`（fused nnz）。
- factor 生成仅保留 CUDA line-integral；16-angle batch，临时显存约 29 MiB。
- Validation-only：`scripts/validate_ctac_pinhole_reference.py` 等抽样对照；不进入生产路径。
- 调试 gate：`doc/CTAC_AutoMu_Validation_CN.md`。
- CT/正则对比：`scripts/run_ct_guided_regularizer_compare.ps1` 固定输出 no-CT MLEM、CTAC MLEM、CTAC BSREM-beta0、CTAC+RDP、CTAC+TV。

## 相关文档

- `../reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html` — 当前 `sieve + CASToR + FWHM 1.3 mm` 原型对齐与五组数据决策记录
- `USAGE_CN.md` — CLI
- `Format_Reader_Matrix_CN.md` — 矩阵格式说明
- `Recon_Interface_Migration_Field_CN.md` — 现场迁移
