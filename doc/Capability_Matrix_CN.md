# SPECT-recon Capability Matrix

> 对齐日期：2026-07-31（收敛重构后）。以 `src/cli/ArgumentParser.cpp`、`src/main/MainProgram.cpp`、compact CUDA 路径为唯一事实来源。

## 证据入口

| 区域 | 代码 |
|---|---|
| CLI 校验 | `src/cli/ArgumentParser.cpp`（`registerHandlers` / `validateArgs`） |
| compact 几何 | `src/io/CompactSymmetryMetadata.cpp`、`ReconDataLoader.cpp::applyCompactMetadataGeometry` |
| 主流程 | `src/main/MainProgram.cpp` |
| compact CUDA | `src/cuda/CompactSymmetricProjectorCUDA.cpp` |
| 回归 | `tests/`、`recon_out/convergence_baseline_20260731/`（冻结金标准） |

## 耐久决策

| 决策 | 结论 |
|---|---|
| 唯一矩阵输入 | **`-chunkCSRDir`**（或 `bin/matrix` 唯一目录自动发现）；v4 `metadata.bin`(88B/96B) + `chunk_*.csrb`，SYM_90Z compact-symmetric |
| 唯一布局 | compact-symmetric；standard 布局与单文件 CSRB/MAT 矩阵入口已整体退场 |
| 算法 | **MLEM / BSREM**；subset=1 语义（全量更新），无 `-subsets` 参数；OSEM/ADMM 退场 |
| 公开正则 | 仅 **TV / RDP**；可选 registered CT 边权，无 CT 时 activity-only |
| 公开 PSF | 仅 image-domain **Operator / None**；`-psf` 为 FWHM mm |
| 稀疏 / Butterworth | 固定 **CUDA / cuFFT**；旧后端选择 CLI 拒绝 |
| CTAC 模型 | **`att[angle, voxel]` fused** line-integral；factor 生成仅 CUDA（16-angle 分批）；AutoMuMap（`-ctacAutoMu` 从早期迭代体生成水 mu-map） |
| 多床位 | 仅 **stitch**：各床独立重建（零 overlap），输出前按 `center66_cosine10`（默认）/ `sigmoid` / `direct` 拼接 + seam postprocess；joint 联合重建已退场 |
| 校正权重 | 坏探测器 mask（`-badDetectorFile`）、normFactor 倒数灵敏度（`-normFactorFile`）、detZ 行权重（`-detZRowWeightsFile`，仅 sensitivity） |
| GPU | **单 GPU**（`-cudaDevice` / `SPECT_CUDA_DEVICE`）；不规划多 GPU |
| 输出 | MHD/RAW 体数据 + 自动日志；退出码 `0` 成功/help/dumpConf、`1` CLI/配置错误、`2` runtime |

## 已退场入口（parser 拒绝）

- 矩阵：`-indexMat` / `-weightMat` / `-loadCSR` / `-mapCSR` / `-saveCSR` / `-compactMeta` / `-matrixLayout` → `Unknown argument`
- 算法：`-subsets`、`-alg`、`-opti admm`（报 `Unknown optimisation algorithm`）
- 其他：`-sparseBackend`、`-conv`、`-psfProj`、`-psfMode ProjDomain|PostSmooth|DistanceDependent`、`-butterBackend`、`-prior`、`-regPower` / `-regSigma` 及 anatomical 正则细调参数
- INI：`[SystemMatrix] matrix_layout` / `map_csr`、`[Reconstruction] subsets`、`[MultiBed] overlap_in_recon`（未知 INI 键被静默忽略）

## 能力表

| 路径 | 特性 | 实现 | 生产签收 | 备注 |
|---|---|---:|---:|---|
| compact 260601 | MLEM CUDA | 是 | **是（B1 gate）** | subset=1 全量更新 |
| compact 260601 | BSREM + TV/RDP | 是 | 路径/parity；beta 标定未完 | 可选 CT 边权 |
| compact 260601 | Operator PSF | 是 | 是 | XY 各向同性 |
| compact 260601 | Butterworth | 是 | 是 | cuFFT，逐 iter |
| compact 260601 | CTAC 单床 | 是 | 圆柱 true/auto gate | fused 默认 |
| compact 260601 | multi-bed CTAC | 是 | 数值 parity 通过 | 每床 fused factors |
| compact 260601 | 多床 stitch | 是 | 是 | center66_cosine10 默认 + seam postprocess |
| compact 260601 | OSEM / ADMM | 否 | — | CLI 拒绝 |

## CTAC 边界（简）

- 生产：`effective = value * att[fullAngle, voxel] / 65535`（fused nnz）。
- factor 生成仅保留 CUDA line-integral；16-angle batch，临时显存约 29 MiB。
- Validation-only：`scripts/validate_ctac_pinhole_reference.py` 等抽样对照；不进入生产路径。
- 调试 gate：`doc/CTAC_AutoMu_Validation_CN.md`。
- CT/正则对比：`scripts/run_ct_guided_regularizer_compare.ps1` 固定输出 no-CT MLEM、CTAC MLEM、CTAC BSREM-beta0、CTAC+RDP、CTAC+TV。

## 相关文档

- `RELEASE_20260721_CN.md` — 当前 release（`RELEASE_20260715_CN.md` 为上一版历史保留）
- `USAGE_CN.md` / `CLI_CHANGES_CN.md` — CLI
- `Format_Reader_Matrix_CN.md` — 矩阵格式说明
- `Recon_Interface_Migration_Field_CN.md` — 现场迁移
