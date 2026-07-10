# SPECT-recon Capability Matrix

> 对齐日期：2026-07-10。以 `ArgumentParser::validateArgs`、`MainProgram.cpp`、compact CUDA 路径为准；过时设计稿已删除，勿再引用 archive。

## 证据入口

| 区域 | 代码 |
|---|---|
| CLI 校验 | `src/cli/ArgumentParser.cpp`（`validateArgs`） |
| compact 几何 | `src/io/CompactSymmetryMetadata.cpp`、`ReconDataLoader.cpp::applyCompactMetadataGeometry` |
| 主流程 | `src/main/MainProgram.cpp` |
| compact CUDA | `src/cuda/CompactSymmetricProjectorCUDA.cpp` |
| 回归 | `tests/`、`doc/Legacy_MLEM_B1B2_Parity_CN.md` |

## 耐久决策（原 refactor / PSF / CTAC 决策合并）

| 决策 | 结论 |
|---|---|
| 默认 compact 矩阵 | **260601** `-chunkCSRDir`（或 `bin/matrix` 唯一目录自动发现） |
| 260525 `-loadCSR -compactMeta` | **不退场**，legacy 兼容 + deprecated 提示；大文件隐式 mmap |
| 公开 PSF | 仅 image-domain **Operator / None**；`-psf` 为 FWHM mm；ProjDomain/PostSmooth/DD 公开入口拒绝 |
| 稀疏 / Butterworth | 生产固定 **CUDA / cuFFT**；旧 backend 选择 CLI 拒绝 |
| 公开正则 | 仅 **TV / RDP** |
| compact CTAC 模型 | 生产保留 **`att[angle, voxel]` fused**；不做 production Siddon / row-bin 表 |
| compact multi-bed CTAC | **不支持**（`validateArgs` 拒绝） |
| 退出码 | `0` 成功/help/dumpConf；`1` CLI/配置错误；`2` runtime（`include/cli/ExitCodes.h`） |
| RegularizerCompat→新梯度 | **拒绝**静默委托；离散化未统一前不迁 |
| v2 sysmat 生成端 | 暂不删；归档条件见 `6module_bak_src/README.md` |

## 能力表

| 路径 | 特性 | 实现 | 生产签收 | 备注 |
|---|---|---:|---:|---|
| standard CSRB/MAT | MLEM / OSEM | 是 | 基线/对齐测试 | `subsets>1`→OSEM |
| standard | BSREM + RDP | 是 | 部分（beta 标定未完） | |
| standard | ADMM + TV | 是 | 未作生产物理签收 | |
| standard | Operator PSF | 是 | 是 | |
| standard | CTAC line-integral | 是 | slab gtest；phantom 见 CTAC 文档 | |
| standard | multi-bed / bProj | 是 | 路径回归 | |
| compact 260601 | MLEM subset=1 CUDA | 是 | **是（B1 gate）** | |
| compact 260601 | BSREM + RDP subset=1 | 是 | 路径/parity；beta 标定未完 | |
| compact 260601 | Operator PSF | 是 | 是 | multi-bed 全局体 |
| compact 260601 | CTAC 单床 | 是 | 圆柱 true/auto gate | fused 默认 |
| compact 260601 | OSEM / ADMM / multi-bed CTAC | 否 | — | CLI 拒绝 |
| compact 260525 单文件 | MLEM/BSREM | legacy | 非 hard gate | 无 CTAC |
| v2 row-wise `chunk_*.bin` | 转换工具 | 是 | reference | `spect_convert_6module_sysmat_to_csrb` |

## PSF 边界（简）

- **Operator**：图像域可分 Gaussian，前/后向与 sensitivity 同口径；compact 要求 XY 各向同性。
- **ProjDomain 等**：内部/历史路径，公开 CLI 已退场；与 Operator **不可互换**。

## CTAC 边界（简）

- 生产：`effective = value * att[fullAngle, voxel] / 65535`（fused nnz）。
- Validation-only：`scripts/validate_ctac_pinhole_reference.py` 等抽样对照；不进入生产表布局。
- 调试 gate：`doc/CTAC_AutoMu_Validation_CN.md`。

## 相关文档

- `RELEASE_20260708_CN.md` — 当前 release
- `USAGE_CN.md` / `CLI_CHANGES_CN.md` — CLI
- `Format_Reader_Matrix_CN.md` — 格式×读者
- `Legacy_MLEM_B1B2_Parity_CN.md` — B1 签收
- `Recon_Interface_Migration_Field_CN.md` — 现场迁移
