# SPECT-recon Capability Matrix（Draft，2026-07-03）

> 状态：第一批 PR 契约草案，待 Opus 4.8 / Fable 5 review。本文只记录当前代码能力，不为对齐文档修改代码。

## 证据入口

- CLI 校验：`include/cli/ArgumentParser.h::validateArgs`
- compact 几何校验：`src/io/CompactSymmetryMetadata.cpp::validateCompactSymmetricInputs` / `validateCompactChunkSymmetricInputs`
- 主流程：`src/main/MainProgram.cpp`
- 标准算法：`src/app/OSEMReconstructor.cpp`、`src/app/BSREMReconstructor.cpp`
- compact CUDA：`src/cuda/CompactSymmetricProjectorCUDA.cpp`
- 回归证据：`tests/test_interfaces.cpp`、`tests/test_psf_modes.cpp`、`tests/test_main_ctac_ops.cpp`、`.agent/CONTINUITY.md`

## 重构不动边界

1. 260601 chunk CSRB 是 canonical compact 路径：`-chunkCSRDir` + 88B/96B metadata + Sym90Z/Sym45Z 校验。
2. 260525 `-loadCSR -compactMeta` 单文件入口仍是 legacy compatibility；三条件签收前不删除，只加安全保护和 deprecated 提示。
3. CTAC line-integral、compact CTAC value baking、Operator PSF sensitivity、BSREM `w_proj/w_img` fail-fast、bProj additive 位置和 sensitivity/numerator/denominator 语义都属于物理回归触发区。
4. 结构拆分 PR 不改变数值顺序；性能 PR 单独立项。

## 当前能力矩阵

| 路径 | 算法/特性 | 公式已实现 | 物理已验证 | 真实矩阵 smoke | 证据 |
|---|---|---:|---:|---:|---|
| standard CSR/MAT/CSRB | MLEM, subsets=1 | 是 | 基线回归 | 是 | `MainProgram.cpp`, `test_data/share` 50 iter 记录 |
| standard CSR/MAT/CSRB | OSEM, subsets>1 | 是 | 算法对齐测试 | 无本批新增记录 | `OSEMReconstructor.cpp`, `tests/test_osem_alignment.cpp` |
| standard CSR/MAT/CSRB | BSREM + RDP/curvature/kappa | 是 | 部分，beta 标定未完成 | 是 | `BSREMReconstructor.cpp`, `tests/test_bsrem_*` |
| standard CSR/MAT/CSRB | ADMM/TV | 是 | 未作为生产物理签收 | 无本批新增记录 | `include/app/ADMMOptimizer.h`, CLI help |
| standard CSR/MAT/CSRB | Operator PSF | 是，含 sensitivity `G^T A^T 1` | 是，模式级回归 | 是 | `ProjectionOps.cpp`, `tests/test_psf_modes.cpp` |
| standard CSR/MAT/CSRB | projection-domain PSF | 隔离实现，单床/subset=1 约束 | 模型差异未物理签收 | 无本批新增记录 | `ReconPreparation.cpp`, `ProjDomainChunked.h` |
| standard CSR/MAT/CSRB | CTAC line-integral | 是 | slab gtest；phantom 定量待做 | 是 | `CTACOps.cpp`, `tests/test_main_ctac_ops.cpp` |
| standard CSR/MAT/CSRB | bProj additive | 是 | 外部项接入，散射 solver 未实现 | 路径 smoke 已有 | `MainProgram.cpp` background projection 分支 |
| standard CSR/MAT/CSRB | multi-bed | 是 | 输出 Z flip/overlap 回归 | 是 | `ReconSetup.cpp`, `BedOffsets.h`, `tests/test_main_*` |
| compact 260601 chunk CSRB | MLEM, subset=1, CUDA | 是 | 是 | 是 | `CompactChunkCSR.cpp`, `CompactSymmetricProjectorCUDA.cpp`, continuity 260601 记录 |
| compact 260601 chunk CSRB | BSREM + RDP, subset=1, CUDA | 是 | 路径和 beta=0 parity；beta 标定未完成 | 是 | continuity 2026-06-05 记录 |
| compact 260601 chunk CSRB | OSEM subsets | 否 | 否 | 否 | `validateArgs` 拒绝 `numSubsets != 1` |
| compact 260601 chunk CSRB | ADMM/TV | 否 | 否 | 否 | `validateArgs` 仅允许 MLEM/BSREM |
| compact 260601 chunk CSRB | Operator PSF | 是，multi-bed global-volume | 是 | 是 | `MainProgram.cpp`, `tests/test_psf_modes.cpp` |
| compact 260601 chunk CSRB | CTAC | 单床 chunk 路径已实现 | 需物理 review | share zero-mu/水圆柱记录 | `CompactSymmetricProjectorCUDA.cpp`, checklist/continuity |
| compact 260601 chunk CSRB | CTAC multi-bed | 否 | 否 | 否 | `validateArgs` 拒绝 compact CTAC multi-bed |
| compact 260601 chunk CSRB | bProj / bProjList | 外部 additive 支持 | 零背景回归；非零散射未物理签收 | 是 | continuity bProjList zero smoke |
| compact 260525 single-file | MLEM/BSREM subset=1 CUDA | legacy compatibility；compact-symmetric 公式路径保留 | 旧数据对照需要保留 | 条件可跑 | `-loadCSR -compactMeta`, 76B metadata |
| old row-wise chunk_*.bin | 转换工具读取 | reference-only | 否 | 否 | `tools/convert_6module_sysmat_to_csrb.cpp`, `6module_bak_src/README.md` |

## 明确不支持或未签收

- compact OSEM subsets、compact ADMM/TV、compact CTAC multi-bed。
- `w_proj` / `w_img` 在 compact/fused BSREM 中不允许静默忽略。
- OSL-style penalized path不是生产推荐口径；不在本批统一。
- projection-domain PSF 与 Operator PSF 模型不同，保持隔离。
- BSREM/RDP beta sweep、Derenzo objective、针孔汇聚射线 Siddon CTAC 属物理专项。
## 本批变更附注

- 260525 single-file 路径的能力不扩展；本批只增加安全保护：大文件默认隐式 `CSRMap`，full-copy 超过可用物理内存一半时拒绝。
- 本批本地修复了 NVRTC attenuation scale 常量作用域问题；这是 kernel source 自包含缺陷修复，不作为新能力项，随 compact CTAC value baking review 交 Opus/Fable 签收。

## 260525 退场条件

260525 单文件 compact 入口只有在以下三项均满足后才能删除：

1. 连续两个发布周期无人使用 `-loadCSR -compactMeta`，有 RunLogger 使用记录或团队确认。
2. 260601 chunk CSRB 路径完成全部物理验收并成为唯一 compact 生产路径。
3. Opus 4.8 / Fable 5 显式签收退场。

在此之前仅允许 deprecated 提示、大文件保护、隐式 mmap 安全路径和文档标注。
