# PSF 模型差异说明（§6.4，2026-07-05）

> 状态：文档-only，待 Fable 5 / Opus 4.8 review。代码路径不在本 PR 修改。

## 目的

说明 SPECT-recon 中 **image-domain Operator PSF** 与 **projection-domain PSF** 的数学差异、适用约束与验收边界，避免混用。

## Operator PSF（`-psfMode Operator`，推荐生产口径）

- **模型**：在图像域对估计体数据 `f` 施加可分离 Gaussian `G`（及 `G^T`），再进入系统矩阵 `A` / `A^T`。
- **典型更新项**：MLEM/BSREM 中 sensitivity 与 numerator 路径对 `G` 与 `G^T` 的对称使用；compact 路径在 branch 累加外层一次性施加。
- **约束**：compact-symmetric Phase 1 仅支持 Operator 或 None；要求 XY 各向同性以与 D4/Z mirror 可交换。
- **证据**：`tests/test_psf_modes.cpp`、compact multi-bed global-volume PSF smoke、share/260601 回归记录。

## Projection-domain PSF（`-psfMode ProjDomain`，隔离实现）

- **模型**：在投影域对 `Af` 施加 `G`，backward 使用 `A^T G^T(p / (G A f))` 形式（与 Operator 路径的 `A^T G^T(...)` 组装顺序不同）。
- **约束**：实现与测试限定在 **单床、subsets=1**；compact 路径 **不支持**。
- **物理签收**：模型差异未作为定量 phantom 验收项；文档层明确“与 Operator 不可互换”。
- **证据**：`ReconPreparation.cpp`、`ProjectionOps.cpp`、`ProjDomainChunked.h`；compact validation 拒绝 `-psfProj` / ProjDomain。

## PostSmooth / DistanceDependent

- **PostSmooth**：重建后滤波，不参与 EM 更新公式。
- **DistanceDependent**：已公开退场；当前 CLI/config 显式拒绝。

## 使用建议

| 场景 | 推荐 |
| --- | --- |
| 260601 compact 生产 | `-psf <sigma_mm> -psfMode Operator` |
| 260525 legacy single-file compact | 同 Operator/None；CTAC 需 chunk 路径 |
| standard CSRB 对比实验 | 可启用 ProjDomain，但需单独回归，不得与 Operator 输出直接对比 |

## 相关文档

- `doc/Capability_Matrix_CN.md`
- `doc/CompactSymmetricBackend_Design_CN.md` §PSF
- `doc/USAGE_CN.md`
