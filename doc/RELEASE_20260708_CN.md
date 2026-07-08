# Release 2026-07-08（本地 RC）

Release date: 2026-07-08
Git: 见包名后缀（`SPECT_recon_rc_20260708_<hash>`）

## 版本要点

- **CUDA-only 生产口径**：稀疏后端固定 CUDA；旧 `-sparseBackend` / `-spBackend` 直接拒绝。
- **公开 CLI 收口**：算法入口保留 `-opti`；PSF 公开入口保留 image-domain `Operator` / `None`；`-psfProj`、PostSmooth/ProjDomain 公开入口退场。
- **CLI 迁移清单**：发布包新增 `CLI_CHANGES_CN.md`，列出已删除/改名参数和替代写法。
- **CTAC**：compact chunk float CTAC 默认走 fused CUDA nnz lookup，`SPECT_COMPACT_CTAC_BAKED_REFERENCE=1` 仅作 debug/parity。
- **CTAC pinhole 决策**：生产路径继续使用 `att[angle, voxel]` fused CTAC；pinhole-center Siddon 只作为 validation-only sampler，不进入生产。
- **打包**：`scripts/package_rc.ps1 -CleanPrevious` 清理旧 `dist/release_candidates` 后生成当前 RC 目录和 zip。

## 相对 2026-07-07 的主要变化

- fused compact CTAC 默认替代 baked values 调度，保留显式 debug oracle。
- 新增 auto-mu / pinhole reference validation 脚本和 CTAC 决策文档。
- 默认日志输出包含数据路径与重建参数摘要，详细 profiling 收敛到 debug 级。
- 公开参数文档同步为 CUDA-only、cuFFT-only、TV/RDP-only、`-opti`、`-psf [fwhm_mm[,N]]`。

## 验证（打包前）

- `spect_tests.exe --gtest_brief=1`：272/272 passed。
- `spect_cuda_scaffold_tests.exe --gtest_brief=1`：27/27 passed。
- `SPECT_recon` Release target relink 成功。
- 已知非阻塞项：MSBuild 的 CUDA scaffold gtest discovery post-step 仍可能返回 1；直接运行测试 exe 作为当前验收来源。

## 保留边界

- 不包含 q16 / CSRBQ v2B。
- 不包含 production Siddon / row-bin attenuation table。
- 不包含 compact multi-bed CTAC。
