# Release 2026-07-08（当前）

Release date: 2026-07-08  
Git / 包名：`SPECT_recon_rc_20260708_<hash>`（例：`ad3c1d4`）

本文是**唯一维护中的 release 边界说明**。更早的 `RELEASE_20260706/07` 已删除；包布局与签收口径以下文为准，细节以代码与 `bin/config` 为准。

## 版本要点

- **CUDA-only 生产口径**：稀疏后端固定 CUDA；`-sparseBackend` / `-spBackend` 直接拒绝。Butterworth 固定 cuFFT。
- **公开 CLI**：算法入口 `-opti`；PSF 仅 image-domain `Operator` / `None`（`-psf [fwhm_mm[,N]]`）；`-psfProj` / PostSmooth / ProjDomain 公开入口退场。
- **默认矩阵**：260601 compact chunk（`metadata.bin` + `chunk_*.csrb`）。RC zip **可不捆绑**矩阵；现场拷到 `bin/matrix/260601`，或 `-chunkCSRDir`。唯一矩阵目录时可自动发现。
- **CTAC**：compact float CTAC 默认 fused CUDA nnz lookup；`SPECT_COMPACT_CTAC_BAKED_REFERENCE=1` 仅 debug/parity。生产衰减表为 `att[angle, voxel]`；**不做** production Siddon / dense `att[row, voxel]`。
- **不含**：q16 / CSRBQ v2B；compact multi-bed CTAC。
- **打包**：`scripts/package_rc.ps1 -CleanPrevious` → `dist/release_candidates/`。

## 包布局与调用

```text
SPECT_recon_rc_*/
  bin/SPECT_recon.exe + DLL
  bin/config/spect_config.ini, algo_*.conf
  bin/matrix/<name>/          # 可选；唯一含 metadata.bin 的目录可自动选用
  PARAMETERS_CN.md, CLI_CHANGES_CN.md, RELEASE_NOTES.md, ...
```

- 在包根执行：`.\bin\SPECT_recon.exe ...`（无需 `-config`；exe 旁 `config/` 自动加载）。
- 目标 GPU：RTX 4070（sm_89）、5060/5070（sm_120）；运行库随包。
- 环境变量：`SPECT_CUDA_SYNC_LEGACY=1` 可恢复旧 pageable D2H（调试用）。

## 签收与验证

| 项 | 口径 |
|---|---|
| B1 / 260601 plain MLEM 50 iter | vs `test_data/sys_260601_recon_ref/recon_iter050.img`，`rmse/ref_mean ≈ 1.06×10⁻⁵`（见 `Legacy_MLEM_B1B2_Parity_CN.md`） |
| B2 / 260525 | reference-only，非 hard gate |
| 打包前测试（本 RC） | `spect_tests` 272/272；`spect_cuda_scaffold_tests` 手动 27/27；MSBuild gtest post-step 失败非阻塞 |

## 相关维护文档

| 文档 | 用途 |
|---|---|
| `USAGE_CN.md` / `CLI_CHANGES_CN.md` | CLI |
| `Capability_Matrix_CN.md` | 能力边界与耐久决策 |
| `Legacy_MLEM_B1B2_Parity_CN.md` | B1 回归 |
| `CTAC_AutoMu_Validation_CN.md` | CTAC/auto-mu 调试 gate |
| `Recon_Interface_Migration_Field_CN.md` | 现场上位机迁移 |
| `Format_Reader_Matrix_CN.md` | 6module 格式×读者 |
