# Release 2026-07-07（正式 RC）

Release date: 2026-07-07  
Git: 见包名后缀（`SPECT_recon_rc_20260707_<hash>`）

## 版本要点

- **目录布局**：开发包 `staging/SPECT_recon_rc_dev/`；冻结归档 `dist/release_candidates/SPECT_recon_rc_*`
- **调用方式**：与 `PARAMETERS_CN.md` 一致——cd 到包根目录，执行 `.\bin\SPECT_recon.exe ...`；配置/矩阵按 exe 同级 `bin/config/`、`bin/matrix/` 自动识别
- **默认系统矩阵**：260601 compact chunk CSRB（包内不捆绑矩阵目录；现场复制 `test_data/6module/260601` 到 `bin/matrix/260601`，或 `-chunkCSRDir`）
- **260601 MLEM 50 iter 签收**：与 `test_data/sys_260601_recon_ref/recon_iter050.img` 对齐（B1/B1，投影 `PSC_6module_24pin_10MBq_600s_test_1.bin`）
- **目标 GPU**：RTX **4070**（sm_89）、**5060 / 5070**（sm_120）；CUDA 13.x 运行库已捆绑

## 相对 RC 20260704 的主要变化

- CLI：PSF 默认值简化、冗余 verbose 日志收敛（`9c95dd1` 及后续工作区改动）
- CTAC：ENG 投影转换脚本与 automu 圆柱体验收脚本（`28bd484`、`2b5cadf`）
- 打包：`scripts/package_rc.ps1` 从 staging 生成自包含 RC（CUDA/MSVC DLL、SHA256、zip）
- 废弃 `out/` 实验构建树；清理 `gtest` 测试 DLL

## 验证（打包前）

- `build_release.bat` → `staging/SPECT_recon_rc_dev/bin/SPECT_recon.exe`
- `-dumpConf` 应显示 `bin/config/spect_config.ini (loaded)`
- 可选：`scripts/run_6module_legacy_b1b2_parity.ps1 -SkipB2`

## 环境变量

- `SPECT_CUDA_SYNC_LEGACY=1` — 恢复 PR21 之前的 pageable D2H 拷贝路径
