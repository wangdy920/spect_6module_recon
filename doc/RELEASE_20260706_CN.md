# Release 2026-07-06（260601 默认矩阵）

## 版本要点

- **默认系统矩阵**：260601 compact chunk CSRB（开发包 `bin/matrix/260601` 联结至 `test_data/6module/260601`）
- **260601 MLEM 50 iter 签收**：与 `test_data/sys_260601_recon_ref/recon_iter050.img` 对齐（`rmse/ref_mean ≈ 1.06×10⁻⁵`，投影 `PSC_6module_24pin_10MBq_600s_test_1.bin`）
- **B2（260525）**：归档 reference 来源暂无法确认，本 release 不以 B2 数值作为 hard gate
- **目标 GPU**：NVIDIA RTX **4070**（Ada, CC 8.9）、**5060 / 5070**（Blackwell, CC 12.0 / sm_120）；需 **CUDA 12.8+** 驱动/工具链以支持 50 系

## 目录说明（避免与 CMake `Release` 配置混淆）

| 目录 | 含义 | 是否重复编译 |
|------|------|--------------|
| `build/Release/` | CMake **编译树**（中间产物 + `bin/Release/SPECT_recon.exe`） | 编译发生在这里 |
| `staging/SPECT_recon_rc_dev/` | **`cmake --install` 开发/QA 包**（RC 布局：`bin/`、`PARAMETERS_CN.md`） | 否，从 build 树**安装复制** |
| `dist/release_candidates/SPECT_recon_rc_*/` | **冻结 RC 归档**（含 zip，不随日常构建更新） | 否 |

不再使用顶层 `release/` 或已废弃的 `out/`。验收与现场运行请 **cd 到包根目录** 后执行 `.\bin\SPECT_recon.exe`（与 `dist/release_candidates/SPECT_recon_rc_20260610/PARAMETERS_CN.md` 一致）；调试可直接跑 `build/Release/bin/Release/SPECT_recon.exe`（内容同源）。

若本地仍为旧路径 `release/`、`install/Release/`，可运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\migrate_release_layout.ps1
```

## 构建

```bat
powershell -ExecutionPolicy Bypass -File scripts\clean_extra_builds.ps1
build_release.bat
```

产物：`staging/SPECT_recon_rc_dev/`（含 `PARAMETERS_CN.md`、`bin/matrix/260601` 联结）

Legacy 对照程序（可选）：

```powershell
nvcc -O3 -std=c++14 `
  -gencode arch=compute_89,code=sm_89 `
  -gencode arch=compute_120,code=sm_120 `
  6module_bak_src\spect_recon_mlem.cu `
  -o staging\SPECT_recon_rc_dev\bin\spect_recon_mlem_legacy.exe `
  -lcusparse -lcufft
```

## 默认重建示例（260601）

与 RC 包调用方式一致——在包根目录执行，矩阵由 `bin/matrix/260601` 自动识别：

```powershell
cd staging\SPECT_recon_rc_dev
.\bin\SPECT_recon.exe `
  -df ..\..\test_data\sys_260601_recon_ref\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -o recon_output `
  -it 50:5
```

## 一致性验证

见 `doc/Legacy_MLEM_B1B2_Parity_CN.md`、`scripts/run_6module_legacy_b1b2_parity.ps1`。
