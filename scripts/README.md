# 脚本入口

这里只保留当前生产路径的构建、回归、CTAC/配准与通用输出工具。矩阵唯一入口为 `-chunkCSRDir`；多床输出固定为 canonical `center66_cosine10`。历史 seam/profile 修补与自动拼接实验已删除，当前结论见 `reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html`。

## Canonical 数值回归

`run_260601_regression.ps1` 运行 260601 compact MLEM/BSREM、`sieve + CASToR + FWHM 1.3 mm`、独立 Butterworth 能力测试与两床 canonical stitch。单床 B1 继续对 `test_data/sys_260601_recon_ref/recon_iter050.img` 做硬阈值比较；`-GoldenDir` 仅用于未改变数学的冻结 case。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_260601_regression.ps1
```

## 设备端慢重建/重复进程采集

安装包内脚本位于 `bin\tools`。它只启动一个重建进程，并把完整命令、版本/hash、配置、矩阵/cache、全部同名 PID/父 PID/命令行、每 5 秒 GPU 状态和重建日志写进一个 UTF-8 文本：

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\tools\collect_device_recon_diagnostics.ps1 `
  -InputPath "D:\data\case\SPECT\Bed1.bin;D:\data\case\SPECT\Bed2.bin" -MultiBedList `
  -OutputPrefix "D:\SPECT_Result\device_test\recon" -Iterations 20:20 `
  -ReportPath "D:\SPECT_Result\device_test\spect_device_diagnostics.txt"
```

`-MatrixDir` 默认从配置或 exe 同级唯一矩阵自动发现。若要固定 denom 文件，追加 `-DenomCachePath ".\bin\cache\compact_denom\<file>.mhd"`。重建失败时已采集内容仍保留；回传这一份 `spect_device_diagnostics.txt` 即可。

相关通用比较器：

| 脚本 | 用途 |
|---|---|
| `compare_compact_legacy.py` | MHD/RAW/headerless IMG 数值比较 |
| `compare_raw.py` | float32 RAW 精度比较 |
| `compare_recon_metrics.py` | 重建体标量相似度比较 |
| `eval_mhd_stats.py` | MHD 统计与读取 helper |
| `validate_6module_compact_inputs.py` | 投影、metadata 与 chunk CSRB 合同校验 |
| `qa_compact_cylinder.py` | compact 圆柱 QA |
| `eval_csrb_crc_var.py` | CSRB 局部 CRC 方差诊断 |

## CTAC、配准与正则化

| 脚本 | 用途 |
|---|---|
| `run_ctac_automu_cylinder_debug.ps1` | true-mu / auto-mu CTAC 配对运行 |
| `qa_ctac_automu_cylinder.py` | auto-mu Dice、体积和 profile QA |
| `validate_ctac_pinhole_reference.py` | validation-only pinhole-center Siddon 抽样参考 |
| `eval_ctac_smoke.py` | CTAC / no-CTAC 配对指标 |
| `prepare_ct_mu_map.py` | 用共用 FrameOfReference / IPP / IOP 把 CT DICOM 重采样到 SPECT 网格；`--profile device160` 对齐设备 108×108×160，`--profile recon220` 对齐主线 120×120×220。推荐 `--fixed-translation-xyz mdp_device --measure-residual --apply-residual`：从 `multipinhole/config/ct_spect_registration.ini` 加载冻结 MDP 脊柱平移后再做 ±2 mm 脊柱+膝 3 自由度细化。0.3 mm 门作用在脊柱/膝残差上，仍为暂定诊断门。写出 `<output>_registration.ini` 供后续直接加载。`--attmap-bed-order recon_bed1_max_z -z-bed-flip` 写出 SPECT_recon 的 `-attMap`。`--aligned-dicom-dir` 在 SPECT DICOM 网格上写出可叠图的 CT / SPECT / fusion 序列（同一 FoR 与 IPP，平移在 CT 体素里）。 |
| `register_ct_spect_global.py` | CT/SPECT 全局平移配准与留出验证 |
| `refine_ct_spect_bone_centers.py` | CT/SPECT 骨中心细化 |
| `refine_ct_spect_bone_distance.py` | CT/SPECT 骨距离细化 |
| `run_ct_guided_regularizer_compare.ps1` | CT 引导正则化配对驱动 |
| `eval_ct_guided_regularizer_compare.py` | CT 引导正则化指标 |

自检：

```powershell
python scripts\qa_ctac_automu_cylinder.py --self-test
python scripts\validate_ctac_pinhole_reference.py --self-test
python scripts\eng_to_6module_projection_bin.py --self-test
python scripts\prepare_ct_mu_map.py --help
```

## 输出与可视化

`export_recon_volume.py` 将 MHD/RAW 导出为 DICOM 三维序列或 PNG 切片；`render_volume_3d.py` 生成调试用等值面图，不替代原始体或 DICOM 验证。

```powershell
python scripts\export_recon_volume.py `
  --image recon\recon_it50.mhd `
  --out-dir recon\export `
  --format both `
  --png-mode slices
```

## L-curve

`run_l_curve.ps1` 封装 `utils/run_l_curve.py`；`l_curve_smoke_test.py` 只校验参数与 `[METRIC]` 解析。调用时把完整 canonical 重建参数逐 token 传入 `-BaseArg`，不要省略 `-chunkCSRDir`。

## 构建与发布

| 脚本 | 用途 |
|---|---|
| `clean_extra_builds.ps1` | 清理冗余构建目录 |
| `build_cuda_scaffold.ps1` | 构建 CUDA scaffold 测试目标 |
| `stage_dev_package.ps1` | 安装开发包到 `staging/SPECT_recon_rc_dev` |
| `package_rc.ps1` | 打包 RC、文档与校验和 |
| `migrate_release_layout.ps1` | 将旧目录布局迁移为 RC 布局 |
| `validate_butter_cufft_build.bat` | 验证 cuFFT Butterworth 构建 |

`monitor_flood_map_concentration.py` 与 `eng_to_6module_projection_bin.py` 分别用于 flood-map 监控和 ENG→6module 投影 BIN 转换。新增脚本必须是可复用入口；数据集特定的一次性实验不进入 `scripts/`。
