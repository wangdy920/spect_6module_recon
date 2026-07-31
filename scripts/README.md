# 脚本入口（与重构流程集成）

本目录提供对外脚本入口，用于把仓库内的辅助脚本（Python/PowerShell/MATLAB）融入到“构建 → 测试 → 验证/分析”的常用流程中。

> 2026-07-31 收敛重构后：矩阵唯一入口为 `-chunkCSRDir`，`-loadCSR`/`-mapCSR`/`-matrixLayout`/`-subsets` 等已删除（见 `doc/CLI_CHANGES_CN.md`）。

## 1) L-curve（β 扫描）

- 脚本：`scripts/run_l_curve.ps1`（封装 `utils/run_l_curve.py`）
- 依赖：Python + `numpy`, `matplotlib`（见 `scripts/requirements.txt`）
- 可选自检：`python scripts/l_curve_smoke_test.py`（只校验参数文件解析与 `[METRIC]` 正则）

示例（推荐用 `-BaseArg` 逐 token 传参，避免 Windows 路径/空格问题）：

```powershell
.\scripts\run_l_curve.ps1 `
  -BaseArg @(
    "-dfList", ".\\gatedata\\MDP_4h\\data_append_Bed1.mat,.\\gatedata\\MDP_4h\\data_append_Bed2.mat,.\\gatedata\\MDP_4h\\data_append_Bed3.mat,.\\gatedata\\MDP_4h\\data_append_Bed4.mat",
    "-dim", "100,100,100",
    "-vox", "0.3,0.3,0.3",
    "-ang", "360",
    "-detN", "40,40",
    "-it", "5:5",
    "-th", "16",
    "-chunkCSRDir", ".\\bin\\matrix\\260601",
    "-butter", "0.35,2",
    "-bedOverlap", "16,16,16,16",
    "-cropEnd", "16,16",
    "-fout", "recon_out\\lcurve_run"
  ) `
  -RegType TVIsotropic `
  -Betas "1e-4:1e-1:8" `
  -Output "recon_out\\l_curve_results.png"
```

## 2) 数值回归金标准（收敛重构验收口径）

- 脚本：`scripts/run_260601_regression.ps1`
- 目标：对 260601 compact canonical 路径（MLEM/BSREM/PSF+Butterworth/两床 stitch）跑出固定 case，并与冻结基线 `recon_out/convergence_baseline_20260731/` 及 archived 参考 `test_data/sys_260601_recon_ref/recon_iter050.img` 对比
- 阈值：b1_plain vs archived 参考 `rmse_over_reference_mean <= 0.02`、`sum_rel_diff <= 0.005`；vs 冻结基线差异应处于 float 噪声量级（~1e-7）

示例：

```powershell
powershell -ExecutionPolicy Bypass -File scriptsun_260601_regression.ps1
```

> 已退役（standard 布局 / 旧单文件 CSRB 时代）：`run_4bed_smoke.ps1`、`run_bsrem_kappa_sweep.ps1`、`run_6module_compact_acceptance.ps1`、`measure_butter_backend.ps1`。

## 3) CTAC / auto-mu 圆柱调试 gate

- 文档：`doc/CTAC_AutoMu_Validation_CN.md`
- 只用于 CTAC/auto-mu 调试；Derenzo 不作为主 gate。
- `scripts/qa_ctac_automu_cylinder.py`：从 260601 reference 圆柱测量直径/高度，生成 true/auto mu-map、Dice/volume/profile 表和 PNG。
- `scripts/run_ctac_automu_cylinder_debug.ps1`：可选运行 true-mu CTAC 与 auto-mu CTAC，并比较输出差异。
- `scripts/validate_ctac_pinhole_reference.py`：validation-only 抽样比较当前 `att[angle, voxel]` 与 pinhole-center Siddon reference，不进入生产路径。
- `scripts/export_recon_volume.py`：重建 `.mhd` 导出 **DICOM 三维序列**（默认 `spect_to_dicom.exe` → `dicom/image_*.dcm`）或 **PNG 切片**（默认全部轴位 `png/axial/`；可选 `views_center.png` 三正交视图）。

```powershell
python scripts\export_recon_volume.py `
  --image recon\mouse_mlem_it30.mhd `
  --out-dir recon\export `
  --format both `
  --png-mode slices
```

- `scripts/run_mouse_spect_multibed.ps1`：小鼠 ENG 预处理 + 多床位 MLEM + 可选 DICOM/PNG 导出（`-ExportFormat both`）。
- `scripts/stitch_multibed_mhd.py`：实验性单床结果拼接器；自动估计共同 overlap/Z 方向，输出无 overlap reference 与 guarded cosine 融合结果，不进入主线重建路径。

```powershell
python scripts\qa_ctac_automu_cylinder.py --self-test
python scripts\validate_ctac_pinhole_reference.py --self-test
powershell -ExecutionPolicy Bypass -File scripts\run_ctac_automu_cylinder_debug.ps1 -IterationList 20,50,100
python scripts\eng_to_6module_projection_bin.py --self-test
```

`-Iterations 1` 仅作为入口 smoke；CTAC/auto-mu 输出差异主记录应使用 20/50/100 或更长 iteration。可加 `-SkipRecon` 只生成/刷新 mu-map、profile 和视图。

## 4) MATLAB 工具

`utils/*.m`：PSF map/phantom 相关辅助脚本（依赖 MATLAB/Octave 环境）。
详见 [`utils/README.md`](../utils/README.md)。

## 5) 3D 体绘制可视化（Python）

`render_volume_3d.py` 把重建 MHD 渲染为真三维等值面 PNG，用于视觉 QA。
这是调试/验证视图，不替代生产体数据或 DICOM 导出。示例：

```powershell
python scripts/render_volume_3d.py `
  --image recon\mouse_mlem_it50.mhd `
  --output recon\qa\mouse_it50_3d.png `
  --fov-diameter-mm 30 --percentile 99 --downsample 2
```

## 6) 其他脚本

下面列出未在第 1–5 节展开的脚本。除明确标注为"实验评估"外，其余均为通用辅助工具，可按需调用。脚本入口约定：PowerShell 用 `powershell -ExecutionPolicy Bypass -File scripts\<name>.ps1`，Python 直接 `python scripts\<name>.py`。

### 构建 / 发布辅助

| 脚本 | 用途 |
|---|---|
| `clean_extra_builds.ps1` | 清理冗余构建目录（根 `README.md` 的 release 流程已引用） |
| `build_cuda_scaffold.ps1` | 构建 `spect_cuda_scaffold_tests` 等 CUDA scaffold 测试目标（`-Build` 触发编译） |
| `stage_dev_package.ps1` | 执行 `cmake --install` 生成 `staging\SPECT_recon_rc_dev\` 开发包 |
| `migrate_release_layout.ps1` | 把旧布局产物迁移到 RC 布局 |
| `package_rc.ps1` | 打包 RC zip + SHA256，并拷贝 `PARAMETERS_CN.md` / `README.md` 进发布包（`doc/RELEASE_20260715_CN.md` 已引用） |

### 对比 / QA（Python）

| 脚本 | 用途 |
|---|---|
| `compare_raw.py` | 对比两个 `.raw` float32 体数据的数值精度（金标准回归在用） |
| `compare_recon_metrics.py` | 对比两个 SPECT MHD/RAW 重建的标量相似度（CTAC 文档已引用） |
| `compare_seam_postprocess_parity.py` | 验证 C++ 接缝后处理与冻结 Python 参考一致 |
| `postprocess_overlap16_auto.py` | overlap16/k12 冻结 Python 参考 |
| `validate_6module_compact_inputs.py` | 校验 6module compact 投影/矩阵输入一致性 |
| `eval_ctac_smoke.py` | 单次 CTAC vs no-CTAC 最小配对 QA |
| `eval_csrb_crc_var.py` | 从 CSRB 近似局部 CRC 方差曲线 |
| `monitor_flood_map_concentration.py` | flood map 浓度监控 |

### CT / SPECT 配准（Python）

| 脚本 | 用途 |
|---|---|
| `prepare_ct_mu_map.py` | 从 DICOM 准备 CT mu-map |
| `register_ct_spect_global.py` | CT/SPECT 全局残差配准（仅全局平移/留出锚点验证，禁止逐床独立配准） |
| `refine_ct_spect_bone_centers.py` | CT/SPECT 骨中心细化 |
| `refine_ct_spect_bone_distance.py` | CT/SPECT 骨距离细化 |

### 实验评估脚本（探索性，非生产路径）

以下脚本用于特定数据集的 seam 修复、phantom 评估和可视化实验，**仅作为离线探索**，不进入 `VolumeOutput.cpp` 主线。脚本内 docstring 记录了各自的实验动机与方法；如需复用结论，请先阅读脚本头注释。

- `run_ct_guided_regularizer_compare.ps1`：CT 引导正则化对比驱动脚本（`Capability_Matrix_CN.md` 已引用）。
- `make_synthetic_phantom.py`、`eval_phantom_21mm_seam_regression.py`：合成 phantom 与 21mm seam 回归评估。
- `eval_mouse_continuity_bridge.py`、`eval_mouse_confidence_fusion.py`：mouse 接缝连续性与置信度融合评估。
- `eval_epguan_*.py`（8 个）：EPguan 仿体 seam 修复/轮廓插值/纹理保留/半径锁定等实验。
- `eval_fov_close_bed_similarity_classifier.py`、`eval_feature_discrimination_profiles.py`、`visualize_epguan_ade_export.py`、`eval_epguan_phantom_feather_resample.py`：分类、特征判别与可视化辅助。

> 新增脚本时请在此表登记用途；标记为实验性的脚本若结论已被主线采纳，应迁入第 1–5 节并删除"实验评估"归类。
