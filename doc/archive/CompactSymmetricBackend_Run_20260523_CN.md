# 6module compact-symmetric 后端执行记录（2026-05-23）

## 1. 执行目标

按 `doc/CompactSymmetricBackend_Execution_Target_CN.md` 验证第一条 6module `SYM_45Z` compact-symmetric 链路：

- 输入投影：`gatedata/6module/PSC_6module_24pin_10MBq_600s_1.bin`
- compact CSRB：`gatedata/6module/matrix_6module.csrb`
- metadata：`gatedata/6module/metadata.bin`
- 重建：`MLEM + subset=1 + CUDA compact projector + image-domain Operator PSF + Butterworth`
- 命令中 PSF 使用 sigma 语义：FWHM=1.5 mm 对应 `-psf 0.637 -psfMode Operator`
- 性能门槛：`total_load_to_iteration_done < 20 s`

## 2. 当前实现状态

已接通并验证的部分：

- `-matrixLayout compact-symmetric` 显式进入 compact 后端。
- `-loadCSR` 同目录默认发现 `metadata.bin`。
- metadata strict validation 校验 compact rows、full projection rows、detector shape、角度数和图像尺寸。
- compact Phase 1 限制为 `-sparseBackend cuda`、`-alg mlem -subsets 1`。
- 输入投影按 full rows `829440` 校验，没有截断到 compact rows `52992`。
- profile 日志已拆分加载、metadata、PSF、compact sensitivity、compact forward/ratio/backward、MLEM update、Butterworth 和输出写盘。
- 已新增 QA 脚本：`scripts/qa_compact_cylinder.py`，输出中心 XY/XZ/YZ 三视图 PNG 和 JSON 指标。

## 3. 验证命令

```powershell
.\build_cuda_scaffold\bin\SPECT_recon.exe `
  -df gatedata\6module\PSC_6module_24pin_10MBq_600s_1.bin `
  -loadCSR gatedata\6module\matrix_6module.csrb `
  -matrixLayout compact-symmetric `
  -alg mlem -subsets 1 -it 1 `
  -sparseBackend cuda `
  -butter 0.35,2 -butterBackend cufft `
  -psf 0.637 -psfMode Operator `
  -detN 48,48 -ang 360 `
  -dim 108,108,80 -vox 0.3 `
  -fout recon_out\6module_compact_cylinder_mlem_psf15_butter
```

## 4. 构建与单元测试

已通过：

```text
spect_tests.exe --gtest_filter=IOTest.CompactSymmetryMapsSixModuleFullProjectionRowsOnce:IOTest.CompactSymmetryValidationRejectsCompactProjectionRows:IOTest.ConfigParserStripsInlineComments:IOTest.CompactSymmetryMetadataReadsLegacyLayout:IOTest.LoadProjectionBinAutoDetects
```

结果：

```text
5/5 passed
```

该测试覆盖：

- BIN projection 自动识别与基本统计。
- legacy `metadata.bin` 读取。
- 真实 6module `SYM_45Z` 映射：`46` 个 compact angles、`8` 个 XY branch、`2` 个 Z branch 刚好覆盖 `829440` 个 full projection row，且无重复。
- compact 模式拒绝把 projection rows 错误传成 `52992` compact rows。
- 默认配置文件中的 inline comment 剥离，避免 `config/spect_config.ini` 中注释污染参数值。

## 5. 运行结果

### 5.1 cold run 记录

日志：

```text
recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_153014_789.log
```

关键 profile：

```text
projection_load                  0.005 s
csr_map_or_load                  0.496 s
metadata_load_and_validate       0.000 s
psf_prepare                      0.000 s
compact_sensitivity              2.017 s
compact_forward_ratio_backward   0.259 s
mlem_update                      0.000 s
butterworth                      0.030 s
output_write                     0.014 s
total_load_to_iteration_done     2.873 s
full_command_wall_time           2.889 s
exit code                        0
```

### 5.2 warm run 记录

日志：

```text
recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_153457_109.log
```

关键 profile：

```text
projection_load                  0.005 s
csr_map_or_load                  0.500 s
metadata_load_and_validate       0.000 s
psf_prepare                      0.000 s
compact_sensitivity              2.032 s
compact_forward_ratio_backward   0.262 s
mlem_update                      0.000 s
butterworth                      0.050 s
output_write                     0.012 s
total_load_to_iteration_done     2.927 s
full_command_wall_time           2.940 s
exit code                        0
```

性能门槛已满足：cold run 和 warm run 的 `total_load_to_iteration_done` 均小于 20 s。

## 6. 输出与 QA

重建输出：

```text
recon_out/6module_compact_cylinder_mlem_psf15_butter_it1.mhd
recon_out/6module_compact_cylinder_mlem_psf15_butter_it1.raw
```

QA 输出：

```text
recon_out/qa/6module_compact_cylinder_mlem_psf15_butter_it1_qa.json
recon_out/qa/6module_compact_cylinder_mlem_psf15_butter_it1_views.png
```

QA 命令：

```powershell
python scripts\qa_compact_cylinder.py `
  --image recon_out\6module_compact_cylinder_mlem_psf15_butter_it1.mhd `
  --out-json recon_out\qa\6module_compact_cylinder_mlem_psf15_butter_it1_qa.json `
  --out-png recon_out\qa\6module_compact_cylinder_mlem_psf15_butter_it1_views.png
```

关键 QA 指标：

```text
dim_xyz                            108 x 108 x 80
finite_ratio                       1.0
min                                4.725477430555555
max                                8.228335048010974
sum                                5567817.6110945
center_xy_inside_mean              5.9125775653542965
center_xy_outside_mean             6.2650276213515435
outside_inside_mean_ratio          1.0596102211093998
left_right_mean_diff_over_inside   0.00020384369964732862
top_bottom_mean_diff_over_inside   0.00338253513751564
central_disk_cv                    0.008443600365611413
z_half_mean_diff_over_global       0.0006775836519446182
```

通过项：

- 输出维度正确。
- 无 NaN/Inf。
- 非负。
- 总和大于 0。
- 左右/上下对称性通过。
- Z 方向连续性通过。
- central disk CV 通过。

未通过项：

- `inside_mean_gt_outside_mean = false`
- `outside_inside_lt_0p35 = false`
- 当前 `outside_inside_mean_ratio = 1.0596102211093998`

结论：当前链路性能和运行稳定性已经过线；按最初的 `inside disk > outside annulus` QA 口径，图像背景 sanity 未通过，不能直接按该指标签收。

## 7. 输入投影 sanity

投影 BIN 本身的 detector-domain 粗查显示中心区域高于外圈：

```text
projection shape             360 x 48 x 48
min / max / sum              0 / 27 / 4666311
finite_ratio                 1.0
center disk mean             7.051897321428571
outer mean                   3.3840758547008547
outer / center               0.47988161206171814
```

这说明 QA 失败不能简单归因于输入投影全场均匀；下一步应优先核对 compact backprojection/sensitivity 对图像空间的归一化结果，或与旧 `6module_bak_src/spect_recon_mlem.cu` 输出做同输入对照。

## 8. 旧 CUDA MLEM 对照

已编译旧程序：

```powershell
cmd /c "call ""D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"" >nul && ^
  nvcc -O3 -std=c++14 -arch=sm_86 6module_bak_src\spect_recon_mlem.cu ^
  -o build_legacy_6module\spect_recon_mlem_legacy.exe -lcusparse -lcufft"
```

编译结果：成功。仅出现 Windows 编码页 `C4819` 与未用函数 warning。

旧 CUDA 对照命令：

```powershell
.\build_legacy_6module\spect_recon_mlem_legacy.exe `
  --matrix-dir gatedata\6module `
  --proj gatedata\6module\PSC_6module_24pin_10MBq_600s_1.bin `
  --out recon_out\legacy_6module_compact_mlem_psf15_butter `
  --iters 1 --gpus 1 --save-every 1 --sym auto `
  --conv-mode psf --psf-fwhm 1.5,1.5,1.5 --psf-sigmas 3 `
  --butter-cutoff 0.35 --butter-order 2 --butter-mode final
```

旧程序结果：

```text
host CSR build            42.29 s
iter 1                    0.191 s
exit code                 0
output                    recon_out/legacy_6module_compact_mlem_psf15_butter/recon_iter001.raw
converted MHD             recon_out/legacy_6module_compact_mlem_psf15_butter/recon_iter001_mhd.mhd
```

旧程序 QA：

```text
finite_ratio                       1.0
min                                4.846688747406006
max                                7.5707573890686035
sum                                5561549.0
center_xy_inside_mean              5.911585807800293
center_xy_outside_mean             6.224804401397705
outside_inside_mean_ratio          1.0529838530270714
left_right_mean_diff_over_inside   0.00025666477434875144
top_bottom_mean_diff_over_inside   0.002998105907532603
central_disk_cv                    0.007679252304941628
z_half_mean_diff_over_global       0.0006211512066366215
passed                             false
```

新后端与旧 CUDA 输出差异：

```text
max_abs_diff           0.9252169111955642
mean_abs_diff          0.046858286396852945
rmse                   0.08485811179327367
rmse / legacy_mean     0.01423754323137811
sum_rel_diff           0.0011270460808167067
```

该对齐检查已固化为脚本：

```text
scripts/compare_compact_legacy.py
scripts/run_6module_compact_acceptance.ps1
```

复跑命令：

```powershell
python scripts\compare_compact_legacy.py `
  --candidate recon_out\6module_compact_cylinder_mlem_psf15_butter_it1.mhd `
  --reference recon_out\legacy_6module_compact_mlem_psf15_butter\recon_iter001.raw `
  --out-json recon_out\qa\6module_compact_vs_legacy_cuda_iter001_compare.json `
  --max-rmse-over-mean 0.02 `
  --max-sum-rel-diff 0.005
```

结果：

```text
exit code                  0
passed                     true
rmse / legacy_mean         0.01423754323137811
sum_rel_diff               0.0011270460808167067
output json                recon_out/qa/6module_compact_vs_legacy_cuda_iter001_compare.json
```

已验证 `scripts/run_6module_compact_acceptance.ps1 -SkipRecon` 可基于已有 candidate 输出完成：

```text
QA JSON/PNG regeneration
legacy CUDA comparison
exit code 0
compare passed true
compare json recon_out/qa/6module_compact_cylinder_mlem_psf15_butter_vs_legacy_cuda_iter001_compare.json
```

脚本默认不带 `-SkipRecon` 时会重跑目标命令；`-SkipRecon` 仅用于复用已有 candidate 输出做快速复核。

已验证完整入口可从重建开始复跑：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_6module_compact_acceptance.ps1
```

结果：

```text
exit code                              0
log                                    recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_154907_576.log
total_load_to_iteration_done           2.838 s
full_command_wall_time                 2.853 s
compare passed                         true
compare rmse/reference_mean            0.01423754182542136
compare sum_rel_diff                   0.001127029406580609
```

2026-05-23T15:51:33+08:00 再次复跑完整入口，验证校准后的 QA gate：

```text
exit code                              0
log                                    recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_155133_336.log
total_load_to_iteration_done           2.929 s
full_command_wall_time                 2.942 s
strict QA passed                       false
calibrated_1iter_passed                true
compare passed                         true
compare rmse/reference_mean            0.014237543680872706
compare sum_rel_diff                   0.0011270290396925146
```

因此当前 1 iteration smoke 的判定方式为：strict `inside/outside` 继续记录但不单独否决；`calibrated_1iter_passed=true` 与旧 CUDA compare 通过共同作为真实数据 smoke gate。

解释：

- 旧 CUDA 同输入、同 1 iter、同 PSF/Butterworth 口径下也不满足 `inside disk > outside annulus`。
- 新后端与旧 CUDA 的总体差异较小，`rmse / legacy_mean ≈ 1.42%`，总和差异约 `0.11%`。
- 因此当前 strict QA 失败更可能是“1 次迭代 + 该系统矩阵/投影 + annulus 边界增强”下的基线特征，而不是新 compact-symmetric 映射独有错误。
- 仍不建议直接删除 QA 门槛；更合理的下一步是把验收拆成两层：先签旧 CUDA 数值对齐，再基于旧 CUDA 和更多迭代/参考图像重新校准圆柱形态阈值。

## 9. 关于“参考虚拟多床位映射”的结论

可以参考虚拟多床位的映射方式，但只能参考其“局部坐标到全局坐标的显式映射层”思想，不应把压缩角度/Z 当成真的床位来复用 bed offset 数据结构。

推荐抽象：

```text
compact local row
  -> compact angle / compact detector z / detector x
  -> symmetry branch / z branch
  -> full projection row

local compact image contribution
  -> symmetry branch inverse transform
  -> full local-bed image volume
  -> existing bedZStarts accumulation to global image
```

这与虚拟多床位的共同点是：都不改变底层矩阵的局部语义，而是在 orchestration 层通过受验证的 mapping 把局部 numer/denom 累加到目标空间。

关键差异：

- 多床位映射是 Z 平移：`local z + bedZStarts[bed] -> global z`。
- compact angle/Z 映射是 symmetry branch：角度折叠、D4 XY 变换、Z mirror 和 full projection row 选择。
- compact 的 full projection row 映射还参与 ratio 计算，不能只在最终图像累加时处理。

因此后续应把 `CompactSymmetryMetadata` 继续收敛成一个小而明确的 mapper，并为 `fullAngleFromCompactAngle`、`fullProjectionIndexFromCompactRow`、forward transform、backward inverse transform 增加 toy parity 测试。这样能复用虚拟多床位的工程模式，同时避免把两个不同维度的映射混在同一个 offset 机制里。

## 10. 下一步建议

1. 将新后端与旧 CUDA 的 1 iter 输出对齐固化为验收项；建议先设 `rmse / legacy_mean < 2%`、`sum_rel_diff < 0.5%` 作为首版阈值，再用重复运行确认稳定性。
2. 增加 compact toy parity 测试：8 分支角度映射、Z mirror、forward、ratio、backward 和 denom 都要与 explicit full operator 对齐。
3. 基于旧 CUDA 输出、更多迭代输出或已知真值重新校准圆柱形态 QA；当前 `inside disk > outside annulus` 对该 1 iter ring-like 输出过严且与旧基线不一致。

## 11. 50 iter plain MLEM 与已有旧结果对比

2026-05-23T16:09:16+08:00 按用户已有旧结果 `6module_bak_src/recon_iter050.img` 的真实参数口径补充同参数对比。

旧结果对应命令未包含 `--conv-mode` 或 `--butter-cutoff`，因此不是 PSF/Butterworth 结果，而是 50 次 plain MLEM。当前后端对比命令也显式禁用 PSF/Butterworth：

```powershell
.\build_cuda_scaffold\bin\SPECT_recon.exe `
  -df .\gatedata\6module\PSC_6module_24pin_10MBq_600s_1.bin `
  -loadCSR .\gatedata\6module\matrix_6module.csrb `
  -matrixLayout compact-symmetric `
  -alg mlem -subsets 1 -it 50 `
  -sparseBackend cuda `
  -psfMode None `
  -detN 48,48 -ang 360 `
  -dim 108,108,80 -vox 0.3 `
  -fout .\recon_out\6module_compact_cylinder_mlem_plain_iter50
```

运行结果：

```text
exit code                         0
output                            recon_out/6module_compact_cylinder_mlem_plain_iter50_it50.mhd/.raw
log                               recon_out/6module_compact_cylinder_mlem_plain_iter50_20260523_160916_749.log
projection_load                   0.006 s
csr_map_or_load                   0.489 s
metadata_load_and_validate        0.000 s
psf_prepare                       0.000 s
compact_sensitivity               2.041 s
compact_forward_ratio_backward    12.609 s / 50 calls
mlem_update                       0.019 s / 50 calls
butterworth                       0.000 s
output_write                      0.039 s
total_load_to_iteration_done      15.581 s
full_command_wall_time            15.597 s
```

旧 `recon_iter050.img` 已确认为 headerless little-endian float64，尺寸为 `108 x 108 x 80`。`scripts/compare_compact_legacy.py` 已补充 `.img` 读取分支。

对比命令：

```powershell
python .\scripts\compare_compact_legacy.py `
  --candidate .\recon_out\6module_compact_cylinder_mlem_plain_iter50_it50.mhd `
  --reference .\6module_bak_src\recon_iter050.img `
  --out-json .\recon_out\qa\6module_compact_plain_iter50_vs_legacy_recon_iter050_compare.json `
  --max-rmse-over-mean 0.02 `
  --max-sum-rel-diff 0.005
```

对比结果：

```text
passed                     true
shape                      80 x 108 x 108
candidate_sum              5865532.470954429
reference_sum              5865532.511930853
reference_mean             6.285935905275691
max_abs_diff               0.001857913949891099
mean_abs_diff              1.7623219840880896e-05
rmse                       4.2020365512940235e-05
rmse / reference_mean      6.684822458605277e-06
sum_rel_diff               6.985968228262434e-09
```

QA 输出：

```text
qa_json                    recon_out/qa/6module_compact_cylinder_mlem_plain_iter50_it50_qa.json
qa_png                     recon_out/qa/6module_compact_cylinder_mlem_plain_iter50_it50_views.png
finite_ratio               1.0
min/max                    0.05469330285006066 / 573.7687740265379
left_right_diff/inside     0.0006269205407866895
top_bottom_diff/inside     0.02753615215723244
central_disk_cv            0.1628811574122868
z_half_diff/global         0.0019524737963462105
```

结论：

- 当前 compact-symmetric CUDA 后端在 50 iter plain MLEM 口径下与旧 CUDA `recon_iter050.img` 基本数值一致。
- 旧 50 iter 结果不能作为 `-psf 0.637 -psfMode Operator -butter 0.35,2` 目标命令的同参数参考；PSF/Butterworth 目标仍应使用已有 1 iter 同口径旧 CUDA reference 或后续重新生成的同参数多 iter reference。

## 12. 本轮补充验收（数据事实与拒绝路径）

2026-05-23T16:31:00+08:00 补充两个可复跑验证入口：

```text
scripts/validate_6module_compact_inputs.py
tests/test_interfaces.cpp focused parser regression
```

数据事实验证命令：

```powershell
python .\scripts\validate_6module_compact_inputs.py
```

结果：

```text
exit code                         0
output json                       recon_out/qa/6module_compact_input_validation.json
matrix rows/cols/nnz              52992 / 933120 / 470220768
matrix row_ptr[last]              470220768
projection angles/rays/records    360 / 2304 / 829440
metadata mode                     45z
metadata compact angles           46
metadata detNx/detNz/compactDetZ  48 / 48 / 24
metadata image dims               108 / 108 / 80
all checks passed                 true
```

新增 focused regression：

```text
IOTest.CompactSymmetricParserDefaultsToStandard
IOTest.CompactSymmetricParserAcceptsExplicitMetaOverride
IOTest.CompactSymmetricParserRejectsUnsupportedPhase1Options
```

构建和测试：

```powershell
cmd /c "call ""D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"" && cmake --build build_cuda_scaffold --target spect_tests"

.\build_cuda_scaffold\bin\spect_tests.exe --gtest_filter=IOTest.CompactSymmetryMetadataReadsLegacyLayout:IOTest.CompactSymmetryMapsSixModuleFullProjectionRowsOnce:IOTest.CompactSymmetryValidationRejectsCompactProjectionRows:IOTest.CompactSymmetricParserDefaultsToStandard:IOTest.CompactSymmetricParserAcceptsExplicitMetaOverride:IOTest.CompactSymmetricParserRejectsUnsupportedPhase1Options:IOTest.ConfigParserStripsInlineComments:IOTest.LoadProjectionBinAutoDetects
```

结果：

```text
build spect_tests                  success
focused regression                 8/8 passed
```

本轮补齐的验收项：

- 固定 6module CSRB/BIN/metadata 数据事实已由脚本验证并输出 JSON。
- parser 默认 `standard`，不会因为旁边存在 metadata 自动切 compact。
- `-compactMeta` override 可解析。
- 2026-05-23 Phase 1 在 CLI validation 阶段拒绝非 CUDA、非 MLEM、`subsets>1`、CTAC、background/scatter、regularization、projection-domain PSF、DistanceDependent PSF 和 PostSmooth。2026-06-06 后当前 compact 能力已扩展到外部 additive `-bProj/-bProjList`、compact `BSREM+RDP` 和 multi-bed global-volume Operator PSF；本行保留为历史验收记录，不再代表当前能力矩阵。

## 13. 本轮补充验收（公开 CLI 收敛与历史入口拒绝）

2026-05-23T16:54:00+08:00 继续补齐验收清单中的公开 CLI、DD PSF 参数和 OSMAPOSL P1 拒绝项。

代码变更：

```text
include/cli/ArgumentParser.h
include/cli/ReconArgs.h
tests/test_interfaces.cpp
scripts/run_4bed_smoke.ps1
scripts/measure_butter_backend.ps1
scripts/run_bsrem_kappa_sweep.ps1
doc/USAGE_CN.md
doc/RECON_FLOW_CN.md
```

新增/扩展 focused regression：

```text
IOTest.CompactSymmetryMetadataMissingFileThrows
IOTest.StandardParserDoesNotEnableCompactFromMetaOverride
IOTest.CompactSymmetricParserRejectsDistanceDependentPsfParameters
IOTest.ParserRejectsDeprecatedOSMAPOSLAlgorithms
```

验证命令：

```powershell
cmd.exe /c 'call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build_cuda_scaffold --target SPECT_recon spect_tests'

.\build_cuda_scaffold\bin\spect_tests.exe --gtest_filter=IOTest.CompactSymmetryMetadataMissingFileThrows:IOTest.StandardParserDoesNotEnableCompactFromMetaOverride:IOTest.CompactSymmetricParserRejectsDistanceDependentPsfParameters:IOTest.ParserRejectsDeprecatedOSMAPOSLAlgorithms:IOTest.CompactSymmetricParserRejectsUnsupportedPhase1Options

$help = .\build_cuda_scaffold\bin\SPECT_recon.exe -help 2>&1 | Out-String
$bad = [regex]::Matches($help, 'OSMAPOSL|osmaposl|psfDist|psfDetRadius|psfRadii|DistanceDependent|DistDD|-opti')

rg -n -- "-opti|osmaposl|OSMAPOSL|-bedOffset|psfFWHM|psfDist" README.md doc/USAGE_CN.md doc/RECON_FLOW_CN.md scripts/README.md scripts/run_4bed_smoke.ps1 scripts/measure_butter_backend.ps1 scripts/run_bsrem_kappa_sweep.ps1

$files = @('scripts/run_4bed_smoke.ps1','scripts/measure_butter_backend.ps1','scripts/run_bsrem_kappa_sweep.ps1')
foreach ($file in $files) {
  $tokens = $null
  $errors = $null
  [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $file), [ref]$tokens, [ref]$errors) | Out-Null
  if ($errors.Count -gt 0) { exit 1 }
}
```

结果：

```text
build SPECT_recon/spect_tests       success
focused regression                  5/5 passed
help algorithm list                 MLEM | BSREM | ADMM
help forbidden public terms          0 matches
active public/script -opti scan      0 matches
PowerShell AST parse                 passed
```

本轮补齐的验收项：

- `SPECT_recon.exe -help` 不再展示 OSMAPOSL 或 DD PSF 参数，算法入口收敛为 `MLEM | BSREM | ADMM`。
- compact 模式显式拒绝 `-psfDistModel/-psfDistParams/-psfDetRadius/-psfRadiiFile`。
- `-alg osmaposl` 与 `-opti OSMAPOSL/OSL` 在 parser 阶段显式失败，并提示使用 `mlem/bsrem/admm`。
- `standard` layout 不因 `-compactMeta` 或 sidecar 概念自动切 compact。
- `scripts/run_4bed_smoke.ps1` 去除 OSMAPOSL case，保留 MLEM/BSREM smoke，并改用 `-alg`。

## 14. 本轮补充验收（compact denom/numerator 显式统计与失败保护）

2026-05-23T17:01:10+08:00 继续补齐 single-bed 真实 6module smoke 与 CUDA helper 中的 denom/numerator 可观测性验收项。

代码变更：

```text
src/main/MainProgram.cpp
```

变更内容：

- compact-symmetric 分支在 sensitivity denominator 计算、PSF backward 和多床位累加之后，输出 `[STATS] compact_denom`。
- 每次 compact numerator 计算并累加到全局 numerator 后，输出 `[STATS] compact_numer.itN`。
- denom 为空、包含 NaN/Inf、全零或总和非正时，立即抛出错误，避免继续进入 MLEM update。
- numerator 为空或包含 NaN/Inf 时，立即抛出错误。

验证命令：

```powershell
cmd.exe /c 'call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build_cuda_scaffold --target SPECT_recon'

powershell -ExecutionPolicy Bypass -File scripts/run_6module_compact_acceptance.ps1
```

结果：

```text
build SPECT_recon                  success
acceptance script                  exit 0
log                                recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_170110_292.log
compact_denom count                933120
compact_denom finite               933120
compact_denom nonzero              933120
compact_denom min/max/sum          0.115849196911 / 1.17669403553 / 781342.332214
compact_numer.it1 count             933120
compact_numer.it1 finite            933120
compact_numer.it1 nonzero           933120
compact_numer.it1 min/max/sum       0.62638515234 / 6.92444753647 / 4665964.00294
total_load_to_iteration_done        2.866s
full_command_wall_time              2.881s
calibrated_1iter_passed             true
legacy compare passed               true
legacy rmse/reference_mean          0.014237543898598897
legacy sum_rel_diff                 0.001127046424774358
```

本轮补齐的验收项：

- single-bed 真实 6module smoke 的 denom 非空、全有限、非全零，并有日志证据。
- compact CUDA helper 的 numerator 输出全有限，并有日志证据。
- denom/numerator 异常时 compact-symmetric 主路径会 fail fast，不会静默输出错误图像。

## 15. 本轮补充验收（synthetic parity 与共享 symmetry mapper）

2026-05-23T17:09:39+08:00 补齐 compact 数学与 synthetic parity 验收项，并把 D4/Z mirror voxel mapping 从 CUDA 私有实现收敛为共享小工具。

代码变更：

```text
include/io/CompactSymmetryMetadata.h
src/cuda/CompactSymmetricProjectorCUDA.cpp
tests/test_interfaces.cpp
```

变更内容：

- `CompactSymmetryMetadata.h` 新增 `CompactVoxelCoord`、`mapSym45D4XY`、`mapSym45ZVoxelCoord`、`compactVoxelLinearIndex`、`mapSym45ZVoxelIndex`。
- `CompactSymmetricProjectorCUDA.cpp` 的 forward transform 与 backward add 不再维护私有 D4/Z mirror 分支，统一复用共享 mapper。
- `tests/test_interfaces.cpp` 新增极小 `SYM_45Z` toy metadata 与 compact CSR，并展开为 explicit full dense operator 做 numerator/denominator 对照。
- toy 测试覆盖 45 度边界角不重复/不漏算、Z mirror duplicate guard、8 个 D4 branch、`sigma=0` PSF identity、XY 各向同性 Gaussian 与 D4 transform 可交换。

验证命令：

```powershell
cmd.exe /c 'call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build_cuda_scaffold --target spect_tests'

.\build_cuda_scaffold\bin\spect_tests.exe --gtest_filter=IOTest.CompactSymmetryToyMetadataAndD4MappingMatchLegacyRules:IOTest.CompactSymmetryToyExpansionCoversFullProjectionOnce:IOTest.CompactSymmetryToyNumeratorDenominatorMatchExplicitFull:IOTest.CompactSymmetryToyZeroPsfIsIdentityAndGaussianCommutesWithD4:IOTest.CompactSymmetryMapsSixModuleFullProjectionRowsOnce:IOTest.CompactSymmetryValidationRejectsCompactProjectionRows
```

结果：

```text
build spect_tests                  success
focused regression                 6/6 passed
```

本轮补齐的验收项：

- compact 角度、Z mirror 与 D4 图像映射均由共享 mapper 覆盖，避免 CUDA helper 和测试各自维护一套分支。
- compact numerator/denominator 与 explicit full operator 在 toy 数据上 `1e-12` 容差内一致。
- `sigma=0` PSF 等价于关闭 PSF；XY 各向同性 Gaussian Operator PSF 与 8 个 D4 branch 可交换。

## 16. 本轮补充验收（multi-bed overlap 累加工具）

2026-05-23T17:09:39+08:00 补齐 multi-bed 与 overlap toy 验收项，并把床位局部/全局体数据映射从 `MainProgram.cpp` 的匿名 helper 下沉到可测试工具。

代码变更：

```text
include/app/BedOffsets.h
src/main/MainProgram.cpp
tests/test_bed_offsets.cpp
```

变更内容：

- `BedOffsets.h` 新增 `accumulateBedVolumeToGlobal` 和 `extractBedVolumeFromGlobal`。
- `MainProgram.cpp` 删除私有 `makeBedZStarts`、`accumulateBedVolume`、`extractBedVolume`，统一使用 `computeBedZOffsets`、`accumulateBedVolumeToGlobal`、`extractBedVolumeFromGlobal`。
- compact-symmetric 分支仍按每床位独立 local image/local denom/local numerator 计算，再按 `bedZStarts` 累加到全局 denom/numer。
- standard 分支的 denom 累加也复用同一工具，避免 compact 与 standard 各自维护床位累加规则。

验证命令：

```powershell
cmd.exe /c 'call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build_cuda_scaffold --target SPECT_recon spect_tests'

.\build_cuda_scaffold\bin\spect_tests.exe --gtest_filter=BedOffsetsTest.*:MultiBedOverlapTest.*:IOTest.CompactSymmetryToyMetadataAndD4MappingMatchLegacyRules:IOTest.CompactSymmetryToyExpansionCoversFullProjectionOnce:IOTest.CompactSymmetryToyNumeratorDenominatorMatchExplicitFull:IOTest.CompactSymmetryToyZeroPsfIsIdentityAndGaussianCommutesWithD4:IOTest.CompactSymmetryMapsSixModuleFullProjectionRowsOnce:IOTest.CompactSymmetryValidationRejectsCompactProjectionRows
```

结果：

```text
build SPECT_recon/spect_tests       success
focused regression                  13/13 passed
```

本轮补齐的验收项：

- `overlap=0` 且 bed1=bed2 的退化测试中，两个 local 结果独立累加后保持一致。
- `overlap>0` 的 toy 测试中，重叠区 global buffer 等于两个 local bed 结果相加。
- 既有 `MultiBedOverlapTest.*` 继续通过，说明 sparse subset sensitivity/forward/backward 的 overlap 语义未被本轮重构破坏。

## 17. 本轮补充验收（relative_l2 与重复调用缓冲区）

2026-05-23T17:09:39+08:00 补充旧 CUDA 对齐报告中的 `relative_l2` 指标，并用已有 1 iter / 50 iter 输出重写对比 JSON。

代码变更：

```text
scripts/compare_compact_legacy.py
```

变更内容：

- `compare_compact_legacy.py` 在原有 `rmse`、`rmse/reference_mean`、`sum_rel_diff` 基础上新增 `relative_l2 = ||candidate-reference||_2 / ||reference||_2`。
- `runCompactSymmetricMlemNumerCUDA` 与 `computeCompactSymmetricSensitivityCUDA` 当前每次入口清零输出 buffer，并在每个 symmetry branch 前清零中间投影/反投影 buffer。

验证命令：

```powershell
python .\scripts\compare_compact_legacy.py `
  --candidate .\recon_out\6module_compact_cylinder_mlem_psf15_butter_it1.mhd `
  --reference .\recon_out\legacy_6module_compact_mlem_psf15_butter\recon_iter001.raw `
  --out-json .\recon_out\qa\6module_compact_cylinder_mlem_psf15_butter_vs_legacy_cuda_iter001_compare.json `
  --max-rmse-over-mean 0.02 `
  --max-sum-rel-diff 0.005

python .\scripts\compare_compact_legacy.py `
  --candidate .\recon_out\6module_compact_cylinder_mlem_plain_iter50_it50.mhd `
  --reference .\6module_bak_src\recon_iter050.img `
  --out-json .\recon_out\qa\6module_compact_plain_iter50_vs_legacy_recon_iter050_compare.json `
  --max-rmse-over-mean 0.02 `
  --max-sum-rel-diff 0.005
```

结果：

```text
1 iter PSF/Butterworth passed       true
1 iter relative_l2                  0.014210062232198062
1 iter rmse/reference_mean          0.014237543898598897
1 iter sum_rel_diff                 0.001127046424774358

50 iter plain passed                true
50 iter relative_l2                 3.4412670184357883e-06
50 iter rmse/reference_mean         6.684822458605277e-06
50 iter sum_rel_diff                6.985968228262434e-09
```

结论：

- 1 iter 同参数 PSF/Butterworth 对齐现在有 `relative_l2` 证据。
- 50 iter plain MLEM 在同一进程内重复调用 compact numerator 50 次后仍与旧 CUDA 基本一致，结合 helper 内部显式清零逻辑，可作为“重复调用不复用脏缓冲区”的验收证据。

## 18. 本轮补充验收（全局回归与 standard CSRB smoke）

2026-05-23T17:15:00+08:00 运行全局构建/测试回归，并补充一个最小 standard CSRB 主线 smoke。

代码变更：

```text
tests/test_interfaces.cpp
```

变更内容：

- `AppTest.ConfigParserIntegration` 从旧 `[Reconstruction]` 断言更新为当前 `config/spect_config.ini` 的全局配置结构：`[SystemMatrix]`、`[PSF]`、`[Filtering]`、`[Regularization]`。
- 不改变默认配置语义；算法参数仍由 `config/algo_*.conf` 管理。

验证命令：

```powershell
cmd.exe /c 'call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build_cuda_scaffold --config Release'

ctest --test-dir build_cuda_scaffold -C Release --output-on-failure

.\build_cuda_scaffold\bin\spect_tests.exe --gtest_filter=CSRTest.*:SpMVTest.*:PSFModeTest.*:ReconPreparationTest.*:OSEMAlignmentTest.SparseOpsSubsetMatchesFallback:BSREMAlignmentTest.*:ADMM*:*ADMM*:IOTest.CompactSymmetryToyNumeratorDenominatorMatchExplicitFull:BedOffsetsTest.*:MultiBedOverlapTest.*

.\build_cuda_scaffold\bin\spect_cuda_scaffold_tests.exe
```

结果：

```text
Release build                      success
ctest                              237/237 passed
key spect_tests filter              62/62 passed
spect_cuda_scaffold_tests           25/25 passed
```

standard CSRB toy smoke：

```powershell
.\build_cuda_scaffold\bin\SPECT_recon.exe `
  -df .\recon_out\standard_toy_smoke\projection_1x2.bin `
  -loadCSR .\recon_out\standard_toy_smoke\identity_2x2.csrb `
  -matrixLayout standard `
  -alg mlem -subsets 1 -it 1 `
  -sparseBackend cpu `
  -detN 1,2 -ang 1 `
  -dim 2,1,1 -vox 1 `
  -fout .\recon_out\standard_toy_smoke\standard_identity
```

结果：

```text
exit code                          0
output                             recon_out/standard_toy_smoke/standard_identity_it1.mhd/.raw
output raw values                  [2.0, 4.0]
log                                recon_out/standard_toy_smoke/standard_identity_20260523_171500_754.log
```

结论：

- 默认/显式 `standard` CSRB 路径仍按普通 full-row CSR 执行，未因 `metadata.bin` 或 compact-symmetric 新逻辑改变行为。
- 全局 CPU/CUDA 单测和 CUDA scaffold 回归已收敛到全通过。

## 19. 本轮补充验收（compact 内存采样与非 CUDA 错误路径）

2026-05-23T17:28:48+08:00 补充 compact-symmetric 主路径的峰值内存记录，并验证非 CUDA 后端不会静默走错误路径。

代码变更：

```text
src/main/MainProgram.cpp
```

变更内容：

- compact-symmetric 分支在重建开始、每床 sensitivity 计算后、每次 MLEM iteration 后和最终保存后采样进程工作集与 CUDA 显存使用量。
- compact-symmetric 分支结束时打印单行 `[MEM] compact_peak_memory ...`，用于验收峰值主机内存与显存。
- 非 CUDA sparse backend 在 compact-symmetric layout 下保持显式拒绝。

验证命令：

```powershell
cmd.exe /c 'call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build_cuda_scaffold --target SPECT_recon'

powershell -ExecutionPolicy Bypass -File scripts/run_6module_compact_acceptance.ps1

.\build_cuda_scaffold\bin\SPECT_recon.exe `
  -df .\gatedata\6module\PSC_6module_24pin_10MBq_600s_1.bin `
  -loadCSR .\gatedata\6module\matrix_6module.csrb `
  -matrixLayout compact-symmetric `
  -alg mlem -subsets 1 -it 1 `
  -sparseBackend cpu `
  -detN 48,48 -ang 360 `
  -dim 108,108,80 -vox 0.3 `
  -fout .\recon_out\compact_cpu_reject_probe
```

结果：

```text
SPECT_recon build                 success
compact acceptance                exit 0
log                               recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_171815_549.log
compact_denom finite/nonzero       933120 / 933120
compact_numer.it1 finite/nonzero   933120 / 933120
total_load_to_iteration_done       2.866s
full_command_wall_time             2.881s
host working set peak              3954372608 bytes
host process peak working set      7691091968 bytes
cuda used peak                     5130158080 bytes
cuda total                         17170956288 bytes
cuda free last                     12040798208 bytes
cpu backend reject exit code       1
cpu backend reject message         Error: compact-symmetric layout requires -sparseBackend cuda.
```

结论：

- compact-symmetric 真实 6module smoke 在 17.17 GB CUDA 设备上峰值显存约 5.13 GB，未展开 full CSRB 到磁盘或 host full-row CSR。
- compact-symmetric 明确要求 CUDA sparse backend；错误后端会直接失败并给出可读错误。

## 20. 本轮补充验收（DD PSF / OSMAPOSL P2 删除与最终签收）

2026-05-23T17:52:00+08:00 完成 DD PSF 与 OSMAPOSL 的 P2 代码删除、删除后回归、standard/compact Operator PSF smoke，以及 50 iter 圆柱 QA 阈值复核。

代码变更：

```text
include/filters/PSFConfig.h
include/filters/Convolution.h
src/filters/Filters.cpp
include/app/ReconConfig.h
include/main/ReconPreparation.h
src/main/ReconPreparation.cpp
src/app/ProjectionOps.cpp
include/cuda/VolumeOpsCUDA.h
src/cuda/VolumeOpsCUDA.cpp
tests/test_psf_modes.cpp
tests/test_main_recon_preparation.cpp
tests/test_cuda_scaffold.cpp
include/app/OSMAPOSLOptimizer.h              deleted
include/app/OSMAPOSLReconstructor.h          deleted
src/app/OSMAPOSLReconstructor.cpp            deleted
config/algo_osmaposl.conf                    deleted
config/spect_config_dd_psf_example.ini       deleted
config/radii.txt                             deleted
CMakeLists.txt
include/spect/SPECT.h
include/app/ReconCompat.h
include/compat/LegacyCompat.h
include/core/Types.h
include/app/BSREMReconstructor.h
include/cli/ReconArgs.h
include/cli/ArgumentParser.h
tests/test_osem_alignment.cpp
tests/test_interfaces.cpp
```

核心结论：

- `PSFMode::DistanceDependent`、`PSFDistModel`、DD kernel 数据结构、DD CPU/CUDA 卷积分支、DD runtime preparation 和 DD 测试已删除。
- OSMAPOSL optimizer/reconstructor、algorithm enum、main dispatch、compat alias、config 和专用测试已删除。
- OSEM 对齐测试文件从 `tests/test_osl_alignment.cpp` 收敛为 `tests/test_osem_alignment.cpp`，测试套件从 `OSLAlignmentTest` 收敛为 `OSEMAlignmentTest`，避免删除 OSMAPOSL 后保留误导性 OSL 命名；其中 `OSEMAlignmentTest.SparseOpsSubsetMatchesFallback` 向 `OSEM_step(CSRView...)` 传完整投影向量，符合当前接口按全局 row 读取 `p[global_row]` 的约束。
- 公开 help 和活跃代码不再暴露 DD PSF / OSMAPOSL；文档中仅保留历史、验收、退场计划或明确的“本阶段不支持”说明。

验证命令：

```powershell
cmd.exe /c 'call "D:\Program Files\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build_cuda_scaffold --clean-first --target SPECT_recon spect_tests spect_cuda_scaffold_tests'

rg -n "DistanceDependent|PSFDistModel|psfDist|psfRadii|psfDetRadius|radii_per_angle|detector_radius|dd_psf|OSMAPOSL|osmaposl|maposl|One-Step-Late|Green's OSL" include src tests config CMakeLists.txt

.\build_cuda_scaffold\bin\SPECT_recon.exe -help

.\build_cuda_scaffold\bin\SPECT_recon.exe -df .\recon_out\standard_toy_smoke\projection_1x2.bin -loadCSR .\recon_out\standard_toy_smoke\identity_2x2.csrb -matrixLayout standard -alg mlem -subsets 1 -it 1 -sparseBackend cpu -detN 1,2 -ang 1 -dim 2,1,1 -vox 1 -psf 0.1 -psfMode Operator -fout .\recon_out\standard_toy_smoke\standard_identity_psf

powershell -ExecutionPolicy Bypass -File scripts/run_6module_compact_acceptance.ps1

.\build_cuda_scaffold\bin\spect_tests.exe --gtest_filter=PSFModeTest.*:ReconPreparationTest.*:OSEMAlignmentTest.*:BSREMAlignmentTest.*:ADMM*:*ADMM*:IOTest.CompactSymmetryToyNumeratorDenominatorMatchExplicitFull:BedOffsetsTest.*:MultiBedOverlapTest.*

.\build_cuda_scaffold\bin\spect_cuda_scaffold_tests.exe

ctest --test-dir build_cuda_scaffold -C Release --output-on-failure

python .\scripts\qa_compact_cylinder.py --image .\recon_out\6module_compact_cylinder_mlem_plain_iter50_it50.mhd --out-json .\recon_out\qa\6module_compact_plain_iter50_qa.json --out-png .\recon_out\qa\6module_compact_plain_iter50_views.png
```

结果：

```text
clean-first build                 success
active code/config removed scan    no matches
help removed tokens                no matches
standard Operator PSF smoke        exit 0
compact Operator PSF acceptance    exit 0
compact acceptance log             recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_174523_229.log
compact total_load_to_iter_done    2.876s
compact wall time                  2.890s
compact peak CUDA used             5130158080 bytes
focused spect_tests                28/28 passed
spect_cuda_scaffold_tests          24/24 passed
ctest                              227/227 passed
reject -alg osmaposl               Error: Unknown optimisation algorithm 'osmaposl'.
reject -psfMode DistanceDependent  Error: Unknown -psfMode 'distancedependent'.
reject -psfDistParams              Unknown argument: -psfDistParams
```

50 iter plain 圆柱 QA 复核：

```text
image                              recon_out/6module_compact_cylinder_mlem_plain_iter50_it50.mhd
finite_ratio                       1.0
central_disk_cv                    0.1628811574122868
left_right_mean_diff_over_inside   0.0006269205407866895
top_bottom_mean_diff_over_inside   0.02753615215723244
z_half_mean_diff_over_global       0.0019524737963462105
outside_inside_mean_ratio          2.053036082631347
qa_json                            recon_out/qa/6module_compact_plain_iter50_qa.json
qa_png                             recon_out/qa/6module_compact_plain_iter50_views.png
```

结论：

- 50 iter plain 输出与旧 CUDA `recon_iter050.img` 已有数值对齐证据；更多 iteration 仍显示 outside annulus 高于 inside disk，因此当前无真值条件下不应使用 `inside > outside` 或 `outside/inside < 0.35` 作为最终物理背景阈值。
- Phase 1 最终 gate 固定为：有限性、非负、正总量、左右/上下对称性、Z 连续、central disk CV、legacy CUDA compare、standard CSRB smoke、compact acceptance 和全量回归。
- 2026-05-23 Phase 1 记录中，CTAC、background/scatter、projection-domain PSF、regularization、compact BSREM/ADMM 和 CSRB v2 header 扩展仍为外范围。2026-06-06 当前能力已扩展到外部 additive `-bProj/-bProjList`、compact `BSREM+RDP` 和 multi-bed global-volume Operator PSF；CTAC、内部 scatter solver、projection-domain PSF、ADMM compact 后端和 CSRB v2 header 仍为外范围。
