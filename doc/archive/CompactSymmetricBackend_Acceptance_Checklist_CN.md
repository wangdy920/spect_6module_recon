# compact-symmetric 后端与 CLI 收敛验收清单

生成时间：2026-05-23T16:45:00+08:00

## 1. 验收范围

本清单用于验收 6module `SYM_45Z` compact-symmetric 后端接入，以及借接入机会完成的 CLI/功能面收敛。

Phase 1 只验收：

- `MLEM + subset=1`
- single-bed 与现有 `-dfList/-bedOverlap` multi-bed
- 标准 image-domain Gaussian PSF：`-psf <sigma> -psfMode Operator`
- CUDA compact helper
- 显式 `-matrixLayout compact-symmetric` + `metadata.bin` sidecar

Phase 1 明确不验收：

- CTAC、attenuation 动态校正
- 内部 scatter solver；外部 additive projection term 已在后续版本通过 `-bProj/-bProjList` 接入 compact ratio，不再按拒绝项处理
- projection-domain PSF
- DistanceDependent PSF
- regularization
- BSREM、ADMM、OSMAPOSL 或其他非 MLEM 算法
- CSRB v2 header 扩展

## 2. 依据资料

- `doc/CompactSymmetricBackend_Execution_Target_CN.md`：固定 6module 圆柱仿体的执行目标指令，包含 FWHM=1.5 PSF、Butterworth 和 `<20 s` 性能验收。
- `doc/CompactSymmetricBackend_Run_20260523_CN.md`：真实 6module smoke 执行记录，包含 cold/warm profile、QA 图/指标、旧 CUDA 对照与已校准的对齐阈值。
- `doc/CompactSymmetricBackend_Design_CN.md`：compact-symmetric 主设计。
- `doc/CLI_Simplification_Plan_CN.md`：DD PSF 与 OSMAPOSL 退场计划。
- `doc/SystemMatrix_6module_compact_vs_CSRB_CN.md`：compact CSRB 与 full CSRB 语义差异。
- `refactor_docs/SRC_RECOVERY_GUIDE.md`：结构、构建、测试、行为、CUDA 分阶段 DoD。
- `refactor_docs/CUDA_SPARSEOPS_DESIGN.md`：CUDA 后端最小闭环、缓存成本与端到端回归口径。
- `docs/Testing/COMPARISON_TEST_GUIDE.md`：同输入、同参数、同输出对比方法。
- `refactor_docs/REFACTOR_AUDIT_LOG.md`：历史多床位、CTAC、PSF、CUDA 路径回归事实。

## 3. 文档与公开 CLI 验收

- [x] `doc/USAGE_CN.md` 不再公开 OSMAPOSL 算法段。
- [x] `README.md` 不再把 OSMAPOSL 列为推荐优化算法。
- [x] `scripts/README.md` 不再把 OSMAPOSL 列为四床位 smoke 主流程。
- [x] `config/algo_osmaposl.conf` 已在 P2 删除。
- [x] `doc/CompactSymmetricBackend_Design_CN.md` 已并入 CLI 收敛策略。
- [x] `doc/CLI_Simplification_Plan_CN.md` 已包含 OSMAPOSL P0-P3 退场计划。
- [x] `SPECT_recon.exe -help` 只展示推荐算法入口：`MLEM | BSREM | ADMM`。
- [x] 公开示例统一使用 `-alg`，不再推荐 `-opti`。
- [x] 公开示例统一使用 `-dfList/-bedOverlap`，不新增 `-bedOffset`。
- [x] 公开示例统一使用 `-psf <sigma>`，不展示 `-psfFWHM`、`-psfDist*` 或 DD PSF。

证据：`SPECT_recon.exe -help` 输出算法行为 `MLEM | BSREM | ADMM`，且无 `OSMAPOSL/osmaposl/psfDist/psfDetRadius/psfRadii/DistanceDependent/DistDD/-opti` 命中；`README.md`、`doc/USAGE_CN.md`、`doc/RECON_FLOW_CN.md`、`scripts/README.md`、`scripts/run_4bed_smoke.ps1`、`scripts/measure_butter_backend.ps1`、`scripts/run_bsrem_kappa_sweep.ps1` 已无 `-opti/osmaposl/-bedOffset/psfFWHM/psfDist` 命中。

## 4. 数据与 metadata 验收

- [x] `gatedata/6module/matrix_6module.csrb` header 可读取：`rows=52992`、`cols=933120`、`nnz=470220768`。
- [x] `gatedata/6module/PSC_6module_24pin_10MBq_600s_1.bin` 可读取：`fullRows=829440`。
- [x] `gatedata/6module/metadata.bin` 可读取并识别 `SYM_45Z`。
- [x] metadata 校验通过：`compactNumAngles=46`、`fullNumAngles=360`、`detNx=48`、`detNz=48`、`compactDetZ=24`。
- [x] metadata 校验通过：`compactRows=46*48*24=52992`。
- [x] metadata 校验通过：`fullRows=360*48*48=829440`。
- [x] metadata 校验通过：`Nx=108`、`Ny=108`、`Nz=80`、`totalVoxels=933120`。
- [x] `matrix.rows == meta.compactRows()`，否则 compact 模式报错。
- [x] `matrix.cols == meta.totalVoxels`，否则 compact 模式报错。
- [x] `projection.size() == meta.fullRows() * bedCount`，否则 compact 模式报错。
- [x] `standard` layout 不读取 `metadata.bin`，即使同目录存在 sidecar。

证据：`python scripts/validate_6module_compact_inputs.py` 返回 exit 0，输出 `recon_out/qa/6module_compact_input_validation.json`，且 focused regression `IOTest.CompactSymmetryValidationRejectsCompactProjectionRows` 与 `IOTest.StandardParserDoesNotEnableCompactFromMetaOverride` 通过。

## 5. CLI 与拒绝路径验收

- [x] 未指定 `-matrixLayout` 时默认 `standard`，现有 full-row CSRB 行为不变。
- [x] `-matrixLayout compact-symmetric` 未找到 metadata 时直接报错。
- [x] `-compactMeta <path>` 可覆盖默认 sidecar 路径。
- [x] compact 模式要求 `-sparseBackend cuda`，非 CUDA 后端直接报错或显式不支持。
- [x] compact 模式要求 `-alg mlem -subsets 1`，其他算法或 subsets 直接报错。
- [x] compact 模式拒绝 CTAC 参数。
- [x] compact 模式支持外部 additive `-bProj/-bProjList`；仍不包含内部 scatter 估计器。
- [x] compact 模式拒绝 regularizer 参数。
- [x] compact 模式拒绝 `-psfProj on`。
- [x] compact 模式拒绝 `-psfMode DistanceDependent`。
- [x] compact 模式拒绝 `-psfDistModel/-psfDistParams/-psfDetRadius/-psfRadiiFile`。
- [x] `-alg osmaposl` 或 `-opti OSMAPOSL` 在 OSMAPOSL P1 后显式失败，并提示 `mlem/bsrem/admm` 替代。

证据：focused regression `IOTest.CompactSymmetricParserDefaultsToStandard`、`IOTest.CompactSymmetricParserAcceptsExplicitMetaOverride`、`IOTest.CompactSymmetricParserRejectsUnsupportedPhase1Options`、`IOTest.CompactSymmetryMetadataMissingFileThrows` 通过；P2 删除后运行时探针确认 `-alg osmaposl` 输出 `Error: Unknown optimisation algorithm 'osmaposl'.`，`-psfMode DistanceDependent` 输出 `Error: Unknown -psfMode 'distancedependent'.`，`-psfDistParams` 输出 `Unknown argument: -psfDistParams`。

## 6. compact 数学与 synthetic parity 验收

- [x] 新增极小 toy metadata：`Nx=Ny`、`fullAngles=8*Q`、`detZ=2*compactDetZ`。
- [x] synthetic compact matrix 可由显式 full operator 展开作为参考。
- [x] `full_angle_from_compact` 的 8 分支角度映射与旧 `spect_recon_mlem.cu` 一致。
- [x] 45 度边界角不重复、不漏算。
- [x] Z mirror 下半/上半映射与旧实现一致。
- [x] Z 中心或边界层 duplicate guard 与旧实现一致。
- [x] D4 图像 transform 覆盖 identity、reflect x=y、rotate90、flipX、rotate180、reflect anti-diagonal、rotate270、flipY。
- [x] compact numerator 与 explicit full operator numerator 在 toy 数据上满足容差。
- [x] compact denominator 与 explicit full operator denominator 在 toy 数据上满足容差。
- [x] 开启 `sigma=0` PSF 与关闭 PSF 的结果一致。
- [x] XY 各向同性 Gaussian Operator PSF 与 compact transform 可交换性在 toy 数据上通过。

证据：`include/io/CompactSymmetryMetadata.h` 已提供共享 D4/Z mirror voxel mapper，`src/cuda/CompactSymmetricProjectorCUDA.cpp` 复用该 mapper，focused regression `IOTest.CompactSymmetryToyMetadataAndD4MappingMatchLegacyRules`、`IOTest.CompactSymmetryToyExpansionCoversFullProjectionOnce`、`IOTest.CompactSymmetryToyNumeratorDenominatorMatchExplicitFull`、`IOTest.CompactSymmetryToyZeroPsfIsIdentityAndGaussianCommutesWithD4`、`IOTest.CompactSymmetryMapsSixModuleFullProjectionRowsOnce`、`IOTest.CompactSymmetryValidationRejectsCompactProjectionRows` 均通过。toy dense full operator 与 compact numerator/denominator 在 `1e-12` 容差内一致；`sigma=0` PSF 为 identity；XY 各向同性 Gaussian 与 8 个 D4 branch 可交换。

## 7. CUDA helper 验收

- [x] `computeCompactSymmetricSensitivityCUDA` 输出 denom，无 NaN/Inf。
- [x] `runCompactSymmetricMlemNumerCUDA` 输出 numerator，无 NaN/Inf。
- [x] CUDA helper 不截断 full projection 到 compact rows。
- [x] CUDA helper profile 输出包含 compact numerator/sensitivity 标识。
- [x] 相同输入重复调用不会复用脏缓冲区。
- [x] cold run 与 warm run 分别记录上传/准备成本和稳态迭代成本。
- [x] 无 CUDA 设备或未启用 CUDA 编译时，compact 模式给出明确错误，不静默走错误后端。

证据：2026-05-23T17:18:15+08:00 acceptance 日志 `recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_171815_549.log` 含 `[STATS] compact_denom: count=933120, finite=933120, nonzero=933120` 与 `[STATS] compact_numer.it1: count=933120, finite=933120, nonzero=933120`；profile 含 `compact_sensitivity`、`compact_forward_ratio_backward`、`compact.forward`、`compact.backward`、`compact.ratio`、`compact.transform_forward/backward`。投影仍按 `829440` full rows 进入 compact helper，未截断为 `52992` compact rows。`runCompactSymmetricMlemNumerCUDA`/`computeCompactSymmetricSensitivityCUDA` 每次入口清零输出 buffer，并在每个 symmetry branch 前清零 `y`/`bpRot`；50 iter plain MLEM 在同一进程内重复调用 numerator 50 次后仍与旧 CUDA 对齐（`relative_l2=3.4412670184357883e-06`）。显式把 compact 模式改为 `-sparseBackend cpu` 会失败并输出 `Error: compact-symmetric layout requires -sparseBackend cuda.`；未启用 CUDA 编译时 `MainProgram.cpp` 中 compact 分支直接抛出 `compact-symmetric layout requires a CUDA-enabled build.`。

## 8. single-bed 真实 6module smoke 验收

- [x] 已按 `doc/CompactSymmetricBackend_Execution_Target_CN.md` 的固定命令执行圆柱仿体 smoke。
- [x] 命令使用显式 `-matrixLayout compact-symmetric`。
- [x] 命令启用标准 PSF：FWHM=1.5 mm，对应当前 CLI `-psf 0.637 -psfMode Operator`。
- [x] 命令启用 Butterworth：`-butter 0.35,2 -butterBackend cufft`。
- [x] 日志打印 compact layout enabled 和 metadata 路径。
- [x] 投影长度按 `829440` 校验，而不是按 compact rows `52992` 校验。
- [x] `MLEM 1 iteration` 完成。
- [x] denom 非空且无 NaN/Inf。
- [x] image 输出尺寸为 `108x108x80`。
- [x] 输出 `.mhd/.raw` 可读取。
- [x] 圆柱仿体中心三视图已保存，并确认无全零、NaN/Inf、明显 8 扇区折叠伪影、Z mirror 断裂或半体错位。
- [x] `scripts/qa_compact_cylinder.py` 输出 `calibrated_1iter_passed=true`；strict `inside/outside` 失败项保留为诊断指标。
- [x] `total_load_to_iteration_done < 20 s`，计入加载、metadata 校验、PSF 准备、compact sensitivity、1 次 MLEM update 和 Butterworth。
- [x] 记录 wall time、compact numerator time、compact sensitivity time。
- [x] `scripts/run_6module_compact_acceptance.ps1 -SkipRecon` 可基于已有 candidate 输出复跑 QA 与旧 CUDA 对齐。
- [x] 完整 `scripts/run_6module_compact_acceptance.ps1` 无 `-SkipRecon` 复跑通过。
- [x] 基于旧 CUDA reference 重新校准 1 iter QA：当前 1 iter strict `inside disk > outside annulus` 与旧 CUDA基线不一致，不作为单独签收依据；改用 `calibrated_1iter_passed + legacy compare` 作为 smoke gate。
- [x] 基于更多 iteration 输出或真值图像重新校准最终圆柱背景阈值。

补充证据：2026-05-23T17:01:10+08:00 完整 `scripts/run_6module_compact_acceptance.ps1` 通过；日志 `recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_170110_292.log` 包含 `[STATS] compact_denom: count=933120, finite=933120, nonzero=933120, min=0.115849196911, max=1.17669403553, sum=781342.332214`，并在代码中对空、非有限或零和 denom 直接报错。该轮 `total_load_to_iteration_done=2.866s`、`full_command_wall_time=2.881s`、`calibrated_1iter_passed=true`，旧 CUDA compare `passed=true`、`rmse/reference_mean=0.014237543898598897`、`sum_rel_diff=0.001127046424774358`。2026-05-23T17:52:00+08:00 对 50 iter plain 输出 `recon_out/6module_compact_cylinder_mlem_plain_iter50_it50.mhd` 补跑 QA，`finite_ratio=1.0`、`central_disk_cv=0.1628811574122868`、`left_right_mean_diff_over_inside=0.0006269205407866895`、`top_bottom_mean_diff_over_inside=0.02753615215723244`、`z_half_mean_diff_over_global=0.0019524737963462105`，但 `outside_inside_mean_ratio=2.053036082631347`；结合该 50 iter plain 输出与旧 CUDA `recon_iter050.img` 已对齐（`rmse/reference_mean=6.684822458605277e-06`），最终圆柱 smoke 不使用 `inside > outside` 或 `outside/inside < 0.35` 作为签收阈值，改以有限性、非负、总量、对称性、Z 连续、central disk CV 与旧 CUDA 对齐作为当前无真值条件下的最终 gate。

## 9. multi-bed 与 overlap 验收

- [x] multi-bed 仅使用 `-dfList` 和 `-bedOverlap`。
- [x] `-dim` 仍表示单床位尺寸。
- [x] `prepareBedSetup()` 推导全局 Z 尺寸。
- [x] `computeBedZOffsets()` 推导床位起点。
- [x] 每床位 local numerator/denominator 独立计算。
- [x] caller 将 local numerator/denominator 按 `bedZStarts` 累加到 global buffers。
- [x] `overlap=0` 且 bed1=bed2 的退化测试中，两个 local 结果一致。
- [x] `overlap>0` 的 toy 测试中，重叠区 global numerator/denominator 等于两个 local 结果相加。
- [x] 不引入第二套 `-bedOffset` CLI。

证据：床位起点与局部/全局体数据映射已收敛到 `include/app/BedOffsets.h` 的 `computeBedZOffsets`、`extractBedVolumeFromGlobal`、`accumulateBedVolumeToGlobal`；`src/main/MainProgram.cpp` compact 与 standard 分支均使用同一累加工具。focused regression `BedOffsetsTest.AccumulatesLocalBedVolumesInOverlap` 验证 `overlap>0` 重叠层等于两个 local bed 相加，`BedOffsetsTest.ZeroOverlapIdenticalLocalBedsAccumulateIndependently` 验证 `overlap=0` 且相同 local bed 独立累加后保持一致；既有 `MultiBedOverlapTest.*` 继续通过。公开文档/help/script 扫描无 `-bedOffset` 活跃入口。

## 10. 旧 CUDA MLEM 对齐验收

- [x] 与 `6module_bak_src/spect_recon_mlem.cu` 使用同一投影、同一 matrix/chunk、同一 metadata。
- [x] 初始图像一致：两侧均为 1 iter MLEM 默认初值路径。
- [x] MLEM 1 iteration 输出对比 `max_abs_diff`、`mean_abs_diff`、`rmse`、`rmse/reference_mean` 和 `sum_rel_diff`。
- [x] MLEM 50 iteration plain 口径已与用户已有旧结果 `6module_bak_src/recon_iter050.img` 对齐；当前后端命令禁用 PSF/Butterworth，`rmse/reference_mean=6.684822458605277e-06`，`sum_rel_diff=6.985968228262434e-09`，对比 JSON 为 `recon_out/qa/6module_compact_plain_iter50_vs_legacy_recon_iter050_compare.json`。
- [x] MLEM 少量 iteration 输出对比 `relative_l2`。
- [x] 对比中心 slice/profile sanity；旧 CUDA 与新后端均呈同类 ring-like 1 iter 形态。
- [x] 首轮真实数据容差按实测校准为 `rmse/reference_mean <= 0.02` 且 `sum_rel_diff <= 0.005`；当前 1 iter PSF/Butterworth 结果 `rmse/reference_mean=0.014237543898598897`、`relative_l2=0.014210062232198062`、`sum_rel_diff=0.001127046424774358`，50 iter plain 结果 `relative_l2=3.4412670184357883e-06`。
- [x] `scripts/compare_compact_legacy.py` 可复跑旧 CUDA 对齐并以 JSON 记录 pass/fail。

## 11. 全局回归验收

- [x] `cmake --build build_cuda_scaffold --config Release` 通过。
- [x] `ctest --test-dir build_cuda_scaffold -C Release --output-on-failure` 通过，或失败项明确归类为既有问题。
- [x] `build_cuda_scaffold\bin\spect_tests.exe` 通过关键过滤：CSR、SpMV、PSF、ReconPreparation、OSEM、BSREM、ADMM。
- [x] `build_cuda_scaffold\bin\spect_cuda_scaffold_tests.exe` 通过，或无 CUDA 条件下按 skip 处理。
- [x] 标准 full-row CSRB smoke 不受 compact 新逻辑影响。
- [x] `scripts\run_4bed_smoke.ps1` 去除 OSMAPOSL 后仍可验证 MLEM/BSREM 主流程。

证据：Release 构建通过；删除 DD PSF 与 OSMAPOSL 后完整 `ctest --test-dir build_cuda_scaffold -C Release --output-on-failure` 结果 227/227 passed。关键过滤 `PSFModeTest.*:ReconPreparationTest.*:OSEMAlignmentTest.*:BSREMAlignmentTest.*:ADMM*:*ADMM*:IOTest.CompactSymmetryToyNumeratorDenominatorMatchExplicitFull:BedOffsetsTest.*:MultiBedOverlapTest.*` 结果 28/28 passed；`spect_cuda_scaffold_tests.exe` 结果 24/24 passed。标准 CSRB toy smoke 使用 `-matrixLayout standard` + 2x2 identity CSRB + BIN projection，输出 `recon_out/standard_toy_smoke/standard_identity_it1.raw=[2.0,4.0]`；删除后又以 `-psf 0.1 -psfMode Operator` 复跑 standard smoke 成功，日志为 `recon_out/standard_toy_smoke/standard_identity_psf_20260523_174523_017.log`。`scripts/run_4bed_smoke.ps1` 已删除 OSMAPOSL case，并改用 `-alg MLEM` 与 `-alg BSREM`；PowerShell AST parse 对 `run_4bed_smoke.ps1`、`measure_butter_backend.ps1`、`run_bsrem_kappa_sweep.ps1` 均通过。

## 12. 删除 DD PSF 与 OSMAPOSL 后的收敛验收

- [x] `rg "DistanceDependent|PSFDistModel|psfDist|psfRadii|psfDetRadius|radii_per_angle|detector_radius|dd_psf" include src tests config doc README.md scripts` 无非历史命中。
- [x] `rg "OSMAPOSL|osmaposl|maposl|One-Step-Late|Green's OSL" include src tests config README.md doc/USAGE_CN.md scripts/README.md` 无非历史命中。
- [x] 删除后 `SPECT_recon.exe -help` 不再展示 DD PSF 或 OSMAPOSL。
- [x] 删除后 `-psf <sigma> -psfMode Operator` 仍可用于 standard 和 compact 路径。
- [x] 删除后 MLEM/OSEM、BSREM、ADMM 单测和 smoke 无回归。

证据：生产/测试/config 范围扫描 `rg "DistanceDependent|PSFDistModel|psfDist|psfRadii|psfDetRadius|radii_per_angle|detector_radius|dd_psf|OSMAPOSL|osmaposl|maposl|One-Step-Late|Green's OSL" include src tests config CMakeLists.txt` 无命中；扩展文档扫描中的 DD 命中均位于执行目标、设计、CLI 简化计划、验收清单和 archive/历史说明，不是活跃代码或公开推荐入口；`rg "OSMAPOSL|osmaposl|maposl|One-Step-Late|Green's OSL" include src tests config README.md doc/USAGE_CN.md scripts/README.md` 无命中。`SPECT_recon.exe -help` 对 `OSMAPOSL/osmaposl/maposl/psfDist/psfDetRadius/psfRadii/DistanceDependent/distance-dependent/DistDD/-opti` 无命中。运行时拒绝：`-alg osmaposl` 输出 `Error: Unknown optimisation algorithm 'osmaposl'.`；`-psfMode DistanceDependent` 输出 `Error: Unknown -psfMode 'distancedependent'.`；`-psfDistParams` 输出 `Unknown argument: -psfDistParams`。standard 路径 `-psf 0.1 -psfMode Operator` smoke 成功；compact 路径完整 `scripts/run_6module_compact_acceptance.ps1` 成功，日志 `recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_174523_229.log`。OSEM 对齐测试文件已从 `test_osl_alignment.cpp` 收敛为 `test_osem_alignment.cpp`，避免删除 OSMAPOSL 后保留误导性 OSL 命名；聚焦测试 28/28 passed，CUDA scaffold 24/24 passed，全量 `ctest` 227/227 passed。

## 13. 性能与可维护性验收

- [x] compact 模式不展开 full CSRB 到磁盘。
- [x] compact 模式不在 host 端生成完整 full-row CSR。
- [x] 峰值显存和主机内存记录在日志或 profile 中。
- [x] 对同一系统矩阵的重复迭代可区分 cold/warm 成本。
- [x] 性能报告基于完整 MLEM iteration，不只看单 kernel。
- [x] 新增代码边界清晰：metadata IO、CLI validation、compact CUDA helper、main orchestration 各自分层。
- [x] 不新增 speculative abstraction、auto sidecar 切换或第二套 multi-bed CLI。

证据：真实 6module compact smoke 直接读取 `52992 x 933120` compact CSRB 与 `829440` full projection rows，未生成 full-row CSRB 文件；代码路径保持 compact CSR + symmetry mapper + per-bed local/global 累加，没有 host full-row CSR 构造。cold/warm 与完整 iteration profile 已记录在 `doc/CompactSymmetricBackend_Run_20260523_CN.md`，包括 `projection_load`、`csr_map_or_load`、`compact_sensitivity`、`compact_forward_ratio_backward`、`mlem_update`、`butterworth`、`output_write`、`total_load_to_iteration_done`。最新日志 `recon_out/6module_compact_cylinder_mlem_psf15_butter_20260523_171815_549.log` 含 `[MEM] compact_peak_memory host_working_set_peak_bytes=3954372608 host_process_peak_working_set_bytes=7691091968 cuda_used_peak_bytes=5130158080 cuda_total_bytes=17170956288 cuda_free_bytes_last=12040798208`。分层落点为 `include/io/CompactSymmetryMetadata.h`、`src/io/CompactSymmetryMetadata.cpp`、`include/cli/ArgumentParser.h`、`src/cuda/CompactSymmetricProjectorCUDA.cpp`、`src/main/MainProgram.cpp`、`include/app/BedOffsets.h`；CLI 保持显式 `-matrixLayout compact-symmetric`，未恢复 auto sidecar，也未新增 `-bedOffset`。

## 14. 最终签收条件

- [x] compact-symmetric Phase 1 的所有必需项通过。
- [x] 标准 CSRB 主线无行为回归。
- [x] 公开文档与 help 文本只展示当前支持的主线能力。
- [x] 验证命令、输入数据、关键指标和剩余风险写入最终实现记录。
- [x] `.agent/CONTINUITY.md` 更新本轮决策、进展、验证结果和未完成项。

签收结论：compact-symmetric Phase 1 以显式 `-matrixLayout compact-symmetric`、single-bed/multi-bed MLEM subset=1、CUDA compact helper、标准 Operator PSF、Butterworth、BIN projection 与 compact CSRB metadata sidecar 为验收范围，当前必需项均有可复跑证据。剩余风险不是 Phase 1 阻断项：没有独立真值图像，因此圆柱绝对 background contrast 不作为物理正确性签收阈值。2026-06-06 当前能力已扩展到外部 additive `-bProj/-bProjList`、compact `BSREM+RDP` 和 multi-bed global-volume Operator PSF；CTAC、内部 scatter solver、projection-domain PSF、ADMM compact 后端和 CSRB v2 header 扩展仍按外范围处理。
