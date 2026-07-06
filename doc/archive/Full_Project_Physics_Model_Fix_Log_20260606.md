# SPECT-recon 物理模型审查修复记录

日期：2026-06-06

本记录承接 `doc/Full_Project_Physics_Model_Review_20260606.md`，按影响优先级逐项记录已完成的代码/文档修复和回归结果。所有 share 回归均使用 `test_data/share/PSC_6module_24pin_10MBq_600s_test_1.bin`、`test_data/6module/260601`，基础 `MLEM`、`subsets=1`、`50` iter、无 PSF、无后滤波、无正则、无 CTAC。

## 节点 1 - compact numerator 全零保护与 PSF 分段统计

- 修改文件：`src/main/MainProgram.cpp`
- 改动：
  - 在 compact-symmetric CUDA path 中记录 `compact_projection` stats。
  - 当 projection 输入为正且 `compact_numer` 出现 `nonzero==0` 或 `sum<=0` 时直接抛错，避免全零图像继续迭代。
  - Operator PSF 激活时记录 `compact_local_image`、`compact_psf_forward`、`compact_local_numer_pre_psf`、`compact_local_numer_post_psf`，用于定位信号在哪一段被清零。
- 验证：
  - 构建 `SPECT_recon` / `spect_tests` 通过。
  - `spect_tests.exe --gtest_filter=PSFModeTest.*:*CompactSymmetry*:*CompactChunk*`：12/12 passed。
  - share 50 iter 输出：`recon_out/share_mlem50_node1/recon_it50.mhd`。
  - 对 `recon_iter050.img`：`rmse_over_reference_mean=1.0632199304259498e-05`，`sum_rel_diff=1.5327786257864078e-09`，passed。
  - 对 `recon_iter050.raw`：同指标，passed。

## 节点 2 - `-psf` 默认启用 Operator 模式

- 修改文件：`include/cli/ArgumentParser.h`、`tests/test_interfaces.cpp`
- 改动：
  - `-psf <sigma>` 在没有显式 `-psfMode` 时默认设置 `PSFMode::Operator`。
  - 显式 `-psfMode None` 仍可覆盖默认 Operator。
  - 新增 parser 回归：`IOTest.PsfFlagDefaultsToOperatorMode`、`IOTest.ExplicitPsfModeNoneOverridesPsfDefaultOperator`。
- 验证：
  - 构建 `SPECT_recon` / `spect_tests` 通过。
  - `spect_tests.exe --gtest_filter=IOTest.PsfFlagDefaultsToOperatorMode:IOTest.ExplicitPsfModeNoneOverridesPsfDefaultOperator:IOTest.CompactSymmetricParser*:PSFModeTest.*:*CompactSymmetry*:*CompactChunk*`：22/22 passed。
  - share 50 iter 输出：`recon_out/share_mlem50_node2/recon_it50.mhd`。
  - 对 `recon_iter050.img`：`rmse_over_reference_mean=1.0632271338903227e-05`，`sum_rel_diff=1.5312725332280123e-09`，passed。
  - 对 `recon_iter050.raw`：同指标，passed。

## 节点 3 - compact denom cache manifest mismatch 保护

- 修改文件：`src/main/MainProgram.cpp`、`include/cli/ArgumentParser.h`
- 改动：
  - manifest 不匹配时报告首个差异行。
  - manifest mismatch 不再作为普通 cache miss 继续重算和覆盖，而是直接报错，要求 no-PSF、PSF/FWHM、Joseph、bed geometry 使用不同 `-compactDenomCache` 路径。
  - CLI help 中补充 cache 分文件要求。
- 验证：
  - 构建 `SPECT_recon` / `spect_tests` 通过。
  - 相关 22 个 parser/PSF/compact tests passed。
  - mismatch smoke：用 no-PSF cache 跑 `-psf 0.637 -it 1:1`，按预期快速失败：
    - `expected 'psf_mode=Operator'`
    - `actual 'psf_mode=None'`
  - share 50 iter 输出：`recon_out/share_mlem50_node3/recon_it50.mhd`。
  - 对 `recon_iter050.img`：`rmse_over_reference_mean=1.0632403483507045e-05`，`sum_rel_diff=1.5091705358060673e-09`，passed。
  - 对 `recon_iter050.raw`：同指标，passed。

## 节点 4 - CTAC approximate 警告和帮助文案降级

- 修改文件：`src/main/MainProgram.cpp`、`include/cli/ArgumentParser.h`
- 改动：
  - `-ctac` 运行时 warning：当前实现是 `exp(-mu)` per-voxel attenuation factor replicated per angle，不是 quantitative line-integral CTAC。
  - post Chang's AC 输出 warning：当前为基于 attenuation map 的 approximate per-voxel correction。
  - CLI help 中将 `-ctac` 和 `-attMap` 文案降级为 approximate attenuation correction。
- 验证：
  - 构建 `SPECT_recon` / `spect_tests` 通过。
  - `spect_tests.exe --gtest_filter=CTACOpsTest.*:IOTest.PsfFlagDefaultsToOperatorMode:IOTest.ExplicitPsfModeNoneOverridesPsfDefaultOperator:IOTest.CompactSymmetricParser*:PSFModeTest.*:*CompactSymmetry*:*CompactChunk*`：26/26 passed。
  - share 50 iter 输出：`recon_out/share_mlem50_node4/recon_it50.mhd`。
  - 对 `recon_iter050.img`：`rmse_over_reference_mean=1.0632198500919838e-05`，`sum_rel_diff=8.330356101270431e-10`，passed。
  - 对 `recon_iter050.raw`：同指标，passed。

## 节点 5 - compact Operator PSF backward 全零保护

- 修改文件：`src/app/ProjectionOps.cpp`、`tests/test_cuda_scaffold.cpp`
- 复现：
  - 真实 share Operator PSF 1-iter smoke 使用独立 cache `recon_out/compact_denom_cache/share_260601_psf_operator_sigma0637_node5_global_denom.mhd`。
  - 修复前 `compact_local_numer_pre_psf.it1.bed0` 非零：`nonzero=933120`，`sum=3676615.43339`。
  - 修复前同一 local numerator 经 `apply_psf_backward` 后全零：`compact_local_numer_post_psf.it1.bed0 nonzero=0, sum=0`，随后 fail-fast。
- 改动：
  - `applyPsfInplace()` 在 CUDA PSF active 时保留原始输入副本。
  - 若 CUDA separable convolution 对非零输入产生有限全零输出，则恢复原始输入并走 CPU `conv3d_separable_inplace()` 重算。
  - 新增 `CudaScaffoldTest.VolumePsfMatchesCpuOnCompactShareGeometry`，覆盖 `108x108x80` 真实尺寸、`-psf 0.637` 对应 kernel、CPU/CUDA double/CUDA float 对齐。
- 验证：
  - 构建 `SPECT_recon` / `spect_tests` / `spect_cuda_scaffold_tests` 通过。
  - `spect_tests.exe --gtest_filter=PSFModeTest.*:IOTest.PsfFlagDefaultsToOperatorMode:IOTest.ExplicitPsfModeNoneOverridesPsfDefaultOperator:*CompactSymmetry*:*CompactChunk*`：14/14 passed。
  - `spect_cuda_scaffold_tests.exe --gtest_filter=CudaScaffoldTest.VolumePsfMatchesCpuOnCompactShareGeometry:CudaScaffoldTest.FloatPsfMatchesDoublePsfWhenCudaAvailable:CudaScaffoldTest.VolumePsfMatchesCpuWhenCudaAvailable`：3/3 passed。
  - 修复后真实 share Operator PSF 1-iter 输出：`recon_out/share_psf_operator_it1_node5_fix/recon_it1.mhd`。
  - 修复后 `compact_local_numer_post_psf.it1.bed0` 非零：`nonzero=933120`，`sum=3681750.52215`；`compact_numer.it1` 同样非零并完成重建。
  - 基础 no-PSF share 50 iter 输出：`recon_out/share_mlem50_node5/recon_it50.mhd`。
  - 对 `recon_iter050.img`：`rmse_over_reference_mean=1.0632185678549273e-05`，`sum_rel_diff=1.542716665202022e-09`，passed。
  - 对 `recon_iter050.raw`：同指标，passed。
- 注意：
  - 本节点解决的是“CUDA PSF 对非零输入异常输出全零”时的安全回退，不把当前运行时间作为性能签收。node5 基础 MLEM 50 iter 用于数值回归，运行时机器负载较高，`total_load_to_iteration_done=246.019s`。

## 节点 6 - 多床位 compact Operator PSF 改为全局体语义

- 修改文件：`src/main/MainProgram.cpp`
- 改动：
  - 对 `compact-symmetric + PSFMode::Operator + bedCount>1`，不再对每个 local bed 独立做 PSF。
  - denominator 先把每床 `A_b^T 1` 累加到 global volume，再执行一次全局 `H^T`，即 `H^T sum_b A_b^T 1`。
  - 每轮 forward 先对 global image 执行 `H f`，再从全局模糊体裁出各 bed active volume 投影。
  - numerator 先把每床 `A_b^T ratio_b` 累加到 global volume，再执行一次全局 `H^T`。
  - compact denom cache manifest 对多床位 Operator PSF 增加 `psf_multibed_scope=global_volume`，避免旧局部 PSF cache 被误用。
  - 单床和 no-PSF 路径保持原逻辑，不增加 manifest key。
- 验证：
  - 构建 `SPECT_recon` / `spect_tests` / `spect_cuda_scaffold_tests` 通过。
  - `spect_tests.exe --gtest_filter=PSFModeTest.*:IOTest.PsfFlagDefaultsToOperatorMode:IOTest.ExplicitPsfModeNoneOverridesPsfDefaultOperator:*CompactSymmetry*:*CompactChunk*:BedOffsetsTest.*:MultiBedOverlapTest.*`：21/21 passed。
  - `spect_cuda_scaffold_tests.exe --gtest_filter=CudaScaffoldTest.VolumePsfMatchesCpuOnCompactShareGeometry:CudaScaffoldTest.FloatPsfMatchesDoublePsfWhenCudaAvailable:CudaScaffoldTest.VolumePsfMatchesCpuWhenCudaAvailable`：3/3 passed。
  - 三床位 logo Operator PSF 1-iter smoke 输出：`recon_out/logo_psf_operator_it1_node6_globalpsf/recon_it1.mhd`。
    - `compact_denom_pre_psf_global nonzero=2799360, sum=206961760.305`
    - `compact_denom_post_psf_global nonzero=2799360, sum=207143276.349`
    - `compact_numer_pre_psf_global.it1 nonzero=2799360, sum=3752008.48354`
    - `compact_numer_post_psf_global.it1 nonzero=2799360, sum=3754522.16645`
  - 基础 no-PSF share 50 iter 输出：`recon_out/share_mlem50_node6/recon_it50.mhd`。
  - 对 `recon_iter050.img`：`rmse_over_reference_mean=1.063214884680605e-05`，`sum_rel_diff=1.7317876746753504e-09`，passed。
  - 对 `recon_iter050.raw`：同指标，passed。
- 注意：
  - 本节点修复的是 multi-bed Operator PSF 的 global-volume 物理语义；仍未实现真实 CTAC line integral。
  - 本轮 `share_mlem50_node6` 的 `total_load_to_iteration_done=120.650s` 仅作为当前机器记录，不作为性能签收。

## 节点 7 - compact additive background 能力文档修正

- 修改文件：
  - `doc/CompactSymmetricBackend_Acceptance_Checklist_CN.md`
  - `doc/CompactSymmetricBackend_Design_CN.md`
  - `doc/CompactSymmetricBackend_Run_20260523_CN.md`
- 改动：
  - 将旧文档中的“compact 拒绝 background/scatter/bProj”改为当前能力边界：支持外部 additive `-bProj/-bProjList`，不支持内部 scatter solver。
  - 同步写明当前 compact 也已扩展到 `BSREM+RDP` 和 multi-bed global-volume Operator PSF。
  - 对 2026-05-23 历史执行记录保留原上下文，但明确其不再代表 2026-06-06 当前能力矩阵。
- 验证：
  - 文档 grep 自检：不再存在活跃的 `compact 拒绝 background/bProj` 表述。
  - 本节点只修改能力说明，不改算法代码；无需重新跑 share 数值回归。

## 节点 8 - multi-bed Operator PSF overlap/halo phantom 自动测试

- 修改文件：`tests/test_psf_modes.cpp`
- 改动：
  - 新增 `PSFModeTest.GlobalVolumePsfSpansBedStartWhereLocalBedPsfMissesHalo`。
  - 构造两床位 `baseNz=5`、overlap `2` 的小 phantom，床位起点为 `{0,3}`。
  - 将 impulse 放在第二床起点前一层；全局体 PSF 会把信号传播到第二床第一层，而先裁第二床再局部卷积会完全漏掉该 halo。
- 验证：
  - 构建 `spect_tests` / `SPECT_recon` 通过。
  - `spect_tests.exe --gtest_filter=PSFModeTest.*:BedOffsetsTest.*:MultiBedOverlapTest.*:IOTest.PsfFlagDefaultsToOperatorMode:IOTest.ExplicitPsfModeNoneOverridesPsfDefaultOperator:*CompactSymmetry*:*CompactChunk*`：22/22 passed。
- 注意：
  - 本节点锁定 global-volume PSF 的最小物理语义，不依赖真实大矩阵。
  - 若后续有真实 overlap 多床位数据，仍建议补真实矩阵 smoke。

## 节点 9 - compact `-bProjList` 多床位真实矩阵 smoke

- 修改文件：无算法代码改动；生成临时零背景投影文件：
  - `recon_out/bproj_zero_logo_b3_node8/zero_bed1.bin`
  - `recon_out/bproj_zero_logo_b3_node8/zero_bed2.bin`
  - `recon_out/bproj_zero_logo_b3_node8/zero_bed3.bin`
- 验证设计：
  - 使用 `test_data/logo/Bed1.bin,Bed2.bin,Bed3.bin` 和 `test_data/6module/260601`。
  - no-background control 输出：`recon_out/logo_b3_nobproj_it1_node8/recon_it1.mhd`。
  - zero-background `-bProjList` 输出：`recon_out/logo_b3_zerobproj_it1_node8_lowmem/recon_it1.mhd`。
  - 零背景应与 no-background control 数值一致。
- 运行细节：
  - 当前默认高内存口径在 GPU free 约 `5.9 GiB` 时会于 `cusparseCreate` 失败；无背景 control 也复现该失败，因此该问题是当前显存余量/cache warmup 约束，不是 `bProjList` 公式错误。
  - 设置 `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES=1073741824` 后，zero-background `-bProjList` smoke 成功，`hostCacheBudgetBytes=1073741824`，输出 finite/nonzero。
- 对比结果：
  - 对 no-background control：`relative_l2=2.590352429193033e-08`。
  - `rmse_over_reference_mean=3.748040427896362e-08`，阈值 `1e-6`，passed。
  - `sum_rel_diff=4.4695896646029283e-10`，阈值 `1e-8`，passed。

## 节点 10 - 实施前 GPT 5.5 Pro / 文献深研分流

- 新增文件：`doc/gpt55_consultation_20260607/10_pending_implementation_triage_and_literature_research_prompt.md`
- 决策：
  - CTAC line-integral、RDP beta/curvature 标度、Derenzo objective/edge-ring penalty 属于高风险物理建模问题，应先咨询 GPT 5.5 Pro 并做文献深研，不应直接开始代码实现。
  - multi-bed PSF overlap phantom、`-bProjList` smoke、cache/显存运行策略属于本地工程验证，可继续直接实现和回归。

## 尚未完成的高优先级节点

- CTAC 真实 line-integral 实现需要新增射线几何和 phantom 验收；当前仅完成误用防护和文案降级，实施前应先做 GPT 5.5 Pro / 文献深研。
- 多床位 Operator PSF 已完成小 phantom halo 自动测试；若后续取得真实 overlap 多床位投影，仍建议补真实矩阵 smoke。
- compact additive background 已完成零背景 `-bProjList` multi-bed 真实矩阵 parity smoke；后续可补非零 background perturbation smoke。
