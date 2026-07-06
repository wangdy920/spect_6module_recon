# 6module compact-symmetric 后端执行目标指令

生成时间：2026-05-23T17:05:00+08:00

## 1. 目的

实现并验证 6module `SYM_45Z` compact-symmetric 后端的第一条生产候选链路：使用现有 compact CSRB 与 full projection BIN，在单床位圆柱仿体数据上完成 `MLEM + subset=1 + CUDA compact projector + 标准 image-domain Gaussian PSF + Butterworth` 重建。

本目标不是重新设计全项目 CLI，也不是一次性接入所有算法。验收重点只有三件事：

1. compact CSRB 与 full projection 通过 runtime symmetry 映射正确闭环，不展开 full CSRB。
2. 结果图像符合圆柱仿体的基本形态与数值 sanity。
3. 单床位从加载数据到完成 1 次 MLEM 迭代、标准 PSF 和 Butterworth 滤波的总耗时小于 20 s。

## 2. 执行前必须查看的 Markdown 文档

按顺序阅读：

1. `doc/CompactSymmetricBackend_Design_CN.md`
   - 主设计文档。
   - 重点看：数据事实、`-matrixLayout compact-symmetric`、metadata sidecar、SYM_45Z 映射、多床位约束、Phase 1 支持/排除范围。
2. `doc/CompactSymmetricBackend_Acceptance_Checklist_CN.md`
   - 总体验收清单。
   - 本执行目标是其中 single-bed 真实 6module smoke 与性能验收的具体化。
3. `doc/SystemMatrix_6module_compact_vs_CSRB_CN.md`
   - 理解 compact CSRB 与 full-row CSRB 的语义差异。
   - 关键结论：`matrix_6module.csrb` 不能按标准 full-row CSRB 直接重建 full projection。
4. `doc/CLI_Simplification_Plan_CN.md`
   - 确认本阶段公开 CLI 只保留标准 Gaussian PSF，不接入 DistanceDependent PSF 或 OSMAPOSL。
5. `doc/USAGE_CN.md`
   - 确认当前公开参数口径：`-butter <cutoff,order>`、`-butterBackend <auto|fftw|cufft>`、`-psf <sigma>`、`-psfMode Operator|PostSmooth|None`。
6. `docs/Testing/COMPARISON_TEST_GUIDE.md`
   - 复用同输入、同参数、同输出对比思路，用于后续与旧 CUDA MLEM 或 explicit-full toy reference 对齐。
7. `refactor_docs/CUDA_SPARSEOPS_DESIGN.md`
   - 复用 CUDA 后端验收口径：先最小闭环，再记录 cold/warm 成本，再做端到端回归。

可选参考：

- `refactor_docs/SRC_RECOVERY_GUIDE.md`：结构、构建、测试、行为、CUDA DoD 模板。
- `refactor_docs/REFACTOR_AUDIT_LOG.md`：历史 CUDA、PSF、multi-bed、full-subset 回归事实。

## 3. 固定输入数据

必须使用以下单床位输入：

```text
projection:
  gatedata/6module/PSC_6module_24pin_10MBq_600s_1.bin

compact CSRB:
  gatedata/6module/matrix_6module.csrb

metadata sidecar:
  gatedata/6module/metadata.bin
```

必须校验的数据事实：

```text
matrix rows = 52992
matrix cols = 933120
matrix nnz  = 470220768

projection full rows = 829440
full angles = 360
detector = 48 x 48

compact angles = 46
compact detector z = 24
compact rays per angle = 1152

image = 108 x 108 x 80
```

若上述任一校验失败，停止执行，不允许 silent fallback 到 standard CSR。

## 4. 必须执行的目标命令

当前公开文档约定 `-psf` 输入为 sigma(mm)，而用户目标给的是 FWHM=1.5 mm。执行时必须换算：

```text
sigma = FWHM / 2.354820045 = 1.5 / 2.354820045 = 0.637 mm
```

因此命令中使用 `-psf 0.637 -psfMode Operator`，不要写 `-psf 1.5`，除非后续 parser 和文档已明确改成 FWHM 语义。

Butterworth 采用当前项目默认公开示例口径：`-butter 0.35,2`，后端优先走 cuFFT：`-butterBackend cufft`。

目标命令：

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

若当前二进制只支持 `-out` 而不支持 `-fout`，实现时应统一 CLI 文档和 help 文本后再替换；不要在文档之间保留两套输出参数示例。

## 5. 实现约束

- 必须显式指定 `-matrixLayout compact-symmetric`。
- 默认 metadata 路径为 `dirname(loadCSR)/metadata.bin`。
- 不允许把 `matrix_6module.csrb` 当 standard full-row CSRB 使用。
- 不允许生成 full-row CSRB 到磁盘作为 Phase 1 主路径。
- 不允许 host 端常驻完整 full-row CSR。
- 不允许接入 DistanceDependent PSF、projection-domain PSF、CTAC、background/scatter、regularization、OSMAPOSL、BSREM、ADMM。
- compact Phase 1 必须使用 CUDA；无 CUDA 时应明确失败。
- 所有不支持参数组合必须在 CLI validation 阶段报错。

## 6. 计时验收标准

必须记录并输出以下 timing 字段，至少能从日志中人工读取：

```text
projection_load
csr_map_or_load
metadata_load_and_validate
psf_prepare
compact_sensitivity
compact_forward_ratio_backward
mlem_update
butterworth
total_load_to_iteration_done
full_command_wall_time
```

硬性验收：

```text
total_load_to_iteration_done < 20 s
```

其中 `total_load_to_iteration_done` 包括：投影加载、CSRB mmap/load、metadata 加载校验、PSF 准备、compact sensitivity、1 次 MLEM forward/ratio/backward/update、Butterworth 滤波。它不包括构建时间；若最终 `.mhd/.raw` 写盘非常慢，应单独记录 output_write 并说明是否计入 full command wall time。

辅助验收：

- `full_command_wall_time` 应同步记录，建议也低于 20 s；若超出但 `total_load_to_iteration_done` 合格，必须说明超时来自输出写盘或日志 IO。
- cold run 与第二次 warm run 都要记录；签收以 cold run 为准，warm run 用于确认缓存收益。
- 日志必须区分 compact sensitivity 与 compact numerator/ratio/backward 时间，不能只给单个总耗时。

## 7. 图像合理性验收标准

该输入为圆柱仿体。1 次 MLEM smoke 的图像质量不要求达到最终临床或论文级结果，但必须通过以下 sanity gate。

### 7.1 必须通过的数值 sanity

- 输出体数据尺寸为 `108 x 108 x 80`。
- 所有体素为有限值：无 NaN、无 Inf。
- 图像非负；若存在负值，最小值必须在数值噪声范围内并说明来源。
- 全图总和大于 0。
- 中心体区域与背景区域的均值关系应结合 1 iteration 旧 CUDA baseline 校准：若旧 CUDA 同输入同参数也呈 ring-like 边界增强，则 `inside_mean > outside_mean` 不作为单独一票否决项，而应由旧 CUDA 数值对齐、左右/上下对称、Z 连续、中心 disk CV 和三视图人工检查共同判定。
- 不能出现全零图、单点爆亮图、整片 NaN/Inf 或明显内存未初始化条纹。

### 7.2 必须通过的圆柱形态 sanity

- 中心 XY 横断面应呈近似居中的填充圆盘或圆环状圆柱截面。
- XZ/YZ 中心切面应呈沿 Z 连续的柱状结构，不应只出现在半个 Z 区间。
- 不应出现明显 8 扇区角度折叠伪影、45 度边界重复亮线、Z mirror 中心层断裂或上下半空间重复错位。
- 圆柱主体外背景应显著低于主体内平均值。
- Butterworth + FWHM=1.5 mm PSF 后图像应比无滤波/无 PSF smoke 更平滑，但不得把圆柱主体抹成全场均匀背景。

### 7.3 建议量化指标

若已有 QA 脚本，可输出以下指标；若没有，首次实现时至少保存中心三视图和横向/轴向 profile。

```text
finite_ratio = 1.0
sum(image) > 0
inside_mean > outside_mean
outside_mean / inside_mean < 0.35        # 初始 smoke 阈值，可由首次基线校准
left_right_mean_diff / inside_mean < 0.20
top_bottom_mean_diff / inside_mean < 0.20
central_disk_cv < 0.35                   # 1 iter + compact smoke 的宽松阈值
```

这些阈值用于发现明显映射错误，不用于最终图像质量评价。首次真实运行后，应把实际指标写入验证记录，并按实测稳定性收紧阈值。

2026-05-23 校准记录：真实 6module 1 iteration 输出与旧 `6module_bak_src/spect_recon_mlem.cu` baseline 均出现 ring-like 边界增强，旧 CUDA 的 `outside/inside` 同样大于 1。因此 1 iteration smoke 的签收 gate 调整为：

```text
finite_ratio = 1.0
sum(image) > 0
left_right_mean_diff / inside_mean < 0.20
top_bottom_mean_diff / inside_mean < 0.20
central_disk_cv < 0.35
z_half_mean_diff / global_mean < 0.20
legacy_cuda_rmse / legacy_mean < 0.02
legacy_cuda_sum_rel_diff < 0.005
```

`inside_mean`、`outside_mean` 和 `outside/inside` 仍必须记录，但在 1 iteration smoke 中只作为诊断指标；后续若使用更多 iteration 或真值图像，应重新校准圆柱背景阈值。

## 8. 最小交付物

完成该目标时必须留下：

1. 重建输出：`recon_out/6module_compact_cylinder_mlem_psf15_butter*.mhd/.raw`。
2. 运行日志：包含输入路径、metadata 路径、layout、PSF sigma/FWHM、Butterworth 参数和 timing 字段。
3. QA 图：中心 XY/XZ/YZ 三视图，建议保存为 PNG。
4. QA 指标：至少包含 finite、sum、inside/outside、左右/上下对称性、中心 disk CV。
5. 验证记录：写入实现 PR/变更说明或后续专门验证 md。

## 9. 一票否决条件

出现以下任一情况，不得签收：

- 未显式启用 `-matrixLayout compact-symmetric`。
- 投影按 `52992` compact rows 截断。
- standard CSR 路径误读 `metadata.bin` 并改变旧行为。
- 结果图像全零、NaN/Inf、明显错维度或明显 symmetry folding 错误。
- `total_load_to_iteration_done >= 20 s` 且无法通过 profile 证明瓶颈是可排除的输出写盘。
- 使用 DistanceDependent PSF、projection-domain PSF、OSMAPOSL、BSREM、ADMM 或 regularization 才能通过目标。

## 10. 执行顺序建议

1. 先实现 metadata 读取和 strict validation。
2. 再实现 compact CUDA sensitivity 和 numerator helper。
3. 先跑 synthetic parity，确认 SYM_45Z、Z mirror、图像 transform 正确。
4. 再跑本文件固定命令，记录 timing 和图像 QA。
5. 最后补充拒绝路径测试与 standard CSRB 无回归测试。
