# SPECT-recon 全项目物理模型与实现审查清单

审查日期：2026-06-06

范围：当前仓库的标准 CSR/CSRView 重建路径、compact-symmetric / chunk CSRB 路径、PSF、MLEM/OSEM、BSREM/RDP、CTAC、多床位、projection/background/additive/multiplicative 校正因子、相关测试与运行日志。本文按影响优先级排序；每项均分离“证据”和“推断/建议”。

## 0. 模型基线

- 当前最清晰的一致模型应写作：`y_i ~ Poisson((A H f)_i + b_i)`。
  - `A`：系统矩阵；标准路径可在 `A` 的值上乘 CTAC attenuation。
  - `H`：Operator PSF，当前实现为 image-domain 3D separable Gaussian，前投影前作用一次，反投影 numerator / sensitivity 后作用一次。
  - `b_i`：additive projection term / scatter / background，只应进入 `yhat = Af + b`，不进入 sensitivity denominator。
  - MLEM：`f <- f * H^T A^T(y / (A H f + b)) / H^T A^T 1`。
  - BSREM/RDP：当前主路径采用 `gradient_diff = numerator - subset_denom - beta*kappa*gradR`，再以 global sensitivity / curvature 缩放。

## 1. P0 - compact Operator PSF 在真实 chunk 矩阵上产生全零 numerator，且被静默接受

- 影响：当前用户报告的核心问题；一旦 numerator 全零，MLEM/BSREM 更新会把图像整体压到零，后续迭代不可恢复。
- 证据：
  - 无 PSF 对照使用同一 `170811_detx_flip.bin` projection 和 `test_data/6module/260601` chunk CSRB，`compact_numer.it1 nonzero=933120, sum=373896575.157`：`recon_out/derenzo_param_search_20260606/coarse_170811_xflip/mlem_plain/recon_20260606_105616_229.log:18-21`。
  - Operator PSF MLEM 同 projection/matrix，denom 非零但 numerator 全零：`recon_out/derenzo_param_search_20260606/coarse_170811_xflip/mlem_psf_s065_crystal/recon_20260606_110022_142.log:18-22`。
  - PSF run 继续记录 `compact_numer.it2...` 全零，说明不是单轮 transient failure。
  - 主程序只拒绝 numerator empty/non-finite，不拒绝 finite all-zero：`src/main/MainProgram.cpp:1749-1755`。
  - MLEM compact 更新直接 `image *= numer / denom`：`src/main/MainProgram.cpp:1804-1823`。
- 机理判断：
  - denominator 经 `computeCompactChunkSensitivityCUDA` 后再 `apply_psf_backward`，保持非零：`src/main/MainProgram.cpp:1562-1576`。
  - numerator 路径先对 local bed image 做 `apply_psf_forward`，再进入 compact CUDA numerator，最后做 `apply_psf_backward`：`src/main/MainProgram.cpp:1655-1705`。
  - 因此当前失败更像是 compact PSF 后的 forward image / compact row mapping / ratio / CUDA float path 某一段把有效信号置零，而不是 denominator 或 projection 输入全零。
- 建议：
  - 立即加 fail-fast：`compact_numer` 若 `nonzero==0` 或 `sum<=0` 且 projection sum > 0，应抛错。
  - 增加 stage-local stats：PSF 后 `localUsed`、每 chunk `yhat`、ratio、`localNumer` before/after PSF backward。
  - 用真实 `test_data/6module/260601` 加 1-iter regression，不要只依赖 toy compact symmetry 测试。

## 2. P0 - CTAC 当前实现不是射线积分 CTAC，文档声称与代码不一致

- 影响：如果用户把 `-ctac -attMap` 结果当作定量 CTAC，物理模型会系统性错误。
- 证据：
  - `computeAttenFactorsPerVoxel` 忽略 `dx/dy/dz/angle`，直接对每个体素计算 `exp(-muMap[idx])` 并复制到所有角度：`src/main/CTACOps.cpp:10-31`。
  - `applyCTACToCSR` 只是把上述 per-voxel factor 乘进 CSR row 的每个 matrix value：`src/main/CTACOps.cpp:35-54`。
  - 多床位 dynamic CTAC 只是从 global mu map 裁出 local bed，再调用同一个 per-voxel factor 函数：`src/main/MainProgram.cpp:1424-1447`。
  - CTAC 测试只覆盖 zero-mu all-ones、zero-mu keeps weights、invalid angle：`tests/test_main_ctac_ops.cpp:1-70`，没有 line-integral / angle-dependent phantom。
  - 文档却写“动态射线追踪 / 全局 mu-map / 已完成”：`doc/MultiBed_Improvements_Summary.md:3-4`。
  - 另一个适配器注释承认是 placeholder：`include/app/ReconContextAdapter.h:127-130`。
- 机理判断：
  - 真正 CTAC attenuation 应为 `exp(- integral mu ds)`，通常依赖角度、ray path、体素尺寸和源/探测几何。当前实现是 `exp(-mu_voxel)`，没有 path length，也没有 ray integration。
- 建议：
  - 在未实现 ray integral 前，把当前功能文档改名为 approximate voxel attenuation factor，不应叫已完成 CTAC。
  - 增加 2D/3D uniform slab phantom：不同角度/path length 应产生不同 attenuation；当前实现会失败，这正好锁定修复目标。
  - compact 路径继续保持 CTAC 禁止，直到 attenuation 的 compact branch 语义被单独定义。

## 3. P0 - compact Operator PSF 冷启动真实 PSC smoke 在 numerator 前 bad allocation

- 影响：即使先解决全零 numerator，当前 PSF denom cold path 也可能在真实矩阵/内存状态下失败。
- 证据：
  - 1-iter PSC Operator PSF smoke：`recon_out/review_compact_psf_operator_it1_20260606/recon_20260606_140402_832.log:1-18`，到 compact chunk directory opened 后 `[FATAL] bad allocation`。
  - chunk matrix nnz 约 `2436868849`，host cache budget `25769803776`：同日志 `:14-15`。
  - PSF denominator miss 会重新构建并保存 cache；PSF/no-PSF 共享同一 cache path 时反复 manifest mismatch：`src/main/MainProgram.cpp:1536-1599`。
- 推断：
  - 这不是数学错误本身，而是工作流和内存风险：PSF/no-PSF 使用同一 `-compactDenomCache` 路径会造成 cold recompute 和覆盖，增加 OOM/bad_alloc 概率。
- 建议：
  - PSF/no-PSF、FWHM、bed geometry 使用不同 denom cache 文件名。
  - 若 manifest mismatch，先报告 expected/actual key 摘要；探索 run 可选择不覆盖现有 cache。

## 4. P1 - `-psf` 命令行只设置 FWHM，不自动启用 Operator 模式

- 影响：用户很容易以为 `-psf <sigma>` 已启用 PSF；实际上若未加 `-psfMode Operator`，运行期可能保持 `PSFMode::None`。
- 证据：
  - `-psf` handler 只设置 `usePSF=true` 和 `fwhm_mm`：`include/cli/ArgumentParser.h:480-482`。
  - `-psfMode` 才设置 mode：`include/cli/ArgumentParser.h:487-503`。
  - `PSFConf` 默认 `mode = PSFMode::None`：`include/filters/PSFConfig.h:36-44`。
  - runtime prepare 只有 mode 非 None 才生成 kernel：`src/main/ReconPreparation.cpp:97-117`。
- 建议：
  - 若 `-psf` 出现且没有显式 `-psfMode`，默认设为 `Operator`，或至少打印 fatal/warn。
  - help 文档中把 `-psf` 与 `-psfMode Operator` 的关系写成强约束。

## 5. P1 - 多床位 + Operator PSF 使用 local bed volume，无 Z 向 halo，重叠边界可能偏置

- 影响：对 multi-bed overlap 的物理边界，PSF kernel 会跨 bed 边界传播；当前先切 local bed 再卷积，边界处理不会看到邻床真实体素。
- 证据：
  - local bed 从 global image 截取：`src/main/MainProgram.cpp:1655-1659`。
  - 在 local `baseNz` 上直接做 PSF forward：`src/main/MainProgram.cpp:1662-1668`。
  - local numerator 又在 local `baseNz` 上做 PSF backward：`src/main/MainProgram.cpp:1698-1705`。
  - 多床位 overlap 工具只是 Z offset 与累加，没有 halo：`include/app/BedOffsets.h:21-60`。
- 推断：
  - 若 PSF kernel z radius > 0，bed 边界处使用局部卷积边界条件，等价于缺失邻床上下文；重叠累加不能完全补偿卷积前后的边界差异。
- 建议：
  - 对 multi-bed Operator PSF 增加 halo extraction，halo 半径为 kernel half-width；只把 active bed 区域累加回 global。
  - 在无 halo 前，multi-bed compact PSF 只能作为 QA/实验，不建议作为物理签收路径。

## 6. P1 - compact denominator cache manifest 足够防错但工作流易被共享路径破坏

- 影响：反复切换 PSF/no-PSF、FWHM、Butterworth/Joseph/bed 参数时，单一路径 cache 会频繁 miss/覆盖；这会放大内存开销并污染审查证据。
- 证据：
  - manifest 包含 matrix/chunk、global dims、bed starts、Joseph、PSF mode/FWHM/kernel：`src/main/MainProgram.cpp:550-617`。
  - cache hit 要求 manifest 全字符串一致：`src/main/MainProgram.cpp:620-681`。
  - PSF run 显示 `cache manifest mismatch` 后保存同一路径：`recon_out/derenzo_param_search_20260606/coarse_170811_xflip/mlem_psf_s065_crystal/recon_20260606_110022_142.log:18-20`。
- 建议：
  - 保留 manifest 严格校验；但将 cache path 规范化为带 `psf_none` / `psf_operator_fwhm065` / `bedN` 的目录。
  - 报告中引用 cache 时同时保存 manifest snapshot，避免后续覆盖后无法复核。

## 7. P1 - BSREM/RDP compact 更新形式合理，但 beta/curvature 标度仍缺少真实矩阵校准

- 影响：BSREM/RDP 当前可以运行，但数值稳定性和正则强度解释依赖 sensitivity scale；真实 6-module 矩阵下必须校准，否则 beta 不是可转移参数。
- 证据：
  - compact BSREM 使用 `termLikelihood = numer - denom`，`termPenalty = beta*kappa*grad`，`denomEff = sensGlobal + beta*curv`：`src/main/MainProgram.cpp:1842-1856`。
  - 标准 BSREM 同样使用 additive gradient-difference update 和 limiter：`src/app/BSREMReconstructor.cpp:609-627`。
  - RDP gradient 与 energy 的有限差分测试存在：`tests/test_regularizers.cpp:219-222`。
  - CUDA RDP gradient 与 CPU 对齐测试存在：`tests/test_cuda_scaffold.cpp:624-665`。
- 建议：
  - 保持当前 BSREM compact update，不退回旧 OSL denominator 写法。
  - 用真实 Derenzo/PSC 做 beta sweep，并记录 likelihood、penalty、central contrast、background CV，而不是只看视觉结果。
  - curvature map `abs(grad)/f` 属启发式，应固定 warmup/freeze 策略后再签收。

## 8. P1 - 标准 CTAC 在 OSEM/BSREM forward/backward/sensitivity 中的位置一致，但 attenuation 本身不正确

- 影响：若只看迭代公式，attenuation 乘在矩阵项里的位置是对的；但 factor 生成见 P0，导致整体 CTAC 仍不可作为定量模型。
- 证据：
  - OSEM CSRView forward/backward 都解析 `attPtr` 并把 attenuation 乘到矩阵值上；backprojection 的非 thread-local 与 thread-local 路径分别见 `src/app/OSEMReconstructor.cpp:492-563`。
  - BSREM forward/backward 同样在矩阵值上乘 attenuation；forward 段见 `src/app/BSREMReconstructor.cpp:245-255`, `src/app/BSREMReconstructor.cpp:270-319`，backprojection 段见 `src/app/BSREMReconstructor.cpp:361-385`, `src/app/BSREMReconstructor.cpp:430-448`, `src/app/BSREMReconstructor.cpp:472-503`, `src/app/BSREMReconstructor.cpp:542-568`。
  - `computeSubsetSensitivity` 会把 attenuation 乘入 sensitivity：`include/app/ProjectionOps.inl:198-343`。
  - CUDA fused full-subset 会对 attenuation 构建 attenuated chunks：`src/cuda/SparseOpsCUDA.cpp:4251-4269`, `src/cuda/SparseOpsCUDA.cpp:4460-4572`。
- 建议：
  - 保留 attenuation 的应用位置；重写 attenuation factor 生成。
  - 增加 standard CPU/GPU/CSRView 对同一 CTAC phantom 的数值对齐测试。

## 9. P2 - Projection-domain PSF 路径不支持 additive background/CTAC/multi-bed/subsets，且仍使用旧 OSL 正则写法

- 影响：如果用户启用 projection-domain PSF，会离开标准 Operator PSF 模型；BSREM 直接禁止，MLEM 路径也只支持更窄条件。
- 证据：
  - BSREM + projection-domain PSF 直接抛错：`src/main/MainProgram.cpp:1221-1224`。
  - `canProjPSF` 要求 `subsets==1`、`bedCount==1`、det geometry 与 projection size 匹配：`src/main/MainProgram.cpp:2153-2159`。
  - Proj-domain MLEM ratio 为 `p / yhat`，随后对 ratio 做 2D PSF，再反投影：`src/main/MainProgram.cpp:969-1007`。
  - 正则项仍使用 `denom += beta*grad/f`，并有 fallback：`src/main/MainProgram.cpp:1017-1028`。
- 建议：
  - 当前不要把 projection-domain PSF 与 BSREM/RDP、multi-bed、CTAC 混用。
  - 如果保留该功能，应独立写明模型是 `A^T G^T(p/(G A f))`，与 image-domain Operator PSF 不同。

## 10. P2 - OSEM/MLEM 的 legacy penalized path 仍使用 `denom += beta*grad/f`，可能非正或与 BSREM 不一致

- 影响：普通 MLEM 加 regularizer 或 projection-domain PSF 加 regularizer 时，penalty 进入 denominator 的方式可能造成非物理更新；BSREM 已换成更合理的 additive gradient step。
- 证据：
  - OSEM_step 两个 overload 都使用 `denom_eff += beta * gradR / f`：`src/app/OSEMReconstructor.cpp:299-305`, `src/app/OSEMReconstructor.cpp:599-605`。
  - compatibility OSL update 也有相同逻辑和 fallback：`src/regularizer/Regularizers.cpp:575-600`。
- 建议：
  - 对 RDP/TV 等非二次 penalty，不建议继续通过 MLEM legacy OSL path 做生产结果。
  - 若需要 penalized MLEM，应统一到 BSREM-style additive gradient update 或明确命名为 legacy OSL。

## 11. P2 - `w_proj` / `w_img` 乘性因子当前没有 CLI 接入；若未来启用，语义不完整

- 影响：这些指针看似支持 projection/image multiplicative corrections，但当前主程序显式置空，且实现语义不等价于常见 normalization/dead-time/efficiency 校正。
- 证据：
  - 主程序设置 `opts.w_proj = nullptr; opts.w_img = nullptr; opts.b_proj = ...`：`src/main/MainProgram.cpp:1467-1469`。
  - OSEM forward 把 `w_proj` 乘到 `yhat`，ratio 仍为 `p/yhat`：`src/app/OSEMReconstructor.cpp:185-188`, `src/app/OSEMReconstructor.cpp:399-402`。
  - OSEM update 后再乘 `w_img`：`src/app/OSEMReconstructor.cpp:306-307`, `src/app/OSEMReconstructor.cpp:606-607`。
  - OSEM fused CUDA 在有 `w_proj`/`b_proj` 时禁用：`src/app/OSEMReconstructor.cpp:46-60`。
  - BSREM fused 只因 `b_proj` 禁用，不检查 `w_proj`；BSREM CPU/sparse forward 也只加 `b_proj`，未处理 `w_proj`：`src/app/BSREMReconstructor.cpp:47-60`, `src/app/BSREMReconstructor.cpp:183-192`, `src/app/BSREMReconstructor.cpp:254-319`。
- 推断：
  - 若 `w_proj` 表示物理 normalization，应进入系统模型和 sensitivity，而不是只乘 predicted projection。
  - 若 `w_proj` 表示统计权重，应改变 likelihood/gradient，而不是这种形式。
  - `w_img` 作为 update 后乘法会改变 fixed point，不能当一般 multiplicative correction。
- 建议：
  - 在 CLI 未接入前保持禁用。
  - 若未来接入，拆分命名：`norm_proj`、`stat_weight`、`image_mask/prior`，并分别定义数学模型与 sensitivity。

## 12. P2 - additive background / scatter 的核心公式正确，但 compact 文档和验收范围已过期

- 影响：使用 `-bProj/-bProjList` 时，核心 EM 位置合理；但文档仍称 compact Phase 1 拒绝 background/scatter，容易误导实验设计。
- 证据：
  - 主程序加载 `-bProj/-bProjList` 并检查 per-bed geometry：`src/main/MainProgram.cpp:1201-1218`。
  - OSEM/BSREM 将 `b_proj` 加入 `yhat`，不进入 denominator：`src/app/OSEMReconstructor.cpp:185-188`, `src/app/BSREMReconstructor.cpp:190-192`。
  - compact ratio 使用 `fullProjection / max(y + fullBackground, eps)`：`src/cuda/CompactSymmetricProjectorCUDA.cpp:484-508`。
  - compact chunk multi-bed 在有 background 时禁用 opt-in device ratio，走 host ratio：`src/cuda/CompactSymmetricProjectorCUDA.cpp:947-960`。
  - 旧验收文档仍写 compact 拒绝 background/scatter：`doc/CompactSymmetricBackend_Acceptance_Checklist_CN.md:19-20`, `doc/CompactSymmetricBackend_Run_20260523_CN.md:511`。
- 建议：
  - 文档改成：compact 当前支持外部 additive background projection，但不支持内部 scatter solver，也不支持 CTAC。
  - 给 `-bProjList` 加 multi-bed compact 真实矩阵 smoke。

## 13. P2 - compact-symmetric 当前只允许 CUDA、subsets=1、MLEM/BSREM/RDP；不要推广到完整物理路径

- 影响：compact 是高性能特化路径，不是标准 CSRB 的完整替代。
- 证据：
  - 参数校验要求 compact 使用 CUDA：`include/cli/ArgumentParser.h:1294-1302`。
  - 只允许 `mlem|bsrem` 且 `subsets=1`：`include/cli/ArgumentParser.h:1303-1309`。
  - compact 禁止 CTAC/attenuation maps：`include/cli/ArgumentParser.h:1310-1313`。
  - compact regularization 只允许 `BSREM + RDP`：`include/cli/ArgumentParser.h:1314-1320`。
  - compact 禁止 projection-domain PSF/PostSmooth：`include/cli/ArgumentParser.h:1322-1326`。
- 建议：
  - 对生产说明写清楚：standard CSRB 是完整物理功能主线；compact 是无 CTAC、subsets=1 的性能路径。

## 14. P3 - PSF 测试目前主要是 toy 级，不覆盖真实 matrix/chunk/Projection 组合

- 影响：toy 测试通过不能阻止本次真实 chunk matrix 的 all-zero numerator。
- 证据：
  - 已通过 focused tests：`spect_tests.exe --gtest_filter=PSFModeTest.*:*CompactSymmetryToyZeroPsf*`，3/3 passed。
  - `tests/test_psf_modes.cpp:66-104` 只验证 Operator/PostSmooth 是否 blur。
  - compact toy zero PSF / Gaussian commutation 不是 `test_data/6module/260601` 的真实 chunk path。
- 建议：
  - 增加真实 chunk 1-iter PSF smoke：断言 denominator nonzero、PSF-forward image nonzero、first chunk yhat nonzero、numerator nonzero。
  - 记录 projection sum 和 output sum，避免“全零但 finite”逃逸。

## 15. P3 - CUDA fused full-subset helper 不处理 b/w correction，依赖外层 gate；BSREM gate 不完整

- 影响：当前 `b_proj` 会禁用 BSREM fused，OSEM 对 `w_proj/b_proj` 都禁用；但如果未来把 `w_proj` 接入 BSREM，会被忽略。
- 证据：
  - CUDA ratio kernel 是 `projection / yhat`，没有 background 或 w_proj：`src/cuda/SparseOpsCUDA.cpp:620-633`。
  - full-subset fused EM/BSREM 共用同一 helper：`src/cuda/SparseOpsCUDA.cpp:4198-4606`。
  - OSEM gate 检查 `w_proj || b_proj`：`src/app/OSEMReconstructor.cpp:46-60`。
  - BSREM gate 只检查 `b_proj`：`src/app/BSREMReconstructor.cpp:47-60`。
- 建议：
  - 在 BSREM gate 同步检查 `w_proj`，即使当前 CLI 不设置。
  - 更好方案是删除/隐藏 `w_proj`，直到模型定义完成。

## 16. P3 - 文档中 compact Phase 1、CTAC、多床位能力的描述需要重新归档

- 影响：文档会误导下一轮实验，尤其是把 CTAC 写成已完成，把 compact background 写成拒绝。
- 证据：
  - CTAC 文档声称 dynamic ray tracing 已完成：`doc/MultiBed_Improvements_Summary.md:4`。
  - compact 旧文档称拒绝 background/scatter，但代码已支持 additive background：见第 12 项。
  - `doc/CompactSymmetricBackend_Design_CN.md:48` 仍描述 Phase 1 支持范围，后续 BSREM/RDP/multi-bed/background 已发生变化。
- 建议：
  - 增加一份当前能力矩阵：standard vs compact，MLEM/BSREM/RDP/CTAC/PSF/bProj/multi-bed/subsets/CUDA。
  - 将“已实现”拆成“公式位置实现”、“物理模型已验证”、“真实矩阵 smoke 已通过”三列。

## 17. 已确认的正向实现点

- 标准 OSEM/MLEM 中 additive background 的使用位置正确：只进入 `yhat`，不进入 sensitivity。
- 标准 CTAC attenuation 的应用位置在 forward/backward/sensitivity 中一致；问题在 attenuation factor 生成，不在 EM 公式位置。
- Operator PSF 的标准路径是前投影前 blur image，反投影后 blur numerator；PostSmooth 不参与 projector，测试覆盖了该语义。
- RDP energy/gradient 有 finite-difference 测试，CUDA gradient 有 CPU 对齐测试。
- 多床位标准路径的 offset/overlap sensitivity 累加有 toy tests，纯 matrix offset 逻辑是可验证的。

## 18. 当前 no-go 清单

- 不要将当前 compact Operator PSF 结果用于任何物理结论；先修复全零 numerator 和 bad allocation。
- 不要把当前 `-ctac` 解释为真实 line-integral CTAC。
- 不要在 compact 路径启用 CTAC；参数校验已经禁止，保持该边界。
- 不要把 `-psf` 单独写入生产脚本；必须显式 `-psfMode Operator`，直到 parser 修复。
- 不要复用同一个 `psc_260601_global_denom.mhd` 给 no-PSF、不同 PSF FWHM、不同 bed geometry 的实验。

## 19. 建议的修复顺序

1. 对 compact numerator 增加 all-zero fail-fast，并记录 local PSF forward / yhat / ratio / localNumer stats。
2. 用真实 `test_data/6module/260601` 加 PSF 1-iter regression，先定位全零发生在 PSF forward、compact CUDA forward、ratio 还是 backward。
3. 将 PSF/no-PSF denom cache 路径隔离，避免 manifest mismatch 覆盖。
4. 修正 `-psf` 默认 mode 或强制 warning/fatal。
5. 将 CTAC 文档降级为 placeholder/approximation；再实现 angle-dependent line-integral attenuation。
6. 给 multi-bed PSF 加 halo，或在文档中明确禁用 multi-bed Operator PSF 物理签收。
7. 同步 compact/standard capability matrix 文档，避免 Phase 1 旧结论继续传播。

## 20. 本次执行过的验证

- `out\build\chunk-csrb-agent\bin\spect_tests.exe --gtest_filter=PSFModeTest.*:*CompactSymmetryToyZeroPsf*`
  - 结果：3/3 passed。
- 读取并对比了以下关键日志：
  - no-PSF compact MLEM：`recon_out/derenzo_param_search_20260606/coarse_170811_xflip/mlem_plain/recon_20260606_105616_229.log`
  - Operator PSF compact MLEM：`recon_out/derenzo_param_search_20260606/coarse_170811_xflip/mlem_psf_s065_crystal/recon_20260606_110022_142.log`
  - PSC Operator PSF 1-iter cold smoke：`recon_out/review_compact_psf_operator_it1_20260606/recon_20260606_140402_832.log`
