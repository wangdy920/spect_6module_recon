# Refactor Decisions（2026-07-05）

> 状态：PR2 本地更新，§2a 正式结论为**不退场**；待 Opus 4.8 / Fable 5 review。

## 第一批本地执行摘要

- GPT-5.5 xhigh / high 范围已本地执行：PR3、PR4、PR5、PR6、PR8。
- Composer-only 范围未纳入本地实现：PR7、PR24。
- 验证：`build_codex` 构建 `SPECT_recon` / `spect_tests`，`spect_tests.exe` 254/254 passed，`build_codex\bin\SPECT_recon.exe -dumpConf` exit 0；`build_codex_cuda` 构建 CUDA `SPECT_recon` / `spect_cuda_scaffold_tests`，`spect_cuda_scaffold_tests.exe` 25 passed / 1 skipped（cuFFT Butterworth unavailable）。

## 第二批本地执行 / review 摘要

- Composer 已完成且本地复核通过：PR7、PR15 move-only、PR24、PR26 §5.6。未发现 `physics-regression-required`。
- GPT-5.5 high/xhigh 本轮补充完成：PR14、PR16 最小退出码接线、PR25。PR24 拆分时遗漏的 `CompactChunkCSRSampleValidationRejectsTailOutOfRangeColIdx` 已补回。
- PR25 真实几何 CUDA scaffold 在本机实际跑到 `test_data/6module/260601` 首个 chunk：resident matrix cache `1 / 1 matrices cached (4720 MiB)`，forward SpMV 抽样 CPU 对照通过。
- 返回码 smoke：`-help=0`、`-dumpConf=0`、parser error `=1`、runtime missing-input error `=2`，符合 `include/cli/ExitCodes.h` 当前契约。
- 260525 smoke 口径：本批实跑为显式 `-matrixLayout compact-symmetric -loadCSR -compactMeta` 的 1-iteration legacy compact loader smoke，用于确认隐式 mmap / legacy compact CUDA path 可运行；不等同于 R50 数值回归或退场签收。
- 最终验证：`build_codex` 构建 `SPECT_recon` / `spect_tests`，`spect_tests.exe` 256/256 passed；`build_codex_cuda` 构建 CUDA `SPECT_recon` / `spect_cuda_scaffold_tests`，`spect_cuda_scaffold_tests.exe` 26 passed / 1 skipped（cuFFT Butterworth unavailable）。
- PR10 review 归因：CUDA `spect_cuda_scaffold_tests` MSBuild target 的 app-local/gtest discovery post-build 失败在 main `2570a34` 干净 CUDA build 同样复现；当前覆盖证据以手动运行 `spect_cuda_scaffold_tests.exe` 为准。

## PR12 BSREM 回归口径与 kappa 裁决（2026-07-04）

- **CUDA BSREM 回归门槛修订**：compact CUDA 路径含 atomic 归约，同一 exe 同输入两次运行 `relative_l2 ≈ 4.28e-08`（实测噪声底）。`<1e-12`/bit 级门槛在该路径不可达。判定口径改为：(a) CPU 级 update 公式对照 gtest bit 级一致（`test_bsrem_update_parity.cpp`，永久保留）；(b) CUDA 端 10-iter BSREM 前后对比落在噪声底同量级（≤1e-7 relative_l2）。归因证据：final exe 复用 baseline denom 缓存后 vs baseline 两次输出均为 4.29e-08（= 噪声底）；此前 8.3e-08 的超额来自两侧独立生成的 denom 缓存（denom 生成本身经 CUDA atomic）。
- **kappa map 双版本裁决**：能力保留（生产 `algo_bsrem.conf` 启用 `kappa_map=true`，6-module 灵敏度动态范围 ~8×，灵敏度自适应正则必要）。历史 compact 变体判定为缺陷：零/负灵敏度体素以 `1/sqrt(eps)` 计入归一化 mean，死体素可将物体内有效正则实质关闭（golden 输入下物体内 kappa 跌至 ~1e-3）。统一到 standard 语义（`computeKappaMapStandard`），compact 变体删除。对严格正灵敏度输入（260601 生产数据 933120 体素全非零）两版逐 bit 一致；统一后 10-iter BSREM 与统一前同 denom 对比 `relative_l2=4.30e-08`（噪声底内），确认零生产影响。

## 第三批本地执行摘要（2026-07-05）

- **§2a 正式决断**：不退场，只做兼容保护；PR28 不启动；转换工具与 260525 路径保留。
- **PR15 后半**：`validateArgs` 拆为 `validateIoArgs` / `validateGeometryArgs` / `validateAlgorithmArgs`；`registerHandlers()` 内所有 flag 绑定统一经 `bindHandlerAliases`，不再直接写 `handlers_["flag"]`；各 handler lambda 内部解析逻辑未改；`-help` 补充 260525 legacy 说明。
- **§1.14 RegularizerCompat**：`compute_reg_gradient()` 继续保留 legacy 实现；委托 `computeGradientNew()` 的尝试已撤销（TVIsotropic/TVHuber FD 回归失败）。迁移必须等能量/梯度离散化口径统一后另开 PR。
- **Epic 8 §4.3**：新增 `tests/test_convert_6module_sysmat.cpp` toy chunk → CSRB 回归。
- **文档**：`doc/PSF_Model_Differences_CN.md`、`doc/Format_Reader_Matrix_CN.md`；README 补 `SPECT_CUDA_SYNC_LEGACY`。
- 此前验证：撤销 RegularizerCompat 委托后，主仓根目录 + `TMP=D:\Temp` 全量 `spect_tests` 261/261 passed。
- Codex 收尾复测（2026-07-06）：`spect_tests` 263/263 passed；`build_release.bat` 通过并生成 `staging\SPECT_recon_rc_dev`；包内 `-dumpConf` 加载 exe-local config 且自动解析 `bin\matrix\260601`；B1/B1 50-iter plain MLEM 对 `sys_260601_recon_ref/recon_iter050.img` 通过（`rmse_over_reference_mean=1.062910420816022e-05`，`sum_rel_diff=1.8900022603998347e-09`）。

## 剩余依赖 / 暂不本地启动

- PR15 `registerHandlers()` 收敛：本轮完成 flag 绑定统一入口，待 Opus review 核对为无语义改动。
- PR28 / §2a 删除：仅在将来三条件齐备且团队明确要求退场时启动。
- PR26 format matrix 正文已迁至 `doc/Format_Reader_Matrix_CN.md`，待 Fable 5 审定措辞。

## §1.6 双轨接口裁决

采用 `ReconContext` + Adapter 作为后续 MainProgram 拆分目标接口。`IReconstructor` / `ReconstructorFactory` / `IReconRunner` 保留一个发布周期但标记为 deprecated，不再从聚合头继续扩大公开面；文件暂不删除。

理由：`ReconContext` 已覆盖 CLI、投影、床位、几何和运行时状态，主路径已有 adapter；补齐另一套未落地主接口会增加第二条演进线。

## §2a 260525 退场条件

**正式决断（2026-07-05）：不退场，只做兼容保护。** PR28 不启动；仅当团队将来明确要求退场且三条件齐备时再评估删除。

退场必须同时满足：

1. 连续两个发布周期无人使用 `-loadCSR -compactMeta`。
2. 260601 chunk CSRB 完成物理验收并成为唯一 compact 生产路径。
3. Opus 4.8 / Fable 5 显式签收。

签收前保留 76B metadata、single-file compact CSR CUDA 路径、转换工具和相关对照脚本。

## §5.3 v2 生成端归档条件

当前结论：不移动、不删除 `spect_sysmat_PSC_6module_24pin_YSO_v2.cu`。

归档必须同时满足：

1. v4 `.csrb` 生成端通过物理验收并被确认为唯一生成端。
2. v2 无独有仍需能力，或能力已迁移。
3. 归档动作由 review 签收；优先移动到 `6module_bak_src/archive_v2/`，不做源码内容改写。

## §1.15 退出码契约

当前外部契约按实测 smoke 与 `include/cli/ExitCodes.h` 统一为 0/1/2：

- 0：成功、`-help`、`-dumpConf`。
- 1：CLI 用户输入、配置和参数校验错误。
- 2：进入运行阶段后的 runtime 错误。

PR14/PR16 只把既有 runtime `return 2` 行为命名化并接线，不改变当前脚本可观察的返回码语义；后续 parser cleanup 不得把 2 静默折回 1。

## 物理回归标记

本批 PR6 只给 CUDA PSF fallback 和 host registration fallback 增加日志，不改变数学顺序。由于触及 Operator PSF fallback 域，标记 `physics-regression-required`，统一交 Opus 4.8 / Fable 5 review。

## 本地物理回归记录

- 2026-07-03 本地 CUDA scaffold 首跑发现 NVRTC `projdomain_chunk_ops.cu` 内 `kAttenUint16Scale` 未定义，导致 compact CTAC / OSEM / BSREM / warmup 相关 12 个 scaffold tests 失败。
- 本地最小修复：在 `src/cuda/SparseOpsCUDA.cpp` 的 NVRTC raw source 内定义同值 `static constexpr float kAttenUint16Scale = 65535.0f;`。这是 host header 无法进入 runtime-compiled kernel 的作用域修复，不改变 attenuation 缩放公式。
- 复测：`spect_cuda_scaffold_tests.exe` 25 passed / 1 skipped（cuFFT Butterworth unavailable）。该修复触及 compact CTAC value baking 域，随 PR6 一并标记 `physics-regression-required`，待 Opus 4.8 / Fable 5 review。
