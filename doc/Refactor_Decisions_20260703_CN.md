# Refactor Decisions（Draft，2026-07-03）

> 状态：PR2 本地草案，待 Opus 4.8 / Fable 5 review。

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

## 剩余依赖 / 暂不本地启动

- PR9-PR13：依赖 PR1/PR2 经 Fable 5 / Opus 4.8 审定后再进入架构/物理链；PR11/PR12 仍需 `physics-regression-required` 双审。
- PR17：Composer CUDA helper 三合一，用户记录为需 Epic 2 稳定；本轮不抢跑。
- PR18-PR23：依赖 PR17、接口契约审定和性能 benchmark，不在当前混合工作树继续拆。
- PR27：依赖 PR18；PR28 / Epic 8 §4.3：依赖 §2a legacy 退场决策；PR26 format matrix 誊写仍需 Fable 5 内容。

## §1.6 双轨接口裁决

采用 `ReconContext` + Adapter 作为后续 MainProgram 拆分目标接口。`IReconstructor` / `ReconstructorFactory` / `IReconRunner` 保留一个发布周期但标记为 deprecated，不再从聚合头继续扩大公开面；文件暂不删除。

理由：`ReconContext` 已覆盖 CLI、投影、床位、几何和运行时状态，主路径已有 adapter；补齐另一套未落地主接口会增加第二条演进线。

## §2a 260525 退场条件

当前结论：不退场，只做兼容保护。

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
