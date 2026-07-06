# SPECT-recon 未修改项重构执行方案（2026-07-03）

> 依据：`doc/Code_Optimization_Checklist_20260703_CN.md`（2026-07-03 审查版）。
> 所有标记"已修改（2026-07-03）"的清单项视为已完成，本方案不重复规划。
> 本方案由 Opus 4.8 / Fable 5 设计，供 GPT-5.5 xhigh / GPT-5.5 high / Composer 2.5 Fast 分工执行。

---

## 1. Executive Summary

### 1.1 当前最大风险

1. **无能力矩阵单点真相（§7.1）**：standard vs compact 两条路径对各算法/特性的支持状态散落在 5+ 份文档和 CLI 校验代码中，任何架构重构都无法安全验收"没有丢功能"。这是所有后续重构的前置契约，**必须第一个做**。
2. **God Object 主程序（§1.1/§1.2/§1.3）**：`MainProgram.cpp` 约 2678 行，`runProgram()` 约 1630 行，compact 路径内藏第三套 BSREM 更新环。任何人（含模型）在此文件内做改动都有极高的意外破坏面，且 compact BSREM 与 `BSREMReconstructor.cpp` 的语义漂移无法被现有测试捕获。
3. **260525 单文件 3.7GB 默认全量 vector 加载（§2.5）**：不加 `-mapCSR` 时对 32GB 级主机是 OOM/交换风险，属于 silent-failure 类正确性风险。
4. **3 模块时代硬编码默认值（§1.9）**：`numAngles=72`、`detectorChannels=40`、默认 200³ 体素——6 模块数据下若 metadata 缺失静默回退，产出的是"能跑但错"的图像，违反"能跑通 ≠ 语义一致"原则。
5. **CUDA 单体 5311 行 + 接口头 910 行（§3.1/§3.2）**：结构不拆，任何性能专项都无法做小 patch 审查。

### 1.2 当前不应继续自动 loop 的原因

清单"Loop 执行状态"节已明确：剩余项均为**多周级重构或需单独立项**。具体原因：

- 剩余项跨文件耦合度高（MainProgram ↔ ArgumentParser ↔ ReconContext ↔ CUDA 后端），自动 loop 的单轮小 patch 粒度无法保证接口一致性。
- 性能类改动需要 `test_data/share` 50-iter 数值回归 + 空闲系统 benchmark 的人工签收，不能无人值守。
- `.agent/CONTINUITY.md` 记录了大量已证伪方向（resident cache、mmap 默认化、SpMM 合并、subchunk、device-ratio 默认化），自动 loop 有重复踩坑风险。
- B 部分物理逻辑已完成并锁定，自动 loop 缺乏"物理语义敏感区"边界意识，可能在清理中破坏 CTAC value baking / PSF sensitivity 链路。

### 1.3 推荐执行策略

**契约先行 → silent failure 防护 → 架构拆分 → CUDA 结构拆分 → 性能专项 → 清理**，共 6 个优先级阶段（详见 §7）。核心原则：

- 每个 Epic 拆成独立可回滚的小 PR（详见 §10），每个 PR 有明确的验收命令。
- 结构拆分与性能改动**严格分离**：结构 PR 不允许改变任何数值路径；性能 PR 单项立项、单项回归。
- 不删除任何 legacy 入口，直到 §2a 的退场条件（确认无人再用 260525 compact 单文件入口）被显式签收。
- 触及 B 部分物理路径（CTAC/PSF/BSREM/RDP/bProj/sensitivity 语义）的任何重构都标记"需要物理回归确认"，由 Opus 4.8 / Fable 5 审查。

### 1.4 模型分工摘要

| 模型 | 角色 | 典型任务 |
|------|------|----------|
| Opus 4.8 / Fable 5 | 方案设计 + 最终 review | 能力矩阵内容审定、接口契约、fail-fast 边界、物理语义审查、每个高风险 PR 的 review |
| GPT-5.5 xhigh | 高风险执行 | MainProgram/Recon Loop 拆分、BSREM 三合一、CUDA 单体拆分、接口迁移、Matrix IO 策略实现 |
| GPT-5.5 high | 中等复杂度执行 | 测试编写（真实几何 CompactChunk 覆盖）、回归脚本、床位路径 util 收敛、NVRTC loader 合并 |
| Composer 2.5 Fast | 低风险机械执行 | 移动纯 helper、测试文件按域拆分、CMake 模块化、文档更新、重复日志块提取、`mat_convert_bin.m` 函数化 |

---

## 2. 已完成项与锁定边界

### 2.1 已完成项（不重新规划）

| 域 | 已完成项 | 锁定内容 |
|----|----------|----------|
| P0 工程基线 | §0.1–0.5 | git baseline（remote `wangdy920/spect_6module_recon`，`main`）、`.gitignore`、README 死链清理 |
| 主程序 | §1.8、§1.14（部分）、§1.16 | `ReconLimits.h::kMaxBeds=5`；`LegacyCompat.h`/`ReconCompat.h` 已删；`AttenuationConstants.h` 统一 uint16 scale |
| IO/格式 | §2.1–2.4、§2.6–2.8、§2.10–2.12 | 双 CSRHeader 标注、metadata 76/88/96B 版本化、MAT 返回值检查、宽字符路径、CSRB version==1 校验、格式文档对齐 |
| 测试 | §4.1、§4.2 | 260601 真实格式可选 gtest（88B metadata / 103680 行 / Sym90Z 双射 / 首 chunk 读入）；compact toy MLEM golden |
| 生成端 | §5.2、§5.4、§5.7 | v4 `coll_modules` 默认 6 + 几何摘要打印；bak_src README 标注 deprecated/reference-only + 编译示例 |
| 物理 | §6.1（一阶段）、§6.3（防护）、§6.6 | line-integral CTAC 默认启用；BSREM w_proj/w_img fail-fast；标准路径 PSF sensitivity 补 `G^T` |
| 文档 | §7.2–7.4 | README 环境变量表、MultiBed 文档 CTAC 行、性能示例路径 |

### 2.2 不能被重构破坏的路径（锁定边界）

以下路径的**行为与数值输出**在重构中必须保持不变，任何 PR 触及即触发对应回归：

1. **260601 chunk CSRB 主链路**：`-chunkCSRDir` → `CompactChunkCSR.*` → `CompactSymmetricProjectorCUDA` → 88B metadata 解析 → 103680 行公式 → Sym90Z 829440 full-row 双射。canonical 地位不变。
2. **260525 legacy compact 单文件入口**：`-loadCSR -compactMeta` + 76B metadata。在 §2a 退场条件签收前**必须保持可用**，只允许加保护（mmap 策略 / fail-fast），不允许删除。
3. **CTAC 全链路**（B.1/§6.1 产物）：`computeAttenFactorsLineIntegral`、`-ctacModel line-integral|per-voxel`、`-ctacRotSign`、slab phantom gtest、compact 路径设备端 value baking（`compact_bake_atten_values` kernel、按 (chunk, branch, 表指针) 缓存、denom cache manifest CTAC 参数）。
4. **PSF sensitivity 语义**（§6.6 产物）：标准路径全局 denom 与 BSREM subsetDenom 的 `apply_psf_backward` 补丁；compact 路径原有正确实现。
5. **BSREM 防护**（§6.3 产物）：fused gate 与 `BSREM_step` 对 `w_proj`/`w_img` 的 fail-fast。
6. **fused full-subset 双缓冲路径**：清单 §3.3 明确指出这是已有的较优实现，是性能改动的模板而非改造对象。
7. **v28 性能基线**：117.6s @ 50 iter；chunk 交替顺序（v28）决策；`.agent/CONTINUITY.md` 中已证伪的优化方向不得重试。
8. **CLI 向后兼容**：所有现有参数（含 `-loadCSR`、`-compactMeta`、`-mapCSR`、`-ctac*`、`-bProj` 等）在整个重构期间保持接受且语义不变。
9. **metadata 版本化语义**（§2.2 产物）：76B legacy / 88B ExtendedV1 / 96B ExtendedV2 按文件大小强制读取、截断/多余字节抛错。

---

## 3. B 部分状态与重构约束

### 3.1 状态声明

**B 部分（后续改进方向 / B. 物理模型与图像质量，B.1–B.5）已由 Fable 5 执行完成。** 本方案：

- 不将 B 部分重新列为待执行主线；
- 不重复设计 B.1–B.5 的任何方案；
- 仅将 B 部分产物作为**依赖与回归边界**处理。

### 3.2 B 部分产物清单（作为约束的锁定内容）

| B 项 | 已完成产物 | 对重构的约束 |
|------|-----------|--------------|
| B.1 CTAC | 平行射线 line-integral（`-ctacModel`/`-ctacRotSign`）+ slab gtest | 触及 `computeAttenFactorsLineIntegral` 或其调用点的重构 → 物理回归 |
| B.4 compact CTAC | 设备端 value baking kernel + (chunk, branch, 表指针) 缓存 + denom cache manifest | 触及 compact SpMV 前处理链、chunk 缓存键、manifest 的重构 → 物理回归 |
| B.3 相关 | BSREM/RDP 路径（beta 标定流程属物理专项，未做，见 §6.2） | 触及 beta/curvature/kappa/updateFactor 计算 → 物理回归 |
| B.2 相关 | `-bProj` additive background 外部接入 | 触及 forward model 加性项 → 物理回归 |
| B.5 相关 | 无代码产物（诊断属物理专项） | 无 |

### 3.3 "需要物理回归确认"触发清单

任何代码重构（包括看似纯机械的移动/拆分）若触及以下路径，PR 必须打 `physics-regression-required` 标记，由 Opus 4.8 / Fable 5 审查签收，**不得归为普通代码清理**：

1. CTAC line-integral（`computeAttenFactorsLineIntegral` 及角度/间距/旋转方向语义）；
2. compact CTAC value baking（`compact_bake_atten_values`、烘焙缓存键、单位表还原语义）；
3. Operator PSF（`apply_psf_forward/backward` 及其在 numerator/denominator 两侧的调用对称性）;
4. projection-domain PSF（`A^T G^T(p/(G A f))` 隔离路径，§6.4）；
5. bProj additive background（forward model 中 `+ b_i` 的位置与缩放）；
6. BSREM/RDP beta/curvature（含 compact 路径第三套实现的收敛改造——见 Epic 4）；
7. legacy OSL penalized path（§6.5，`denom += beta*grad/f`）；
8. sensitivity / numerator / denominator 语义（任何改变 `A^T 1`、`A^T (y/ŷ)` 计算顺序、精度或 PSF 参与方式的改动）。

### 3.4 物理专项待确认池（不混入代码清理 epic）

清单中以下项属物理专项，本方案**只登记、不规划**，等待物理侧单独立项：

- §6.1 剩余：针孔汇聚射线 Siddon 积分；物理 phantom 定量验证与 `-ctacRotSign` 方向确认。
- §6.2：BSREM/RDP beta sweep 标定。
- §6.4：projection-domain PSF 保持隔离 + 文档写明模型差异（文档部分可并入 Epic 10，代码不动）。
- §6.5：OSL → BSREM-style 统一（触及物理语义，须物理侧签收后才能立项）。
- B.4 剩余：compact OSEM subsets（角度子集 × 90Z 折叠交互需设计）、ADMM/TV、compact CTAC 多床。

**重要**：Epic 4（BSREM 三合一）虽然是代码收敛，但因触及 BSREM 更新公式，整个 Epic 按物理语义敏感区对待（详见 Epic 4）。

---

## 4. 剩余问题总览

严重度沿用清单定义。"Composer 适用" = 可交给 Composer 2.5 Fast 做机械执行（可能仅限该项的部分步骤，详见各 Epic）。

| 编号 | 问题 | 严重度 | 归属 Epic | 执行模型 | 数值回归 | 性能回归 | 物理回归 | Composer 适用 |
|------|------|--------|-----------|----------|----------|----------|----------|----------------|
| §7.1 | 缺能力矩阵 | 高 | E1 | Fable 5 起草 + Opus 4.8 审定 | — | — | — | 否（内容需语义判断） |
| §1.1 | MainProgram God Object 2678 行 | 高 | E2 | GPT-5.5 xhigh | 是 | 否 | 触及 compact CTAC 装配时是 | 否 |
| §1.2 | runProgram 1630 行 / lambda 1166 行 | 高 | E2 | GPT-5.5 xhigh | 是 | 否 | 同上 | 否 |
| §1.3 | compact 第三套 BSREM 环 | 高 | E4 | GPT-5.5 xhigh | 是 | 否 | **是** | **禁止** |
| §1.4 | ArgumentParser.h 1788 行 inline | 高 | E3 | GPT-5.5 xhigh（结构）+ Composer（纯搬移步骤） | 否（CLI 行为测试） | 否 | 否 | 部分 |
| §1.5 | ArgumentParser 内 std::exit | 高 | E3 | GPT-5.5 xhigh | 否 | 否 | 否 | 否 |
| §1.6 | 双轨接口技术债 | 高 | E2 | Opus 4.8 决策 + GPT-5.5 xhigh 执行 | 是 | 否 | 否 | 否 |
| §1.7 | OSEM/BSREM 重复 subset/forward 逻辑 | 高 | E4 | GPT-5.5 xhigh | 是 | 是（AVX2 路径） | 否（不改公式前提下） | 否 |
| §1.9 | 3 模块硬编码默认值 silent error | 中 | E5 | GPT-5.5 high | 否（fail-fast 测试） | 否 | 否 | 否（fail-fast 策略需审定） |
| §1.10 | 床位路径解析两处重复 | 中 | E2 | GPT-5.5 high | 否 | 否 | 否 | 部分（搬移步骤） |
| §1.11 | CUDA chunk timing 日志重复 ~80 行×2 | 中 | E2 | Composer 2.5 Fast | 否 | 否 | 否 | **是** |
| §1.12 | 全局 timing/verbosity 状态 | 中/低 | E2 | GPT-5.5 high | 否 | 否 | 否 | 否 |
| §1.13 | 裸指针生命周期靠约定 | 中 | E2 | GPT-5.5 xhigh | 是 | 否 | 否 | 否 |
| §1.14 | RegularizerCompat 迁移（剩余） | 低 | E3 | GPT-5.5 high | 否 | 否 | 否 | 否 |
| §1.15 | 错误处理不一致 | 中 | E3 | Opus 4.8 定边界 + GPT-5.5 xhigh 执行 | 否 | 否 | 否 | 否 |
| §2.5 | 260525 3.7GB 默认全量加载 | 高 | E5 | GPT-5.5 xhigh | 是（260525 对照） | 否 | 否 | **禁止** |
| §2.9 | validateChunkColIdxRange 全量线性扫描 | 中 | E5 | GPT-5.5 high | 是 | 是（冷加载时间） | 否 | 否 |
| §2a | 260525 退场清单 | — | E5/E10 | Opus 4.8 决策；执行待签收 | — | — | — | 删除动作**禁止** |
| §3.1 | SparseOpsCUDA.cpp 5311 行单体 | 高 | E6 | GPT-5.5 xhigh | 是 | 是（确认无退化） | 否 | 否 |
| §3.2 | ISparseOps.h 910 行接口膨胀 | 高 | E6 | GPT-5.5 xhigh | 是 | 否 | 否 | 否 |
| §3.3 | 基础 SpMV 全同步 + 全局 mutex | 高 | E7 | GPT-5.5 xhigh | **是（50-iter）** | **是** | 否 | **禁止** |
| §3.4 | VolumeOps 默认流 / cuFFT 无 stream | 中 | E7 | GPT-5.5 xhigh | **是（50-iter）** | **是** | 否 | **禁止** |
| §3.5 | NVRTC 样板重复 + 部署依赖 | 中 | E6(合并)/E7(fatbin) | GPT-5.5 high | 是 | 是（首调延迟） | 否 | 否 |
| §3.6 | 无 cudaSetDevice，多卡永远 device 0 | 中 | E6 | GPT-5.5 high | 是（单卡不变性） | 否 | 否 | 否 |
| §3.8 | 析构不清错 / hostRegister 静默 / PSF catch-all 静默降级 | 中 | E5(降级日志)/E6 | GPT-5.5 high | 是 | 否 | PSF 降级路径是 | 否 |
| §3.9 | legacy CSRView(double) 全量 host copy | 中 | E7 | GPT-5.5 xhigh | **是（50-iter）** | **是** | 否 | **禁止** |
| §3.10 | 硬编码资源参数 | 低 | E6 | GPT-5.5 high | 是 | 是 | 否 | 否 |
| §3.11 | CUDA helper 三份重复 / ifdef 双份结构 | 低 | E6 | Composer 2.5 Fast（helper 合并）| 是（编译级） | 否 | 否 | **是**（仅 helper 合并） |
| §4.3 | convert 工具无 gtest | 中 | E8 | 视 §2a 决策 | — | — | — | 是（若补最小回归） |
| §4.4 | test_interfaces.cpp 84KB 巨型文件 | 中 | E8 | Composer 2.5 Fast | 否（测试计数守恒） | 否 | 否 | **是** |
| §4.5 | ChunkWarmup 无真实几何覆盖 | 中 | E8 | GPT-5.5 high | — | — | — | 否 |
| §5.1 | 三种 chunk 格式互不兼容 | 高 | E9 | Fable 5 文档 + Opus 4.8 审定 | — | — | — | 部分（文档誊写） |
| §5.3 | v2/v4 55–65% fork 重复 + 硬编码物理常数 | 中 | E9 | Opus 4.8 决策归档；参数外置属 A.2 前置 | — | — | 参数外置是 | **禁止**（归档决策） |
| §5.5 | 输出数据混在源码目录 | 低 | E9 | GPT-5.5 high（需先核对 share 同名参考） | — | — | — | 否（有误删风险） |
| §5.6 | mat_convert_bin.m 硬编码文件名 | 低 | E9 | Composer 2.5 Fast | 否 | 否 | 否 | **是** |
| §7.5 | CMakeLists.txt 762 行单体 | 低 | E10 | Composer 2.5 Fast | 否（构建等价） | 否 | 否 | **是** |

---

## 5. Epic 设计

### Epic 1: Capability Matrix 与重构契约

- **目标**：建立 `doc/Capability_Matrix_CN.md` 作为单点真相，为后续所有重构提供"重构前后能力守恒"的验收基准。
- **涉及问题编号**：§7.1。
- **不动边界**：本 Epic 零代码改动，仅新增文档。发现文档间冲突时以代码 + gtest 实际行为为准，不得为对齐文档改代码。
- **推荐执行模型**：Fable 5 起草（需读代码验证每格状态）；Opus 4.8 审定。**不适合 Composer 2.5 Fast**（每格需语义判断）。
- **推荐 reviewer 模型**：Opus 4.8。
- **分步执行计划**：
  1. 枚举维度：行 = 特性组合（standard vs compact × {MLEM, OSEM, BSREM, ADMM} × {subsets, PSF, CTAC, bProj, multi-bed} × {CPU, CUDA}）+ 三种矩阵格式（260601 chunk CSRB canonical / 260525 legacy compact 单文件 / 旧 row-wise bin 工具链）。
  2. 每行三列状态：**公式已实现 / 物理已验证 / 真实矩阵 smoke 已通过**，逐格从代码（CLI 校验逻辑、gtest、`MainProgram` 分支）取证，注明证据位置（文件 + 测试名或文档节）。
  3. 对 compact 路径明确标注已知不支持项：OSEM subsets（B.4 剩余）、ADMM/TV、compact CTAC 多床、`w_proj`/`w_img`（§6.3 fail-fast）。
  4. 标注 legacy 状态列：260525 单文件 = legacy compatibility；旧 row-wise bin = reference-only（仅转换工具读取）。
  5. 增加"重构不动边界"节，引用本方案 §2.2 / §3.3。
- **验收标准**：每个格子有证据引用；与 CLI 校验代码（`validateArgs` / `validateCompactSymmetricInputs`）无矛盾；Opus 4.8 审查确认无"标已验证但实际仅 smoke"的夸大。
- **回滚策略**：纯文档，git revert 即可。
- **风险等级**：低（但优先级最高）。

### Epic 2: MainProgram / Recon Loop 架构拆分

- **目标**：把 `src/main/MainProgram.cpp`（~2678 行）按职责拆为可独立测试的编译单元，`runProgram()` 降到 <300 行的编排层；`runReconstruction` lambda 提升为命名类；compact CUDA 迭代分支抽成与 standard 路径对称的 `CompactSymmetricReconLoop`。同时收敛双轨接口（§1.6）。
- **涉及问题编号**：§1.1、§1.2、§1.6、§1.10、§1.11、§1.12、§1.13。
- **不动边界**：
  - **不改任何数值计算顺序、精度、公式**——本 Epic 是纯搬移 + 接口化。
  - 不动 compact BSREM 更新环内部逻辑（留给 Epic 4）。
  - 不动 CTAC/PSF/bProj 装配语义；若搬移触及 compact CTAC 装配代码（value baking 调用点、denom cache manifest），该 PR 打 `physics-regression-required`。
  - CLI 参数与输出文件布局完全不变。
- **推荐执行模型**：GPT-5.5 xhigh（主拆分）；§1.11 日志块提取可交 Composer 2.5 Fast 先行；§1.10 床位路径 util 交 GPT-5.5 high。
- **推荐 reviewer 模型**：Opus 4.8（重点审查：搬移过程中的行为漂移、悬挂指针、初始化顺序变化）。
- **分步执行计划**：
  1. **[决策步，Opus 4.8]** §1.6 双轨接口裁决。推荐方案：`IReconstructor`/`ReconstructorFactory`/`IReconRunner` 头文件标记 deprecated 并从 `SPECT.h` 聚合头移出（不删除文件）；`ReconContext` + Adapter 作为拆分目标接口，MainProgram 拆出的模块以 `ReconContext` 为参数传递上下文，消除床位/overlap 双份计算。理由：`ReconContext` 已完善且有 Adapter，接入成本低于把三个空头文件补全实现。
  2. **[Composer]** 提取 `logCudaChunkTimingSnapshot()` 消除两处 ~80 行重复日志块（§1.11）。纯机械，先行落地降低后续 diff 噪声。
  3. **[GPT-5.5 high]** 新建 `include/app/BedPathUtils.h` + `src/app/BedPathUtils.cpp`，`sortPathsByBedOrder`/`extractBedNumberFromPath` 两处实现合一（§1.10），补 gtest（现有两处行为若有差异，以 ReconSetup 版为准并在 PR 说明）。
  4. **[GPT-5.5 xhigh]** 拆出数据加载/准备阶段：`ReconDataLoader`（投影/mu-map/mask/bProj 加载 + 多床排序），MainProgram 调用其接口。验收：`test_main_recon_preparation` 等现有测试不改断言全过。
  5. **[GPT-5.5 xhigh]** `runReconstruction` lambda 提升为命名类 `StandardReconLoop`（standard CSR 路径），成员持有原捕获变量，构造参数为 `ReconContext`。逐段搬移，每段编译 + `spect_tests` 全过再搬下一段。
  6. **[GPT-5.5 xhigh]** compact CUDA 迭代分支（~684 行）抽成 `CompactSymmetricReconLoop` 编译单元。**该 PR 打 `physics-regression-required`**（含 CTAC baking 装配与 denom cache 逻辑）。BSREM 更新环此步原样搬移不合并。
  7. **[GPT-5.5 xhigh]** Profiling/输出收尾拆出 `ProfilingReporter` + `ReconOutputWriter`；`g_osemStageTiming`/`g_bsremStageTiming` 改经 `StepParams` 传出，`gVerbosity` 注入上下文（§1.12）。
  8. **[GPT-5.5 xhigh]** 裸指针治理（§1.13）：`ReconOpts::mask/b_proj/w_proj` 与 `args.reg.anatomical_image` 改 `std::span` 或 `shared_ptr`，写所有权注释。放在拆分完成后做，避免与搬移 diff 混杂。
- **验收标准**：每步编译通过 + `spect_tests` 全量通过；步骤 5/6 完成后跑 260601 chunk CSRB smoke + compact toy MLEM golden；步骤 6 额外跑 share 50-iter 数值回归（`rmse_over_reference_mean` ~1e-5 量级）+ compact CTAC 相关 gtest；`MainProgram.cpp` 最终 <500 行；CLI 输出逐字节可比（日志时间戳除外）。
- **回滚策略**：每步独立 PR，git revert 单个 PR 不影响其他步骤；步骤 5/6 内部按"编译单元搬移"分 commit，可 revert 到任意中间态。
- **风险等级**：高（步骤 5/6）、中（步骤 4/7/8）、低（步骤 2/3）。

### Epic 3: ArgumentParser 与错误处理收敛

- **目标**：`ArgumentParser.h` 实现移至 `.cpp` 并表驱动化；消除 parser 内 `std::exit`；建立全项目统一的退出码表与"可 warn 降级 / 必须 abort"边界。
- **涉及问题编号**：§1.4、§1.5、§1.15、§1.14（RegularizerCompat 剩余迁移）。
- **不动边界**：所有 CLI 参数名、别名、默认值、校验规则语义不变；`-dumpConf` 输出格式不变；退出码 0/1 的现有外部脚本契约不变（新增细分退出码只能扩展不能改变现有 0/1 含义）。
- **推荐执行模型**：GPT-5.5 xhigh（§1.5 控制流改造、§1.15 边界实施）；Composer 2.5 Fast 仅可做"inline 函数体原样搬到 .cpp"这一机械步骤（不重排、不改逻辑）；GPT-5.5 high 做 RegularizerCompat 迁移。
- **推荐 reviewer 模型**：Opus 4.8（§1.15 边界清单必须由 Opus 4.8 / Fable 5 审定，见本方案 §9）。
- **分步执行计划**：
  1. **[Opus 4.8 定契约]** 定义 `enum class ParseResult { Ok, ExitSuccess, ExitFailure }` 与退出码表（0 = 成功 / 1 = 用户输入错误 / 2 = 运行期错误），写入 `include/cli/ExitCodes.h` 头注释。
  2. **[GPT-5.5 xhigh]** §1.5：4 处 `std::exit` 改为返回 `ParseResult`，`main`/`runProgram` 统一处理。补 gtest：`-dumpConf` 返回 ExitSuccess 且不再直接退出；校验失败返回 ExitFailure。
  3. **[Composer + GPT-5.5 xhigh 分工]** §1.4：先由 Composer 把 inline 实现**原样**搬至 `src/cli/ArgumentParser.cpp`（禁止任何逻辑修改）；再由 GPT-5.5 xhigh 把 614 行 `registerHandlers()` 改表驱动（`{name, arity, handler, help}` 数组），`validateArgs()` 按 IO/几何/算法分组拆为三个私有方法。
  4. **[GPT-5.5 xhigh]** §1.15：按 §9 边界清单统一错误处理——parser 层只返回错误码；setup/recon 层用异常；`main` 捕获并映射退出码。逐文件替换混用点。
  5. **[GPT-5.5 high]** §1.14 剩余：`RegularizerCompat` 使用点迁移到正式接口后删除该兼容层（若仍有真实使用者则标 deprecated 缓删）。
- **验收标准**：`spect_tests` 中所有 parser 测试通过；新增 `-dumpConf`/校验失败路径的返回码 gtest；现有脚本（`scripts/` 下引用退出码者）行为不变；`ArgumentParser.h` 降到 <400 行。
- **回滚策略**：步骤 2/3/4 独立 PR，各自可 revert；表驱动改造保留旧 registerHandlers 于同一 commit 历史，revert 即恢复。
- **风险等级**：中。

### Epic 4: OSEM / BSREM / compact 更新逻辑收敛

- **目标**：消灭第三套 BSREM 实现（§1.3）；提取 OSEM/BSREM 共享的 subset 判断与 forward/backproject 内核（§1.7）。**整个 Epic 视为物理语义敏感区。**
- **涉及问题编号**：§1.3、§1.7。
- **不动边界**：
  - BSREM 更新公式（alpha/beta/curv/kappa/updateFactor）逐项数值语义不变——收敛是"提取共享函数"，不是"重新推导"。
  - §6.3 的 w_proj/w_img fail-fast 防护不得被共享化削弱。
  - AVX2 forward 路径的向量化结构不动（性能敏感）。
  - 不顺带做 §6.5 OSL 统一（物理专项，另行立项）。
- **推荐执行模型**：GPT-5.5 xhigh。**明确禁止 Composer 2.5 Fast 参与本 Epic 任何步骤**（触及 projection/backprojection 更新公式）。
- **推荐 reviewer 模型**：Opus 4.8 + Fable 5 双审（物理语义 + 数值路径保持）。
- **分步执行计划**：
  1. **[GPT-5.5 xhigh]** 先写对照测试：抽取 compact BSREM 环（`MainProgram.cpp` L1965~2054）与 `BSREMReconstructor.cpp` L592~632 的当前行为，构造小规模输入下两者 updateFactor 逐元素对照 gtest（容差 0 或 1e-15 级），**锁定现状**。若发现两者已有数值差异，停下上报（这是潜在 bug，需物理侧裁决哪个正确）。
  2. **[GPT-5.5 xhigh]** 提取 `include/app/BsremUpdate.h`：`applyBsremUpdate(image, numerator, denom, precomputed, BsremParams)` 纯函数，`BSREMReconstructor` 先切换到它，跑 BSREM 全量测试 + share 回归。
  3. **[GPT-5.5 xhigh]** compact 路径（Epic 2 步骤 6 产物 `CompactSymmetricReconLoop`）切换到 `applyBsremUpdate`，删除第三套实现。跑步骤 1 的对照 gtest + compact toy MLEM golden + share 50-iter。
  4. **[GPT-5.5 xhigh]** §1.7：提取 `include/app/SubsetUtils.h`（`isOrderedFullSubset`、`canUseCudaFusedFullSubset*` 单份）+ `SubsetForwardKernel`/`SubsetBackprojectKernel`（AVX2 + 衰减 + 多床 offset 的 6 处重复合一）。每合并一处跑 OSEM/BSREM alignment 测试；AVX2 路径合并后跑一次 benchmark 确认无回退。
- **验收标准**：步骤 1 对照 gtest 永久保留；`spect_tests` 全过；share 50-iter `rmse_over_reference_mean` ~1e-5；BSREM/OSEM alignment 测试（`test_bsrem_alignment`/`test_osem_alignment`）不改断言全过；benchmark 无 >2% 回退；全仓库 BSREM 更新公式只剩一份实现。
- **回滚策略**：步骤 2/3/4 独立 PR。步骤 3 revert 后 compact 路径回到自带环，不影响 standard。
- **风险等级**：高（物理语义敏感 + 数值敏感）。
- **依赖**：Epic 2 步骤 6（compact 分支先成为独立编译单元）。

### Epic 5: Matrix IO / legacy 260525 策略 / silent failure 防护

- **目标**：消除大文件默认全量加载风险（§2.5）；nnz 校验分级（§2.9）；3 模块硬编码默认值 fail-fast（§1.9）；GPU 降级路径日志化（§3.8 的降级部分）；制定 260525 退场决策条件（§2a）。
- **涉及问题编号**：§2.5、§2.9、§1.9、§3.8（降级日志部分）、§2a（决策，不执行删除）。
- **不动边界**：
  - 260601 chunk CSRB canonical 地位与读取语义不变。
  - 260525 `-loadCSR -compactMeta` 入口保持可用，只加保护不删除。
  - `CompactChunkCSR.*`、`CompactSymmetryMetadata.*`、`loadCSRBinary`/`CSRMap` 核心不动（§2a 明确保留项）。
  - `.agent/CONTINUITY.md` 已证伪"mmap 默认化"针对的是 **260601 chunk 路径的性能优化**；本 Epic 对 260525 单文件的 mmap 是**安全性**措施，二者不冲突，但 PR 必须显式引用并说明区别。
- **推荐执行模型**：§2.5 GPT-5.5 xhigh；§2.9/§1.9/§3.8 GPT-5.5 high。**§2.5 禁止 Composer**（矩阵格式读取语义）。
- **推荐 reviewer 模型**：Opus 4.8（重点审查：fallback 滥用、阈值合理性、fail-fast 消息可操作性）。
- **分步执行计划**：
  1. **[GPT-5.5 xhigh]** §2.5 大文件策略（推荐三级）：
     - compact-symmetric 单文件路径按文件大小判断：`size >= SPECT_CSR_MMAP_MIN_BYTES`（默认 1GiB）时**默认启用 `CSRMap`**（等效隐式 `-mapCSR`），stderr 打印 `[INFO]` 说明；
     - mmap 打开失败时**不静默回退全量加载**：`size >= 可用物理内存 × 0.5` 时 fail-fast（消息给出 `-mapCSR` 排查建议与环境变量覆盖方式），否则 `[WARN]` 后回退（与 §2.4 既有口径一致）;
     - 显式 `-mapCSR` 行为不变。
  2. **[GPT-5.5 high]** §2.9 校验分级：默认抽样校验（首/尾 + 均匀抽样约 1e6 个 colIdx）+ 环境变量 `SPECT_CHUNK_VALIDATE=strict|sample|off`（默认 sample）；strict 保留全量扫描。抽样命中越界仍 fail-fast。补 gtest：注入越界 colIdx 的合成 chunk 在 sample 模式被首/尾抽样捕获，strict 模式全捕获。
  3. **[GPT-5.5 high]** §1.9 fail-fast：`numAngles`/`detectorChannels`/体素维度在"6 模块矩阵入口（`-chunkCSRDir` 或 compactMeta 存在）且 metadata/BIN header 未提供"时抛错并提示所需字段，**不再静默用 72/40/200³**；非矩阵路径（如纯工具模式）保留默认值但打 `[WARN]`。gtest 锁定两种行为。
  4. **[GPT-5.5 high]** §3.8 降级日志：GPU PSF `catch(...)` 改为 `catch(std::exception&)` + 记录异常消息与降级原因（引用 fix log 节点 5 的"全零 numerator"事故域）；`cudaHostRegister` 失败打 `[WARN]` 含错误码；析构 `cudaFree` 后 `cudaGetLastError()` 清错（归 Epic 6 结构 PR 顺带）。**PSF 降级路径 PR 打 `physics-regression-required`**（触及 PSF 路径选择）。
  5. **[Opus 4.8 决策文档]** §2a 退场条件写入 Capability Matrix：(a) 连续 2 个发布周期无人使用 `-loadCSR -compactMeta`（通过 RunLogger 使用记录或团队确认）；(b) 260601 路径通过全部物理验收；(c) 显式签收。三条件齐备前，260525 仅标 deprecated（CLI help + 运行时一次性 `[DEPRECATED]` 提示）。
- **验收标准**：`spect_tests` 全过；260525 数据存在时跑一次 `-loadCSR -compactMeta` 对照重建确认输出不变（数值回归）；大文件模拟测试（构造 >阈值稀疏文件或用环境变量降阈值）验证三级策略；§2.9 冷加载时间实测下降且 strict 模式可恢复全量。
- **回滚策略**：每步独立 PR；策略均带环境变量开关，紧急情况可用环境变量恢复旧行为而无需 revert。
- **风险等级**：中。

### Epic 6: CUDA 后端结构拆分

- **目标**：`SparseOpsCUDA.cpp` 5311 行拆为 5~7 个编译单元；`ISparseOps.h` 默认实现下沉；NVRTC 样板合并；helper 去重；`cudaSetDevice` 支持。**全程不改变数值路径与 kernel 内容。**
- **涉及问题编号**：§3.1、§3.2、§3.5（loader 合并部分）、§3.6、§3.8（析构清错部分）、§3.10（参数集中，不改值）、§3.11。
- **不动边界**：
  - 所有 kernel 源码字符串逐字节不变；launch 配置数值不变（§3.10 只做"集中定义"，不改 blockSize=256 等值）。
  - fused full-subset 双缓冲路径不动。
  - 不引入任何 stream/async 改动（留给 Epic 7）。
  - compact chunk 缓存键、CTAC baking 缓存逻辑搬移时逐行保持（触及则打 `physics-regression-required`）。
- **推荐执行模型**：GPT-5.5 xhigh（§3.1/§3.2 主拆分）；GPT-5.5 high（§3.5/§3.6/§3.10）；Composer 2.5 Fast 仅限 §3.11 的 `checkNvrtc`/`checkCuDriver`/`checkCufft` 三份合并到 `include/cuda/CudaDriverHelpers.h` 与 ifdef 双份结构体去重（编译级验证即可）。
- **推荐 reviewer 模型**：Opus 4.8（重点审查：搬移引入的静态初始化顺序、缓存生命周期、锁范围意外变化）。
- **分步执行计划**：
  1. **[Composer]** §3.11：helper 三合一 + `SparseOpsBackend.h` 结构体无条件单份定义。
  2. **[GPT-5.5 xhigh]** §3.1 拆分（每步一 PR，纯搬移）：
     a. NVRTC kernel 源码字符串 → `SparseOpsCudaKernels.h/.cpp`（~240 行）；
     b. CSR 构建 → `CudaCsrBuilder.cpp`（~600 行）；
     c. chunk streaming / host 缓存 → `CudaChunkStreaming.cpp`；
     d. proj-domain kernels → `CudaProjDomainKernels.cpp`；
     e. stub → `SparseOpsCudaStub.cpp`；
     f. 剩余 `SparseOpsCUDAState` 状态机留在 `SparseOpsCUDA.cpp`（目标 <1500 行）。
  3. **[GPT-5.5 xhigh]** §3.2：`ISparseOps.h` ~600 行 private 模板默认实现下沉到 `src/` 下的实现文件（显式实例化或 pImpl）；接口拆分 `ISpmvCore` + subset/atten 扩展接口**先出接口契约文档**（一页纸：方法签名、所有者、CSRView/io::CSRView 重载归属），由 Opus 4.8 审定后再动代码。
  4. **[GPT-5.5 high]** §3.5：三模块共用 `CudaNvrtcLoader`（create/compile/log/load 样板合一，编译产物缓存逻辑不变）。fatbin 预编译属性能/部署改动，**移入 Epic 7 单独立项**。
  5. **[GPT-5.5 high]** §3.6：启动期读 `SPECT_CUDA_DEVICE` 环境变量 + `-cudaDevice` CLI 参数调用 `cudaSetDevice`；未设置时行为与现状完全一致（device 0）；非法 device id fail-fast。
  6. **[GPT-5.5 high]** §3.10：硬编码参数集中到 `include/cuda/CudaLaunchConfig.h`（**值不变**）；host cache 预算改按 `cudaMemGetInfo`/物理内存比例属行为变化，用环境变量 opt-in，默认保持 24GiB/6GiB 现值；补全 README 环境变量表。
  7. **[GPT-5.5 high]** §3.8 析构部分：析构 `noexcept` + `cudaFree` 后清错。
- **验收标准**：每 PR 编译通过 + `spect_tests` 全过（含 `test_cuda_scaffold`、`test_spmv`）；260601 chunk CSRB smoke；拆分全部完成后跑一次 share 50-iter 数值回归 + benchmark **确认零性能回退**（结构拆分不应带来收益也不应带来回退，±2% 内）；`ISparseOps.h` <300 行。
- **回滚策略**：每个编译单元搬移独立 PR，revert 单元级；§3.2 接口拆分在契约文档签核前不落代码。
- **风险等级**：中（步骤 2）/ 高（步骤 3，接口迁移）。
- **依赖**：建议在 Epic 2 主体完成后进行，避免 MainProgram 与 CUDA 两侧同时大 diff。

### Epic 7: CUDA 性能专项

- **目标**：在结构拆分完成后，对 §3.3/§3.4/§3.9 及 §3.5(fatbin) 做**逐项立项**的性能改动。**禁止一次性大规模重写 CUDA 路径。**
- **涉及问题编号**：§3.3、§3.4、§3.9、§3.5（fatbin 部分）。
- **不动边界**：
  - fused full-subset 双缓冲路径是模板不是对象。
  - `.agent/CONTINUITY.md` 已证伪方向（resident cache、mmap 默认化、SpMM 合并、subchunk、device-ratio 默认化）**不得重试**；每个性能 PR 描述中必须显式声明"是否触及已证伪方向"并说明理由。
  - v28 chunk 交替顺序决策不动。
  - 数值容差口径：share 50-iter `rmse_over_reference_mean` ~1e-5 量级；超出即 revert，不允许"放宽容差"。
- **推荐执行模型**：GPT-5.5 xhigh（全部）。**明确禁止 Composer 2.5 Fast 与 GPT-5.5 high 参与**（并发/流语义错误隐蔽且致命）。
- **推荐 reviewer 模型**：Opus 4.8（重点审查：race、stream 依赖遗漏、锁粒度缩小后的 cache 一致性、数值漂移来源）。
- **分步执行计划**（每项独立立项，一项验收后才启动下一项）：
  1. **[项 7a]** §3.3 基础 SpMV async 化：H2D/D2H 绑 computeStream + pinned staging；全局 `mutex_` 缩小为仅保护 cache 更新（以 fused 路径的双缓冲实现为模板）；`executeChunkedWholeMatrixSpMV` 的每 chunk 全流同步改批量 async D2H + 单次同步。
  2. **[项 7b]** §3.4 VolumeOps 专用 stream + 去除每 pass `cuCtxSynchronize`（仅最终 D2H 前同步）；`ButterworthCUDA` 加 `cufftSetStream`。
  3. **[项 7c]** §3.9 legacy `CSRView(double)` 统一走 io::CSRView 直传口径，消除双倍 host 内存；`uploadDouble` 补 pinned + async。注意该路径是 legacy standard 入口，需同时跑 standard 路径回归。
  4. **[项 7d]** §3.5 fatbin：生产构建 `nvcc -fatbin` 预编译主 kernel，NVRTC 仅作未知 arch fallback；验收含"首次调用无编译延迟"与"无 nvrtc DLL 环境可运行"。fallback 到 NVRTC 时必须打 `[INFO]` 日志（不允许静默）。
- **每项验收标准（强制四件套）**：
  1. `test_data/share` 50-iter 数值回归，`rmse_over_reference_mean` ~1e-5 量级；
  2. 空闲系统 benchmark（对 v28 基线 117.6s @ 50 iter），记录前后对比；
  3. PR 描述明确记录是否触及 `.agent/CONTINUITY.md` 已证伪方向；
  4. `spect_tests` 全过 + 260601 smoke + compact toy golden。
- **回滚策略**：每项独立 PR + 环境变量 kill-switch（如 `SPECT_CUDA_SYNC_LEGACY=1` 恢复同步路径），benchmark 或数值不达标立即 revert。
- **风险等级**：高（全部）。
- **依赖**：Epic 6 完成（小文件才能小 patch review）。

### Epic 8: 测试体系拆分与补洞

- **目标**：拆分 84KB 巨型测试文件（§4.4）；补真实几何 CompactChunk warmup 覆盖（§4.5）；按 §2a 决策处理转换工具回归（§4.3）。
- **涉及问题编号**：§4.3、§4.4、§4.5。
- **不动边界**：拆分过程测试**数量与断言逐条守恒**（不新增、不删除、不放宽）；测试 fixture 共享代码提取时行为不变；CTest 注册名可变但需在 PR 中列出映射表。
- **推荐执行模型**：
  - §4.4 拆分：**Composer 2.5 Fast**（纯机械搬移，验收口径客观：测试计数守恒）。
  - §4.5 真实几何测试：**GPT-5.5 high**（需理解 260601 chunk 格式与 warmup 语义，参照 §4.1 已有可选 gtest 的"数据存在时运行"模式编写）。
  - §4.3：视 Epic 5 步骤 5 的 §2a 决策——若走弃用路线则**不补**；若确认保留则由 Composer 2.5 Fast 补最小 2 行 chunk 转换回归（合成输入 → 转换 → 校验输出头 + nnz）。
- **推荐 reviewer 模型**：GPT-5.5 high 复核拆分守恒；§4.5 由 Opus 4.8 review（涉及真实几何语义断言）。
- **分步执行计划**：
  1. **[Composer]** `test_interfaces.cpp` 按域拆为 `test_compact_symmetry.cpp`、`test_parser.cpp`、`test_matrix_io.cpp`、`test_multibed_cli.cpp`（具体切分按文件内 test suite 前缀归组），共享 fixture 提到 `tests/common/` 头文件。每拆一个文件一个 commit，CMake 同步注册。
  2. **[GPT-5.5 high]** §4.5：`test_cuda_scaffold.cpp` 增加可选测试（`test_data/6module/260601` 存在时运行）：真实 chunk 目录首块 warmup → 执行一次 forward SpMV → 与 CPU 参考对照（复用 §4.1/§4.2 的对照口径）。与已有合成小矩阵测试并存，不替换。
  3. **[待定]** §4.3 按 §2a 决策执行或跳过。
- **验收标准**：拆分前后 `ctest` 测试总数一致且全过；拆分后单文件均 <25KB；§4.5 新测试在有数据机器上通过、无数据机器上 skip；构建时间无显著增加。
- **回滚策略**：拆分按文件粒度 revert；新测试独立 PR。
- **风险等级**：低。

### Epic 9: 6module_bak_src 归档与生成端整理

- **目标**：固化"v4 `.csrb` + 主线 `-chunkCSRDir` 为 canonical"的格式矩阵文档（§5.1）；制定 v2 归档决策条件（§5.3 归档部分）；参考数据安全迁移（§5.5）；`mat_convert_bin.m` 函数化（§5.6）。
- **涉及问题编号**：§5.1、§5.3、§5.5、§5.6。
- **不动边界**：
  - 生成端 `.cu` 源码**本 Epic 不做任何代码修改**（v2/v4 去重与几何参数外置属 A.2 统一几何真相源立项，且触及物理常数——物理语义敏感，不在本方案范围）。
  - `test_data/share/` 已有参考数据不可覆盖。
  - v2 源码与独立 MLEM 参考实现在归档决策签收前保留原位。
- **推荐执行模型**：§5.1 Fable 5 起草 + Composer 誊写；§5.3 Opus 4.8 决策；§5.5 GPT-5.5 high（含核对步骤）；§5.6 Composer 2.5 Fast。
- **推荐 reviewer 模型**：Opus 4.8。
- **分步执行计划**：
  1. **[Fable 5]** §5.1 格式矩阵写入 `6module_bak_src/README.md`（或 Capability Matrix 附录）：v2 `chunk_*.bin`（row-wise，legacy，仅转换工具读）/ v4 `chunk_*.csrb`（canonical，主线 `-chunkCSRDir`）/ 独立 `spect_recon_mlem.cu`（只读 `.bin`，reference-only，已标 deprecated）。明确"v4 矩阵无法直接喂给独立 MLEM"为已知且接受的限制。
  2. **[Opus 4.8 决策文档]** §5.3 v2 归档条件：(a) v4 已通过物理验收成为唯一生成端；(b) v2 独有能力（若有）已确认不需要；(c) git 历史保留即视为归档，可将 v2 源码移至 `6module_bak_src/archive_v2/` 而非删除。签收前不动。
  3. **[GPT-5.5 high]** §5.5 数据迁移：先用哈希（SHA256）比对 `6module_bak_src/recon_iter050.*`、`sensitivity.raw` 与 `test_data/share/` 同名文件——**相同则删 bak_src 副本仅留 README 指引；不同则移入 `test_data/legacy_ref/` 并在 README 记录来源（2026-05-23 legacy 对齐参考）**。任何删除动作在 PR 中列出哈希证据。
  4. **[Composer]** §5.6：`mat_convert_bin.m` 改为 `function mat_convert_bin(inputMat, outputBin, varargin)`，保留一个示例调用脚本兼容旧用法；不改转换逻辑。
- **验收标准**：格式矩阵与 Capability Matrix 交叉引用一致；数据迁移 PR 含哈希对照表；`mat_convert_bin.m` 旧调用方式仍可用；未删除任何未签收内容。
- **回滚策略**：文档与 .m 直接 revert；数据迁移用 git（若入库）或迁移前备份清单恢复。
- **风险等级**：低（但 §5.5 有误删风险，故不给 Composer）。

### Epic 10: CMake / 文档 / 发布清理

- **目标**：CMakeLists 模块化（§7.5）；文档收尾（§6.4 模型差异文档、deprecated 标注、环境变量表补全）；§2a 退场执行（仅当 Epic 5 步骤 5 条件签收后）。
- **涉及问题编号**：§7.5、§6.4（仅文档部分）、§2a（执行部分，条件触发）。
- **不动边界**：构建产物、编译选项、target 名称完全等价；`build.bat`/`CMakePresets.json` 调用方式不变；§2a 删除动作必须有 Opus 4.8 签收记录，且 Capability Matrix 同步更新。
- **推荐执行模型**：§7.5 Composer 2.5 Fast；§6.4 文档 Fable 5；§2a 执行 GPT-5.5 high（**删除 legacy 入口不是机械任务，禁止 Composer**）。
- **推荐 reviewer 模型**：GPT-5.5 high 复核 CMake 等价性；Opus 4.8 审 §2a 执行。
- **分步执行计划**：
  1. **[Composer]** §7.5：762 行拆为 `cmake/Dependencies.cmake`、`cmake/CudaConfig.cmake`、`cmake/Targets.cmake`、`cmake/Testing.cmake`、`cmake/Install.cmake`，根文件只留 project + include。验收：拆分前后 `cmake --build` 全 target 成功、`ctest` 列表一致、install manifest 一致。**必须排在 Epic 2/6 之后**（避免拆分期间源文件列表频繁变动造成冲突）。
  2. **[Fable 5]** §6.4：在 Capability Matrix 或专门文档中写明 projection-domain PSF 与 Operator PSF 的模型差异（`A^T G^T(p/(G A f))` vs 标准）与 subsets=1/单床限制。代码不动。
  3. **[条件触发, GPT-5.5 high]** §2a 执行：删除 `tools/convert_6module_sysmat_to_csrb.cpp` + CMake target、`run_260601_chunk_comparison.ps1` 对照段；简化 `-loadCSR` compact 分支、`validateCompactSymmetricInputs()`、非 chunk compact CSR CUDA 路径、76B 可选读逻辑。保留 §2a"保留"清单全部内容。每删一块跑全量测试。
- **验收标准**：构建等价性三项（target/ctest/install）；§2a 执行后 260601 全链路回归 + share 50-iter 无变化；文档交叉引用无死链。
- **回滚策略**：CMake 拆分单 PR revert；§2a 执行按删除块分 commit。
- **风险等级**：低（§7.5/文档）/ 中（§2a 执行）。

---

## 6. 模型分工

### 6.1 Opus 4.8 / Fable 5（设计 + review）

负责：

- 方案设计与 Epic 边界维护（本文档的迭代）；
- 架构边界裁决：§1.6 双轨接口二选一、`ISparseOps` 接口契约签核、§2a/v2 归档决策条件；
- fail-fast 策略与 no-fallback 策略审定（§9 清单为准）；
- 物理语义审查：所有 `physics-regression-required` PR（Epic 4 全部、Epic 2 步骤 6、Epic 5 步骤 4、CTAC/PSF 触碰点）；
- 最终 review 检查项（每个高风险 PR 必查）：
  1. **silent failure**：新代码是否吞异常、吞错误码、静默用默认值；
  2. **compatibility break**：CLI 参数、退出码、文件格式、输出布局是否变化；
  3. **physics semantic drift**：CTAC/PSF/BSREM/bProj/sensitivity 计算顺序或参与项是否变化；
  4. **numerical drift**：浮点累加顺序、精度转换、归约方式是否变化；
  5. **fallback abuse**：是否新增了未在 §9 白名单内的降级路径。

### 6.2 GPT-5.5 xhigh（高风险执行）

负责：

- Epic 2 主拆分（StandardReconLoop / CompactSymmetricReconLoop / ReconDataLoader）；
- Epic 3 控制流改造（std::exit 消除、表驱动、错误处理统一）；
- Epic 4 全部（BSREM 三合一、SubsetUtils/Kernel 提取）；
- Epic 5 §2.5 大文件策略；
- Epic 6 §3.1/§3.2 主拆分与接口迁移；
- Epic 7 全部性能专项；
- 要求：**每个任务给出清晰 patch 边界**（本方案各 Epic 的"分步执行计划"即边界），禁止顺带重构边界外代码、禁止"顺手优化"、禁止改动测试断言来让测试通过。

GPT-5.5 high 承接其中中等复杂度子任务：BedPathUtils、§2.9/§1.9/§3.8 防护、NVRTC loader、cudaSetDevice、§4.5 真实几何测试、§5.5 数据迁移、§2a 执行。

### 6.3 Composer 2.5 Fast（低风险机械执行）

**允许清单**（每项都有客观可验证的守恒条件）：

- Epic 2 步骤 2：`logCudaChunkTimingSnapshot()` 日志块提取（§1.11）；
- Epic 3 步骤 3 前半：ArgumentParser inline 函数体**原样**搬移到 .cpp（禁止改逻辑）；
- Epic 6 步骤 1：CUDA helper 三合一 + ifdef 结构体去重（§3.11）；
- Epic 8 步骤 1：`test_interfaces.cpp` 按域拆分（§4.4，测试计数守恒）；
- Epic 8 §4.3：若保留工具，补最小转换回归；
- Epic 9 步骤 4：`mat_convert_bin.m` 函数化（§5.6）;
- Epic 9 步骤 1 誊写：格式矩阵文档排版（内容由 Fable 5 定）;
- Epic 10 步骤 1：CMake 模块化拆分（§7.5）；
- 各类文档更新、README 环境变量表补全、deprecated 标注誊写。

**明确禁止**（任何情况下不得交给 Composer 2.5 Fast）：

- 修改 CTAC / PSF / BSREM 数学语义（Epic 4 全部步骤禁入）；
- 修改 CUDA 性能路径（Epic 7 全部禁入，含 stream/async/pinned/mutex/缓存任何一项）；
- 修改 matrix 格式读取语义（§2.5、metadata 解析、CSRB 版本校验）；
- 修改 compact symmetry 映射（`CompactSymmetryMetadata`、Sym45Z/90Z 双射、branch 变换）；
- 修改 projection / backprojection 更新公式（含 AVX2 forward、subset 内核合并）；
- 删除 legacy 入口（§2a 执行、v2 归档移动）；
- 设计或实现 fallback 策略（含"mmap 失败后怎么办"这类判断）；
- 任何需要判断"两处重复代码行为是否真的一致"的合并（如 §1.10 两处床位解析——已知可能有差异）。

---

## 7. 推荐执行顺序

| 阶段 | 内容 | Epic | 前置 |
|------|------|------|------|
| **P0** 能力矩阵与契约 | Capability Matrix 全表落地；§1.6 接口裁决文档；§2a/v2 归档条件文档；退出码表 | E1 全部、E2 步骤 1、E3 步骤 1、E5 步骤 5、E9 步骤 2 | 无 |
| **P1** silent failure 防护 | §2.5 大文件三级策略、§1.9 硬编码 fail-fast、§2.9 校验分级、§3.8 降级日志化、260525 deprecated 标注 | E5 步骤 1–4 | P0（fail-fast 边界经 §9 审定） |
| **P2** 架构拆分 | MainProgram 拆分、ArgumentParser 下沉与表驱动、错误处理统一、BSREM 三合一、subset 内核合并 | E2 步骤 2–8、E3 步骤 2–5、E4 全部 | P0；E4 依赖 E2 步骤 6 |
| **P3** CUDA 结构拆分 | SparseOpsCUDA 拆编译单元、ISparseOps 契约化下沉、NVRTC loader、cudaSetDevice、参数集中 | E6 全部 | P2 主体完成（避免双侧大 diff 冲突） |
| **P4** 性能专项 | 7a SpMV async → 7b VolumeOps stream → 7c CSRView 直传 → 7d fatbin，逐项立项 | E7 | P3 完成 |
| **P5** 清理与文档 | 测试拆分补洞、bak_src 归档执行、CMake 拆分、§2a 执行（条件触发）、文档收尾 | E8、E9 步骤 1/3/4、E10 | E8 可与 P2 并行；E10 步骤 1 须在 P2/P3 后 |

并行性说明：

- E1（文档）与 E5 步骤 2–4（防护）可并行启动；
- E8 步骤 1（测试拆分，Composer）可在 P1 期间并行——它不与源码 diff 冲突；
- E7 各项之间**严格串行**（一项验收后启动下一项）；
- 同一时间窗内 MainProgram 与 SparseOpsCUDA 不得同时有未合并大 PR。

---

## 8. 每个阶段的验收口径

所有 PR 的基础门槛 + 按触及范围叠加：

| 验收项 | 命令/口径 | 适用范围 |
|--------|-----------|----------|
| 编译通过 | `build.bat`（或 CMake preset 全 target） | 所有 PR |
| spect_tests 全量通过 | `ctest`（`spect_tests` + `spect_cuda_scaffold_tests`，227+ 用例） | 所有 PR |
| 260601 chunk CSRB smoke | `-chunkCSRDir test_data\6module\260601` 跑短迭代重建，输出维度/非零/无 NaN | 触及 IO、CUDA、MainProgram、compact 路径的 PR |
| compact toy MLEM golden | §4.2 既有 gtest：`CompactSymmetryToyMlemOneIterationMatchesExplicitFull` | 触及 compact 路径的 PR |
| share 50-iter 数值回归 | `test_data/share` 50 iter 对 `recon_iter050.img`，`rmse_over_reference_mean` ~1e-5 量级 | E2 步骤 6、E4 全部、E6 收尾、E7 每项、§2a 执行 |
| 260525 legacy 对照 | `-loadCSR -compactMeta` 短迭代输出 vs 改动前 bit 级/容差一致 | E5 步骤 1、§2a 执行前基线 |
| CUDA benchmark | 空闲系统，v28 基线 117.6s @ 50 iter；结构类 ±2% 内、性能类须记录收益 | E6 收尾、E7 每项、E4 步骤 4（AVX2） |
| 物理回归触发 | §3.3 清单任一路径被触及 → CTAC slab gtest + compact CTAC 分支 gtest + share 零 mu 回归（rel_l2 ~1e-8 口径）+ Opus/Fable 签收 | `physics-regression-required` PR |
| CLI backward compatibility | 现有参数全部可解析且语义不变；退出码 0/1 契约不变；`-dumpConf` 输出格式不变 | E3 全部、E2 步骤 5–7 |
| 测试守恒 | 拆分前后 `ctest` 用例总数一致、断言不放宽 | E8 步骤 1 |
| 构建等价 | target 列表 / ctest 列表 / install manifest 三项一致 | E10 步骤 1 |

---

## 9. Fail-fast / Warning / Fallback 边界清单

### 9.1 必须 fail-fast（发现即抛错终止，禁止继续）

1. metadata 缺失或截断：76/88/96B 之外的大小、尾部多余字节（§2.2 已实现口径，重构中保持）；
2. matrix version mismatch：CSRB `version != 1`、未知 metadata 版本（§2.10 已实现口径）；
3. 6 模块矩阵入口下 `numAngles`/`detectorChannels`/体素维度无 metadata 来源（§1.9，本方案 E5 步骤 3 落地）；
4. unsupported DOI / compact mode：运行时收到 SYM_45Z/SYM_90Z 之外的 sym 模式；
5. compact 路径收到 `w_proj`/`w_img`（§6.3 已实现，保持）；
6. chunk colIdx 越界（无论 sample 还是 strict 模式命中）；
7. 大文件（≥可用物理内存 × 0.5）mmap 失败后的全量加载尝试（E5 步骤 1）；
8. `nnz > UINT32_MAX` 写回 uint32（§2.6 已实现口径）；
9. 非法 `SPECT_CUDA_DEVICE` / `-cudaDevice`（E6 步骤 5）；
10. geometry 与 metadata 不一致（行数公式、模块数校验失败）。

### 9.2 可以 warning 后继续

1. mmap 回退到普通读取——仅限文件小于内存阈值（§2.4 既有口径 + E5 收紧）；
2. `cudaHostRegister` 失败降级 pageable（须打 `[WARN]` + 错误码，E5 步骤 4）；
3. 非矩阵路径（纯工具模式）使用硬编码默认值（`[WARN]`，E5 步骤 3）；
4. 260525 legacy 入口使用时的一次性 `[DEPRECATED]` 提示；
5. NVRTC fallback（fatbin 未覆盖的 arch，须打 `[INFO]`，E7 项 7d）;
6. 环境变量显式选择的非默认行为（如 `SPECT_CHUNK_VALIDATE=off`——用户自担风险，打 `[WARN]`）。

### 9.3 禁止 silent fallback（现存的要修，新增的一律 reject）

1. GPU PSF `catch(...)` 无信息降级 CPU（§3.8——修复后降级必须带原因日志；该路径曾与"全零 numerator"事故同域）；
2. 大文件静默全量 vector 加载（§2.5）；
3. metadata 缺失时静默使用 3 模块默认值（§1.9）；
4. 任何"CUDA 失败自动转 CPU 且不告知"的新路径；
5. 校验失败后"记 warning 但结果照常输出"的模式（结果文件不可在已知校验失败下正常命名产出）；
6. review 时发现边界外新增 try/catch-and-continue 一律要求改为 fail-fast 或列入 §9.2 白名单。

### 9.4 legacy compatibility 允许范围

1. 260525 `-loadCSR -compactMeta` 单文件入口：保留 + deprecated 标注 + E5 大文件保护，直至 §2a 三条件签收；
2. 旧 row-wise `chunk_*.bin`：仅转换工具读取，reference-only，不新增任何消费者；
3. legacy `CSRView(double)` 路径：保留至 E7 项 7c 统一直传口径，之后内部实现变化但接口不删；
4. 76B metadata 可选读逻辑：保留至 §2a 执行时统一 88B；
5. v2 生成端源码 + 独立 MLEM：reference-only 原位保留，归档须 Opus 4.8 签收；
6. deprecated 双轨接口头（`IReconstructor` 等）：移出 `SPECT.h` 但文件保留一个发布周期。

---

## 10. PR 切分建议

基础测试命令缩写：**T** = `build.bat` + `ctest` 全量；**S** = 260601 chunk smoke；**G** = compact toy MLEM golden；**R50** = share 50-iter 数值回归；**B** = 空闲系统 benchmark；**P** = 物理回归（§3.3 触发项 + Opus/Fable 签收）。

| # | 标题 | 涉及文件 | 问题编号 | 执行模型 | Reviewer | Composer 适用 | 测试 | 风险 |
|---|------|----------|----------|----------|----------|---------------|------|------|
| PR1 | 新建 Capability Matrix | `doc/Capability_Matrix_CN.md` | §7.1 | Fable 5 | Opus 4.8 | 否 | 文档交叉校对 | 低 |
| PR2 | 决策文档：接口裁决 + 260525 退场条件 + v2 归档条件 + 退出码表 | `doc/`、`include/cli/ExitCodes.h`（仅头注释） | §1.6/§2a/§5.3/§1.15 | Opus 4.8 | Fable 5 | 否 | 文档 | 低 |
| PR3 | 260525 大文件三级加载策略 | `include/io/CSRBuilder.h`、`src/io/` | §2.5 | GPT-5.5 xhigh | Opus 4.8 | **禁止** | T + 260525 对照 + 大文件模拟 gtest | 中 |
| PR4 | 6 模块入口硬编码默认值 fail-fast | `include/cli/ReconArgs.h`、`src/main/ReconSetup.cpp` | §1.9 | GPT-5.5 high | Opus 4.8 | 否 | T + fail-fast gtest | 中 |
| PR5 | chunk 校验分级（sample/strict/off） | `src/io/CompactChunkCSR.cpp` | §2.9 | GPT-5.5 high | Opus 4.8 | 否 | T + S + 越界注入 gtest + 冷加载计时 | 低 |
| PR6 | GPU PSF 降级日志化 + hostRegister WARN | `src/cuda/CompactSymmetricProjectorCUDA.cpp`、`SparseOpsCUDA.cpp` | §3.8(部分) | GPT-5.5 high | Opus 4.8（**P**） | 否 | T + S + G + **P** | 中 |
| PR7 | 提取 logCudaChunkTimingSnapshot | `src/main/MainProgram.cpp` | §1.11 | Composer 2.5 Fast | GPT-5.5 high | **是** | T + S | 低 |
| PR8 | BedPathUtils 收敛床位路径解析 | `include/app/BedPathUtils.h`、`src/app/`、`ArgumentParser.h`、`ReconSetup.cpp` | §1.10 | GPT-5.5 high | Opus 4.8 | 否（两处行为需比对） | T + 多床测试 | 低 |
| PR9 | 拆出 ReconDataLoader | `src/main/`、`include/app/` | §1.1(部分) | GPT-5.5 xhigh | Opus 4.8 | 否 | T + S + G | 中 |
| PR10 | runReconstruction → StandardReconLoop | `src/main/MainProgram.cpp`、新编译单元 | §1.2 | GPT-5.5 xhigh | Opus 4.8 | 否 | T + S + R50 | 高 |
| PR11 | compact 分支 → CompactSymmetricReconLoop | `src/main/`、新编译单元 | §1.1/§1.2 | GPT-5.5 xhigh | Opus 4.8 + Fable 5（**P**） | 否 | T + S + G + R50 + **P**（CTAC 装配） | 高 |
| PR12 | BSREM 更新对照 gtest + applyBsremUpdate 提取 + 三合一 | `include/app/BsremUpdate.h`、`BSREMReconstructor.cpp`、`CompactSymmetricReconLoop` | §1.3 | GPT-5.5 xhigh | Opus 4.8 + Fable 5（**P**） | **禁止** | T + G + R50 + 对照 gtest + **P** | 高 |
| PR13 | SubsetUtils + 统一 forward/backproject 内核 | `include/app/SubsetUtils.h`、`OSEMReconstructor.cpp`、`BSREMReconstructor.cpp` | §1.7 | GPT-5.5 xhigh | Opus 4.8 | **禁止** | T + R50 + alignment 测试 + **B**（AVX2） | 高 |
| PR14 | ArgumentParser 消除 std::exit | `include/cli/ArgumentParser.h`、`src/main/` | §1.5 | GPT-5.5 xhigh | Opus 4.8 | 否 | T + 返回码 gtest | 中 |
| PR15 | ArgumentParser 实现下沉 + 表驱动 | `include/cli/ArgumentParser.h`、`src/cli/ArgumentParser.cpp` | §1.4 | Composer（搬移）+ GPT-5.5 xhigh（表驱动） | Opus 4.8 | 部分（仅原样搬移步骤） | T + parser 测试全量 | 中 |
| PR16 | 错误处理统一（退出码表实施） | `src/main/`、`src/app/`、`include/cli/` | §1.15/§1.12(部分) | GPT-5.5 xhigh | Opus 4.8 | 否 | T + CLI 兼容检查 | 中 |
| PR17 | CUDA helper 三合一 + 结构体去重 | `include/cuda/CudaDriverHelpers.h`、三个 cuda cpp、`SparseOpsBackend.h` | §3.11 | Composer 2.5 Fast | GPT-5.5 high | **是** | T + S | 低 |
| PR18 | SparseOpsCUDA 拆分（a–f 子 commit） | `src/cuda/SparseOpsCUDA.cpp` → 5~7 编译单元 | §3.1 | GPT-5.5 xhigh | Opus 4.8 | 否 | T + S + G；收尾 R50 + **B**(±2%) | 高 |
| PR19 | ISparseOps 契约文档 + 默认实现下沉 | `include/interfaces/ISparseOps.h`、`src/` | §3.2 | GPT-5.5 xhigh（契约先经 Opus 4.8 签核） | Opus 4.8 | 否 | T + S + R50 | 高 |
| PR20 | CudaNvrtcLoader 合并 + cudaSetDevice + 参数集中 | `src/cuda/*`、`include/cuda/CudaLaunchConfig.h` | §3.5(loader)/§3.6/§3.10 | GPT-5.5 high | Opus 4.8 | 否 | T + S + 单卡不变性 | 中 |
| PR21 | 性能 7a：基础 SpMV async 化 + mutex 缩小 | `src/cuda/SparseOpsCUDA*` | §3.3 | GPT-5.5 xhigh | Opus 4.8 | **禁止** | T + S + G + R50 + **B** + CONTINUITY 声明 | 高 |
| PR22 | 性能 7b：VolumeOps/cuFFT stream | `src/cuda/VolumeOpsCUDA.cpp`、`ButterworthCUDA.cpp` | §3.4 | GPT-5.5 xhigh | Opus 4.8 | **禁止** | T + R50 + **B** + filters golden | 高 |
| PR23 | 性能 7c：CSRView(double) 直传统一 | `src/cuda/SparseOpsCUDA*` | §3.9 | GPT-5.5 xhigh | Opus 4.8 | **禁止** | T + standard 路径回归 + R50 + **B** | 高 |
| PR24 | 测试拆分：test_interfaces 按域拆 4 文件 | `tests/`、`CMakeLists.txt` | §4.4 | Composer 2.5 Fast | GPT-5.5 high | **是** | ctest 计数守恒 + 全过 | 低 |
| PR25 | 真实几何 ChunkWarmup 可选测试 | `tests/test_cuda_scaffold.cpp` | §4.5 | GPT-5.5 high | Opus 4.8 | 否 | 有数据机器过 / 无数据 skip | 低 |
| PR26 | bak_src 格式矩阵文档 + 参考数据哈希迁移 + mat_convert 函数化 | `6module_bak_src/`、`test_data/legacy_ref/` | §5.1/§5.5/§5.6 | Fable 5 + GPT-5.5 high + Composer 分工 | Opus 4.8 | 部分（仅 .m 与誊写） | 哈希对照表 | 低 |
| PR27 | CMake 模块化拆分 | `CMakeLists.txt`、`cmake/*.cmake` | §7.5 | Composer 2.5 Fast | GPT-5.5 high | **是** | 构建等价三项 | 低 |
| PR28 | （条件触发）260525 退场执行 | §2a 删除/简化清单全部 | §2a/§4.3 | GPT-5.5 high | Opus 4.8 | **禁止** | T + S + G + R50 + 签收记录 | 中 |

依赖链摘要：PR2 → PR3/PR4（fail-fast 契约）；PR9 → PR10 → PR11 → PR12；PR12 → PR13 可并行于 PR14–16；PR17 → PR18 → PR19 → PR20 → PR21 → PR22 → PR23（严格串行）；PR27 排在 PR18 合并后；PR28 仅在 §2a 三条件签收后。

---

## 11. 不建议现在做的事项

1. **一次性重写 CUDA 后端**：5311 行单体必须先按纯搬移拆分（E6），任何"重写式"PR 直接 reject。
2. **同时改结构和性能**：同一 PR 混合搬移与 stream/async 改动无法定位数值漂移来源，一律拆开。
3. **未建立 Capability Matrix 前迁移接口**：§1.6 裁决与 §3.2 接口拆分都依赖能力矩阵确认"迁移后能力守恒"，PR1/PR2 未合并前不启动 PR9+/PR19。
4. **未确认 260525 是否仍使用前删除 legacy**：§2a 三条件（无使用记录 + 260601 物理验收 + 显式签收）齐备前，PR28 不启动；期间只做 deprecated 标注与安全加固。
5. **让 Composer 2.5 Fast 处理物理语义或 CUDA 性能**：见 §6.3 禁止清单；分派前逐条核对。
6. **重复规划 B 部分**：B.1–B.5 已由 Fable 5 完成，仅作为回归边界（§3）；发现 B 相关 TODO 只登记入 §3.4 物理专项池。
7. **重试 CONTINUITY 已证伪方向**：resident cache、260601 mmap 默认化、SpMM 合并、subchunk、device-ratio 默认化。
8. **顺带做 §6.5 OSL 统一或 §6.2 beta 标定**：物理专项，须物理侧单独立项，不得混入 E4。
9. **生成端 v2/v4 代码去重与几何参数外置（§5.3 代码部分）**：属 A.2 统一几何真相源立项，触及物理常数，本轮只做归档决策文档。
10. **standard 与 compact 路径的"直接合并"**：两者语义差异（对称折叠、CTAC baking、fused gate 限制）未在能力矩阵中完全刻画前，只做"对称的编排结构"（E2），不做数据通路合并。

---

## 12. 最终推荐

**第一周**：

1. 合并 PR1（Capability Matrix，Fable 5 起草 / Opus 4.8 审定）与 PR2（四项决策文档）——这是全部后续工作的契约。
2. 并行启动 PR3/PR4/PR5/PR6（silent failure 防护四件套，E5）与 PR7（Composer 热身任务）。
3. Composer 2.5 Fast 并行做 PR24（测试拆分）——与源码改动零冲突，尽早减小 test_interfaces 巨型文件对后续 PR review 的干扰。

**第一批 PR**（合并顺序）：PR1 → PR2 → {PR3, PR4, PR5, PR6, PR7, PR24 并行} → PR8。

**交给 GPT-5.5 xhigh**：PR3、PR9–PR16（架构主线）、PR18/PR19（CUDA 结构）、PR21–PR23（性能专项，逐项串行）。每个任务附本方案对应 Epic 的"分步执行计划 + 不动边界"作为 patch 边界约束。

**交给 Composer 2.5 Fast**：PR7、PR15 的原样搬移步骤、PR17、PR24、PR26 的 .m 函数化与文档誊写、PR27。逐条核对 §6.3 禁止清单后再分派。

**必须交 Opus 4.8 / Fable 5 review**：所有 `physics-regression-required` PR（PR6、PR11、PR12）双审；PR10/PR13/PR18/PR19/PR21–PR23 高风险单审；review 按 §6.1 五检查项（silent failure / compatibility break / physics semantic drift / numerical drift / fallback abuse）逐项出具结论。

**里程碑判定**：P2 结束时 `MainProgram.cpp` <500 行、BSREM 单实现、parser 无 exit；P3 结束时 CUDA 单文件 <1500 行、`ISparseOps.h` <300 行、benchmark ±2% 内；P4 每项性能 PR 有 R50 + benchmark 双记录；P5 结束时 Capability Matrix 与代码状态一致、§2a 决策有明确签收状态。






