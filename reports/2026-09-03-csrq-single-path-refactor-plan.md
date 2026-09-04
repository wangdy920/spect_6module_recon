# 设计记录：系统矿阵单路径重构方案（未采纳执行）

日期：2026-09-03。状态：**不执行**。用户决策：CSRB 路径保留为生产路径，CSRQ 只做实验用。本文件仅作为设计记录保存，第 2–8 节描述的是被搁置的方案，不代表当前计划。

## 决策更新（2026-09-03，用户）

- **CSRB 设备路径继续作为生产路径**，仍是受保护的数值基线；**CSRQ（含双能窗联合似然、流式驻留）只做实验用途**，不进入生产配置。
- 因此本方案的阶段 3（删除 CSRB 设备路径与实验分支）**无限期推迟**；阶段 0 中为 CTAC 迁移准备的 CSRB-CTAC 基线**不再需要**；阶段 2 的"CTAC 迁入 CSRQ"降为可选实验项。
- 仍然成立且值得做的部分：第 3 节的接口/解耦思路可在不删除任何路径的前提下逐步落地（先把驱动里的 `format()` 分叉收拢到一个工厂/接口后面）；第 4 节的 K 与值编码模板化只影响实验路径；第 6 节阶段 4 的 (angle, z 切片) backward 仍是实验路径的效率杠杆。
- 生产路径的守护动作：今天对共享代码（状态类投影槽扩容、`refreshCompactProjection` 增加背景参数、驱动通道包装、重建循环通道化）的改动需要用 CSRB 端到端回归确认无行为变化；结果记录在 `.agent/CONTINUITY.md`。

## 1. 背景与问题

当前设备端存在两条并行执行路径：

| 路径 | 存储/传输 | 计算 | 独占能力 |
|---|---|---|---|
| CSRB（fp32 CSR） | chunk 流式、设备 chunk 缓存、常驻矿阵集、pinned 暂存 | cuSPARSE SpMV/SpMM、packed-K forward、CSR_ALG2 转置 | CTAC（衰减因子融合进 nnz 循环） |
| CSRQ（run 索引 + u16 sqrt 值） | 常驻或 pinned 流式（两槽双缓冲） | 自定义 K=8 内核 | 背景项、双能窗联合似然、常驻 MLEM |

代码体量：`CudaSparseOpsState.h` 4758 行（单个类），`CompactSymmetricProjectorCUDA.cpp` 2120 行，`CompactSymmetricReconLoop.cpp` 约 1700 行。今天核查到的事实：

- cuSPARSE 那套 compact API 的**唯一调用者是驱动** `CompactSymmetricProjectorCUDA.cpp`；`tools/transpose_bench.cpp` 直接用 cuSPARSE 做转置实验。
- 状态类第 3539–4424 行的全矿阵/subset 路径（`warmupFullSubsetChunkCache`、`executeChunkedWholeMatrixSpMV`、`runFullSubsetEmNumer` 等）在 `SparseOpsCUDA.cpp` 中没有转发函数，**已是死代码**。
- `CompactTransposeMode`（csrt/cscn/alg1/alg1pre/spmvpre）等被 CONSTRAINTS 标为"已拒绝"的实验分支仍在源码里。
- 本地全部矿阵（Tc99m、Lu177-113、Lu177-208）均为 `v7.9`、`sym_mode=90`、`use_z_symmetry=1` → 分支数 K = 4 × 2 = 8，且 K 在 `CompactSymmetryMetadata::xySymmetryCount()/zBranchCount()` 中是硬编码。
- 磁盘上没有任何 CTAC 参考结果或 μ 图；删除 CSRB 设备路径前必须先冻结一份 CSRB-CTAC 基线。

痛点：格式与内核硬耦合。CSRQ v1 的 run 字宽（6 bit）、u16 sqrt 解码、K=8 同时固定在转换器、`readChunkCSRQHeader`、`CompactRunU16View`、6 个内核、启动器参数表、状态类上传形状、测试写入器七处；任何新格式都要沿"loader → view → 驱动分叉 → 状态类 → 内核 → 测试"整条链再走一遍。

## 2. 目标与非目标

目标：

1. 设备端只保留**一条执行路径**（CSRQ 家族），CSRB 退回为规范的离线归档格式，只被转换器和参考工具读取。
2. 重建循环只依赖一个**抽象算子接口**；磁盘格式的变化止步于 loader/解码器和转换器。
3. 预期的格式变化——压缩角度（对称模式 → 分支数 K）与数据类型（值编码）——各自只触碰一处枢轴（模板参数 / 枚举），不新增分支。
4. CTAC、背景项、双能窗、多床、常驻 MLEM、加权灵敏度在同一条路径上全部可用。
5. 效率不退：接口粒度放在"整遍"级别（一次 forward/backward 跨所有 chunk 和所有床），热点全部留在内核。

非目标：不改变 MLEM/BSREM 算法、PSF、输出滤波、多床拼接；不重新设计 CSRQ v1 文件格式（v1 继续被读取；新编码通过 header 的 `value_encoding`/`run_len_bits` 字段扩展）。

## 3. 目标架构

```
io/
  ChunkDirectory            发现 chunk_*.csrb/csrq、metadata（88/92 B）、逐块读取、校验
  decoders/                 每种磁盘编码一个 decoder → DeviceMatrixLayout（内核消费的布局）
cuda/
  DeviceSystemMatrix        抽象算子接口（下文）
  CsrqMatrix                唯一实现：常驻 | 流式（pinned + 双槽），持有 K 与值编码
  kernels/                  forward / backward / ratio / pack，模板参数 <K, ValueEncoding>
                            backward 内部可选 (angle, z 切片) tile 版本（P3 后继）
                            CTAC：att[fullAngle(row, b)][M_b(col)] 查表融合进 nnz 循环
recon/
  与算子无关的循环：通道 × 床、联合 MLEM/BSREM、灵敏度/denom 缓存、诊断
```

接口（整遍粒度，参数均为设备端已就位的缓冲）：

```cpp
class DeviceSystemMatrix {
public:
  // 上传/固定；resident 或 streamed 由实现决定（或被强制）
  virtual void load(const ChunkDirectory&, ResidencyPolicy) = 0;
  // 所有床、K 个分支：yhat[bed][row][k]
  virtual void forward(const BranchBatchInput&, int bedCount, const AttenuationTables* = nullptr) = 0;
  // yhat -> weight*p/(weight*yhat+b) 就地；bed 范围
  virtual void ratio(const ProjectionSlots&, double weight, double eps) = 0;
  // 累加 weight*Aᵀratio 到每床双精度累加器（含对称变换）
  virtual void backwardAccumulate(Accumulators&, const AttenuationTables* = nullptr) = 0;
  virtual void sensitivity(Accumulators&, double weight, const AttenuationTables* = nullptr) = 0;
  int branchCount() const;   // K
  int cols() const; size_t rows() const;
};
```

驱动层现有的 `CompactMatrixChannel` 保留（通道 = 矿阵 + 权重 + 投影/背景），`runCompactChunkMlemNumerMultiBedCUDA(channels…)` 变成对接口的纯编排；所有 `format()` 分叉删除。

值编码与 K 的落点：

- 内核以 `template <int K, class Value>` 实例化进同一个 fatbin：K ∈ {2, 4, 8, 16}（`xySymmetryCount × zBranchCount`），Value ∈ {u16_sqrt（v1）, u8_sqrt, fp16, fp32}。启动器按 metadata 的对称字段和 header 的 `value_encoding` 选函数名（如 `csrq_forward_k8_u16`）。`run_len_bits` 作为运行时参数（一次位移，无成本）。
- `[voxel][K]` 打包输入：K=4 一次 float4，K=8 两次（32 B sector，现状），K=16 四次。
- `CompactSymmetryMetadata::xySymmetryCount()/zBranchCount()` 改为由 `sym_code`/`use_z_symmetry` 推导，不再硬编码。

## 4. 预期格式变化 → 触碰点映射

| 变化 | 现状触碰点 | 目标架构触碰点 |
|---|---|---|
| 压缩角度（sym90→sym45 / 取消 z 对称，K 8→16/4） | 硬编码 K=8 的 6 个内核、pack 布局、状态类校验、驱动的 `branches.size()` 检查 | 元数据推导 K；启动器选 `<K>` 实例；其余不动 |
| 值类型（u16 sqrt → u8 sqrt / fp16 / fp32） | 转换器、header 校验、`CompactRunU16View::valuesQ` 类型、内核解码、上传形状、测试写入器 | 转换器加模式；decoder 按 `value_encoding` 产出对应 `Value` 数组；内核 `<Value>` 实例；header 校验改为枚举分派 |
| run 字宽（6 bit → 5/7/8） | header 拒绝非 6 | 运行时参数 `runLenBits` 传入内核 |
| metadata 布局（88/92 B） | 已兼容 | 不变 |
| chunk 数 / 角度分块 | 已由目录层处理 | 不变 |
| 全新布局（如按线分组） | 新格式 + 新路径 | 新 decoder 解到同一设备布局；若需新内核则作为 `CsrqMatrix` 内部策略，不新增顶层路径 |

## 5. 删除与保留清单

删除（阶段 3）：

- 状态类：cuSPARSE compact 执行（`executeCompactDeviceRatio*`、`executeCompactBedBranchBatch*`、`executeCompactSensitivityBranchBatch*`、`executeCompactDeviceBackward*`、`executeCompactAttenSpMVFloat`、`executeSpMVFloat`）；`CompactTransposeMode` 全部变体与 csrt 主机转置缓存；`DeviceChunkCache`、`ResidentMatrixSetCache`、transient range / streaming range / subset / atten 缓存；packed-K forward 与 `compactPackedForwardKMem/Tile`；全矿阵 subset 路径（已死代码）；`PinnedHostCSRData`、`CudaCsrBuilder` 中仅为 CSRB 服务的部分。
- 驱动：CSRB 分支（host ratio、device ratio、branch-batch、per-branch、baked CTAC）、`applyCompactCsrLocalityHint`、`warmCompactChunkResidentDeviceCache`、`prefetchNextCompactChunk`、`kCompactChunkResidentReserveBytes`。
- 环境变量分叉：`SPECT_COMPACT_T_SPMM`、`SPECT_COMPACT_BRANCH_BATCH`、`SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES`（mmap 只服务 CSRB）；主机 chunk 缓存预算逻辑（CSRQ 目录已为 0）。
- 工具：`spect_transpose_bench`；`tools/search_compact_perf.py` 的 mmap/branch-batch 变体。
- 文档：README 环境变量表、CONSTRAINTS 中的 packed-K/CSR_ALG2 保护基线描述改写为 CSRQ 基线。

保留：

- `io/CompactChunkCSR` 的 CSRB **读取**（转换器的对照、测试的主机参考、`analyze_csrb_structure.py`），以及 CSRB→CSRQ 转换器（Python）。
- 公共 API 中与算子无关的部分：`refreshCompactProjectionCUDA`、`prepareCompactBranchBatchInputFromGlobalCUDA`、`finishCompactResidentMlemIterationCUDA`、`downloadCompactResidentImageCUDA`、`compactDeviceBackwardBegin/DownloadCUDA`、`releaseCompactExecutionCUDA`、计时快照。
- 常驻 MLEM 更新、PSF 卷积、体素变换等 `volume_ops` 内核。

## 6. 分阶段执行步骤与验证门

每个阶段结束都要过三道门：`spect_tests` 全绿；同输入主体积 relL2 ≤ 1e-5（对阶段 0 冻结的基线）；`iteration_core_s` 不劣于阶段 0 记录值的 5%。每阶段独立可回滚（按阶段提交）。

### 阶段 0 — 冻结基线（单矿阵即可，GPU 被共享时也能做）

- Tc99m CSRQ：Derenzo 182222 单床 MLEM 200:50（已有 `recon_out/csrq_resident_20260903/derenzo_csrq_bm`，it200 vs fp32 CSRB relL2 1.47e-6）；mouse_ch6o6p2 五床 MLEM 50（已有，relL2 6.2e-7）。重构后以这两组为对照。
- CSRB-CTAC 基线：用 `-ctacAutoMu <早期重建体积>` 在 Derenzo 单床上跑 CSRB CTAC MLEM 50，输出与 μ 图一并冻结到 `recon_out/refactor_baseline_20260903/`。这是阶段 2 CTAC 迁移的唯一对照，必须在删除 CSRB 设备路径前完成。
- 记录三者的 `iteration_core_s` 与 `[PROFILE]`。

### 阶段 1 — 立接口、包旧路径（无行为变化）

- 新建 `sparse/DeviceSystemMatrix.h`（接口）与 `CsrqMatrix`（把状态类 CSRQ 段搬入，含常驻/流式、chunk 迭代、计时）；CSRB 路径临时包成 `CusparseCsrMatrix` 只为让驱动统一调用。
- 驱动 `runCompactChunkMlemNumerMultiBedCUDA(channels…)`、灵敏度、前投影全部改为对接口编排；`format()` 分叉消失；lease/slot 规划迁入 `CsrqMatrix` 工厂。
- 门：阶段 0 三组 relL2 ≤ 1e-5，计时不变。

### 阶段 2 — 内核泛化 + CTAC 迁入 CSRQ

- 内核模板化 `<K, Value>`，先实例化 K=8 × {u16_sqrt}，再加 u8_sqrt/fp32 实例与 decoder；元数据推导 K；单元测试增加 K=4 合成矿阵（无 z 对称）与 u8 编码。
- CTAC：forward 的 `v` 乘 `att[fullAngle(row,b)*nvox + M_b(col)]`（每分支一次 u16 查表），backward 同；灵敏度按床分别调用。以合成矿阵对照 CSRB-CTAC 设备路径（此时仍在），再用阶段 0 的 Derenzo CTAC 基线做端到端 relL2 ≤ 1e-5 门。
- 门：新增测试全绿；CTAC 端到端过门；非 CTAC 路径计时不变。

### 阶段 3 — 删除 CSRB 设备路径与实验分支

- 按第 5 节清单删除；`CudaSparseOpsState.h` 按组件拆为 `CsrqMatrix.h`、`CompactBatchBuffers.h`（分支批输入、投影/背景槽、累加器）、`ResidentMlemUpdate.h`；`SparseOpsCUDA.cpp` 只剩转发。
- 测试中依赖 CSRB 设备执行做参考的用例改为主机参考（toy/合成矿阵可直接用 CPU SpMV），CSRB 文件读取保留。
- 更新 CONSTRAINTS：受保护基线改为阶段 0 的 CSRQ 结果；删除 packed-K/CSR_ALG2/mmap/三驻留条目；README 环境变量表相应缩减。
- 门：全量测试；阶段 0 三组 relL2；计时。

### 阶段 4 — 效率杠杆（在单路径上做，不再是分支）

- (angle, z 切片) tile backward（P3 后继）：预热时建 (row, z) run 索引；全切片 fp32 tile（opt-in smem）；4 个 xy 分支累加进未旋转累加器；z 镜像第二遍；替换 `csrq_backward_k8` + transform-add。预期 backward 150 → 60–70 ms/遍。
- 去掉窗与床之间的 `cudaStreamSynchronize`，只在一遍结束同步；可选：流式矿阵的上传流与计算流事件链复用。
- 门：relL2 ≤ 1e-5；单床/五床/双窗计时对比记录。

## 7. 风险与对策

- **CTAC 数值等价**：CSRB 用 cuSPARSE fp32 累加，CSRQ 用 fp32 原子 + 每 chunk 双精度累加；差异预期 ~1e-6（与今天单窗一致）。若超门，先在合成矿阵上二分定位。
- **性能回退**：接口本身零成本；风险在 CTAC 查表让 forward/backward 变慢（每 nnz 多 K 次 u16 随机读）。CTAC 仅在启用时付费，非 CTAC 路径不受影响。
- **删除面过大**：阶段 3 单独提交，阶段 1/2 已把行为固定在接口后，回滚只需恢复一次提交。
- **GPU 共享导致显存判定不稳**：`-csrqResidency streamed` 已可强制；阶段 1 把规划逻辑收进 `CsrqMatrix` 工厂后行为不变。
- **基线数据缺失**：Sampling_Approach 门已由用户放弃；CTAC 基线用 `-ctacAutoMu` 自造，不依赖外部 CT。

## 8. 当前可执行的测试（单矿阵，GPU 被共享期间）

1. 阶段 0：Tc99m CSRQ 单床 Derenzo 20 迭代快速回归（对照 `probe_kpass2_fast`），确认 chunk-outer 重构后的数值/计时；CSRB-CTAC `-ctacAutoMu` 基线各 50 迭代（CSRB 路径需要约 9 GB 显存 + 6 GiB 主机缓存，与另一个 GPU 任务共存时若显存不足则等其结束）。
2. 阶段 1 开工后的每一步都用同一条 Derenzo 单床命令回归。
3. 双矿阵（208 常驻 + 113 流式）的 auto 计时等 GPU 空闲再补。
