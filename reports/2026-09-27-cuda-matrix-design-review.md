# SPECT-recon CUDA 与系统矩阵设计审查

日期：2026-09-27。初次审查源码：`b206e14`，开始时工作树干净。第1–5节为只读审查，范围为 multipinhole 实际入口、矩阵读者、子集、CUDA 投影/更新及缓存。第6节起记录随后获授权的修改与回归；发布包和原始数据始终不变。

问题：在保持当前数值合同与 CSRB 生产地位的前提下，怎样以最小改动减少重复工作，并兼容不同矩阵编码和对称性？证据为源码追踪、既有测试及极小畸形输入复现；不运行大矩阵性能实验，不给未经测量的提速承诺。停止点为列明问题、最小修正、验证条件及不应开展的重构。

## 1. 优先发现

### P1：CSRB 权值缺少入口合法性检查，strict 也不检查

`multipinhole/src/io/CompactChunkCSR.cpp:572` 的整块读取、`:598` 的分段读取、`:651` 的映射读取都只调用列索引检查；没有权值有限性/非负检查。相对地，CSRQ 在 `:764` 检查 rowMax 非负且有限。

直接编译当前 CompactChunkCSR.cpp 并设置 `SPECT_CHUNK_VALIDATE=strict`，四个单非零元输入结果如下：

| 权值 | 读者结果 |
|---|---|
| 1 | 接受 |
| -1 | 接受 |
| NaN | 接受 |
| +Inf | 接受 |

这是已复现的读者边界缺陷，不表示已证明现有生产矩阵损坏。后续分母/更新中的非有限检测不能替代入口拒绝：负值可能相互抵消而不立即触发，错误也失去 chunk/nnz 定位。

最小修正：在现有 payload 校验处统一检查实际读入的权值，报路径与 nnz 位置；覆盖整块和 subset。新外部矩阵首次接入必须全量校验，再复用已确认身份。当前列索引默认是 sample（`:148`），也不能把它当作完整有效性证明。不要为了每轮检查而反复遍历已确认且不可变的整矩阵。

验收：负值/NaN/Inf 拒绝、零和正值接受、subset 与 whole 一致；合法矩阵输出不变。风险为首次加载成本，应记录在 prepare 而非藏入迭代。

### P2：双窗 CSRB 重复上传不变的投影，已有通道槽位没有被完整利用

`CompactSymmetricReconLoop.cpp:1269` 已按 channel×bed 预上传全部投影。随后 `CompactSymmetricProjectorCUDA.cpp:1800` 在每个子集每次切窗时仍向 per-bed 槽 refresh。`CudaSparseOpsState.h:2454` 主动清空指针缓存，再同步上传；`:1247` 还会重新扫描零 bin 比例。

例如 PSC 五床、双窗、50×N10，投影本身的迭代上传量按调用逻辑为：`360×2304×8×5×2×50×10 = 33,177,600,000 bytes = 30.899 GiB`，不含背景及启动上传。这是静态字节推算，不是实测 PCIe 流量或节省时间。

最小修正：CSRB 内部执行传入通道 projection-slot base，复用已经存在的 channel×bed 槽；accumulator 仍按 bed 累加。现有 CSRB slot 范围检查也要同步，不能只删除 refresh。保留每次新任务或显式输入变化的失效语义，不能退回“host 指针相同就认为内容相同”的旧错误。

验收：固定双窗任务进入迭代后投影 H2D 次数接近零；high/low 切换、同地址内容改变、background 有/无切换回归通过；同输入数值门和实际耗时分别报告。

### P2：ROI+PSF 退回 host 路径；满视野圆柱快速路径隐含矩阵支持区假设

`CompactSymmetricReconLoop.cpp:861` 建立局部支持区，`:870` 通过 ROI 尺寸判断是否裁剪矩阵；`:1488` 在 compactRoiPrunesMatrix 时禁用 resident input，导致图像下载、host bed 抽取/准备以及 host 更新。此处不表示所有 PSF 算子都在 CPU 执行，而是数据路径不能保持全驻留。

因此，小 ROI 虽减少矩阵工作量，却不自动意味着最快的执行路径。SC60 36×36mm 支持区相对 36×36×96mm 网格，以及此前 32.8×12mm 设置，都应检查日志的 input/update 路径，不能把旧耗时直接代表完全驻留 GPU 的上限。

另一个独立正确性边界：代码注释明确承认“满视野圆柱依靠矩阵自己的圆柱支持区”。矩形网格中有圆柱外角落；新格式/新 producer 若在角落有非零列，不能仅凭 ROI 外接尺寸认定无需 post-PSF mask。这不是已验证现有矩阵错误，而是未来兼容必须处理的显式假设。

最小方向：沿现有 GPU bed/PSF helper 加入同一局部 mask，严格保留前向 PSF 后、反向 PSF 前的掩膜顺序；若消除隐含假设，需要该路径也正确处理满视野圆柱。不要删除 mask，也不要把列裁剪偷偷改成更新后截断。

验收：现有 host 路径作为参考，覆盖非中心 ROI、边界穿越的 PSF、小 ROI 和圆柱外非零列；验证伴随关系、单床/多床同输入结果及内外拷贝次数。支持区改变属于算子改变，不是单纯显示设置。

### P2：低主存模式的预取仍有同步磁盘读取；逐块计时限制提交深度

`CompactSymmetricProjectorCUDA.cpp:861,877` 在提交当前 chunk 运算前直接调用 loadSubset 获取下一块；`CompactChunkCSR.cpp:998` 的未缓存读取发生在该调用线程。`CudaSparseOpsState.h:1055` 每个 CSRB chunk 又等待 transform 事件完成以收集计时。完整 host cache 的异步 H2D 仍然存在，不能说当前完全没有重叠；但部分 host cache 下，磁盘读下一块不能自然与当前块计算重叠。非完整缓存 payload 也不注册为 pinned（Projector `:430`）。

最小方向：先在真实部分缓存任务测磁盘读取、pageable staging、H2D 等待和计算的时间占比。只有读取确实占优时，才引入固定容量的一块前读和有限 pinned staging；只有提交/同步明显占优时，才考虑延迟读取计时事件。不能直接删 synchronize：当前设备槽重用、源数据生命周期和错误检查依赖这些完成边界，必须有 compute-done/upload-done 事件代替。

此项改动范围大于双窗投影复用；不优先引入 CUDA Graph、多生产线程或更深队列。单独的每子更新 flags D2H（State `:2192`）承担失败检测，不以省同步为由随意去掉。

### P2：分母缓存只使用文件 stat 身份，对多来源同名矩阵不够强

`CompactSymmetricReconLoop.cpp:278` 的文件身份由 basename、长度、mtime 构成；`:405` 用于全部 chunk，缓存命名又保留矩阵目录 basename。不同来源矩阵若同名、同长度且保留相同时间戳，其他配置相同，就无法凭此区分 payload，可能复用旧 sensitivity。

这是条件性风险，未声称现有缓存已经碰撞。最小方向是在矩阵导入/发布清单中一次性计算内容摘要，并将已验证的矩阵身份纳入分母键；不能在每次迭代重新 SHA 整矩阵，也不能以 SN 名称替代内容身份。缺少清单的 legacy 输入仍需明确的身份验证规则。

验收：同名同大小同 mtime、不同值的两个 toy 矩阵不能命中同一分母；相同矩阵仅迁移安装根路径可以按既有便携合同复用。

## 2. 第一性原理与应保留的主干

Poisson 重建的核心是一个非负线性响应算子及其反投，不是某一种文件后缀。抽象为每床每窗 `lambda = Kx + background`；OSEM 子集分子为各观测的 `K^T(y/lambda)` 之和，灵敏度为匹配算子下的 `K^T 1`。当前预处理校正、denominator-only 行权重和窗权重有自己的已冻结合同，不能在重构时机械改为另一套归一化公式。

必须保留：

- 前/反投使用相同矩阵、行选择、对称映射和支持区，PSF/CTAC 顺序一致；分母使用同一个有效算子。
- 子集有独立 S_m，且它们之和匹配 S。MLEM/OSEM 与 BSREM 更新不因共享基础计算就变成同一公式。
- 多床与对称分支复用上传的 chunk；不把 ROT90/ROT90_Z 展开成巨大全矩阵。
- 图像、分子、灵敏度、mask、正则梯度优先驻留 GPU；仅保存/必要检查下载。
- CSRB 为权威 float32 格式，CSRQ 为派生实验格式；两种编码可以有不同 CUDA 内核，但不能有不同的物理语义。

对单个 RHS，CSR float32 的每非零元至少读取 value4B+index4B、完成约2 FLOP，忽略其他访存的算术强度约0.25 FLOP/B。多分支复用能提高强度，但反投原子冲突、列局部性、输出清零、PCIe/磁盘流量都会改变实际瓶颈。因此，优先减少反复读取/上传和中间体积，不先追求 occupancy、Tensor Core 或最大批宽。

N10 是改变更新频率和收敛轨迹；每个完整 epoch 仍需覆盖全部子集。相同 epoch 数的总矩阵遍历量不会自动降到十分之一，100×N10 也不能直接当作与1000×N1数值等价。

当前 packed-K forward（1/2/4）以及按精确零 bin 比例选择 scatter-pair/cuSPARSE 反投应保留。源码 State `:1471` 留有宽 SpMM 负面测量；既有项目记录还包含 scatter 的同输入/重复性能证据。它们不是本次新测量，但足以反对未经新证据重开宽 SpMM、CSC 全转置、FP16、三块驻留或 CSRQ 全替代。

## 3. 多格式应怎样收敛

| 层次 | 当前支持 | 不应混淆的边界 |
|---|---|---|
| metadata 字节布局 | 88B、92B double voxel、124B SMMD v3 | 字节不同可以归一；不是不同重建算法 |
| 对称语义 | ROT90、ROT90_Z | 无对称、纯 Z、其他角度对称尚不支持；不能只加配置放行 |
| payload 编码 | CSRB float32、实验性 CSRQ run/u16-sqrt | CSRQ 当前仅 ROT90_Z；不应强制都展开 float32，也不强制量化 CSRB |
| 子集 | compact-angle interleaving | 当前 N 必须整除 compact angle count；绝对角度表必须保留 |
| 通道 | single 或双窗 joint，同任务编码相同 | 混合 CSRB/CSRQ 明确拒绝；当前没有实际需求，不急着实现 |
| 索引容量 | 文件 row_ptr i64，host u32，CUDA chunk/slice nnz≤INT32_MAX | 已有上传边界拒绝，不是 silent overflow；新 producer 应按执行上限分块 |
| CSRQ CTAC | N1 路径有检查 | CTAC+N>1 明确拒绝，不得宣传与 CSRB 能力完全相同 |

建议继续使用已有三层，不另建插件系统：

```text
各 metadata/payload 读者
        -> CompactSymmetryMetadata + chunk 信息 + 已确认的能力约束
        -> CSRB/cuSPARSE+现有核 或 CSRQ/专用核
        -> 共用 Poisson 数据项、OSEM/BSREM 调度、保存
```

协议至少清楚区分：文件版本/索引与值类型、投影 row/voxel column 编号、网格与单位、对称分支映射、支持区、矩阵内容身份、窗响应/权重归属。设备 SN 负责选择资源，不应进入每个 CUDA kernel 判断“PSC还是SC60”。现有 loader/目录/调度器分工已接近这个形态，优先补明确缺口而非重写一个 UniversalMatrixManager。

未来新对称性先取得 producer 与极小样本，证明展开后的映射、覆盖次数及伴随关系，再增加少数已知 profile 的特化。不要为了“通用”让每个非零元走解释型变换或构造巨型索引表。ROI、角度次序、镜像中心和插值如果不一样，属于算子差异，不能仅靠 metadata 版本归一掩盖。

最简维护上，`SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES` 仍可启动已拒绝的 mmap 路径；`SPECT_COMPACT_BRANCH_BATCH=0` 仍保留非默认执行路径。它们不是死代码，当前文档和历史实验工具有引用。后续可从正式运行时删除或隔离到实验构建，保留 Git/历史二进制可复现；本次不直接删，也不把 CUDA 数学参考测试误当成应删除的纯 CPU 后端。

## 4. 执行顺序与验收

1. 补矩阵合法性边界与内容身份，防止新格式“能读但语义不对”。这属于正确性工作，不要求先证明速度收益。
2. 修双窗静态投影槽复用，局部改动、观察次数/字节并做数值回归。
3. SC60 小 ROI+PSF 路径保持 GPU 驻留，保留严格 mask/PSF 顺序；对现有 host 数学路径做对照。
4. 只有真实部分缓存时间线证实瓶颈后，优化前读/锁页 staging/计时同步。对 GPU 全缓存、host 全缓存、部分缓存分别评价。
5. 最后清理已拒绝实验的运行时开关。不重写所有 kernel、不改默认迭代/PSF、不提升 CSRQ 为生产。

后续性能验收需冻结矩阵/输入/EXE/配置、明确 cache hit/miss、记录其他 CPU/GPU 负载，交错重复运行。同时报总墙钟、prepare、iteration、save 与应用读取/H2D 字节；不能用 GPU 利用率或单次冷启动代替吞吐。结构改动沿用对应既有同输入门；正则长期轨迹会放大原子噪声，不能把短迭代门机械套用所有正则长程结果。

## 5. 本次验证和证据

- 已有测试：`ctest --test-dir build/csrb-native-20260921 -C Release --output-on-failure -R "Compact|Csrq|CSRQ|Smmd|JointMetadata|RealSc60"`，66/66 通过，11.37秒。使用现有构建，本次未修改/重编生产 EXE；不能代替所有格式/实机/实采验收。
- 独立读者复现：`recon_out/architecture_review_20260927/probe_values.cpp`，直接用当前 CompactChunkCSR.cpp 编译，日志 `probe_values.log`。仅生成该目录内的4个极小派生文件，不触碰真实矩阵。
- 编译：在上述证据目录、VS x64 developer shell 中运行 `cl /nologo /EHsc /std:c++17 /MD /I D:\Projects\SPECT-recon\multipinhole\include probe_values.cpp D:\Projects\SPECT-recon\multipinhole\src\io\CompactChunkCSR.cpp /Fe:probe_values.exe`，再运行 `probe_values.exe`。
- 没有性能 A/B、没有 Nsight 时间线、没有新的画质/定量结论；缓存身份和圆柱支持区问题是源码可见的条件性风险，不是宣称已经在现有数据中复现的错误。
- NVIDIA 技能目录已检查，没有与本原生 cuSPARSE 审查足够匹配的专用项；未安装技能。核对了官方 CUDA 异步/锁页规则与 cuSPARSE 文档，不以厂商一般建议替代本项目的历史实验。

参考：[CUDA 异步执行与 pageable/pinned 行为](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)，[cuSPARSE 13.2](https://docs.nvidia.com/cuda/archive/13.2.0/cusparse/index.html)。转置路径/原子求和不能因选择 ALG2 就宣称逐位确定性；以项目数值容差和代表性输入验收。

## 6. 实施与验证合同（2026-09-27）

用户已授权按本审查修复、优化和回归。允许修改源码、测试及本文档；原矩阵、投影、历史结果和 SPECT_Core 发布目录不变。基线使用 `staging/SPECT_recon_b9f007c/bin/SPECT_recon.exe`，SHA256 `f847481048208bbaa153f0e1a30add2ffe63b33e53be84cb660d7d842356c9a7`。

- 正确性：所有现有测试，加非法权值、相同 stat 不同内容、静态双窗零重复上传、非对称局部支持区与 PSF 顺序测试。原有 CPU 数学参考保留。
- 首个实际性能对照：SC60 `matrix_roi32p8_h12`、`121314derenzo_1200s.bin`、20×N10、PSF sieve/castor 1.5 mm、32.8×12 mm ROI、8线程。先各一次暖机，随后 A/B/B/A/A/B 三对，独立分母缓存均命中，记录实际 host cache 覆盖率。
- 比较相同最终状态与输出，relL2≤1e-5；迭代效率独立于 prepare 和保存报告。新增性能复杂度的收益门沿用≥5%，不以冷启动或 GPU 利用率证明提速。
- 内容身份先使用每个任务准备期一次全文件 SHA256，不信任仅按 stat 复用的旁车摘要；自动缓存升级会换键重算分母，显式指定旧缓存则拒绝不匹配内容，要求调用方选择新路径。额外扫描成本明确计入 prepare。跨运行免扫描的可信导入/清单机制不是这一步已完成的能力。
- 部分缓存预取只在真实测量显示收益时改动；未达到证据门保留现有同步边界。CSRQ 仍实验性，不改变 CLI、默认配方或投影模型。
- 输出和运行记录放在 `recon_out/cuda_matrix_optimization_20260927`。本轮完成点为修改可编译、回归结果与重复计时落盘，并列出尚未关闭的验收项；失败立即停止后续批次并定位，不发布未经验证的候选。

## 7. 修改与首批结果

### 7.1 实现

1. CSRB 整块和 subset 读取始终检查所有读入权值，拒绝负数、NaN、Inf，允许零；subset 报原文件 nnz 索引。已缓存且未变的块不在迭代中重复校验。列索引 sample/strict 既有接口保留，首次外部矩阵仍需 `SPECT_CHUNK_VALIDATE=strict`；不能把 SHA256 当作数值合法性证明。
2. 分母缓存升级为 `compact_global_denom_v5_payload_sha256_local_support`。chunk 内容用 SHA256，而非大小/mtime 作为身份；Windows 使用原生 BCrypt，非 Windows 构建链接 OpenSSL Crypto（本轮仅验证 Windows）。新增 `[MATRIX_IDENTITY]` 单独记录摘要扫描字节和耗时，这些字节不重复计入矩阵 loader 的 payloadReadBytes。
3. CSRB 将 channel×kMaxBeds 传至现有投影/背景缓存，accumulator 仍按 bed；静态双窗进入 prepared 路径后投影 H2D 次数和字节均不再增加。显式 refresh 仍使同地址输入失效，不改变权重或窗模型。
4. 局部 mask 复用现有 CUDA 前向变换核和反向 double→float 核，无新增独立全体积 kernel；mask 随首次图像上传刷新，驻留迭代复用。小 ROI 与满视野圆柱均显式应用 mask，不再依赖矩阵自行消除圆柱外角落。保留 PSF 后前向 mask、PSF 前反向 mask 的原顺序。
5. 删除 mmap reader、映射 payload 状态、非合批 sensitivity/numerator 的正式调度及旧环境开关；设置退役开关时 CLI 明确报错。保留 CUDA 数学参考接口与历史冻结 EXE，不改 CSRB/CSRQ 支持边界。实验搜索工具不再默认运行已退役变体。

### 7.2 SC60 重复对比

三对 A/B 顺序为 A/B/B/A/A/B；每次 20×N10、sieve/castor 1.5 mm、ROI 32.8×12 mm，最终保存相同完整矩阵网格。矩阵 18 chunks、26,960,069,784 文件字节；两组均自动完整 host cache，注册约 25,711 MiB，非 GPU 全驻留。分母均命中各自缓存。以下为三次中位数，秒：

| 指标 | 基线 A | 优化 B | 解释 |
|---|---:|---:|---|
| 20轮 `[ITER_TIME]` 合计 | 41.419 | 23.558 | 完整迭代墙钟减少 43.1%，含输入准备与更新 |
| `iteration_core_s` | 29.480 | 23.426 | 原有核心计时减少 20.5%；它不包含全部 host 输入准备 |
| `denom_and_matrix_prepare_s` | 31.587 | 72.228 | 新增全文件摘要校验增加准备时间 |
| EXE 进程墙钟 | 75.481 | 98.150 | 本次短任务总耗时增加 30.0%，不能称端到端提速 |
| 最终图像 relL2（各对最大） | — | 1.18001e-7 | 小于预设 1e-5，mask 输出完全相同 |

两组在迭代内 loader 读取字节均为 0；20轮矩阵 H2D 均为 539,135,041,920 字节，迭代收益不是减少矩阵遍历。候选路径为 input=device、update=device。每图保存时间约为 `full_command_s - load_to_iteration_done_s`，本批约 0.15–0.22 秒，不能用进程墙钟减 core 推断保存成本。

这些是在实际共享机器上交错重复的结果，不是独占空闲主机的最高吞吐。10秒采样中其他进程 CPU 占用估算峰值约 15.8–20.8 个百分点，已保留逐样本记录；没有将 GPU 利用率作为验收指标。不能直接把20轮结果换算成已经实测100轮或长期正则轨迹。

初测候选冻结为 `recon_out/cuda_matrix_optimization_20260927/SPECT_recon_measured.exe`，确切 SHA256 见同目录 `provenance.json`。这是二进制证据，不是独立发布包，运行依赖对应 build/bin 的 DLL。随后删除非默认实验路径、增加摘要耗时日志，最终构建的额外20轮为 core 23.646秒、进程97.137秒，与初测候选输出 relL2=6.58878e-8；记录为 `final_checks.json`。初测和最终二进制不得混称同一 SHA。

### 7.3 最终验证

- Release 编译成功，架构 sm_86/89/120；实测设备 RTX 4070 Ti SUPER。已有 LNK4098 运行库警告仍在，未将它描述为无警告构建。
- 全套 CTest：322项，318通过、4项既有条件跳过，0失败，48.34秒。随后仅补同地址投影修改、背景移除/恢复、局部 mask 开/关回归，重编 spect_tests，受影响的12项全部通过，2.42秒。未变的 EXE 不因测试源码修改重跑完整性能批次。
- 合成双窗两床 CLI：合法整数床位通过，未知床位与非整数层步进拒绝；完整输出网格与已知体素值符合，最大 relL2=6.58956e-8。不是 SC60 真实多床采集验收。
- 真实 CLI 缓存碰撞：相同 basename/长度/mtime、不同权值的矩阵拒绝复用显式旧缓存；使用独立缓存重新计算成功；相同内容迁移根路径后命中。退役环境开关两项均明确报错。
- `tools/search_compact_perf.py --self-check` 通过；迁移时发现 self-check 仍引用 mmap 旧变体，已一并修正。保留旧日志解析能力仅供历史记录使用。
- 最终 EXE：`build/csrb-native-20260921/bin/Release/SPECT_recon.exe`，SHA256 `1ee973137e50524a7236f89152ac78cd7ec73950583201ebb97873bedbfc1b80`。未提交 Git、未部署或替换 SPECT_Core，CLI 和默认配方不变。

### 7.4 PSC 控制与部分缓存诊断

PSC 使用既有 `matrix_PSC_PIN26x28x20_208keV_120x120x220_ZFOLD` 和 mouse_normal_lu 的5床历史合并投影、MLEM50×N1、sieve/castor 1.3 mm、16.8 mm步进与既有输出滤波。它是冻结的性能/数值控制，不是把 Lu177 双窗生产方案改为合并窗。

| PSC 50轮单次冷缓存配对 | 基线 | 最终候选 |
|---|---:|---:|
| core（秒） | 74.435 | 73.522 |
| prepare（秒） | 61.029 | 117.320 |
| 进程墙钟（秒） | 141.221 | 197.217 |
| 最终原始重建 relL2 | — | 9.22944e-8 |
| 最终 Butterworth 输出 relL2 | — | 1.38059e-7 |

两组分母均 miss，迭代内 loader 读取均为0。仅一次冷缓存配对，用于回归，不是独立的速度收益结论，也不代替既有要求 cache hit 的 MLEM50 性能门。候选完整 SHA256 扫描35,199,642,800字节耗时58.524秒，已显式记录。

SC60 8 GiB host cache **预算**诊断：1×N10、分母 hit、完整迭代墙钟33.554秒、core33.523秒，迭代内 loader 读取23,992,347,768字节，矩阵 H2D26,956,752,096字节。与完整缓存同轮输出 relL2=4.73113e-8。预算不等于实际装入8 GiB，也不等于磁盘物理读取量；这是一次诊断样本，不作重复性能验收。读取、pageable staging、H2D 等待尚未被时间线充分分离，未引入额外前读线程或删除现有同步点。

### 7.5 结论与剩余边界

已完成：权值入口修复、内容身份保护、静态双窗槽复用、ROI+PSF 驻留及旧实验入口清理。数值回归通过；SC60 完整缓存迭代墙钟有本机重复收益，默认数学路径未改算法、精度、迭代数或窗权重。

尚未闭环：**每任务全量 SHA256 扫描的准备成本**。当前保证内容身份正确，但本次20轮短任务整体变慢；后续可信导入/部署清单必须有可靠的验证绑定，不能为省这一步退回 basename/size/mtime 信任。还未完成部分缓存异步前读/锁页 staging 的受控候选比较、实际双 CSRB 大矩阵的重复性能验收、长期 BSREM/RDP 及新 SC60 实采多床验收。CSRQ 不提升为生产，新对称性也未放行。

证据均在 `recon_out/cuda_matrix_optimization_20260927`：`summary.json`、`final_checks.json`、`psc_checks.json`、`final_cache_cli.json`、`final_ctest.log`、`cache_refresh_tests.log`、`provenance.json`、`final_input_identity.json`、`implementation.patch`。权威缓存用例为 `cache_collision_v3`；前两次脚本试跑分别暴露了夹具缺少 algoConf、以及误将显式缓存不匹配预期为自动 miss 的测试假设，已修正，不计为产品回归失败。

复现使用本轮临时驱动 `compare.py`（暖机及三对SC60）、`verify_final.py`（最终CLI/缓存、SC60桥接/部分缓存及PSC控制）；它们不是新增正式维护入口，均拒绝覆盖既有证据目录。常规代码回归仍使用现有命令：

```powershell
cmake --build build/csrb-native-20260921 --config Release --target SPECT_recon spect_tests --parallel 6
ctest --test-dir build/csrb-native-20260921 -C Release --output-on-failure
python tools/search_compact_perf.py --self-check
```

## 8. 2026-10-08 统一与内存效率优化（代码候选完成，GPU 验证待续）

本节记录首轮候选。其第 1 项 SHA 方案已被随后用户明确的核心职责边界取代，当前实现见第 9 节；其他三项仍待 GPU 验证。历史测量与冻结 EXE 不改写。

用户授权四项优化，并要求减少专用内核维护、优先融合。范围：矩阵准备、分块读取与传输、缓冲生命周期、实验性 CSRQ ROT90；不改变 CLI、默认配方、PSF/ROI/校正顺序、精度、原始矩阵或发布包。源码基线为 b206e14 加本节之前的未提交修改。冻结 EXE：`recon_out/unified_cuda_20261008/SPECT_recon.baseline.exe`，SHA256 `fa37676edb7b15df69d423e6443ffc4596a6e31328c0a287772a23daee44c3cc`。

问题与端点：保持内容校验时减少重复 I/O；先分离部分缓存的载荷读取与 CUDA 等待，再验证有界前读；共用 4/8 分支实现，不复制 SC60 内核。CSRQ 仍为实验格式，文件量化误差不替代重建误差。优先将 SHA 与可缓存块首次读取合并，不以未经可靠绑定的旁置摘要信任任意可变文件。

验证契约：现有测试加格式/对称/缓存失效边界；SC60 ROI 32.8×12 mm、N10、PSF1.5 完整缓存重复配对，并记录部分缓存；PSC 208 五床 MLEM50 N1 使用既有性能门。原生路径 primary relL2 <=1e-5；历史 host-pin 路径保持原保护条件；CSRQ 先与独立解码 CSRB 比较算子，再运行真实输入，分别报告量化与结构变化。性能复杂度采用原约束 >=5% core 收益门，纯内存/结构收敛单独说明。数据、配方、分母命中、设备和干扰均记录。任何数值失败停止该候选并诊断，不部署、不自动提交。

### 8.1 本轮代码范围

1. **准备期读取与摘要合并。** 可进入主机缓存的 CSRB/CSRQ 块，在首次完整读取时同步计算 SHA256，覆盖原始头部、载荷和 CSRQ padding；分母缓存继续使用 v5 内容身份。无缓存或剩余预算不足仍独立扫描，不信任大小/mtime 跨任务复用摘要。已消除可缓存块的重复读取，但未实现跨进程可信矩阵清单，也未消除每任务摘要计算。
2. **部分缓存有界前读。** 原生 CSRB numerator 循环使用一个 host future 读取下一物理块的 subset；I/O worker 不调用 CUDA，异常在消费结果时传播。完整缓存路径保留既有预取。新增 `loader_work_s` 是可与 GPU 重叠的累计 loader 工作时间，不是可与 core 相加的墙钟阶段。未新增锁页 staging、第三个 GPU slot，也未删除现有 CUDA 同步边界；前读收益尚未实测。
3. **CSRQ 工作缓冲按实际需求分配。** X/Y 按实际 K 分支数分配，ratio 由全矩阵行数缩为各 channel 的最大 chunk 行数。streamed slot 预算使用各数组最大尺寸之和，避免不同块分别取得最大值时低估。日志新增 K 和 scratch 字节；这是分配量记录，尚无真实峰值 RAM/VRAM 测量。
4. **CSRQ ROT90/ROT90_Z 共用 K=4/8 实现。** 分支数从已校验 metadata 贯穿目录、驻留预算、launcher 与设备布局；SC60 不新增一套专用 kernel。保留前向投影与 Poisson ratio 的融合，普通 pack 合并到现有带可选衰减的 pack kernel，删除重复普通 pack kernel。既有两种 backward 布局保留，不宣称已合并所有算子。CSRQ 格式和量化规则未变，仍为实验候选。

### 8.2 已完成的非 GPU 验证

用户最新指示为“暂时继续其他实验，本轮先完成代码和非 GPU 检查”。已停止本轮等待中的配对驱动，没有停止、修改其他 GPU 实验，也不自动排队新的 GPU 批次。

- Release EXE 和测试目标编译通过，fatbin 覆盖 `sm_86 / sm_89 / sm_120`；既有 C4244/LNK4098 警告未作无关修复。
- **169/169 非 GPU 测试通过，0 失败、0 跳过**，XML 记录 0.149 秒。覆盖 CLI、IO、subset、配置、manifest 等；新测试证明融合 SHA 等于独立全文件 SHA、同大小/mtime 的新任务内容变更可被识别、CSRQ padding 也参与身份计算。
- K=4/8 参数化 CUDA 测试已编译，包含算子、joint subset、CTAC/伴随和驻留 BSREM；本轮未执行它们。已静态核对 driver API 参数数组与设备签名，但这不替代 GPU 执行。
- 候选 EXE SHA256：`1bf33b0363e29f1b93677f08c694e36dfe9c57eec3a91e59622d6796d440f951`；测试 EXE SHA256：`26ce099f1862740df8023e3fdfe8298f34a0ad13e9a1838eafcdc20c97d9187e`。
- 证据：`recon_out/unified_cuda_20261008/build.log`、`non_gpu_tests.log`、`non_gpu_tests.xml`；`implementation.patch` 是相对 b206e14 的完整 tracked 工作树快照，包含本轮之前的已有修改，不是仅本轮差异。`git diff --check` 通过。首次基线性能试跑因既有 GPU ownership lock 退出；重试驱动在等待阶段按用户要求停止，无成功 CUDA 数值或性能样本。

```powershell
cmake --build build/csrb-native-20260921 --config Release --target SPECT_recon spect_tests --parallel 6
build/csrb-native-20260921/bin/Release/spect_tests.exe --gtest_filter="CompactSubsets.*:CsrqCompactMatrixFormat.*:IOTest.*:CLITest.*:SieveParserTest.*:BedOffsetsTest.*:ReconSetupTest.*:RunManifestTest.*-IOTest.Real*" --gtest_output=xml:recon_out/unified_cuda_20261008/non_gpu_tests.xml
```

### 8.3 续跑与采用边界

当前完成的是**代码候选、编译和非 GPU 检查**，不是 CUDA 正确性或性能验收。原始数据、矩阵、冻结基线和已部署 SPECT_Core 均不变；本轮不提交、不发布、不替换生产包。

待用户允许 GPU 时段后，先运行新旧 K=4/8、驻留/流式、joint、CTAC 和 BSREM 回归，再进行完整缓存与部分缓存的冻结 A/B 比较及实际内存记录。部分前读只有满足本节预先约定的性能与数值门才能被采用；失败时定位该机制，不改门限。SC60 CSRQ 还需真实输入对照，不能用文件量化误差代替重建验收。

临时驱动 `recon_out/unified_cuda_20261008/compare.py` 保留续跑参数与拒绝覆盖逻辑，不是正式维护入口。续跑应使用新的 phase 目录并核对候选 SHA；不能覆盖已中断的 `identity` / `identity_retry`，也不能把第 7 节历史性能收益归给本轮代码。

## 9. 2026-10-08 移除运行期矩阵 SHA

**用户决定与范围：**重建核心验证当前传入的投影 BIN、系统矩阵及参数能否配合计算，遵守支持的 CSRB/CSRQ、metadata、几何、索引和数值规范；不认证矩阵的历史版本、设备标定来源。系统/核素与实际路径由 wrapper/调用方选择，不要求历史矩阵库或不可变版本目录。这取代第 6/8 节每任务全文件 SHA 及“可信导入注册”方向，但不取消离线实验和发布证据的核验。仍不运行 GPU、不修改原始输入、不发布。

### 9.1 代码与接口

- 删除全矩阵 SHA 实现、读取时附带摘要、摘要字段和 BCrypt/OpenSSL 链接依赖；没有换成另一种全载荷哈希，也没有新增开关或追溯文件。CSRB/CSRQ 原有格式/范围/权值检查保留。CSRB 列索引仍按已有 strict/sample/off 策略检查，不能把默认 sample 描述成完整内容认证。
- 保留主机/GPU 缓存和跨任务分母缓存，不为取消摘要而强制每任务重算分母。分母协议改为 `compact_global_denom_v6_path_stat_local_support`，按本次规范化完整路径、文件大小/修改时间、已解析 chunk 布局/metadata 和计算参数分区；只读取轻量文件属性，不扫描载荷。小型 manifest 文件名继续用既有 FNV，不是矩阵内容摘要。
- 日志改为 `[MATRIX_CACHE] key=path_stat payload_scan=none`。目录 I/O 层仍校验本次输入，cache 信息生成不会提前把矩阵装入 host cache；因此后续性能比较必须分别记录准备和首轮读取，不能只比较某一阶段就宣布总耗时收益。
- CLI、配置值、CSRB/CSRQ 文件版本及计算公式不变。正常属性变化会产生新自动缓存；旧 v3/v4/v5 自动缓存一次性 miss。显式旧缓存仍按既有规则拒绝 manifest 不匹配，由调用方换缓存路径或清除旧缓存。

**明确的保证边界：**路径/大小/mtime 不是内容身份。若调用方保留这些属性原地换矩阵，须一并使旧分母缓存失效；核心不负责发现这种历史变化。运行中的输入不得改写。仅凭文件格式有效和几何相容，不能证明矩阵标定、能窗对应或物理模型正确。

### 9.2 验证与证据

- Release EXE 与 spect_tests 编译通过；既有 LNK4098 警告保留。新 EXE：`build/csrb-native-20260921/bin/Release/SPECT_recon.exe`，构建证据 SHA256 `5bb8b7e4faf3a8bf2b815077036077059113ecbf7556d78d780550127ced0326`。这是小型发布二进制证据，不是重建时的矩阵摘要。
- **169/169 非 GPU 测试通过，0 失败/0 跳过**。新增/替换测试验证：相同 basename/属性但不同完整路径区分缓存；规范化等价路径复用；mtime 变化失效；缓存信息读取不加载 payload；文件丢失/尺寸变化失败；同 stat 的有效新 payload 可正常读入且不被当作历史身份异常；CSRQ padding 不影响解码值。已有非法权值、截断、混合格式、CLI/几何等检查保持通过。
- 全部运行期源码/头文件/CMake 不再包含矩阵 SHA 或加密库调用。原来的内容身份测试按新合同替换，不把其原有结论继续套用 v6。
- 证据目录：`recon_out/unified_cuda_20261008/no_payload_sha/`，含 `build.log`、`non_gpu_tests.log`、`non_gpu_tests.xml`。修改前 EXE 冻结为 `SPECT_recon.before.exe`，SHA256 `1bf33b0363e29f1b93677f08c694e36dfe9c57eec3a91e59622d6796d440f951`。`implementation.patch` 是包含此前已有修改的完整 tracked 工作树快照，不是仅本次差异。

非 GPU 复现沿用第 8.2 节命令，将 XML 输出目录改为本节证据目录。尚未进行 GPU 回归、实际缓存 miss/hit 重建或重复计时；续跑时先为候选预热独立 v6 缓存，不能把旧 v5 缓存直接作为候选 cache-hit 基线。已部署 SPECT_Core 不变，无提交、无 GPU 任务排队。

## 10. 2026-10-08 最小职责审查与收敛（非 GPU 验证完成）

用户要求在第 9 节职责边界下继续优化并审查，避免未经需求证明的机制。范围限定为当前 compact 输入读取、缓存消费、CSRB/CSRQ 预热调度和显存分配公式；不改物理模型、CLI、矩阵格式、量化、更新顺序和 CUDA 内核。沿用先前非 GPU 限制，其他实验不受影响。基线为第 9 节 EXE 和本轮前脏工作树，原始矩阵/投影/发布目录不变。

完成点为 Release 编译、非 GPU 回归和预热脚本自检通过，以及列出未运行的 CUDA 验证。不以代码行数或单次运行声称提速，不继续增加未经测量的内核/异步机制。

### 10.1 审查发现与修复

| 优先级 | 原问题与影响 | 最小修复及证据 |
|---|---|---|
| P2 | CSRQ header.cols 被读出后丢弃，后续 view 直接使用 metadata.totalVoxels，列维数矛盾的输入也被接受 | 在共同头读取入口比较两者；新增测试先复现两个目录打开重载均未抛错，修复后均拒绝 |
| P2 | CSRQ 双窗规划以最大整块字节数估算流式 slot，实际分配却取各数组最大长度；不同块贡献不同最大值时预算偏小 | 复用现有 CsrqChunkShape，规划器和分配器共用 deviceBytes/includeDimensions，删除重复公式和维度结构；合成两块分别 392/320B，但 slot 实际需 544B，CPU 测试覆盖该反例 |
| P2 | denom 预热工具把 auto_x.manifest.txt 错映射为 auto_x.manifest.mhd，--force 没有清除真实缓存；缺少旧 chunk_dir 字段时筛选又过宽 | 修正双后缀映射，按当前 chunk_0_path 或旧显式 chunk_dir 匹配矩阵目录与床数；排除 joint/CTAC 和无法绑定路径的旧缓存。临时文件自检覆盖应删三件套与应保留的其他缓存 |
| P2 | host cache 预算解析忽略尾随垃圾；部分 NaN 文本可进入浮点转整数，负值/错误值可能静默变成 auto | 同一解析函数检查完整消费、有限非负及合法单位；保留 auto、0、非负字节和 K/M/G/T，无新增配置。CPU 测试覆盖非法/非有限值、0、小数字节单位及 auto |
| P3 | CSRB 预热也创建 CSRQ 规划向量、查询槽状态；每次相关算子预热都会经过无关分支 | 在现有 warmCompactChunks 入口按已校验格式分流，CSRB 直接沿用原来的 CSRB 预热，CSRQ 保留原有规划；未改 kernel 或数学顺序 |

合法 CLI/格式不变；拒绝此前遗漏的非法输入属于边界校验修复。文件原值、量化与投影映射未变。README/工具说明清除退役 mmap/批处理开关与旧跨路径缓存承诺，并明确 SPECT_CHUNK_VALIDATE=off 只关闭 CSRB 列索引范围检查，不关闭结构或权值校验。预热辅助脚本仍只支持 88/92B metadata；SMMD v3/SC60 由正常重建入口建立缓存，不为本轮扩大脚本能力。

### 10.2 验证与剩余边界

- **Release EXE 与 spect_tests 编译通过**；既有 LNK4098 警告仍在。候选 EXE 为 `build/csrb-native-20260921/bin/Release/SPECT_recon.exe`，二进制证据 SHA256 `ae6f46dad5de66547c2cecb429ff29833accc3001ea420a43437a402f0f1d3cb`。本轮前二进制冻结为证据目录中的 `SPECT_recon.before.exe`，SHA256 `5bb8b7e4faf3a8bf2b815077036077059113ecbf7556d78d780550127ced0326`；均非运行期矩阵摘要。
- **172/172 非 GPU 测试通过，0 失败/0 跳过**；XML 记录 0.118 秒。新增 3 项分别覆盖 header.cols、流式 slot 每数组最大值、host budget 非法值。测试集与第 8.2 节相同，未启动 CUDA 测试 fixture。
- `python tools/prep_compact_denom_cache.py --self-check --work-dir recon_out/unified_cuda_20261008/minimal_review` 通过；只写临时夹具，没有清理真实缓存。未提供矩阵目录，metadata 可选检查明确显示未解析，本结果不算完整真实矩阵预热验证。
- 运行期 src/include/cmake 中扫描 MatrixSha/sha256/BCrypt/OpenSSL 无匹配；未恢复任何矩阵全载荷摘要或历史身份机制。`git diff --check` 通过。
- 证据目录 `recon_out/unified_cuda_20261008/minimal_review/`：`repro_build.log`、`repro_header_cols.log` 保留修复前失败复现；`build.log`、`non_gpu_tests.log/.xml`、`prep_self_check.log` 保留修复后证据。`implementation.patch` 是相对 b206e14 的完整 tracked 工作树快照，含此前修改，并非本轮独立差异。

本轮是限定读取/缓存/调度边界的审查，不是全仓库零缺陷认证。显存预算反例是分配公式证据，不是实际峰值显存测量；CSRB 减少无关 host 工作也未测得耗时收益。第 8 节的 K4/K8、异步前读与融合候选仍需用户允许 GPU 后进行数值和重复性能回归。本轮不运行 GPU、不提交、不部署或替换 SPECT_Core；不新增矩阵注册表、导入服务、内核变体或配置开关。

## 11. 2026-10-08 按架构优化原则进行三轮交叉审查

依据：`E:/Xiang_Project/PET/super_cuda_test_link/Repository Architecture & Performance Optimization.md`，本轮完整读取。遵循其第 3/12 节先分析真实调用链、第 6 节 fail-fast、第 10 节性能证据，以及第 14 节最小修复要求。用户本次要求再次审查，本轮执行 audit，不修改正式源码/配置；仅新增隔离反例程序并更新本文和连续性记录。未审查 E: 仓库的预处理实现，未将 PET 算法或目录规则移植进来。

问题：在第 9 节职责边界下，现有输入是否真正按同一合同消费，前一轮是否遗漏跨调用者问题，哪些复杂度有真实消费者或收益。基线是第 10 节工作树与 EXE `ae6f46dad5de66547c2cecb429ff29833accc3001ea420a43437a402f0f1d3cb`；隔离测试链接其现有 Release 库，不运行 CUDA。完成点为有源码位置和最小反例的审查结论；GPU 数值/性能问题只定位，不推定验收通过。原始投影、矩阵、已有缓存、发布包和其他实验均不变。

### 11.1 Current Architecture

正式链路为 `src/main/main.cpp -> pipeline/MainProgram.cpp::runProgram`：CLI/配置及矩阵路径解析 → CUDA 环境/进程锁 → 独立 metadata 与双窗共享坐标校验 → 床位设置 → 投影/背景读取与 mask/校正 → CompactChunkCSRDirectory → CompactSymmetricReconLoop → 公共 Poisson q/S 接口 → 更新/保存。

- metadata reader 归一 88/92/124B；chunk reader 按头与目录校验 CSRB/CSRQ、连续角度范围、实际载荷与索引。生产 CSRB 和实验 CSRQ 有不同存储/上传方式，不是两套不同的重建数学定义，不应强行删掉一种。
- CSRB：完整 host cache 或按 subset 分段读取；最多两个 GPU 矩阵缓存槽；静态投影按 channel×bed 保留，图像/分子/更新可驻留。部分 host cache 的单 future 前读仍是未通过 GPU 验收的候选。
- CSRQ：离线转换；同一 K4/K8 参数化实现；GPU 完整驻留或 pinned-host 双槽流式；前投影与 ratio 融合。普通和 CTAC 输入共享打包实现，但不是所有打包调用均已消除。
- sensitivity 先按全局 S、最细子集 S_m 准备并缓存；阶段使用指针引用或按既定顺序合并更粗子集。MLEM/OSEM 和 BSREM 保持各自既有更新定义，保存阶段负责 sieve/输出滤波/几何变换，不混入输入适配。
- Python converter 是离线格式生产者；预热脚本是辅助入口，不是重建器内部依赖。核心不需要历史矩阵注册、运行期全矩阵 SHA 或统一框架来完成这些职责。

### 11.2 三轮证据

| 轮次 | 交叉方式 | 结果 |
|---|---|---|
| 1：调用链与合同 | 从 MainProgram 向下跟踪四类输入（第一窗、第二窗、两窗背景），再由最终矩阵几何校验向上核对其可见信息 | 发现通用角度截取掩盖原始 BIN 角度数，以及第二窗/背景只比较总行数 |
| 2：生命周期与职责 | 对照 manifest 与实际权重准备、host/GPU cache 所有权、流式槽、子集引用和 pack 调用 | 确认重复 norm/坏探头解析；未发现理由移除现有格式分工或强行融合 CUDA 内核；异步前读与逐块 pack 的收益尚无新测量 |
| 3：反例与反证 | 隔离原生程序直接调用现有 Release 输入库和几何校验，再将相同文件交给 norm reader；独立 Python converter 与原生 CSRQ reader 互验 | 三类输入缺口复现；norm reader 拒绝这些反例；空 chunk 转换疑点被反证排除 |

原生检查未调用 runProgram（它会初始化 CUDA）。第一类反例直接通过真实读取器和真实最终几何校验；第二类用 MainProgram 同样的两条长度判据与真实读取结果比较，再以完整几何校验反证。它们是可复现的输入层行为证据，不是 GPU 重建后的图像证据。

### 11.3 Main Problems

以下 P1/P2 为缺陷严重度，不与下一节文档规定的优化实施优先级混用。均尚未修复。

**A1 [P1] 生产输入可能被静默截短后判为相容。**

`MainProgram.cpp:507` 将已由矩阵覆盖的 args.numAngles 传入 loadProjectionSet；`ProjectionLoader.cpp:89-104` 将其夹到文件角度数并只读取此前缀，返回的是截取后的 numAngles。后面的矩阵检查已看不到原文件 totalAngles。反例：矩阵 8 angles×4 rays，BIN 12×4，实际读入 8×4 并通过 validateCompactChunkSymmetricInputs；丢弃尾部 16 个值，未报告输入不匹配。

最小修复应在生产输入边界按原始头核对完整几何，不破坏通用 loadProjection 的显式截取能力。现有 `IOTest.LoadProjectionBinAutoDetects` 和 `ReorderProjectionLimitAngles` 明确依赖截取，所以不能简单删除 min/clamp。验证需要单窗/双窗/背景以及 time-segment 的多角度、少角度、准确匹配用例；合法输入应保留同一数组值与顺序。

**A2 [P1] 第二能窗与背景以总长度代替完整布局校验。**

`MainProgram.cpp:573` 背景只比较 raysPerBed；`:595-596` 第二窗只比较 raysPerBed 与 data.size，未比较 numAngles、raysPerAngle 或已知 detector shape。反例：第一窗 8×4，第二窗 4×8，均为 32 个值，当前长度判据通过，但实际行到角度/探测器的对应完全不同；独立调用完整几何检查会拒绝。背景具有相同缺口。第一窗后续严格校验不能替第二窗背书。

最小修复是所有实际消费的观测/背景都与同一 metadata 比较几何；复用现有几何语义，不增加另一套格式猜测器。仅已知 detector shape 才比较 detZ/detX，旧 BIN 不表达二者时仍由矩阵给出形状，但其 header angles/rays 必须准确匹配。测试需包含等长但异布局、跨床、两窗分别失败和合法控制。

**A3 [P2] 投影/背景只有有限性检查，没有非负性检查。**

`ReconDataLoader.cpp:275-279` 只要求 std::isfinite；隔离文件首值 -1、其余 31 值为 1，被完整保留。下游 `CompactSymmetricReconLoop.cpp:1159-1162` 的 sum>0 不能代替逐元素非负，CSRB/CSRQ ratio 直接消费该计数（`projdomain_chunk_ops.cu:215,1323`）。负计数不符合当前 Poisson 观测模型；负背景会进入预测均值。CUDA 是否在具体数据上稍后报错/截断取决于累积结果，本轮未执行，不声称已观察到错误图像。

最小修复在实际观测输入边界要求有限且 >=0，并在合并时间段前检查每份输入，不能通过正负相抵掩盖非法段，也不能静默夹到 0。norm 已有更严格的 >0 校验，不能把它的正值要求机械套给允许零计数的投影。验证覆盖 emission/background、两窗、多床、分段及 NaN/Inf/负值/零值。

**A4 [P2] 缓存描述阶段再次解析已消费的校正输入。**

`CompactSymmetricReconLoop.cpp:870` 首次读取 norm1，双窗时 `:965` 再读同一文件只为取得 size，manifest 在 `:508` 第三次读；norm2 在 `:968/:517` 读取两次。坏探头列表也分别在 MainProgram、权重准备和 manifest 解析。这些都是启动阶段，不是每次迭代，不应夸大成主要 CUDA 热点。

最小简化为复用已校验的值/计数和实际 sensitivity 权重；manifest 应描述输入，不负责重新读取物理输入。先保持现有 cache 字段和数学权重不变，不引入 registry 或永久解析缓存。可用临时文件与调用计数证明减少解析，用 manifest/权重逐元素比较证明语义保留；实际墙钟收益未知。

### 11.4 Optimization Priority

此表使用所引用原则中的 P0/P1/P2：高收益低风险、明确收益中等修改、可选优化；不表示安全漏洞等级。

| 批次 | 内容与最小改法 | 风险 / 预期收益 | 验证 |
|---|---|---|---|
| P0 | 修复 A1/A2：统一正式输入边界的完整几何检查，保留通用截取 API | 只拒绝不匹配输入；避免错误角度映射，非提速目标 | 上述原生反例转为拒绝测试；有效 BIN/MAT、单/双窗/背景、time-segment 不变 |
| P0 | 修复 A3：复用当前有限性遍历同时查负值，错误定位文件/索引 | 不新增整数组遍历或状态；保护 Poisson 合同 | 非法输入在矩阵载荷加载/迭代前拒绝，合法含零计数保持原值 |
| P0 | 简化 A4：已读数据传给 manifest，删除二次/三次解析 | 少维护一处读取职责；减少启动 I/O/分配，耗时收益尚未量测 | 原字段和权重精确比较，cache miss/hit 随 GPU 验证再检查 |
| P1 | 第 8 节 partial-cache async 前读先完成现有数值/计时门，再决定采用 | 每物理块创建一个 std::async，增加调度与生命周期复杂度；当前无重复性能收益证据 | 原生 relL2<=1e-5，冻结 A/B，按既有复杂度增加 >=5% core 门；不另建线程池 |
| P2 | CSRQ 无 CTAC 分支逐 chunk×bed 再打包相同图像，见 CudaSparseOpsState.h:2786-2800 | 源码证明存在重复调用；消除它可能要求另一种 buffer 布局/容量，不能仅凭次数判断值得修改 | 先 GPU 时间线测 pack/传输/计算比例，再考虑现有 buffer 复用；不增加按系统型号区分的 kernel |

### 11.5 排除项与验证状态

- **空 CSRB chunk 转换：已排除。** 合成完整矩阵包含一个 nnz=0 chunk 和一个 nnz=1 chunk。原生 CSRB reader、converter、原生 CSRQ reader 均成功。初次隔离脚本错误预期 NumPy 会拒绝空 memmap，触发的是夹具断言，不是产品错误；原日志保留为 converter_audit_initial.log，修正证据判断后的交叉检查通过。
- **最细子集额外全量复制：已排除。** enterStage 在最细阶段引用 finestSubsetDenoms，并没有再复制 M 份体积；GPU upload 逐指针拷贝，没有额外 flattened host volume。更粗阶段求和是当前已定义计算，不应作为重复状态直接删除。
- **移除所有 fallback：不成立。** device memory 不足时受控流式矩阵传输、诊断/不支持 CUDA gradient 时使用既有 host update 都有实际用途，不等于纯 CPU 重建。仍有宽泛 CUDA memory query catch 与预读失败重试值得在错误注入范围内精简，但未据此宣称 CUDA 数值失效，也未扩大本轮修复范围。
- **重新引入 SHA 或改用另一种载荷摘要：不需要。** 以上三个输入缺口都可由现有维度/值检查解决；norm reader 的交叉控制证明无需历史身份来做正确输入校验。小型缓存键及实际计算输入的既有描述不等于全矩阵身份认证。

隔离原生程序构建成功；input_audit.log 记录三类缺口、有效/NaN 控制和 norm reader 三项拒绝控制；converter_audit.log 记录 CSRB→CSRQ→原生 reader 的空块/非空块成功；既有 converter --self-check 通过（17031 nnz、500 runs，文件解码 relL2 1.073e-5，不是重建误差）。第 10 节 172 项非 GPU 回归是此前同一代码状态的结果，本轮未把重复运行当作独立审查证据。

证据位于 `recon_out/cross_review_20261008/`，CMakeLists.txt/input_audit.cpp/converter_audit.py 仅为此次审查反例，不是新增正式维护入口。tracked_state.patch 与第 10 节 implementation.patch 精确相同，确认没有修改正式源码；EXE 身份不变。没有 GPU 运行、性能 Before/After、重新打包、提交或发布。因此本轮结论是**定位了需补齐的输入合同和可删的重复解析**，不是性能优化已通过验收或全仓库零缺陷声明。

复现（只调用 host 输入库，不初始化 CUDA）：

```powershell
cmake -S recon_out/cross_review_20261008 -B recon_out/cross_review_20261008/build -G "Visual Studio 18 2026" -A x64
cmake --build recon_out/cross_review_20261008/build --config Release --parallel 2
$env:PATH = (Resolve-Path 'build/csrb-native-20260921/bin/Release').Path + ';' + $env:PATH
recon_out/cross_review_20261008/build/Release/input_audit.exe recon_out/cross_review_20261008/input_fixtures
python recon_out/cross_review_20261008/converter_audit.py
python tools/convert_csrb_to_csrq.py --self-check
```

## 12. 2026-10-08 四项输入边界与重复解析修复

用户授权修复第 11 节 A1–A4。按 code-optimization-convergence v2.0，将 A1/A2/A3 视为已确认的输入合同问题，A4 为次要简化；这取代第 11 节的实施排序，不改写历史观察。范围仅为生产输入读取、校正信息复用及对应测试，不改 CLI/wrapper 参数、默认配置、数学更新、矩阵格式或 CUDA 内核。继续禁止 GPU 测试/重建/性能测量，未部署 SPECT_Core。

### 12.1 实际修改

- A1/A2：内部 `loadProjectionSet` 接收矩阵 metadata，完整读取每份源投影后，在合并和消费前复用已有几何检查。第一窗、第二窗、两窗背景及所有床位使用同一入口。BIN 比较完整角度数和行数；MAT 已知形状时同时比较 detZ/detX。底层 `io::loadProjection` 的显式截取 API 不变。第二窗仍检查床位数量。
- A3：在既有 BIN/密集 MAT 读取/合并遍历中要求每份计数有限且 >= 0，保留求和溢出检查；不夹零、不增加 BIN 完整数组扫描。旧 MAT 事件表另查原始计数字段，避免重排时的跳过/聚合掩盖非法值。错误包含文件及从 0 开始的元素/记录索引。有效输入保留原值、顺序和求和次序。
- A4：MainProgram 读取的坏探头列表通过已有 reconstruction context 传入；权重计算与 manifest 复用列表。第一窗 norm/行权重的数量直接复用，第二窗 norm 在实际读取时计算原有小型缓存摘要并复用。没有新增矩阵摘要、永久缓存或配置选项；manifest schema/字段不变。

CLI parser 的既有预检保持不变，因为它承担解析阶段拒绝非法文件的公开行为。读取次数的源码核对如下，不把它包装为耗时收益：

| 文件 | 原重建准备/manifest 阶段 | 修改后 | 加上保留的 CLI 预检后，整个调用 |
|---|---:|---:|---:|
| 第一窗 norm（双窗） | 3 | 1 | 4 → 2 |
| 第二窗 norm | 2 | 1 | 3 → 2 |
| detZ 行权重 | 2 | 1 | 3 → 2 |
| 坏探头列表 | 3 | 1 | 4 → 2 |

### 12.2 验证与边界

- `SPECT_recon` 与 `spect_tests` Release 编译成功；原有 LNK4098 警告仍在。
- 34 项针对性非 GPU 回归通过，0 失败。运行时设置 `CUDA_VISIBLE_DEVICES=-1`，避免测试总入口末尾的 CUDA 同步接触设备；没有 GPU 数值或计时证据。
- 新增/扩展用例覆盖：四类输入标签、单床/第二床、多余和不足角度、等长异布局、负值/NaN/Inf、错误文件与索引、时间段负值抵消、合并溢出、跨时间段布局错误、MAT 形状与重排、通用显式截取兼容。合法双床 BIN 数组逐字节相同，含零与负零。MAT 事件表的 Nx3/3xN 两种布局均复现了负值抵消和非有限记录被跳过的缺口（`event_before.log`，新增检查前测试失败），修复后全部拒绝，合法重复记录保持原结果。
- 隔离 host-only 程序分别包含修复前后实际 manifest 函数，同一微型矩阵/校正文件下比较无校正、行权重、单窗 norm、双窗 norm/坏探头四种配置；输出逐字节一致。它不运行重建或 CUDA，不证明真实图像/性能结果。
- `git diff --check` 通过。未修改原始投影、系统矩阵、既有分母缓存、其他实验或发布包。前一轮 K4/K8、流式/部分缓存数值与配对计时门仍待 GPU 时段。

证据：`recon_out/input_contract_fix_20261008/` 中的 `build.log`、`tests.log/xml`、`manifest_before.txt`、`manifest_after.txt`。隔离 CMake/probe 与修复前源码快照仅用于此项缓存兼容性核对，不是正式维护入口。

复现编译：`cmake --build build/csrb-native-20260921 --config Release --target SPECT_recon spect_tests --parallel 2`。

复现非 GPU 检查：设置 `CUDA_VISIBLE_DEVICES=-1`，运行 `build/csrb-native-20260921/bin/Release/spect_tests.exe`，过滤器为 `IOTest.ProjectionDataEmpty:IOTest.ReorderProjection*:IOTest.LoadProjectionBinAutoDetects:IOTest.TimeSegment*:IOTest.Reconstruction*:IOTest.Normalization*:IOTest.DetZRowWeights*:IOTest.BadDetector*:IOTest.EmptyBadDetector*:IOTest.MultiBed*:RunManifestTest.*`。

## 13. 2026-10-08 CTAC 五项修复

### 13.1 范围与验收

用户授权修复 CTAC 跟进审查的五项缺口。按共享输入边界、缓存、输出坐标、CT 准备、格式能力顺序推进，不改重建数学/执行次序、CLI 参数名、默认算法、CSRB/CSRQ 二进制，不新增 GPU 内核、SHA 或缓存框架。保持当前 GPU 禁用边界，不部署、不提交，不改原始数据。

基线反例位于 `recon_out/ctac_followup_20261008/probe_final.log`：0.8 mm μ 图被 0.4 mm 重建接受；只更新 RAW 后 manifest 不变且命中旧合成因子；负值/NaN/Inf μ 被接受；旋转显示体被 AutoMu 接受且未还原坐标；CSRQ CTAC N10 到 CUDA 消费端才拒绝。该目录早期 `probe.log` 的字节序夹具有误，不能替代 final 日志。原 22 项非 GPU 测试不覆盖这些反例。

验收规则在执行前确定：原反例应拒绝或令缓存失效，合法体素值不变，CT HU 保留负值，输出 RAW 数学不变；SC60 使用合成 CT 坐标坡度检验全网格和床块映射。合成重采样容差为 HU 绝对误差 <0.001、μ 误差 <=2e-7 mm^-1，仅容纳 float32 重采样舍入，不是配准误差或临床定量门。

### 13.2 实施结果

| 项目 | 现在的行为 | 验证 |
|---|---|---|
| CTAC 缓存只识别头文件 | manifest 构建归到现有 CTACFactorCache；v2 同时记录 MHD/HDR 和实际 RAW/IMG 路径、大小、mtime，补 AutoMu 阈值/μ/圆柱参数 | 头文件不变、只改 RAW 后缓存 miss；同 mtime 改大小也失效；HDR→IMG 时间变化失效；缺数据文件报错 |
| μ/CT 输入未严格检查网格与值 | 共用 header reader；尺寸、有限正间距精确匹配（1e-6*max(1,spacing) mm 舍入容差）；μ 与活动必须有限非负，HU 只要求有限；缺 spacing 不再当成 1 mm | 合法 double 值相同；错误维度/间距、缺 spacing、零 spacing、负 μ、NaN/Inf 均拒绝；负 HU 保留 |
| AutoMu 消费输出坐标错误 | MHD 增加 SPECTAutoMuCompatible；旋转/翻转/裁剪/端层置零/拼床/大端输出为 0。只接受明确兼容的内部体；不逆插值显示图 | 未变换全网格可读；无标记旧体、±旋转、X/Z 翻转、裁剪、同尺寸拼床均拒绝。普通输出数值测试仍通过 |
| CT→μ 脚本硬编码 PSC | 同一 prepare_ct_mu_map.py 增加 --recon-manifest，消费现有全网格运行记录和 DICOM 坐标；原 PSC profile 保留。拒绝变换参考体、冲突参数、缺步进和错误网格 | 真尺寸 SC60 90×90×240/0.4 mm、非 PSC 两床/局部 Z 反转、原 PSC 108×108×160/0.3 mm 合成端到端检查通过 |
| CSRQ CTAC N>1 拒绝过晚 | 打开 compact 目录获知格式后立即检查；在矩阵载荷预热/因子生成前报错。保留下层保护，不自动改 N | CSRB CTAC N10、CSRQ 无 CTAC N10、CSRQ CTAC N1 均通过能力检查；CSRQ CTAC N10 和 10→1 阶段调度拒绝 |

兼容性变化是有意的输入收紧：旧无标记 AutoMu 图需重新生成坐标正确的参考体，不盲目补标记；普通 MHD 消费者可忽略新字段。run manifest 增加 `output.z_per_bed_flip`，使 CT 准备不猜测单床逐床翻转是否实际生效。CT 准备入口暂只消费无输出变换的完整矩阵网格 DICOM，不承诺从旋转/裁剪图恢复坐标；也不自动复用 PSC 的配准标定。新入口自动处理多床 local-Z，但不将单床配置中的未生效 z_bed_flip 错当成输入 μ 图反转。

缓存只使用文件属性，不是内容身份认证；保留同路径/大小/mtime 的原地改写仍须调用方清缓存。公开调用说明集中在 `multipinhole/doc/USAGE_CN.md`；能力表和仍维护的参数/迁移摘要已同步 CSRB/CSRQ 边界。历史 AutoMu 验证文档保留旧证据并标明旧入口不能直接代表新 EXE。

### 13.3 验证与剩余边界

- Release `SPECT_recon` 和 `spect_tests` 编译通过，仅原有 LNK4098 链接警告。
- 最终 78 项非 GPU 测试通过、0 跳过/0 失败，覆盖本次 CTAC/输出及上一轮投影合同、manifest 回归。测试进程设置 CUDA_VISIBLE_DEVICES=-1，未运行任何 GPU 测试/重建。
- 合成 CT→μ 三组检查均通过。HU 最大绝对误差分别为 SC60 1.07e-13、两床 0、PSC 2.85e-14；床块切取与 μ 局部 Z 方向也匹配独立数组参考。错误输出变换/维度/间距/步进，以及 DICOM 与 manifest 间距不一致均拒绝。
- 另以 `b206e14` 中原版 CT 准备脚本作控制，同一组合成 PSC DICOM、同一旧 CLI 下，修改前后的 μ、HU、attMap、regCT 共 8 个 MHD/RAW 文件逐字节一致（每份 RAW 7,464,960 字节）。这证明该旧单床调用的输出兼容，不外推为所有实际扫描已验证。
- 这是代码修复及非 GPU 验证完成，不是最新 CTAC kernel、真实 SC60 配准、AutoMu 物理适用性或性能采用证明。既有 K4/K8、CTAC 有/无 CT、多床和驻留/流式 GPU 回归仍待用户允许时段；CSRQ 仍实验，CTAC 多子集仍不支持。

证据：`recon_out/ctac_fixes_20261008/build_final.log`、`non_gpu_final.log/xml`、`python_grid_final.log`、`legacy_parity.log`；该目录内 check_legacy_parity.py 是一次性控制比较，不是新增生产脚本。实际二进制位于 `build/csrb-native-20260921/bin/Release/`，SPECT_Core 未替换。无性能加速结论。

复现编译同第 12 节；CPU 测试过滤器见 `non_gpu_final.log` 首行。CT 准备回归命令：`python scripts/test_prepare_ct_mu_map.py`（默认使用临时目录；`--output <新目录>` 可保留合成 DICOM/MHD/摘要）。

## 14. 2026-10-09 CSRQ 多子集 CTAC 支持

用户授权继续实现 CSRQ CTAC 的 N>1 支持。本节取代第 13 节的 N1 限制，不放宽 AutoMu 输入坐标规则。范围是既有因子分解路径的子集调度，不是新重建算法或性能采用；保留 CLI、CSRB/CSRQ 编码、矩阵合法性检查、窗权重/背景/衰减指数和 N1 全量语义。基于 `b206e14` 加现有未提交修改；不运行 GPU、不发布、不改原始数据。

### 14.1 变更与复用

- CSRQ 的 CTAC 计划按 `(子集数, 子集号)` 保存当前矩阵每块的绝对 compact 角度及 branch→factor slice 映射。复用 `compactSubsetAngles`，增加半开角度范围以支持非零块偏移和空选择；整除规则仍基于整个矩阵，不要求每个 chunk 被 N 整除。
- 前投影/ratio、灵敏度与反投影仅遍历选中角度。矩阵 run/value 继续沿用已有子集传输，CTAC 也只搬运该子集涉及的因子切片。沿用 K4/K8、packed/branch-major 反投影及现有双槽事件同步；没有新增 CUDA kernel、格式、运行开关、SHA 或长期缓存文件。
- 全量 forward 诊断使用独立 N1 计划，不能误用最近一次 N>1 的切片索引；反投影 ratio 仍按原 chunk 行偏移索引。计划只缓存映射，不缓存某床/能窗的衰减数值，因而各床表与逐窗 exponent 继续独立。
- 删除准备阶段和底层两处 N>1 拒绝，以及只服务该拒绝的 helper/测试；没有留下无操作能力检查。矩阵所有权显式禁止复制、允许移动，解决 MSVC 对新增 map 成员后的 vector 扩容误选复制构造问题。

### 14.2 验证状态

| 检查 | 结果与边界 |
|---|---|
| Release `SPECT_recon` / `spect_tests` | 编译通过；原有 LNK4098 警告未处理。首次编译的 C2280 已修复，保留初次与最终日志 |
| 非 GPU 回归 | 91 项通过，0 失败/跳过；进程设置 `CUDA_VISIBLE_DEVICES=-1`。包括子集范围/拼接、非法范围、格式、CTAC cache、AutoMu/输出与投影合同 |
| 扩展的 GPU 用例 | 已编译，**未执行**。K4/K8、驻留/流式、packed/branch-major、两床两矩阵、背景和行权重、不同床因子表、0/1 因子、第二窗 exponent=1.35；算子 N10/5/2/1 与重新访问 N10 |
| GPU 预设验收 | 对解码为同值 CSRB 的独立子集切片参考，q/S 与全量 forward 的 relL2 <=1e-5；逐体素所有子集 S 求和等于 N1 S（同容差）。沿用已有容差，不是量化前后误差门 |
| 静态复核 | 绝对角度→原 chunk 行偏移、选中角度序号→slice 索引、流式物理 run/value 偏移分别核对；双槽完成事件与释放顺序保持原实现。`git diff --check` 通过 |

上述 N 序列用于直接算子测试，不改变公开递减调度的整除约束；合法 CLI 阶段可用 N10→N5→N1，不能据测试序列把 N5→N2 当成合法阶段。GPU 测试保留 CSRB 参考结果，在后续模式连续切换子集与诊断，不在每次调用前清空 CSRQ 计划，覆盖缓存复用。

证据：`recon_out/csrq_ctac_subsets_20261009/build_final.log`、`non_gpu_tests.log/xml`。编译命令同第 12 节；非 GPU 过滤器记录在日志首行。恢复允许的 GPU 时段后，最小新增数值检查为 `spect_tests.exe --gtest_filter=*JointSubsetsWithAndWithoutCtacMatchDecodedCsrb*`，随后仍需既有 CTAC/BSREM 与真实数据端到端回归。不要在该 GPU 检查中设置 `CUDA_VISIBLE_DEVICES=-1`。

结论：**支持代码、编译及非 GPU 检查完成，GPU 数值/显存/耗时和生产采用待验证**。AutoMu 无标记/变换体仍拒绝；CSRQ 保持实验性；SPECT_Core 未替换。此次没有可报告的速度提升或峰值显存实测。

## 15. 2026-10-09 六项后续优化

用户授权按顺序执行：残差模型、现有 GPU 组合回归、矩阵/能窗/校正语义、norm+background、配对性能、逐针孔 CTAC/空间 PSF 测量。当前 GPU 空闲，新的授权取代此前仅非 GPU 检查的边界；不停止外部实验、不改变默认配方/CLI/磁盘格式、不部署、不提交。证据集中于 `recon_out/six_followups_20261009/`，基础 revision 为 `b206e14` 加既有未提交工作；本轮修改前 EXE/DLL/配置在 `baseline/`。

验收区分：输入和掩码/行数要求精确；原生小矩阵对独立 EM/残差参考 relL2/逐值 rtol=1e-5（float32 算子已有门限）；既有 CUDA q/S/forward 沿用原测试门限。性能用同参数、暖分母缓存和交错重复，图像 relL2<=1e-5，复杂性增加的候选仍须满足原 >=5% 迭代收益门；无速度收益不伪称内核加速。物理测量先做描述性分析，不从路径积分或图像宽度直接推出真实图像质量/绝对分辨率。不以缺少 CT 阻止 SPECT-only 毛细管定位，源位置可由已有重建估计，但估计位置不冒充硬件真值。

### 15.1 残差与背景模型

- 修改前真实 EXE 的反例：屏蔽行被输出为 count=2、measured_sum=0、expected_sum=216.826，产生约 -2.17e12 的 relative_bias；同一夹具还覆盖效率不为 1 的预测。修复后按有效行计算 `e*w*A*x+b`，复用已加载效率，并将前向 PSF 后的局部 ROI mask 与实际迭代对齐。
- norm 与加性背景的组合在准备阶段计算一次 `b/e`，原始背景保留用于残差。既有比值/反投内核保持原样；无背景默认路径不变。双窗必须成对提供各窗 norm，背景必须是同能窗/床位/曝光的预计探测计数，不能先乘 c 或从 y 中减去后夹零。
- `scripts/test_compact_observation_model.py` 用 all-one 小矩阵的独立标量 EM 解检查真实 EXE 的输出体及逐角度/探测器残差。它不依赖待测 CUDA 实现计算参考；geometry 置换/稀疏行/CTAC 的非平凡数值另由已有 CUDA suite 覆盖。
- `Format_Reader_Matrix_CN.md` 维护唯一数学/物理合同，USAGE/PARAMETERS 仅说明调用与链接。不新增身份 registry、二进制协议版本或默认单位；实际 Lu v7.9 生成权重来源未知项保持未知。

### 15.2 验证状态

残差修复的原生 16 组通过；最终为 K4/K8、CSRB/CSRQ、单/双窗、1/2 床、N1/2、有效行屏蔽的 **32 组无背景 + 32 组有背景通过**。另以独立 NumPy separable convolution 检查 operator PSF 后重新施加 ROI 的残差闭合，有/无背景两组通过；夹具保证 ROI 外模糊量足以暴露原遗漏。

CSRQ 多子集 CTAC 两个 profile 的专项 GPU 测试通过。全套首次仅失败于旧日志 `[8]` 断言，修正为共享实现的 `[K]`，未放宽数值容差；norm+background 和最终预取简化后重跑为 **339 项总计，335 通过、4 跳过、0 失败**。跳过原因准确区分：一个五数据集滤波 parity 未启用，三个历史 260601 实矩阵检查因 `test_data/6module/260601` 不存在。没有用当前另一份矩阵冒充该历史基线。日志/XML：`gpu_suite_final.*`；原生闭合检查：`closure_no_background.log`、`closure_background.log`。

Release EXE 与 tests 编译通过；既有 LNK4098 警告仍在，没有把它说成零警告构建。此次验证证明软件组合与数值实现，不替代真实 CT 配准、物理校正标定、长迭代正则或原五床生产图像门。

### 15.3 配对性能

冻结条件：RTX 4070 Ti SUPER 16 GB；SC60 Derenzo 投影，18 chunks、26,960,069,784 字节 ROI32.8×12 派生 CSRB，PSF sieve/CASToR1.5、N10、8 threads、相同末轮输出。完整 host cache 40 GiB 用20轮 ABBA，部分缓存2 GiB用3轮 ABBA；每臂独立分母缓存，暖机与正式 hit 分开。机器有其他 CPU 任务，正式采样估计其他进程 CPU 最高约20.4个百分点；未运行第二项 GPU 重建。没有 OS 磁盘缓存清空，不声称物理冷盘带宽；配对顺序、资源采样、原始数值均保留。

为避免把不同对照混在一起，以下用 A0/B0/C 命名：A0=`unified_cuda_20261008/SPECT_recon.baseline.exe`，B0=本轮修复后、保留逐块 async 前读的 EXE，C=删除该前读、统一复用已有双槽预取的最终 EXE。A0→B0 是此前多项优化的累计效果，不能归因于本轮残差或背景功能。

| 正式配对 | core 两次，秒 | prepare 均值，秒 | 进程总墙钟两次，秒 | 解释 |
|---|---:|---:|---:|---|
| 全缓存 A0，20×N10 | 24.025 / 24.070 | 51.813 | 73.452 / 82.831 | 旧基线 |
| 全缓存 B0，20×N10 | 23.860 / 24.286 | 26.019 | 52.132 / 52.221 | core基本不变；总墙钟均值减少33.23%，主要来自准备阶段 |
| 部分缓存 A0，3×N10 | 60.431 / 61.556 | 23.579 | 84.029 / 85.777 | 旧基线 |
| 部分缓存 B0，3×N10 | 68.031 / 73.370 | 2.338 | 70.735 / 76.043 | 总墙钟减少13.56%，但core变慢15.91%，触发进一步判别 |
| 仅预取判别 B0，3×N10 | 65.573 / 83.013 | 2.375 | 68.364 / 85.683 | 单因素对照的原实现 |
| 仅预取判别 C，3×N10 | 58.129 / 59.942 | 2.347 | 60.806 / 62.607 | core均值减少20.54%，两对分别减少11.35%/27.79%；总墙钟均值减少19.89% |

两次结果差异较大，因此保留范围与配对结果，不将均值当普适提速比例。A0/B0 全缓存最大图像 relL2=1.219e-7；部分缓存6.079e-8；B0/C最大6.152e-8，均满足预设1e-5。C未改变完整缓存的执行分支，不把部分缓存收益推广成全缓存迭代收益。

B0/C三轮的应用 payload 读取同为71,990,314,632字节、240次，矩阵 H2D同为80,870,256,288字节。改善不是少读矩阵，而是恢复已有预取调度、删除额外逐块线程路线后的实测效果；尚不能独立区分线程开销与传输重叠各占多少。保持两 GPU 槽与部分缓存 pageable 来源策略，没有新 kernel、配置开关或第三槽。10秒系统显存采样不足以证明峰值显存下降，本轮不作该声明。

**变体结论：completed / validated（当前实测范围） / retained in worktree。** 删除未获得收益的 async 前读候选，不保留第二条可切换生产路径；冻结 EXE 是回滚对照，不是运行期开关。未部署。证据 `csrb_full/`、`csrb_partial/`、`prefetch_partial/`、`metrics.json`；最终二进制备份 `SPECT_recon.final.exe`，带前读对照 `SPECT_recon.async_baseline.exe`。

![SC60配对耗时](D:/Projects/SPECT-recon/recon_out/six_followups_20261009/performance.png)

### 15.4 逐针孔 CTAC 测量

实际 SC60 矩阵的审计文件提供6个 PM 中心：X≈±10.905 mm，Y≈29.225 mm，Z≈0/±12.8 mm；metadata 的 detectorY=90 mm 是探测器平面，不能代替针孔。更新既有验证脚本支持88/92/124B metadata，并修掉微型网格依据角度数误猜45° profile的问题。旧单针孔入口现在要求显式 `--detector-y` 给出针孔平面Y，不再默用探测器位置；这是验证工具的输入收紧，不改重建 CLI。

合成体为半径16.2 mm、半高43.2 mm的水圆柱（μ=0.0152/mm）及局部高 μ 插入（0.03/mm）。3个Z层、5个XY位置、8等间隔角度、6针孔、2仿体，共1440个未加权样本。用同一 Siddon 积分器比较平行路径与实际针孔中心路径，另分开记录当前生产采样器的差异，避免将采样误差全部归因于针孔几何。独立轴向/斜向均匀体解析长度自检通过。

| 仿体 | 相对衰减因子差的绝对值中位数 | P95 | 最大值 |
|---|---:|---:|---:|
| 水圆柱 | 11.40% | 32.98% | 44.44% |
| 高 μ 插入 | 12.64% | 37.76% | 59.93% |

定义为 `abs(exp(I_pinhole-I_parallel)-1)`，即以针孔路径衰减因子为参照；不是重建图像误差。远离轴向中心的样本差异更大。当前总矩阵没有逐针孔贡献标签，以上样本不按实际 A 的权重/可见性筛选，不能直接推导观测投影或图像的偏差。**发现了值得继续检查的几何误差机制，但不据此更改生产 CTAC**。下一项最小物理判别需具有逐针孔贡献的矩阵/模拟投影，再比较重建端点，而不是给现有总矩阵随意乘一个针孔平均因子。

![SC60针孔路径差异](D:/Projects/SPECT-recon/recon_out/six_followups_20261009/ctac_geometry/path_difference.png)

### 15.5 无 CT 的毛细管相对位置与表观宽度

已核对 `pre_tc2/preprocess_run.json` 的原始目录确为 `F:/DATA/SC60_luzhou/sc60_maoxi/SPECT`；Bed2_asis BIN与pre_tc2/Bed2.bin逐字节相同。采用已有同参数 Bed1/Bed2/Bed3 的5×N10、PSF sieve1.5、无CTAC、0.4mm网格、无输出X/Z翻转结果，不要求CT，也不重新猜测摆放坐标。

全体积 Bed1/Bed3 存在巨大的边缘极值，Bed2无限制30轮也被远端极值主导。因此完整图仍保留，同时用历史 `cyl36` 所对应的中心36mm高度作**探索性观察ROI**（XY半径16mm），不修改原始重建、不把排除边缘当作图像改善。对这个重叠区域做平移 NCC，搜索Z±32mm、XY±4mm，要求至少25%的ROI交叠；已用已知整数平移合成控制验证符号/轴顺序，误差<0.15体素。

| 移动图→Bed2 | X平移 mm | Y平移 mm | Z平移 mm | 重叠NCC |
|---|---:|---:|---:|---:|
| Bed1 | −0.268 | −0.036 | +21.577 | 0.9704 |
| Bed3 | +0.015 | −0.223 | −21.603 | 0.9769 |

这是图像存储坐标中的相对平移，不是CT世界坐标或机械步进标定。图像估计约21.6mm与记录的机械21mm并不完全一致；未据此修改步进或轴方向配置。Bed3只保留较少重叠结构，不能由高NCC推定完整配准精度。

从Bed2 cyl36图最大的3个长条连通结构估计加权中心/主轴，在两条垂直主轴的截面测FWHM。阈值为图像最大值15%，沿结构主轴−0.4/0/+0.4mm三处平均，三线性采样、局部端点基底；这是描述性分析规则，没有质量采用阈值。中心XYZ约为(0.515,7.650,2.368)、(9.433,−2.458,2.928)、(0.603,2.582,7.553)mm，以矩阵中心为零。

| 已有重建 | 6个横向截面的表观FWHM范围 |
|---|---:|
| Bed2 5轮，无额外cyl36重建约束 | 1.776–1.880 mm |
| Bed2 30轮，cyl36 | 1.599–1.684 mm |

两组均已使用1.5mm sieve PSF，且迭代数/ROI同时不同，不能将差值归因于空间PSF或宣称改算法后分辨率改善。源内径未知、0.4mm体素和有限位置覆盖也限制绝对PSF反演；0.02mm采样间距不代表测量精度。**相对摆放位置可由SPECT估计；当前证据不足以引入空间变化PSF、修改本组1.5mm设置或发布默认值。** 后续实际标定只需补源内径、相同ROI/迭代和无额外sieve模糊的多位置图像，CT并非必需。

![SPECT内部配准](D:/Projects/SPECT-recon/recon_out/six_followups_20261009/maoxi_review/registered_overlay.png)

![毛细管表观宽度](D:/Projects/SPECT-recon/recon_out/six_followups_20261009/maoxi_review/apparent_width_profiles.png)

### 15.6 复现与边界

主要维护入口仍为重建源码/现有文档；新增 `scripts/test_compact_observation_model.py` 是共享观测模型回归，现有 `scripts/validate_ctac_pinhole_reference.py` 是验证工具。`recon_out/six_followups_20261009/paired.py`、`inspect_maoxi.py`、`summarize.py` 是此次证据复现脚本，不是发布/上位机接口。完整命令、JSON、CSV、PNG、日志和冻结二进制留在同一证据目录。

```powershell
cmake --build build/csrb-native-20260921 --config Release --target SPECT_recon spect_tests --parallel 8
build/csrb-native-20260921/bin/Release/spect_tests.exe
python scripts/test_compact_observation_model.py --exe build/csrb-native-20260921/bin/Release/SPECT_recon.exe --output recon_out/observation_recheck
python scripts/test_compact_observation_model.py --exe build/csrb-native-20260921/bin/Release/SPECT_recon.exe --output recon_out/observation_bg_recheck --background
python scripts/validate_ctac_pinhole_reference.py --self-test
python scripts/validate_ctac_pinhole_reference.py --chunk-dir test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV --geometry-audit test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV/pinhole_geometry_audit.txt --out-dir recon_out/ctac_geometry_recheck
python recon_out/six_followups_20261009/inspect_maoxi.py
```

六项本轮端点已完成：残差修复、现有套件GPU回归、明确语义、norm+background、实测后删减低效前读、物理模型观察。**没有新增逐针孔CTAC或空间变化PSF生产模型，没有真实CSRQ量化前后图像质量签收，也没有重跑受保护的Lu五床50轮门。** CSRQ仍实验；原始矩阵/投影与SPECT_Core未改动，未提交或部署。

## 16. 2026-10-11 SC60 多床联合重建、16 床与 `-beds`

问题：SC60 每床有效轴向范围约 10 mm（XY 求和灵敏度 FWHM 16.8 mm、FW10% 30.4 mm，中心 10 mm 内占 52.8%），一次扫描需 10–15 床。修改前 matrix-grid 上限 5 床；且各床独立重建后用宽余弦融合，步进 ≤10 mm 时每床边缘的低灵敏度发散体素会被带入扫描区中部。

实现：matrix-grid 多床改为单一全局体积 `Nz=baseNz+(N-1)·step` 的联合 OSEM——每次更新中重叠床的灵敏度与反投影相加，输出即全局体积，无融合、无增益拟合；尺寸与方向与旧输出相同，重叠区内容语义改变。CTAC/AutoMu/`-maskFromAtt` 在该模式显式拒绝。上限 16 床（PSC 仍 5）。`-beds N|A:B` 从单个 `-df`/`-df2` 目录选取 `Bed<k>.bin`，缺床报错。分母缓存键包含 `bed_z_starts`。

合成验证（软件正确性，非 SC60 真实多床）：90×90×NZ/0.4 mm 圆柱+Derenzo+球体仿体，CSRQ 前投影，步进 20 层=8 mm，`-zbedflip 0`，无 PSF，OSEM 10×10；评估区为联合灵敏度切片和 ≥50% 最大值的 z 43.2–84.8 mm、r≤14 mm；“发散体素”定义为 >20×真值最大值；尺度为最小二乘（剔除发散体素）。

| 5 床对照 | 评估区发散体素 | relL2 |
|---|---:|---:|
| 无噪声 旧（独立+融合） / 联合 | 0 / 0 | 0.378 / 0.368 |
| 泊松 ×15 旧 / 联合 | 322 / 0 | 0.828 / 0.374 |

![旧融合与联合重建](D:/Projects/SPECT-recon/recon_out/sc60_multibed_review_20261011/fig_5bed_old_vs_joint.png)

性能（RTX 4070 Ti SUPER 16 GB、64 GB RAM，CSRB 62.5 GiB 部分 host cache）：CSRB 每轮成本由矩阵 I/O 决定，1/5/12 床的 10 轮 H2D 均为 671,027,529,760 字节；同二进制同配置重复间 core 差达 40%（磁盘吞吐波动，C/D 读取字节完全相同）。

| 臂 | 床 | 矩阵 | 轮×子集 | iteration_core s | 峰值显存 MiB（整卡，含空闲 2017） | 峰值工作集 GiB |
|---|---:|---|---|---:|---:|---:|
| F | 1 | CSRB | 10×10 PSF1.5 | 544.9 | 13194 | 45.4 |
| G | 1 | CSRQ | 10×10 PSF1.5 | 10.6 | 4713 | 19.5 |
| D | 5 | CSRB | 10×10 | 509.7 | 10794 | 40.8 |
| E | 12 | CSRB | 10×10 | 521.7 | 14482 | 45.7 |
| H | 12 | CSRQ | 10×10 | 45.9 | 5658 | 19.8 |
| I | 16 | CSRB | 2×10 | 85.7 | 14937 | 45.8 |
| J | 16 | CSRQ | 10×10 | 58.4 | 6011 | 19.9 |

CSRB 显存峰值随缓存预算/时序浮动（同旧 EXE 两次 11.5/14.4 GiB），不是每床刚性需求。CSRQ 与 CSRB 同输入：单床仿真 Derenzo（PSF1.5）全体积 relL2 1.38e-7，通过 1e-5 配对门；12 床评估区为 2.76e-6，但全体积为 0.092589，**不通过全体积门，不能宣称整体数值等价**。原分析将差异定位到评估区外的发散体素；排除它们后的 ROI 结果不能替代原全体积门。

**结论：联合重建已合入共享树（未提交）；CSRQ 保持 experimental。** 本组单次观测的同端点 core 比值，1 床 F/G 约 51 倍、12 床 E/H 约 11 倍，工作集约降至 1/2.3；未做控制 I/O 状态的交错重复，不能视为稳定加速保证。16 床 I/J 的迭代数不同，不作同端点速度比较。真实多床数据、噪声水平质量门及全体积差异仍待验证，`z_bed_flip`/步进须由真实 SC60 多床采集确定。`spect_forward_project` 的 CSRB 非 CTAC 路径在部分缓存下反复读矩阵（单床读 715 GB 后终止），该 I/O 效率问题未修；其偶发缓存误命中另见第 17 节。

复现：`recon_out/sc60_multibed_review_20261011/` 内 `make_phantom.py --nz 460|540`、`spect_forward_project -chunkCSRDir <CSRQ> -dim 90,90,240 -beds 12|16 -bedOverlap 220,... -zbedflip 0 -vox 0.4`、`make_noisy.py --scale 15`、`run_arm.ps1`（各臂 `runs/<臂>/command.txt`）、`analyze_synth.py --sens <联合分母缓存>`、`make_figure.py`；冻结修改前 EXE 在 `baseline_bin_independent_blend/`。

## 17. 2026-10-11 部分缓存前向投影的所有权修复

承接线程 `01a12ec7-87e3-7510-8db1-3cb7a8197394`。范围是既有 ROT90 部分缓存偶发数值错误和 PSC 五床接口回归；不改矩阵格式、算法、CLI、默认值、锁页策略，不部署或替换上一轮交付包。基础为 `b206e14` 加共享树既有修改；原测试 EXE/DLL 保存在 `recon_out/partial_cache_fix_20261011/baseline_bin/`。沿用独立稠密 oracle 的 relL2 <=1e-5，不放宽门限，不进行昂贵全矩阵性能试验。

根因：`uploadIoCsrView` 同步上传后没有保留 `CSRView::sourceOwner`，而 GPU 缓存键使用主机数组地址。部分缓存释放源 chunk 后，分配器可能把同一组地址给不同矩阵块，使后续调用误用旧设备矩阵。异步整块和子集上传已保留所有权，仅同步公共入口漏掉。修复为同样保存现有 `sourceOwner`，直到该设备槽被替换或释放；不新增哈希、矩阵身份注册、缓存槽或内核。

| 验证 | 结果 |
|---|---|
| 冻结基线，原 ROT90 用例单独重复 50 次 | 1 次失败，零主机缓存前向投影 relL2=0.07142548；分母/更新之前已失败 |
| 新增确定性生命周期测试，修复前 | 两次同步上传后源存储均提前释放，测试失败 |
| 修复后，生命周期+ROT90 各重复 50 次 | 100/100 通过；替换和最终 release 后旧源存储释放 |
| 修复后，原 ROT90 用例独立重复 50 次 | 50/50 通过，未依赖新增用例的清理改变测试间状态 |
| 完整 CUDA/CPU 套件 | 348 项：344 通过、4 跳过、0 失败；包括 SC60 联合床更新、16 床/选床、PSC 上限/输出、CSRB/CSRQ CTAC 多子集 |
| PSC 冻结交付配置与新 EXE | 旧五床命令及冲突 scanner 两项的全部有效配置一致，仅归一化 EXE 相对路径；Lu merged 单矩阵、OSEM50:N10、PSF1.3 sieve、zbedflip=1、xoutflip=0、XY180、center66_cosine10 保持 |
| Release 构建 | `SPECT_recon`、`spect_tests`、`spect_forward_project` 均通过；保留既有 LNK4098 警告 |

完整套件跳过项：未启用的 `FrozenFiveDatasetsWhenRequested`，以及缺少历史 `test_data/6module/260601` 的三项实矩阵检查。没有用另一份矩阵代替该历史输入。PSC 核对属于接口配置和既有软件回归，不代表已重跑真实五床 MLEM50 图像/性能门；现场 5060 Ti 尚未实测。此次是正确性修复，没有速度提升结论；CSRB 非 CTAC 全量 forward 的按 branch 重读仍待独立优化，CSRQ 全体积/真实多床采用门仍未关闭。

证据目录 `recon_out/partial_cache_fix_20261011/` 保存 baseline/candidate repeat 日志、`ownership_before.log`、`full_suite.log/xml`、构建日志、`psc_config_parity.json`。`verify_psc_config.py` 是只读冻结配置的任务内复核脚本，不写原交付包。复现：

```powershell
cmake --build build/csrb-native-20260921 --config Release --target SPECT_recon spect_tests spect_forward_project --parallel 4
build/csrb-native-20260921/bin/Release/spect_tests.exe --gtest_filter=CompactSubsetCudaTest.Rot90N10MatchesIndependentOracleWithPartialCaches --gtest_repeat=50
build/csrb-native-20260921/bin/Release/spect_tests.exe
python recon_out/partial_cache_fix_20261011/verify_psc_config.py
```

## 18. 2026-10-12 本地提交与干净构建发布

用户要求继续优化并提交，避免 dirty 版本。本轮收敛已有已验证实现，不增加新算法、内核、运行期矩阵哈希或性能策略；第 16/17 节的“未提交”描述是当时状态。

- `1469d27` 提交矩阵执行、校正、缓存生命周期、多床及对应回归/文档。
- `0805148` 提交发布入口约束：`package_rc.ps1` 拒绝脏工作树，也拒绝不能用 `--version` 标识当前干净提交的暂存 EXE，包括 `-SkipBuild`。不通过更改包名或手改版本文件掩盖状态。
- 预处理仓库的独立未提交实验不纳入这次包；包中九份 Python 工具逐项确认与预处理提交 `49e24ce` 的 Git blob 相同。预处理 EXE/DLL/资源沿用旧包，不声称预处理整个工作树已经清理。

在 `0805148` 干净工作树显式重新配置 CMake、构建 EXE/测试/前投影工具后：完整套件 **348 项，344 通过、4 跳过、0 失败**；发布身份隔离检查 6/6；观测模型无背景/有背景各 32 例及两次 operator-PSF/ROI 残差闭合均通过；CT/Mu 几何、针孔参考和 CSRQ 转换器自检通过。四个跳过项仍为未启用的冻结输出滤波数据门和缺少历史 260601 输入的三项检查。链接器既有 LNK4098 警告未在本轮改动。

最终设备包入口为 `E:/Xiang_Project/SPECT/6Module/release/devices/{PSC30-MPH6,SC60-MPH3}/SPECT_Core.zip`；旧 dirty 包留在 `release/history/20261012_device_packages`。仅升级干净重建 EXE 和对应版本/来源记录，设备配置、预处理工具及资源保持原包字节不变。PSC 按用户最新纠正保持 `x_out_flip=0,z_bed_flip=1,z_out_flip=0,xy_out_rotation_deg=0`，不用早期候选的 XY180；SC60 沿用 21.2 mm 和 -60 度的历史试验边界。

本次证据统一保存在 `recon_out/clean_release_20261012/`：`full_suite.xml/log`、两组 observation 结果、`package_smoke_results.json`、设备 verify 日志、`delivery.json`。封包使用记录在该目录的 `refresh_packages.py`，复用既有验收入口；实际封包提交及 EXE 版本以包内 `version.txt/release_manifest.json` 为准。文档提交后再次配置/构建版本身份，不能复用上一提交的旧 EXE。

这是源代码/发布过程收敛，不是新的速度提升结论。真实 SC60 多床、CSRQ 全体积采用门、Lu 五床生产性能门及 RTX 5060 Ti 现场计时仍保留原边界。

## 19. 2026-10-12 PSC / SC60 多床吞吐优化

范围：原生 CSRB 不改精度、算子、子集顺序或配置默认值；复用现有 CUDA 内核与锁页 lease，不改变 CSRQ 实验地位。控制为干净提交 `0c71ecb` 的程序副本；候选为本节局部修改。用户新要求全项目默认禁止 SHA，本轮仅记录路径、大小、修改时间、版本及实际参数，不执行摘要计算。

最终保留两项独立机制：无 CTAC 完整前投影改为 chunk 外循环，一次加载后处理全部 4/8 对称分支；OSEM 首次从全量块进入较小子集时释放原整块 GPU 数组容量，再按切片分配，后续子集继续复用。前者额外保留 4/8 个 float 旋转图像（本次 SC60/PSC 约多 22/85 MiB 主机空间），换取每次完整前投影至多一次读入各块；后者复用 `CachedMatrix::release()` 及已有上传同步，不增加内核、缓存层或开关。普通迭代已按 chunk 外循环在所有床位间复用矩阵，本轮没有再建一个多床实现。

验证合同：同输入 primary relL2 <= 1e-5；原有独立稠密 oracle、CSRB/CSRQ/CTAC 回归不降级。PSC 使用既有 Lu208 矩阵及五床历史合并投影，N10/sieve1.3；SC60 使用完整 CSRB 和既有合成 12/16 床投影，N10/sieve1.5，8 mm 为该合成数据的步进，不作机械标定。PSC/SC60 分别固定 30/40 GiB 主机预算以隔离部分缓存路径；这不是自动预算的生产性能门。

两臂先暖分母，再 ABBA 各 3 轮；分别记录 prepare、全部迭代、首轮后的稳态迭代、进程总墙钟、应用读取/H2D、工作集和 GPU 总占用。单臂上限 900 秒，失败即停止对应比较；不清空 OS 文件缓存、不把单次 GPU 利用率当作收益。实现新增复杂度的收益门仍为 5%，低于门限的锁页候选不默认采用。SC60 16 床边界另做一次同输入配对，不据此宣称重复性能。

执行证据：`recon_out/throughput_20261012/`，任务驱动 `paired.py`、汇总 `summarize.py`；基线 EXE 保存在 `baseline/`。最终保留方案 D 的完整套件 **350 项，346 通过、4 个原有条件跳过、0 失败**（`retained_suite.xml/log`）。新增检查覆盖 K4/K8 完整前投影的独立稠密 oracle 与每次读块次数上界，以及两个 64 MiB 整块 GPU 槽变为 32 MiB 切片后至少释放 48 MiB（预留 16 MiB 驱动波动）的实际显存检查。原有 CSRB/CSRQ、CTAC、部分缓存和多床回归保留；CSRQ 不因这些测试升级为生产方案。

中间候选 B（前投影调整加部分前缀预载/锁页，不含 GPU 容量修复）：PSC 五床 ABBA 稳态均值 7.984 -> 6.325 秒/轮（-20.78%），三轮进程 74.111 -> 69.593 秒（-6.10%），最大 relL2 1.467e-6。SC60 第一个候选正式样本正常，第二个样本首轮 191.040 秒，loader_work 从 71.969 增至 225.666 秒；后两轮恢复为 39.305/38.512 秒。这个读取长尾保留在记录中，不剔除、不认定为稳定加速，也不认定已证明由锁页造成。中间 EXE 保留于 `pinned_candidate/`。

中间候选 C 同时加入 GPU 容量释放，但仍有部分缓存预载/锁页。PSC 两次两轮进程耗时 184.748/134.753 秒，而控制为 67.105/66.150 秒；一次首轮 132.172 秒，另一次 prepare 112.241 秒。SC12 C 两轮表现正常（115.450/120.847 秒），不能抵消 PSC 的复现长尾。**不采用部分缓存预载/锁页**，恢复原来“完整主机缓存才尝试锁页”的策略；已经验证的锁页拒绝回退仍在。这里拒绝的是本次默认实现，未证明所有锁页方法都无效，也未断言长尾的底层因果已定位。C 的源差异保存在 `candidate_C.patch`，EXE 在 `subset_capacity_candidate/`；`psc5_final/`、`sc12_final/`、`sc16/` 这些任务目录名中的 final 不是最终采用标识，均属 C，不能并入 D 的结果。

D 配对测试使用同输入、同配置、同缓存预算与已有分母缓存：PSC5/SC12 各 ABBA 两轮，SC16 一次 AB 两轮（只验证边界及记录耗时，不作重复加速结论）。输出目录为 `psc5_retained/`、`sc12_retained/`、`sc16_retained/`，全部退出码为 0、输入文件大小/时间未变。硬件为本机 RTX 4070 Ti SUPER 16 GiB、64 GiB RAM，不是设备端 RTX 5060 Ti。

| 场景 | 首轮后单轮秒 A -> D | 两轮进程墙钟秒 A -> D | 稳态 GPU 总占用采样峰值 GiB A -> D | 最大 primary relL2 |
|---|---:|---:|---:|---:|
| PSC 五床，两对 | 7.864 -> 7.767 | 63.252 -> 65.460 | 10.04 -> 5.35 | 1.255e-7 |
| SC60 十二床，两对，保留慢样本 | 39.498 -> 40.572 | 126.003 -> 199.212 | 10.83 -> 8.04 | 4.722e-8 |
| SC60 十六床，一对 | 43.107 -> 44.080 | 133.252 -> 133.142 | 11.18 -> 8.42 | 4.670e-8 |

计时列为全部配对样本的均值，不是筛选最快值。GPU 总占用包含桌面等其他进程，以 5 秒间隔采样；稳态从第二轮开始，不代表最低显存要求。整次运行的 GPU 采样峰值 A/D：PSC 10.05/8.97 GiB、SC12 10.83/8.47 GiB、SC16 11.23/8.44 GiB。进程峰值工作集分别约 30.0/30.0、41.0/41.0、41.2/41.2 GiB，**没有主机常驻内存下降结论**；Windows private commit 分别为 39.06/34.34、51.02/48.20、51.73/48.78 GiB，不能与工作集或显存重复相加。其他 CPU 占用采样均值约 5.6%~7.8%，没有宣称整机独占。

SC12 D 的两次进程分别为 **271.663/126.761 秒**；前一次首轮 217.556 秒、`loader_work_s=221.413`，另一次首轮 74.275 秒、`loader_work_s=79.540`，两者读取字节数和 H2D 字节数相同。恢复原部分缓存策略后仍出现该长尾，排除了“只有部分锁页候选才会发生”的解释，未定位系统层根因。**本轮确认显存与完整前投影读块次数改善，不宣称普通 OSEM 稳定加速或读取长尾已修复。** N1 路径和受保护的自动缓存 Lu208 五床 MLEM50 门未被本次固定预算 N10 短跑替代；该生产性能门未重跑。保留 D 是共用缓冲的容量修复与前投影遍历简化，不增加需要达到 5% 核心加速门的新执行策略；部分锁页扩展不进入生产实现。

SC12/16 每轮 CSRB H2D 均为 **67,102,752,976 bytes（62.49 GiB）**，两轮应用读取均为 85,833,298,112 bytes；增加床位没有增加矩阵传输次数，但图像/投影存储和计算仍随床数增加。普通迭代每轮未常驻的约 24.3 GiB 仍需由文件路径获取，不等同于全部落到物理磁盘。本次验证的是完整矩阵、合成 12/16 床的软件/容量边界，上限仍为 16 床；真实 SC60 多床采集、步进/朝向标定、长轮次收敛、CTAC 多床及 RTX 5060 Ti 现场性能均未由此关闭。

复现使用本节证据目录的 `paired.py`：`psc5 --label <new-name> --iterations 2 --reuse-cache <psc5-dir>`、`sc12` 同样参数；`sc16 --iterations 2 --order AB --reuse-cache <sc16-dir>`。省略 `--order` 是 ABBA；每个输出目录必须全新。脚本和原始 JSON/日志属于任务证据，不新增生产维护入口。完整套件入口仍为 `build/csrb-native-20260921/bin/Release/spect_tests.exe`。本轮仅本地提交、干净重建，不推送，不替换已有 SPECT_Core ZIP。

## 20. 2026-10-12 矩阵读入热点的后续配对验证

控制为干净提交 `5808455`，冻结 EXE/DLL 在 `recon_out/throughput_20261012/baseline_5808455/`。沿用第 19 节矩阵、投影、PSF、床位顺序及合成 SC60 8 mm 步进；N10 主机预算仍为 PSC 30 GiB、SC60 40 GiB。每臂 3 轮，ABBA 两对，分母缓存命中、verbosity=1；每臂上限 900 秒。数值门限仍为 primary relL2 <=1e-5，新增性能路径要求受影响工作负载核心耗时改善至少 5%，其余既有路径不得以降质换速度。无 SHA、无矩阵精度/格式/内核/配置默认值修改。

先用 SC60 十二床 `-vb 2` 分阶段定位：3 轮 loader_work=71.954 秒，forward=7.798 秒，图像更新=0.054 秒。该次分母 miss 且 debug 会改变 host/device 更新路径，只用于热点诊断，**不进入正式配对计时**；loader_work 含读入、分配和校验，不是纯物理磁盘耗时，各 profiler 项不能不加区分地相加。该次错误复用 `sc12_retained` 而不是实际 `sc12/cache_A` 的缓存位置，驱动正确拒绝将其登记为正式命中样本。

### 20.1 顺序读取提示：拒绝

尝试用共享的 stdio payload 入口和 Windows `rbS` 顺序缓存提示读取 CSRB/CSRQ，所有原检查保留。微软将 `S` 定义为顺序访问缓存提示，并非硬件吞吐承诺，见 [CRT 文档](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/fopen-s-wfopen-s?view=msvc-170)。SC12 实际两对结果：三轮核心均值 149.726 -> 159.013 秒（慢 6.20%），稳态 39.241 -> 39.501 秒/轮，进程 160.685 -> 168.162 秒；最大输出 relL2=5.58e-8。首次迭代也都变慢。完整套件 346 通过、4 原有条件跳过，但正确不等于更快。**代码已撤回，不添加运行期开关，不扩大到 PSC 或作为正式文件读取实现。** 证据：`sc12_sequential/`、`sequential_candidate.patch/exe`、`sequential_suite.xml/log`。

### 20.2 有限非负值校验：保留局部优化

唯一生产代码差异为 `validateChunkValues` 的 7 行快速检查：对每个值累积 `value >= 0 && value <= FLT_MAX` 的真假值，不在正常数据循环内分支或抛异常；失败后仍用原循环定位第一个非法值、保留源 nnz 下标和错误文本。NaN、正负无穷和负数仍拒绝，负零、非负次正规数和最大有限 float 仍接受。没有省略校验、接受历史身份、缓存“已验证”结论或增加线程/内存；CSRB 整块和磁盘子集共用这一入口，CSRQ 的 row_max/run 校验未改。

任务内 MSVC Release 微测读取真实 SC60 `chunk_000.csrb` 的前 256 MiB values，ABC CBA 顺序重复 3 次。原循环约 0.043 秒，单线程累积检查约 0.014 秒，8 线程探针约 0.004~0.008 秒；多线程分支不进入生产代码。微测仅定位机制，采用判断使用下面实际重建配对结果。`validation_probe/` 是可复现证据，不新增维护入口。

| 场景，N10，各 ABBA 三轮 | 三轮核心均值秒 A -> B | 核心降幅 | 首轮后稳态秒/轮 A -> B | 进程均值秒 A -> B | 最大 primary relL2 |
|---|---:|---:|---:|---:|---:|
| PSC Lu208 五床，30 GiB 部分缓存 | 59.399 -> 56.902 | 4.20% | 7.787 -> 7.539 | 72.917 -> 69.907 | 1.42e-7 |
| SC60 十二床，40 GiB 部分缓存 | 148.972 -> 140.061 | 5.98% | 38.759 -> 36.998 | 162.294 -> 151.921 | 6.03e-8 |

SC12 两次候选核心 140.020/140.101 秒，配对基线 149.489/148.455 秒，两对均超过 5%；PSC 是较小的附带收益，不单独宣称达到 5%。稳态降幅为 PSC 3.17%、SC12 4.54%，不能把首轮加载也在内的三轮核心降幅说成长序列稳态收益。两组新增工作集基本为零：PSC 约 30.04 GiB、SC12 约 41.03 GiB，GPU 稳态采样峰值分别约 5.4/8.1 GiB；不作显存优化结论。每臂矩阵读取和 H2D 字节数相同；SC12 三轮 loader_work 均值 98.564 -> 90.064 秒，与值检查的干预方向一致。

本机 RTX 4070 Ti SUPER 16 GiB、64 GiB RAM；其他 CPU 负载采样均值 PSC 约 4.8~4.9%、SC12 约 5.6~5.8%。这是同主机短轮配对，不是 RTX 5060 Ti 现场结果，也不是 SC60 实际机械步进、真实多床图像质量或 1000 轮收敛结论。所有实际保存的输出（PSC primary/Butterworth、SC60 primary）均通过门限；不能由短轮声称所有长序列 bitwise 相同。

N1 受保护的 Lu208 五床 MLEM50 已补测：同输入、PSF sieve/castor 1.3 mm、16 线程、自动主机缓存、分母命中，ABBA 每臂完整 50 轮，均自动容纳并锁页全部 6 块（33568 MiB），迭代期零载荷重读。A 两次核心 71.912/72.843 秒，B 为 72.385/72.236 秒，均达到原目标 <=120 秒；核心均值 72.378 -> 72.311 秒，变化约 0.09%，**不宣称 N1 稳态加速**。进程均值 121.856 -> 121.359 秒，prepare 均值 46.005 -> 44.574 秒。最大 primary relL2=1.403e-7，Butterworth=1.819e-7，均小于 1e-5；这是数值等价门，不是 bitwise 承诺。证据 `psc5_validation_n1/`，没有用 N10 短跑代替该门。

当前候选完整套件 **350 项，346 通过、4 原有条件跳过、0 失败**，扩充原权重合法性用例以覆盖负零、次正规数、FLT_MAX 和负无穷；原 CSRB/CSRQ、CTAC、多床、部分缓存独立 oracle 保留。结论：保留这 7 行局部校验优化，拒绝顺序读取候选；性能证据仅支持上述工作负载，不把较小的 PSC 收益外推为所有系统 6% 加速。仅本地提交及干净重编译，不推送，不更新现有 SPECT_Core ZIP。

证据：同根目录 `psc5_validation/`、`sc12_validation/`、`validation_candidate.patch/exe`、`validation_suite.xml/log`、`metrics.json`。中间候选未经重新配置版本头，其 EXE banner 仍为基线 `5808455`，不能仅凭该 banner 区分；必须使用保存的候选补丁和对应 EXE 路径/大小/时间。交付构建须在最终提交后重新配置，不能将实验 EXE 当干净发布版。

复现：`python recon_out/throughput_20261012/paired.py <psc5|sc12> --label <新目录> --iterations 3 --baseline recon_out/throughput_20261012/baseline_5808455/SPECT_recon.exe --reuse-cache D:/Projects/SPECT-recon/recon_out/throughput_20261012/<psc5|sc12>`。受保护 N1 使用 `psc5 --iterations 50 --subsets 1 --threads 16 --auto-host-cache`，其余参数相同。默认 ABBA，所有实际命令、输入轻量属性、逐轮与进程时间及数值差异均在对应目录。

## 21. 2026-10-12 CSRB 读取、上传和计算重叠

状态：通过回归和配对采用门，保留；未部署。控制为干净 `a635d10`，EXE/DLL 受保护副本在同一证据根的 `baseline_a635d10/`。继续使用第 20 节 PSC5/SC12 的矩阵、投影、N10、PSF 和 30/40 GiB 部分主机缓存；每臂 3 轮、ABBA、verbosity=1、分母命中。primary relL2 <=1e-5；新增调度沿用 >=5% 核心收益门，并检查 PSC 非退化、SC16 边界和受保护 PSC5 N1x50 自动缓存门。单臂 900 秒上限，失败停止对应比较；不改原始数据、精度、子集或配置，不计算 SHA，不覆盖设备包。

### 21.1 正常路径时间线

用 Nsight Systems 2025.6.3 对控制的 SC12 三轮重建采集 CUDA API、内核和传输；禁用 CPU 采样及线程切换跟踪，应用仍为 verbosity=1。以第 10 和第 30 个 `compact_mlem_update` 结束为边界，取稳态第 2、3 轮的事件区间并集，避免重叠时间重复相加。该受采集运行只定位机制，不进入无采集的性能采用比较。

控制区间 76.1959 秒，GPU kernel 并集 22.2074 秒，H2D 并集 9.2822 秒，kernel/H2D 交集仅 0.02375 秒；所有记录的 GPU kernel/memcpy/memset 并集 31.5169 秒，无所列 GPU 活动的时间 44.6790 秒。主线程 memcpy API 区间并集 9.3483 秒。最后一项不是纯 DMA 时间，GPU 空闲也不全等于物理磁盘等待。证据 `sc12_timeline_a635d10/timeline.nsys-rep`、`timeline.sqlite`、`timeline_stats.json`。

代码对应：numerator 的 `processChunk` 原先在提交当前块 GPU 工作之前同步调用下一块的 `loadSubset` 和预取，之后每块又等待 GPU 完成。即使名为 Async，pageable 源上传也可能阻塞主线程；这是 [CUDA 同步语义](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) 允许的行为，不能仅凭 Async 后缀声称有重叠。

候选仅把同一主线程的下一块加载/预取移到当前块所有计算提交之后、原完成等待之前。继续使用现有 GPU 槽、上传 stream、sourceOwner 和事件；不增加 CPU worker、预载缓存、锁页策略或内核。普通 CSRB 与 CTAC 共用该调度，CSRQ 不改。当前槽不能成为下一块上传目标；加载/上传异常先同步现有计算和上传，再抛出，不继续重建。内部 loader 回调仅做主机加载，不允许重入 sparse CUDA 状态。

候选完整套件 351 项，347 通过、4 个原有条件跳过、0 失败；新增独立小矩阵检查正常预取与加载异常后当前 numerator，并检查异常出口流已完成；该检查及 ROT90 部分缓存 oracle 重复 5 次均通过。Nsight 采集驱动最初只检查采集器 stdout，误报缺少成功标志；实际 recon 独立日志、输出体和 Nsight 文件齐全，驱动已改为检查应用日志，没有把该检查错误描述成重建失败。任务脚本 `profile_io.py`、`timeline_stats.py` 属于可复现证据，不增加生产入口。

### 21.2 无采集配对结果和长尾边界

PSC5 的 ABBA 三轮比较一致：核心均值 59.160 -> 52.924 秒（-10.54%），稳态 7.778 -> 6.331 秒/轮（-18.61%），进程 72.857 -> 66.053 秒（-9.34%）。两对核心分别 58.180 -> 53.599、60.140 -> 52.249 秒，均超过 5% 门。最大保存体 relL2=1.162e-7；包括 primary 和 Butterworth，不只比较一张图。

SC12 首轮 ABBA 三轮出现显著读取长尾：A 两次首轮 191.256/257.388 秒，B 为 66.999/177.534 秒；A 还有一个稳态样本 106.813 秒。全部样本保留在 `sc12_overlap/`，不把核心均值 339.213 -> 188.014 秒解释为稳定 44.57% 加速，也不宣称磁盘长尾被修复。为增加稳态观察，每臂改为 6 轮再做一次 ABBA，输入、参数及门限不变，见 `sc12_overlap_long/`。

SC12 六轮两对核心为 261.263 -> 218.771 秒（-16.26%）、380.312 -> 229.259 秒（-39.72%）。后一对控制第 5 轮仍有 153.941 秒长尾，不能把均值降幅 30.17% 当稳定吞吐承诺。每臂的首轮之后共 10 个样本全部纳入：A 均值/中位数 50.488/38.786 秒，B 为 31.503/31.392 秒；中位数是观察到长尾后增加的描述性统计，**不是替换原 5% 核心采用门的事后标准**。核心两对均通过原门，较少长尾的第一对中 loader_work 158.857 -> 160.974 秒，并未变快，而核心减少 42.492 秒，与隐藏等待而非减少读取的干预方向相符。最大保存体 relL2=3.840e-7。

SC16 仅作一次 AB 两轮边界检查：全部退出 0，核心 117.221 -> 99.921 秒，第二轮 41.441 -> 33.876 秒，最大保存体 relL2=4.662e-8。未重复，不单独宣称稳定加速。SC12/16 的矩阵 H2D 仍为 67,102,752,976 bytes/轮，应用读取字节和加载次数对应相等；没有用减少床位或截断矩阵换性能。

PSC5 N1x50 自动缓存保护门 ABBA：A 核心 75.186/76.029 秒，B 76.272/76.146 秒，均低于 120 秒目标。均值 75.608 -> 76.209 秒（+0.60 秒，+0.80%），稳态 1.5011 -> 1.5052 秒/轮（+0.27%），进程 122.902 -> 124.280 秒。全量 6 块/33568 MiB 均锁页，迭代期零文件载荷重读、H2D 总量相等，最大保存体 relL2=1.655e-7。这里**没有全缓存加速结论，也不把轻微耗时增加写成零退化**；原保护性耗时和数值门均满足。为这一量级增加独立调度分支没有采用依据，保留一个共用实现。

主机峰值工作集 A/B 基本相同：PSC 部分缓存约 30.04 GiB，SC12 41.03 GiB，SC16 41.19 GiB，PSC 全缓存约 33.21 GiB。稳态 GPU 总占用采样峰值分别为 PSC 5601/5580 MiB、SC12 六轮 8367/8363 MiB、SC16 8813/8801 MiB、PSC N1 14560/14607 MiB；含其他进程，以 5 秒间隔采样，不作为最低显存要求。其他 CPU 平均负载约 7.6%~10.3%，不宣称整机独占。未改变矩阵精度、PSF、子集、存储格式、默认值、wrapper 或 CLI；本机仍为 RTX 4070 Ti SUPER/64 GiB，不外推为 RTX 5060 Ti 实测。SC60 步进 8 mm 仍只是合成验证参数，真实多床标定和长轮次图像质量不由本轮关闭。

### 21.3 重叠复核、采用与复现

候选同条件 Nsight 稳态两轮区间 62.6884 秒：kernel 并集 22.6214 秒、H2D 9.7399 秒，两者交集 **5.9452 秒**；未记录到该进程 GPU kernel/memcpy/memset 的时间为 **36.2272 秒**。对应控制为 76.1959、22.2074、9.2822、0.02375、44.6790 秒。该窗口 H2D 字节相同（133,464,363,016 bytes）；kernel 和 DMA 本身并未变快，而其重叠增加。窗口以 update 完成为锚，跨子集预取可能落在边界外，不用它替代应用记录的整轮字节计数。两次受采集运行只证明调度干预确实增加了重叠，不替代无采集的配对采用门。

保留此共用调度，不增加第二套内核、CPU 读线程、锁页缓存扩展或运行期开关。数值/格式/床位/能窗语义保持原合同；CSRQ 没有变更。既有小型独立 oracle 覆盖 ROT90/ROT90Z、部分缓存、CSRB/CSRQ、CTAC、双窗及多床；新增异常检查验证当前计算完成后才向上抛错。Compute Sanitizer memcheck 对新增检查和 ROT90 部分缓存 oracle 报告 0 errors；不能据此扩展成所有实机和长序列的绝对正确性证明。PSC N1 的轻微耗时增加、SC60 文件读取长尾和现场 5060 Ti 未测均保留为边界。

复现：同根 `paired.py` 使用 `--baseline recon_out/throughput_20261012/baseline_a635d10/SPECT_recon.exe --reuse-cache D:/Projects/SPECT-recon/recon_out/throughput_20261012/<psc5|sc12|sc16>`；PSC5 3 轮 ABBA、SC12 6 轮 ABBA、SC16 2 轮 AB，PSC5 保护门加 `--iterations 50 --subsets 1 --threads 16 --auto-host-cache`。每次用新的 `--label`，不能覆盖已有样本。Nsight 用 `profile_io.py <新目录> --exe <EXE>`，随后 `nsys export --type sqlite` 和 `timeline_stats.py <SQLite>`。原始证据为 `psc5_overlap/`、`sc12_overlap/`、`sc12_overlap_long/`、`sc16_overlap/`、`psc5_overlap_n1/`、`sc12_timeline_*/`、`overlap_suite.xml/log`、`overlap_memcheck.log` 及 `metrics.json`。

实验 EXE `overlap_candidate.exe` 及 `overlap_candidate.patch` 冻结候选；实验时未重新配置版本头，banner 仍为 `a635d10`，以实际路径/大小/时间和补丁区分，不将其当正式干净构建。最终提交后须重新配置并构建发布 EXE，复查完整套件和 `--version`；本轮不推送，也不修改 SPECT_Core ZIP 或现场部署。
