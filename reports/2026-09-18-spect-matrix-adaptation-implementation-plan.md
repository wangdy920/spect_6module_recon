# SPECT 系统矩阵通用适配：审查结论与实施任务书

日期：2026-09-18  
状态：设计审查与实施规格；不是已修改、已编译或已完成实机验证的生产补丁。

## 0. 结论与范围

采用“多格式 reader → 统一 MatrixContract → 能力校验 → 一次性 ExecutionPlan → 现有高效执行路径”。

首轮保护 PSC `ROT90_Z` 生产路径，增加 SC60 `ROT90` 的 CSRB 路径。保留现有 CSRB v1 chunk 字节布局，不重写 SC60 大矩阵，不经 CSRQ 中转，不根据文件夹名字决定数学语义或 GPU 调度。

将三个问题分开：

1. **字节兼容**：能否解释 metadata 与 chunk 编码。
2. **算子兼容**：行/列编号、对称变换、子集和伴随是否一致。
3. **执行效率**：在设备约束下选择 RHS 分块、流式读入和缓存策略。

此次仅获得契约草案及此前调度设计资料，未获得当前主线 C++/CUDA 文件、真实矩阵 payload 或本次运行环境。因此不声称已排除所有内核中的固定 8 分支假设，也不虚构 88/92/124 B 的完整字段偏移。

## 1. 来源与已知事实

### 1.1 本次草案支持的事实

依据 `2026-09-18-csrb-universal-matrix-contract-design.md` 第 1、2、6 节：

- 88 B 与 92 B metadata 已由主线支持；124 B SMMD v3 属于 SC60 原型格式，指定主线 EXE 在 metadata 读取处失败，未开始迭代。
- 三类 metadata 对应的 CSRB chunk 都使用 28 B 头、i64 row_ptr、i32 col_idx、f32 values。
- PSC 为 ROT90_Z：4 个 XY 分支 × 2 个 Z 分支。
- SC60 为 ROT90：4 个 XY 分支 × 1 个 Z 分支，compact_det_z=det_nz=96。
- 现有 CSRB 分支枚举调用了分支计数方法，但 metadata 的方法返回值仍固定为 4 和 2。
- CSRQ 是实验格式，当前 reader 和 kernel 仍有格式及固定 8 分支限制。

本次草案列出的几何可推导：

| 项目 | PSC 示例 | SC60 示例 |
|---|---:|---:|
| image_voxels | 3,168,000 | 1,944,000 |
| stored_rows | 103,680 | 829,440 |
| expanded_rows | 829,440 | 3,317,760 |
| 分支数 B | 8 | 4 |
| 一个 FP32 图像向量 | 12.085 MiB | 7.416 MiB |

这些是根据文档几何的计算值，不是性能测量。两套矩阵的探测器尺寸、图像网格及物理系统不同，不能将速度或文件大小差异单独归因于 Z 对称。

小型算术审计：草案给出的 67,106,070,664 B / 2^30 = 62.4973985 GiB，按两位小数为 62.50 GiB；草案中的 62.48 GiB 是近似数值上的小偏差，不影响路线判断。

### 1.2 延续的历史执行约束

依据 `2026-08-15-sysmat-dispatch-design-context.md` 第 2、3、5 节：

- 已采用 ROW-packed 2-RHS forward + CSR_ALG2 transpose/odd remainder + prefetch。
- 历史 Sampling 50-iter iteration_core_s=148.051 s；这是旧日期、旧工况的记录，不是本次代码状态或 SC60 的实测值。
- 不重开 mmap payload、第三个持久常驻 CSR chunk、FP16 values、运行时 NVRTC、目录名调度。
- 保留既有两个持久 CSR 槽和 6 GiB 设备预留策略；新增 scratch 必须进行总预算核对。
- 不将列有序设为 packed-K 正确性的必要条件；不在重建启动时对整套 CSR 排序。
- packed-4 与新的 transpose 优化不属于本轮兼容改造的默认项。

以上旧资料中的 generator/compact “v4”，不能直接当作本次新提出的 `SMMD metadata version 4`。metadata version、CSRB payload version、generator version 和 engine version 必须分别记录。

## 2. 必须先修正的契约问题

### 2.1 v4 header 字节数不一致

草案有 6 个标量 u32 加 reserved[3]，合计 36 B，而不是声明的 32 B。

建议仍采用 32 B，修正为：

```text
byte offset  type  field
0            u32   magic = 0x444D4D53
4            u32   version = 4
8            u32   header_size = 32
12           u32   core_size
16           u32   tlv_size
20           u32   flags = 0
24           u32   reserved0 = 0
28           u32   reserved1 = 0
```

全部小端。逐字段序列化，禁止以编译器 struct padding 或 sizeof(struct) 作为磁盘协议。

发布 v4 writer 前必须冻结 core 的逐字段类型、字节偏移、单位、枚举值与 core_size。草案当前只列字段名字，尚不足以直接实现互操作的独立 reader/writer。

文件长度必须满足经溢出保护的：

`file_bytes = header_size + core_size + tlv_size`。

当前 v4 必须校验受支持的 header_size/core_size、flags/reserved；不能将更大的未知 core 静默当作已支持语义。

### 2.2 reader 校验与 payload 校验分层

metadata reader 不应为每次启动先完整扫描约 62.5 GiB 文件再由重建器读第二遍。

建议分三层：

- **ContractValidation**：metadata 字段、版本、有限正 voxel、维度与乘法溢出、profile 与几何关系。
- **ChunkInventoryValidation**：所有 chunk 的头、长度、角度区间、rows、nnz 汇总和覆盖关系；只需少量读取。
- **PayloadValidation**：row_ptr、col_idx、values 的实际内容，结合首次必要读入验证；另提供离线深度审计。

采样可用于性能特征，不可代替完整正确性证明。已验证状态的跨运行复用需要可靠的矩阵身份/完整性机制，不能只靠同名目录。

### 2.3 CSRB v1 没有显式 n_cols

草案中的 `chunk matrix cols == image_voxels` 不能通过 chunk 头直接验证，因为该头没有 n_cols。

应改成：

- 算子的列维数由 MatrixContract.image_voxels 定义。
- 实际 payload 的每个 col_idx 满足 `0 <= col_idx < image_voxels`。
- 不要求 `max(col_idx) == image_voxels-1`；ROI 外或未被照射的列可以完全不出现。
- 若未来列索引从完整体素网格改为 ROI 压缩编号，必须显式提供列映射并获得执行支持，不能只附一个 optional ROI_INFO。

### 2.4 几何/编号语义不能只有 layout 名字

首轮仍沿用生产约定，但需要在 profile 规范与 golden fixture 中明确：

- x-fastest 的列编号公式、轴方向、体素中心和旋转/镜像中心。
- 完整与 compact 的行编号，角度正方向、起始角约定、detector x/z 的顺序。
- 对每个 branch，完整投影行与 compact 行如何对应，是否有 detector index 反转。
- 奇偶尺寸与固定点规则；不能将 detector Z 的奇数中心行简单按两个独立分支重复使用。

旧文件中无法恢复的字段须标为旧 profile 隐含约定或未声明，不能为了“通用”填上未经证明的 0°、1°、原点等数值。

未来非均匀角度或其他行枚举若影响测量行对应关系，应使用 core 或关键扩展；不能只放人读 metadata.txt。

### 2.5 数值编码不等于物理含义

FLOAT32_LE 只说明字节如何变成浮点数，不说明系数是每次衰变探测概率、单位活度响应或已含曝光/归一化的量。

作为后续扩展建议，可增加关键 `MODEL_SEMANTICS`，描述系数单位及矩阵已经包含的物理效应；不把 OSEM、运行时 norm 文件、滤波参数塞进 metadata。

旧矩阵不具备这些信息时保留现有语义并标记 provenance，不反向虚构定量约定。当前首轮兼容仍不等于已完成跨系统定量验证。

## 3. 建议的软件分层

```text
Legacy88Reader / Legacy92Reader / SmmdV3Reader / SmmdV4Reader
                         |
                         v
              immutable MatrixContract
                         |
         ContractValidation + ChunkInventory
                         |
            EngineCapabilities::validate
                         |
          build ExecutionPlan once per run
                         |
       +-----------------+-----------------+
       |                                   |
 CSRB v1 ChunkSource              future payload adapters
       |                         (not automatically supported)
       +-----------------+-----------------+
                         |
          existing forward / ratio / backward
          + profile-specific transformations
```

### 3.1 MatrixContract

只描述矩阵如何解释：payload 编码、图像/投影维数、voxel_xyz、profile、row/column layout、必要语义与来源。

不要包含：GPU 型号、packed-K、prefetch、显存槽数、OSEM 子集数、PSF 参数、bad detector、运行时 norm factor。

### 3.2 MatrixTraits

描述从实际 payload 或可信生成记录获得的可选性能属性：行长分布、列跨度、局部性、有序/唯一状态及证据等级。

采样后可以得到“未发现乱序”，但不能因此声明全矩阵 sorted。优化路径的正确性不得依赖未经验证的属性。

### 3.3 ExecutionPlan

由 contract、engine/device capability、recipe 和已测量策略构建：

- profile 对应的前向及伴随变换函数。
- B=数学分支数，K=一次处理的 RHS 数；分开命名。
- 后端函数/模板实例、scratch 大小及布局、CSR 槽预算。
- 每个 OSEM subset 的行范围及对应 chunk slice。
- 缓存键、传输依赖与错误诊断信息。

只在 CPU 构建计划时分派一次。禁止将 metadata 版本判断、文本解析、通用虚函数或 TLV 解释放进每个 nonzero 的循环。

已有 CSRB adapter 原样提供其数组，不因接口抽象增加全矩阵复制、全矩阵转置或排序。

## 4. 对称 profile 与执行能力

### 4.1 首轮维持封闭 enum

保持草案 ID 0..7 的 registry；首轮执行白名单只放入已验证的 ROT90_Z 和新增并验证的 ROT90。

“profile ID 已登记”“metadata 可解析”“引擎允许执行”是三个不同状态。认识 ROT45_SECTOR 的名字不代表能正确执行。

不同时在磁盘存 profile、use90、use45、useZ 和独立 branchCount 等冗余真相。旧布尔字段只在 adapter 边界归一一次，并对冲突组合失败。

### 4.2 branchCount 不是对称语义

ROT90 与 ROT180_Z 都有 4 branches，但变换不同。执行分派必须保留 profile 的完整身份，不能只写 `run<4>()` 后默认代表 ROT90。

建议变换核按 Profile 专门化；与对称无关的稀疏计算继续复用现有 packed-2/ALG2 实现。后续 CSRQ 的 branch specialization 同样须携带已验证的映射语义。

对于基于索引置换的 ROT90，Nx=Ny、vx=vy 及旋转中心一致是重要条件。首轮仍可按引擎现有边界拒绝任意 anisotropic voxel；不要把引擎暂不支持误写成所有数学 profile 必须 vx=vy=vz。

### 4.3 ROT45_SECTOR 延后启用

草案的 `full=8*(compact-1)` 只提供数量关系，无法确定每个边界行归谁、反射方向和 detector 映射。

以 full=360 推导 compact=46 为例，若简单复制 8 个包含两端点的 sector，会得到 368 个候选角度，而不是 360；必须定义边界 ownership 或精确权重。

这也不能被替换为“图像任意旋转 45° 再插值”：那会改变原算子及其伴随。具体是否是旋转/反射组合，需要核对生成器和原型实现，本次资料不足以给出其真实映射。

首轮允许审计显示该 profile，但运行时拒绝。未来启用前提供 explicit row ownership、图像映射和伴随 reference。

### 4.4 把前向与伴随一起定义

设计表达式：

`A x = sum_b R_b C T_b x`

其中 C 是 compact 矩阵，T_b 为图像侧变换，R_b 为 compact 到 full 的行映射/选择。对应：

`A^T y = sum_b T_b^T C^T R_b^T y`。

无 Z 适配要覆盖前向、ratio 行选择、反投、transform-add、灵敏度、子集行映射、mask/norm 广播和缓存键，而不只是一个 zBranchCount()。

对于 OSEM，subset sensitivity 必须对应实际 `A_s^T 1` 或引擎采用的等价完整校正算子。不能无条件用完整 sensitivity 除以子集数，也不能沿用 Z 分支的乘 2 捷径。

## 5. v4 扩展、身份与版本策略

### 5.1 TLV

保留 unknown critical fail、unknown non-critical skip。建议注册表明确 tag、criticality、value schema、最小/最大长度、singleton/repeatable。

- 重复检测应按去除 critical bit 后的 base tag ID，避免同一 singleton 的两个 critical-bit 变体绕过检查。
- 每个可能独立演进的结构化 TLV 在 value 中定义 schema version。
- 关键 tag 的未知 schema version 也失败。
- 改变既有 tag 的含义时不能只因为“tag 已知”就让旧 reader 误读；使用新 schema/version 并遵守关键性规则。
- length 与剩余字节比较，所有偏移和长度运算溢出保护。

### 5.2 身份与 recipe

保留 human-readable IDENTITY/GENERATOR 等审计信息；其非关键属性不应被解释为生产中可以任意混配矩阵。

建议 recipe 绑定预期 matrix identity/fingerprint，并对系统、能窗、探测器编号方案进行一致性检查。单纯 shape 一致不足以防止串矩阵。

legacy 无 IDENTITY 时可绑定已登记的 manifest/fingerprint，而不是从目录名推测系统。

SC60 首轮显式使用空或 SC60 专用 bad-detector 配置，不沿用 PSC detector 68；索引落在合法范围内，也不代表物理含义正确。

### 5.3 缓存键

凡影响 sensitivity/denominator 的输入都必须进入缓存身份：矩阵 fingerprint、profile/mapping 版本、subset 归属、bed 几何、mask/bad detector、norm、以及实际进入算子的其他校正和 PSF 等。

磁盘元数据/索引表的快速一致性检查和全 payload 的强 hash 不是一回事。强 hash 可生成期或离线一次建立，运行时不要为命中缓存再扫描全量 62.5 GiB。

### 5.4 版本独立

仅增加执行后端、B/K 模板或 GPU scratch 布局，不改 CSRB chunk version。只有磁盘负载字节解释变化才升级 payload version。

v4 metadata 是新 reader 的输入之一，不是让 SC60 先转换 metadata 才能运行的强制前置。先实现 124 B reader/no-Z 可独立验收，随后实现 v4 writer 与 roundtrip。

## 6. 保持效率：本轮必须审查的路径

### 6.1 不把 B、bedCount 和 K 混为一谈

B 是对称分支数，RHS 总数还可能乘以 bedCount；K 是一次稀疏扫描并行计算的 RHS tile。

SC60 的 B=4 不意味着 K=4 最优，也不意味着要换掉 packed-2。历史 0.60*L2 的预算只作为待验证候选护栏，不是已经证明的最优定律。

按草案 SC60 网格，一个 FP32 向量约 7.416 MiB；4 个约 29.663 MiB。对于历史 48 MiB L2 的设备，0.60*L2=28.8 MiB。这个简单候选模型本身也不能直接批准 K=4；是否有收益仍需要同设备同工况 A/B。

### 6.2 OSEM 逻辑遍历不等于物理 I/O

草案“不增加每轮矩阵遍历量”需限定为逻辑上处理的有效行/nnz；它不能保证磁盘读取、CPU 复制或 H2D 字节量不增加。

若 subset 按 `compact_angle % N` 交错，而磁盘 chunk 按连续角度切分，同一个文件可能服务多个 subset。逐 subset 全量加载这些文件会出现读取放大。

仅作示例，不是 SC60 实测：如果 18 块恰好各包含 5 个连续角度，N=10，则一块可能分属 5 个子集。在整块加载且无有效跨 subset 缓存的条件下，一轮可发生约 5 次整矩阵等量流量，即约 312.49 GiB，而不是约 62.50 GiB。实际放大系数须读取 chunk 头与实际 subset 规则后计算。

建议计划中生成 `SubsetSlice`：

- 从 compact angle/row 区间确定所属 subset。
- 使用 row_ptr 定位对应 col_idx 和 values 的连续字节段。
- 在现有可复用缓冲中拼接所需 row segments 并 rebase row_ptr。
- 保留磁盘中的列顺序；这不是 CSR 排序，也不改写原矩阵。
- 不超过既有两个持久 GPU CSR 槽；实际 slice 大小也参与预算。

子集内图像固定时可对 slice 执行 forward → ratio → backward 累加，待该 subset 的全部贡献完成后再更新图像。

禁止为方便 I/O 把所有 subset 都用同一张旧图像算完再更新；这不再是相同的 OSEM。也不应为了顺序读把交错子集静默替换成连续角度子集。

### 6.3 流式和显存预算

保留已验证的 pinned prefetch 和现有流水线，审查其是否兼容 variable profile、不同最大 chunk 和 subset slices。

预算必须以最大 chunk/slice 与 scratch 峰值计算，不能只看 18 块的平均大小。62.5 GiB 矩阵不能在 64 GiB 主机上假定全缓存可用；host cache 应有显式上限并给系统与重建状态留空间。

CUDA 官方文档支持用适量 page-locked staging 和适当 streams 实现传输/计算重叠，也提醒 pinned memory 过量会影响系统。这里不建议 pin 整套矩阵，也不因文档存在某个更宽 SpMM API 就替换本项目已验证的 packed-2。

### 6.4 必须记录的性能量

```text
contract_parse_s
inventory_validate_s
payload_validate_s
iteration_core_s
end_to_end_s
matrix_disk_read_bytes
matrix_host_copy_bytes
matrix_h2d_bytes
host_cache_hit_bytes
host_peak_bytes
device_peak_bytes
forward_s / ratio_s / backward_s / transform_s
```

注意异步重叠后各阶段计时不能简单相加当作总墙钟时间。

旧 gate 保持其已冻结输入与统计口径；新增 SC60 自己的 1/少量迭代正确性测试、短迭代诊断和较长性能测试。不要将 PSC 与 SC60 或不同 bedCount/OSEM/MLEM 的绝对秒数直接比较为优化收益。

## 7. 文件级改造与提交顺序

下表是基于文档源码指针的建议；具体类名/API/调用点须在当前仓库确认。

| 提交 | 修改范围 | 关键输出 |
|---|---|---|
| A | 新 MatrixContract、readers；CompactSymmetryMetadata.h/.cpp | 88/92/124 读入并归一；旧字段兼容接口；不改生产执行语义 |
| B | ReconDataLoader.cpp、能力检查、recipe/缓存身份 | metadata 几何优先；profile/布局/编号能力校验；隔离 SC60 配置；避免旧 sensitivity 误命中 |
| C | CompactSymmetricProjectorCUDA.cpp、CudaSparseOpsState.h、涉及变换与 ratio 的 CUDA 核 | ROT90_Z 不变；新增 ROT90；同一组映射驱动前向/伴随/灵敏度/子集 |
| D | ExecutionPlan、既有加载与预取路径 | 检查固定 8 的内存与步长假设；记录 I/O；必要时加入 subset slice，不改 OSEM 语义 |
| E | SMMD v4 reader/writer、生成器与协议文档 | 字段偏移冻结；32 B header；TLV；roundtrip；修正文档 88/96 与实现 88/92 的漂移 |
| 后续 | CSRQ converter/kernel、更多 profile | 在 CSRB 已正确且有实测瓶颈后单独验收，不并入首轮兼容补丁 |

v4 writer 不应阻塞 A–C 的 SC60 适配，但一旦发行 v4，必须具有独立测试与完整 wire schema。

## 8. 验收测试

### 8.1 格式与错误输入

- 88 B/92 B 的归一结果与当前 reader 相同；124 B 使用真实 fixture，与 SC60 原型 reader 字段逐项对照。
- 124 B 中 decimal voxel 字段按既有真实字段范围解析，要求完整、有限、正值，避免 locale 引入逗号等歧义。
- v4 roundtrip、未知 critical/non-critical tag、未知 tag schema、重复 singleton、错误长度、非零保留字段、未知 profile。
- 截断 chunk、非法 magic/version、负 rows/offset/count、行区间缺失或重叠、nnz/文件长度溢出。
- CSRB v1 文件长度：`28 + 8*(rows+1) + 8*nnz`，运算有溢出保护。
- row_ptr[0]=0，row_ptr[rows]=nnz，非负单调、范围合法；col_idx 不越界；values 有限且符合当前非负物理矩阵语义。
- 空行/未出现的体素列不是结构损坏；有计数却无响应等物理不一致另行报告，不默默修数据。
- 区分 unsorted 与 duplicate；不将乱序拒绝为坏矩阵。重复列的支持须明确，不能默认所有后端都保证正确。

### 8.2 算子正确性

- 建立小尺寸 CPU float64 显式展开 reference，对每个支持 profile 分别比较 Ax 与 A^T y。
- 做伴随内积测试，但不能只做伴随测试：前后都用了同一个错误置换时，仍可能互为伴随。
- 采用非中心点源、非对称图像、边角及轴向样例；不要只用均匀圆柱判断 no-Z 是否正确。
- 独立测试完整与每个 subset 的 sensitivity，mask/bad detector、bed 变换、PSF 等按实际组合覆盖。
- 对 N=10 验证每个有效完整投影行的覆盖/归属，不重复、不遗漏；对于后续 sector profile，额外覆盖端点 ownership。
- 再进行真实 SC60 短迭代，检查值有限、shape、计数和 sensitivity；“不崩溃”只算 smoke，不算通过算子验收。

### 8.3 冻结回归与性能

- 纯 reader 重构优先要求 contract 与输出不变。
- 共享执行路径保持现有 same-input relL2<=1e-5 门；不要在新格式失败后自动放宽。
- 先测同一旧二进制重复运行的数值/时间波动；选择 ALG2 不能直接推导 transpose 必然 bitwise deterministic。
- 性能比较要同设备、同输入、同迭代/子集/bed/PSF、相同 cold/warm 和 denom cache 状态；旧 gate 的口径不变。
- 可以新增“旧生产路径不出现超出测量噪声的稳定退化”的发布门。具体百分比先根据重复测量确定，不能伪造为既有门槛。
- SC60 用其独立基线衡量优化，记录读取量与显存，不承诺 B 从 8 减到 4 就加速一倍。
- CSRQ 的 relL2<=1e-5 是待达到的验收条件，不是 u16 companding 对任意矩阵/迭代次数的数学保证；不达标就不进入生产。

## 9. 原草案八个开放问题的决策

| 问题 | 建议 |
|---|---|
| profile enum 还是两轴组合 | 首轮封闭 enum；内部 traits 可分解 XY/Z，但磁盘不存多套冗余真相。 |
| 任意角度/transform table | 留关键扩展接口；首轮不执行。能描述和能执行分开。 |
| 45° sector 边界 | 比例公式不够。启用前明确 row ownership/映射/伴随；本轮禁止执行。 |
| anisotropic voxel | core 保留三元组；先保持引擎现有限制。未来按实际算子能力与 profile 放开，不靠 metadata 自称“支持”。 |
| TLV version | critical bit 保留，结构化 value 采用明确 schema version；未知关键 schema 失败。 |
| system family / matrix ID | 审计身份可留非关键 TLV；生产 recipe 绑定预期 fingerprint。影响行列/数值解释的字段必须关键化。 |
| CSRQ 的 B/K 泛化 | 后续采用有限的编译期专门化与一次运行期分派；B 与 K 分开。只有字节解释改变才升级 chunk version。 |
| 生成端 ROI | 可作为独立派生矩阵项目，但 ROI 误裁会改变 A，不是无损加速。首轮先量化 I/O 放大；保留原矩阵、映射与定量/图像回归。 |

## 10. 最终实施边界

本轮要交付的是：**新 reader 读旧格式，统一契约选择正确的已验证算子，既有高效核不退化，SC60 no-Z CSRB 可正确运行。**

本轮不交付也不声称：任意系统矩阵已经通用、任意 45° 旋转已支持、CSRQ 已满足所有误差门、非均匀角度/任意 voxel 已通过、SC60 性能已达某个秒数、或生产源码已经完成修改。

### 参考资料

1. 用户本次文件：`2026-09-18-csrb-universal-matrix-contract-design.md`，重点第 1–9 节。
2. 历史文件：`2026-08-15-sysmat-dispatch-design-context.md`，重点第 2、3、5、10、12 节。历史测量不代表本次代码状态。
3. NVIDIA cuSPARSE 官方文档（本次查阅页面显示 13.4），CSR、cusparseSpMV、cusparseSpMM 章节：CSR 乱序列支持与重复列的区别；transpose/算法的确定性边界。没有据此要求升级项目 CUDA 版本。
4. NVIDIA CUDA C++ Best Practices Guide（本次查阅页面显示 13.4），10.1.1–10.1.2：page-locked memory、异步传输与计算重叠及 pinned memory 资源限制。

## 11. 本地源码复核与实施记录（2026-09-20）

本节补充并在冲突处修正前述文档建议。基线为 `635dec6e61e7c8f9f9bec4accfc8015c673a87ab`；原 Release `build/clean-635dec6-20260914` 保留，候选独立构建于 `build/sc60-adapter-20260920`。当前是工作树候选，不代表生产采用。

### 11.1 复核结论

- 原源码已有 `CSRSubsetView`、GPU 子集行段上传和跨子集预取；尚无磁盘 payload 分段读取。应扩展现有路径。
- 原预取函数从临时 shared_ptr 构造非拥有 slice。超过 host cache 预算时，slice 指向已释放 payload；全缓存小矩阵测试未覆盖这一条件。
- 原 numerator 主循环与预取搜索都先加载整 chunk，再判断 slice 是否为空。SC60 实际 18 块各 5 角度，共 67,106,070,664 B。N10 无主机缓存假设下，仅主循环即 180 次整块加载、约 624.97 GiB/完整迭代，另有预取；第 6.2 节 5x 示例只适用于加载前先过滤无关 chunk 的路径，不能视为原实现上界。这是源码推算，不是磁盘实测。
- 原真实运行的 sensitivity 检查只比较全体素总和；已有 toy CUDA 测试另有向量 relL2 检查。两类证据不可混淆。
- 28B compact CSRB chunk header 与历史 24B CSRHeader 不是同一 wire 格式。

### 11.2 首轮实现范围

沿用 `CompactSymmetryMetadata` 作为归一模型，增加封闭 `Rot90/Rot90Z` profile 和 124B SMMD v3 reader；不新增平行 MatrixContract 类、ExecutionPlan 框架或 v4 writer。旧格式计算精度及 PSC 对称路径保留，CSRQ 对 ROT90 显式拒绝。

切片与 CUDA 缓存保留源数据所有权；加载前跳过空 slice，弱引用复用活跃 chunk。加入应用读取和 H2D 计量，运行时 sensitivity 改为体素向量 relL2 校验，阈值沿用原检查的 1e-4；toy oracle 仍使用 1e-5。

新增 SC60 单床 recipe，并增加显式 matrix_grid_output 选项，解决 PSC 固定 100x100 裁剪与 SC60 90x90 网格冲突。矩阵本体不改写。原 metadata、投影作为只读输入。

### 11.3 验证契约与状态

先执行 reader/错误输入、零/部分缓存生命周期、独立 ROT90 明文展开 reference、N1/N10 numerator/sensitivity 测试及既有回归；然后在真实 SC60 两个 BIN 上进行短迭代。使用 90x90x240、0.4 mm、PSF 1.5 mm，保存 primary，要求有限、非零且输出网格正确。运行参数、EXE/输入身份、日志与统计进入 `recon_out/sc60_adapter_20260920`。

短迭代与计量只证明此次输入的执行兼容性，不代表 1000 轮收敛、影像质量、定量标定或生产采用。长迭代须在短验证通过后另行记录；不把原型 CU 当作生产执行入口。

### 11.4 已完成的验证

后续说明（同日）：第 11 节短测 EXE 已按下述 SHA 保存在 `recon_out/sc60_reference_compare_20260920/SPECT_recon.pre-timing.exe`。构建目录中的当前 EXE 后续仅增加逐迭代/子更新计时输出；新的 100 轮参考图像比较见 `2026-09-20-sc60-reference-comparison.md`，不要将当前可执行文件路径当作不可变身份。

- 新 Release EXE 与 spect_tests 均完成全项目构建，CUDA 架构 86/89/120。最终 EXE SHA256：`0A7C28393CA8AD237DA5C2D5A3B5F36CAC511B15D044CFCD9F5A7637EFEB5134`。
- 全套测试 312 项：308 通过，4 跳过（显式 opt-in 冻结输出滤波门和 3 项本机缺失 260601 数据）。日志与机器结果：`recon_out/sc60_adapter_20260920/all-tests.log`、`all-tests.json`。
- ROT90 独立显式 XY 排列表覆盖 N1/N10、18 个 5-angle chunk、零/部分/全 host cache，验证 forward、numerator 和带非均匀权重/屏蔽的 sensitivity，relL2 <=1e-5。零缓存另断言实际重读次数超过 chunk 数，避免全驻留掩盖生命周期问题。
- 既有 resident OSEM/BSREM host 对照扩展到 ROT90 与 ROT90_Z；matrix-grid 单床保留及多床拒绝有专门测试。GPU warmup 明确最多两个常驻 CSR 块，遵守现有资源边界。
- Derenzo `1:1` 成功：1944000 个体素全部有限，1434359 个非零，min=0，max=223.513015747，sum=26770883.9659。MHD 确认 90x90x240、0.4 mm、MET_DOUBLE。该次使用修正日志文字前的 EXE，SHA256 `46BAA48809788C46CD422DA5E9A8708F97B77A56990B06A00CC689F9C1FDE152`；之后仅修正输出模式日志并补单床测试。
- N1 单次描述性计量：denom miss，准备 277.012 s，iteration_core 97.207 s，full_command 374.759 s；迭代阶段应用 payload 读取 41,065,782,680 B / 11 次，矩阵 H2D 59,679,075,984 B。不是与旧版本配对的性能比较，也不是物理磁盘吞吐测量。

输入 SHA256、矩阵文件大小/时间清单分别在 `input-hashes.json`、`matrix-inventory.json`；矩阵 payload 未做全量 SHA256 重读。原保护 EXE SHA256 为 `160E6E3300AFBBDACD2B892CF98CF976543DBDF8CEF2C812D695F2E87FAF2EBB`。

状态：首轮源码适配、全构建/单元回归、Derenzo N1/N10 与第二份 BIN N10 短测均完成。1000 轮、PSC 五床性能门、SC60 CSRQ 及 v4/TLV writer 均未在本轮闭合；保持候选，不替换保护 Release。

Derenzo N10：1 个完整迭代、10 个子更新，1944000 个体素全部有限，1126634 个非零，min=0，max=1656.793701172，sum=16521565.6467。磁盘缓存 `sum_m S_m` 对全量 S 的逐体素 relL2=`3.228521313994865e-7`；运行时校验亦通过（日志以六位小数显示）。

N10 单次描述性计量：global denom hit、10 个 subset denom miss，准备 322.585 s，iteration_core 273.583 s，full_command 597.497 s。迭代阶段应用 payload 读取 205,074,551,640 B / 55 次、H2D 67,102,752,976 B，空 slice 跳过计数 180（含前台与预取搜索，不等同于 180 个唯一 chunk）。本机自动 host 预算 27,228 MiB，未全缓存；仍存在整块读取放大。未执行旧/新配对重复测试，不将这些数字写成加速比或无性能回退结论。

第二份 `sen_100MBq_1200s.bin` N10：1 个完整迭代成功，1944000 个体素全部有限，1476193 个非零，min=0，max=288.172576904，sum=50883483.9795。使用同一 90x90x240 / 0.4 mm / PSF sieve-castor 1.5 mm 设置，global 与全部 subset denom cache hit；准备 11.527 s，iteration_core 660.022 s，full_command 673.745 s。自动 host 预算 33,531 MiB，迭代 payload 193,902,573,576 B / 52 次，H2D 67,102,752,976 B。它与 Derenzo 的投影稀疏度、反投分派、内存/OS 缓存状态不同，不能作为代码性能 A/B；未对两次耗时差做因果归因。

三份输出的机器检查保存在 `output-validation.json`，三者均为全矩阵网格且非负、有限、非零。输入及旧/新 EXE 哈希已复核；原矩阵 payload 未改写。候选源差异为 `source.patch`，recipe 与全局 INI 身份见 `config-hashes.json`。本轮无 GPU 进程遗留，无生产替换或 Git 提交。实际 BIN 短测只证明执行兼容；无独立全尺寸参考图像，尚不构成绝对重建正确性或影像质量结论。

### 11.5 复现入口

在仓库根目录 PowerShell 中运行，使用 multipinhole 的新 EXE，不执行 SC60 原型 CU：

```powershell
$common = @(
  '-config', 'D:/Projects/SPECT-recon/multipinhole/config/spect_config.ini',
  '-algoConf', 'D:/Projects/SPECT-recon/multipinhole/config/profiles/sc60_osem1000.conf',
  '-chunkCSRDir', 'D:/Projects/SPECT-recon/test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV',
  '-n', 'sc60', '-th', '8', '-vb', '2'
)
& ./build/sc60-adapter-20260920/bin/Release/SPECT_recon.exe @common `
  -df D:/Projects/SPECT-recon/test_data/SC60_luzhou/sim_data/121314derenzo_1200s.bin `
  -o D:/Projects/SPECT-recon/recon_out/sc60_adapter_20260920/derenzo_n10 `
  -it 1:10 -save-every 1
```

N1 对照使用 `-it 1:1` 和独立输出目录 `derenzo_n1`。第二份投影用 `sen_100MBq_1200s.bin` 和独立目录 `uniform_n10`。不得覆盖既有结果；重跑时更换输出目录。

正式长跑参数为 `-it 1000:10 -save-every 50`（也已写入 recipe），不是本节短测命令。检查命令：`python recon_out/sc60_adapter_20260920/validate_outputs.py`，输出网格/有限性/非零断言及磁盘缓存的逐体素 sensitivity partition relL2；此脚本不评价孔分辨率或标定精度。

## 12. 当前框架与协议审查（2026-09-20）

### 范围与结论

用户将优先级明确为现有框架审查、协议明确及局部架构优化，不开展论文 LM-MEW 迁移。此次沿 CLI/recipe → metadata → chunk directory → channel → q/S → subset update 路径审查。允许最小输入校验修复及现有文档更新；保护 SC60 已验证 EXE、原始数据、数据项/PSF/对称公式及 kernel 顺序。不做大矩阵长跑或速度提升声明。

本轮把责任明确为：reader 归一几何，pipeline 验证跨通道共同坐标，directory 管理存储与生命周期，projector 计算 q/S，loop 负责子集/PSF/更新/保存。沿用已有类型，只在 metadata 模块新增一个共享兼容性校验函数；没有 MatrixContract/ExecutionPlan 新框架，没有新增二进制格式或通道机制。当前正式合同集中在 `multipinhole/doc/Format_Reader_Matrix_CN.md`，历史 v4/TLV 提案加醒目状态说明，不继续与现行规格并列作为 authority。

### 审查发现与处理

| 发现 | 风险 / 证据 | 处理 |
|---|---|---|
| P1：第二矩阵 metadata 未读取 | `MainProgram` 用第一窗 meta 打开第二目录；行数相同但体素间距等不同会被套入同一坐标。真实旧 EXE 在 voxel mismatch fixture 上越过此检查、到投影读取才停止 | 两窗独立读取；在投影与 payload I/O 前比较归一后的 grid、detector、angle/index、profile 及四个几何浮点字段；不匹配/缺失均失败 |
| P2：正无穷能窗权重通过 `>0` 校验 | CLI 和 projector 公共入口未要求 finite，可将非法权重带入均值和灵敏度 | 两层均改为 finite 且 >0；合法值行为不变 |
| P2：接口注释仍称每轮只更新一次 | OSEM 实际是累计所有能窗后，每个 angle subset 更新一次；原 S 注释也未展示 sensitivity-only W | 修正接口注释与 q/S 文档，区分 channel、subset、symmetry branch |
| 协议缺口：旧 metadata 没有完整物理属性 | 未记录能窗边界、发射谱、跨窗响应、每光子/每衰变/活度/时间的包含关系；相同 shape 不能证明物理等价 | 列为生成/校准来源必须明确的合同；不从路径猜测，不填伪默认值 |
| 物理边界：norm + background（续查更正） | 当前 CLI 已禁止单窗或双窗任一 norm 与背景组合；W 只进入 sensitivity 的 norm 实现适用于无背景效率抵消模型 | 保留显式拒绝并补测试；不能将尚未开放的组合写成当前可运行缺陷。普通 detZRowWeights 与背景的物理来源仍需单独约定 |

兼容性变化已明确接受：双窗现在要求第二目录存在合法 metadata，且已表达的共同坐标精确一致。比较发生于 reader float32 归一之后，因此允许 88/92/124B 不同编码；不比较能窗间可以不同的 maxNnz、chunk 切分及权值。未添加重采样容差。当前实际 Lu-177 113/208 两份 92B metadata 的相关字段一致，并通过真实入口检查。该 geometry 检查不新增 CSRB 双窗执行支持。

### 验证

- 独立构建 `build/matrix-contract-audit-20260920`，Release EXE 和 spect_tests 都重新构建，CUDA 86/89/120。最终 EXE SHA256 `D89EB9943F7CF52BCA6CA22092E06D15A0FBD9E3FA63E36AE861A2978E69C632`。随后仅追加接口注释，不改变已测二进制行为。
- 全套 314 项：310 通过，4 项既有跳过（一个 opt-in 冻结输出门、三项缺失 260601 数据）。包含新增共享坐标检查、CLI 非有限权重拒绝，以及扩充的双窗 projector 非有限权重检查。既有 ROT90、双窗 CSRQ、subset/CTAC/PSF 数值测试均通过。
- `check_entrypoint.py` 用真实 EXE 完成五个负向/边界场景：匹配 metadata 到投影阶段、voxel mismatch 早停、缺失第二 metadata 早停、旧 EXE 的遗漏复现、实际 Lu 两窗 metadata 接受。故意提供不存在的投影，证明校验发生于大数据读取前，不冒充成功完成真实双窗重建。
- 原 SC60 计时 EXE SHA256 仍为 `8C9B004E6286FD3EE814A8D8A60A3A78F73447709B5DFA187BAE87555D15CF2D`，路径未覆盖；此前 100 轮图像和计时结果保留。本次没有再次运行全尺寸 SC60 100 轮或五床生产门，不做新性能/质量/生产采用判断。
- 初次新构建的配置失败留下空 CXX flags，缺少基线 `/EHsc` 和 Release 优化，产生 19 项失败；已保留为 `tests-invalid-build-flags.*`，按基线 cache 恢复所有 CMAKE_CXX_FLAGS 后重新完整构建，全部非跳过测试通过。没有为这些环境性失败改动无关实现或测试。

证据：`recon_out/matrix_contract_audit_20260920/tests.log|json`、`entrypoint-checks.json`、各入口日志及 `check_entrypoint.py`；构建记录为 `build/matrix-contract-audit-20260920/build-verified-flags.log`。脚本复跑 `python recon_out/matrix_contract_audit_20260920/check_entrypoint.py`；测试 `build/matrix-contract-audit-20260920/bin/Release/spect_tests.exe --gtest_output=json:recon_out/matrix_contract_audit_20260920/tests.json`。构建使用已有 vcpkg 安装、`VCPKG_MANIFEST_INSTALL=OFF`，CXX flags 与 SC60 基线 cache 相同；清理进程环境键的大小写重复后调用 MSBuild。

### 后续边界

下一项优先核查实际两窗生成/标定记录，闭合矩阵权重包含关系和 norm/background 的空间与单位；这是物理合同问题，不能用重命名字段替代。ROI+PSF 的 device-resident 优化、CSRQ ROT90、双窗 CSRB 属于三个独立变更，不在本轮附带实现。尤其性能优化应保留当前约 180 s/100 轮的 SC60 基线，先测分阶段开销，再决定最小修复。

状态：此次有界审查、局部修复、协议澄清和测试已完成；代码仍在工作树，未提交、未替换生产版。未实现能力不因文档完善而升级为“已支持”。

## 13. Lu-177 权重与归一化来源续查（2026-09-20）

本轮限于源码、实际 metadata、冻结 recipe 和既有实验记录；不读取全部矩阵 payload、不重新拟合权重、不运行重建、不更改生产公式。完成条件是区分已证实的实现合同与缺失的物理来源，并为既有关键拒绝边界留下测试。

### 结果

- 更正第 12 节原先过宽的 norm/background 疑点：`ArgumentParser.cpp::validateAlgorithmArgs` 已显式拒绝 norm 与任一窗背景投影共用，双窗 norm 必须成对。`CompactSymmetricReconLoop.cpp` 将校正 c 取倒数用于 sensitivity；无背景时效率在 EM 分子抵消，推导已写入当前格式合同。这不是新修复，也不证明校正文件自身标定正确。
- `05_mouse_joint_rdp30.conf` 保持 208/113 次序及 `0.1036,0.0617`，两窗 norm 为空。projector 将同一 channel weight 传给均值/反投和 sensitivity，属于一致模型中的配套使用，而不是把输入计数再乘一次。是否与实际矩阵内含系数重复，不能从运行代码单独判断。
- 实际两窗 metadata.bin 字节相同，SHA256 均为 `5F541644B5063C501CD98F575B9239F1691BD1CE8D23339D20E20B04443D0756`。这只闭合存储的几何，不闭合物理单位。
- 113 文本记录 113 keV、`mu_collimator_per_mm=5.605232`、`mu_detector_per_mm=0.211820`；208 文本缺这三项，并记录 half-voxel 展开/Z-fold 后处理。两者均称 v7.9；仓库 v4 备份生成器不能直接认定为实际生成源。未确认发射分支比、采集时间、全局 scale 和完整能窗效率的包含关系。
- 旧 `2026-09-13-subset-release.md` 的 113 有效权重 `0.132025131625` 是同数据拟合敏感性结果，不是独立校准。本轮不推广它、不重新做该实验。

输入目录均在 `test_data/6module/Lu177_sysmat/`：`matrix_PSC_PIN26x28x20_113keV_120x120x220` 和 `matrix_PSC_PIN26x28x20_208keV_120x120x220_ZFOLD`。对应 metadata.txt SHA256 为 `241A5BE91B66B761642B5A23C3052B0C43EDB74F8DBC2582626A043886F18192`、`B039B4F30BC3BF2A67E60AD9C8CA8CBB0226AB074DD80FEE7A337C40D58B1808`；未对多 GB payload 重算 hash，此次不作 payload 身份或完整性新声明。

### 改动与验证

只增加 `test_multibed_cli.cpp::NormalizationRequiresPairedWindowsAndNoBackground`，更新现行协议和本报告；生产实现、recipe、原始矩阵及三个受保护 EXE 均未修改。单个表驱动测试覆盖单窗/双窗合法 norm、6 种背景参数组合拒绝、两种缺半边 norm 拒绝及单窗误带第二 norm 拒绝。

在原独立审查构建目录重建 `spect_tests` 成功；运行 `build/matrix-contract-audit-20260920/bin/Release/spect_tests.exe --gtest_filter=IOTest.Normalization* --gtest_output=json:recon_out/matrix_contract_audit_20260920/norm-contract-tests.json`，2/2 通过。构建记录 `build/matrix-contract-audit-20260920/build-norm-contract.log`，测试记录 `recon_out/matrix_contract_audit_20260920/norm-contract-tests.log|json`。本轮未重跑此前 314 项全套，也未将局部测试表述为真实数据重建验证。

当前缺口最小闭合输入：实际 v7.9 生成程序/源码身份、生成参数与全局归一化定义、Z-fold/CSRQ 派生命令和输入身份，以及逐窗 norm 的生成与计数空间定义。未取得这些之前，保留现有运行系数和能力边界；不新增 v4/TLV、通道注册表或猜测性校准逻辑。

## 14. CSRB 子集分段读取（2026-09-21）

### 契约与改动

用户重申 CSRQ 仅实验性，优先其余格式兼容和验证。此阶段处理现有 SC60 原矩阵超过 host cache 时的整块重复读取，不扩展 CSRQ、不改变字节格式、对称公式、子集次序或重建算法。允许修改现有 loader 和测试，原矩阵/BIN/参考输出/既有 EXE 均只读。验收为子集列号与权值保持、既有 CUDA 数值门 `relL2 <= 1e-5`、实际应用读取字节减少；不是生产采用或绝对影像质量门。实际 A/B 每次最多 600 s，总链 1800 s；随后原矩阵数值检查另限 600 s。

`CompactChunkCSRDirectory::loadSubset` 现在优先复用完整 host chunk；若完整块能进入剩余预算，或请求覆盖该块全部角度，仍走原整块加载。其他 CSRB 请求读取完整 row_ptr 后，仅 seek/read 所选角度的列索引和值区间。完整角度位置保留在私有 backing storage，未选行为空，继续由既有 `subsetView` 产生相同绝对角度/重基行指针/segment 顺序。该 backing 不进入完整 chunk cache。每块一个弱引用复用正在被预取/GPU 缓存持有的切片，避免预取后再次读取；不新增永久缓存配置或 mmap。

输入检查覆盖 CSR 格式、raysPerAngle 与行数/角度数一致、严格递增的角度列表、完整 row_ptr、实际读取的列索引及短读。未读取的 payload 不冒充已经验证。全缓存、N1 保持整块路径；CSRB 仍为正式源格式，CSRQ 路径未改。应用读取统计按实际读取的 row_ptr 加选中 payload 计数，不等于物理磁盘流量。

### 验证结果

独立构建 `build/csrb-slice-20260921`，Release EXE 和测试均全项目构建成功，CUDA 架构保持 86/89/120。候选 EXE SHA256 `52CE940048025DE0F7E7AF7549B80EDCA0F7D42DF140C385498926A18067EAF5`，工作树基于 `635dec6`，没有提交或发布替换。

- 最终全套 316 项：312 通过，4 项既有跳过。分段 host 测试比较连续/不连续角度、尾部空行、全空角度、整块缓存、所有权、无序/重复角度、错误行布局和截断文件；ROT90 CUDA oracle 覆盖 N1/N10、零/部分/全缓存及全空选中切片。既有 ROT90_Z 两床 subset/CTAC oracle 改用零 host cache，验证同一新读取路径，数值门通过。
- 真实 SC60 ROI 矩阵约 25.1 GiB，Derenzo、N10、sieve/CASToR PSF 1.5 mm、32.8x12 mm ROI、8 线程。顺序 A1/B1/B2/A2，每次 1 个完整迭代、10 次子更新；A 为保护计时 EXE，B 为新候选，均强制零 host cache。灵敏度 11 个 cache 全命中，排除准备阶段差异；随后追加 B 全缓存检查。

| 运行 | 完整迭代 wall / s | 迭代应用读取 / B | 相对 A1 的图像 relL2 |
|---|---:|---:|---:|
| A1 零缓存 | 115.266837 | 125835038136 | 0 |
| B1 零缓存 | 26.133352 | 25191765872 | 4.7103e-8 |
| B2 零缓存 | 25.862423 | 25191765872 | 4.7496e-8 |
| A2 零缓存 | 111.699690 | 125835038136 | 4.7036e-8 |
| B 自动全缓存 | 2.452892 | 0 | 4.7383e-8 |

零缓存下本轮迭代应用读取下降 **79.9803%**，两次结果相同；图像差异与旧程序重复运行差异同数量级，满足预先数值门。各运行前 CPU 占用约 74.8%-82.1%，没有控制或停止其他任务，未做全程占用采样。耗时是该负载环境下的重复观察，不是无干扰吞吐保证，更不能外推到此前全缓存约 180 s/100 轮的加速比。

- 原始未裁剪 SC60 矩阵补测：原 18 块/67,106,070,664 B、90x90x240、0.4 mm、recipe 原 36x96 mm 支持区、PSF 1.5 mm、1:10、零 host cache。与 `sc60_adapter_20260920/derenzo_n10` 的同参数历史输出比，relL2 **4.6892e-8**；全部体素有限、非负、非零。当前 epoch 87.437502 s，process 100.115708 s，迭代读取 62,674,532,056 B。此项不是与历史部分缓存耗时的性能 A/B，不计算跨环境加速比。
- 以上六次真实运行全部退出 0；原始投影、配置、metadata 身份已核对，payload 使用已有来源与文件清单，未重新对全部大文件做 SHA。三个既有 EXE 的 SHA 保持第 11-13 节记录不变。

证据：`recon_out/csrb_slice_20260921/identity.json`、`results.json`、`PASS.json`、各 run.log/command.json/primary/manifest，以及 `B_original_zero/result.json`。可重放脚本 `run_gate.py` 和 `run_gate.py --full-matrix`，拒绝覆盖已有运行目录，重跑须先选择新 evidence 目录。构建及最终测试为 `build/csrb-slice-20260921/build*.log`、`tests-final.log|json`。

### 仍待完成

本阶段执行完成、局部数值及读取量证据通过，保留独立候选，生产采用 pending。以下不因本阶段通过而升级：SC60 多床输出合同及端到端验证、双窗 CSRB、完整 1000 轮长跑、PSC 五床配对生产回归/性能门。任意新对称 profile 和 SMMD v4/TLV 仍需明确生成端格式与样例，不凭形似支持；CSRQ 的扩展/提升生产地位继续暂缓。原生 CSRB 兼容不以 Lu-177 物理标定资料补齐为前置条件。

## 15. 原生双窗与完整网格多床（2026-09-21）

### 范围与实现

沿用原生 CSRB、现有 CUDA 数据项、迭代器和输出融合，不新增格式、插件框架或 CPU 重建后端。原矩阵、投影、历史图像及已发布 EXE 保持只读。用户确认 SC60 仍在组装、暂无真实多床数据，床步进须等待采集后确定；本阶段多床证据仅覆盖软件合同，不推荐实际步进。

- `matrix_grid_output=true` 允许 1..5 个连续床号，跳过 PSC 固定 ROI 覆盖和中心裁剪。多床要求显式有限正步进，且与整数体素层对齐（1e-4 mm），范围 1..baseNz 层。输出为 `Nx × Ny × [baseNz+(床数-1)*step]`，不做床间增益拟合和 XY seam alignment；暂不支持不等距、分数层或有间隙床位。SC60 recipe 的 `bed_step_mm=0` 为未知，单床 manifest 不再伪报 16.8 mm。
- 复用既有 cosine 累加。完整网格模式用体素中心采样的渐变权重，按有效掩膜归一化，避免另一床掩膜外的零值压低真实覆盖区；单层重叠和三床重叠均有测试。PSC 默认裁剪、gain-fit 和端点权重保持不变。
- 两窗现在允许 CSRB/CSRB，继续拒绝混合 CSRB/CSRQ，CSRQ 仍为实验路径。逐窗执行已有 projector，累加 `q=sum w*A^T[y/(w*A*x+b)]`、`S=sum w*A^T W*1` 后交给既有更新器；背景不随窗权重缩放，逐窗 sensitivity W 与 CTAC 指数独立。没有引入第二套联合重建算法。
- 两份 metadata 仍先独立校验共同坐标。CSRB 主机缓存按两矩阵合计大小计算一份预算并按大小分配，显式环境缓存上限也是两窗总额；GPU 仍使用既有两块缓存。切窗时刷新 per-bed 投影/背景槽。此功能不承诺双窗与单窗等速，也不开放 CLI 原先拒绝的 subset+CTAC 组合。
- 复查修复了既有通用预热中 `csrqAssignedSlot` 对 CSRB accelerator lease 的错误类型读取：访问 CSRQ lease 前先验证格式。首轮真实回归因此主动中止，旧输出保留为未完成记录；不是把该次中止当作算法失败。

### 已完成的软件验证

独立候选 `build/csrb-native-20260921/bin/Release/SPECT_recon.exe`，SHA256 `72E55729F0E5C2F29BAE81DD3EF1FF0BAAFDE6C55F72D1D64DD6AE1F81B3861D`，源树基于 `635dec6`，CUDA 86/89/120，未提交或部署。构建及全套测试记录为该 build 下 `build-lease-guard-final.log`、`tests-lease-guard.log|json`：**319 项，315 通过，4 项既有跳过**。

新增 CUDA oracle 覆盖 ROT90/ROT90_Z、不同矩阵及 chunk 划分、1/2 窗非单位权重、背景有/无、每窗不同 W、两个床位、不同 CTAC 指数、N1/N3、零/完整 host cache、正/逆 chunk 顺序、预上传图像以及 `refreshProjectionInputs=false` 的跨窗槽刷新；q/S 相对误差门为 1e-5，全部通过。已有 N10 和 resident OSEM/BSREM 测试仍通过。oracle 验证数学/索引合同，不验证真实核素窗权重标定。

真实 EXE 合成闭环位于 `recon_out/csrb_native_20260921/entrypoint_lease_guard/`，驱动为上一级 `check_entrypoint.py`。两个 SMMD v3/CSRB 矩阵、两窗两床 BIN、2 次 N2、每次保存，输出 2×2×6；相对已知体素坐标/强度的误差 < 1e-5。额外检查 ROI 掩膜不稀释覆盖信号、400 B 总缓存分为 200+200 B、未知步进拒绝、非整数层步进拒绝、manifest 的禁用 gain-fit 和步进信息。合成步进 2 mm 不是 SC60 机械建议。首版夹具头字段顺序错误和 manifest 文件名错误已修正，旧失败日志保留，不归因于生产 reader。

### 真实回归与长跑

最终候选使用 `recon_out/csrb_native_20260921/run_real.py` 串行执行，证据写入 `verified/`，拒绝覆盖旧记录。输入身份、矩阵清单、硬件、实际命令、源码 patch、占用采样均落盘；大矩阵只核对文件清单与既有来源，不冒充本轮完整 payload SHA 校验。

1. PSC 冻结 Lu208 五床，旧 `clean-635dec6-20260914` 为 A、新候选为 B，MLEM 50:1、PSF1.3、既有默认输出，A1/B1/B2/A2；显式独立 sensitivity cache，warmup 不计门。数值门 relL2 <= 1e-5，iteration_core 目标 120 s、底线 180 s，单次上限 900 s。
2. SC60 使用此前获准的 32.8×12 mm ROI 派生矩阵，N10、sieve/CASToR PSF1.5、1000 个完整迭代、每 50 轮保存，共 20 个检查点。单次上限 5400 s；要求全矩阵 host cache、迭代读取为 0、所有检查点有限/非负/非零，it100 与历史同参数输出 relL2 <= 1e-5。记录迭代及进程耗时，不借此宣称原始 67 GB 全支持矩阵完成了 1000 轮。

PSC 配对门已完成：A1/B1/B2/A2 核心耗时分别为 84.861 / 78.600 / 74.202 / 74.426 s，全为 sensitivity cache hit；候选两次均达到 120 s 目标。B1/B2 对 A1 的 primary relL2 为 1.548142e-7 / 7.703139e-8，A2 对 A1 为 7.672423e-8，均通过 1e-5 数值门。四次进程耗时为 140.824 / 135.057 / 121.157 / 120.864 s；其他 CPU 任务占用均值约 71.36 / 48.60 / 7.50 / 11.19%，因此不将均值差解释为实现提速。

SC60 ROI 长跑完成，1000 个完整迭代、10000 次子更新、20 个保存点全部通过网格、有限、非负、非零检查；运行掩膜与派生矩阵支持区一致，ROI 外为零。18 个矩阵块全部进入主机缓存，迭代应用 payload 读取 **0 B**。最终输出仍为 90×90×240、0.4 mm，而不是把 ROI 冒充矩阵尺寸。

| SC60 计时/数值项 | 结果 | 含义 |
|---|---:|---|
| 完整迭代计时合计 | 1942.590 s | 1000 个 `ITER_TIME.wall_s` 之和 |
| 平均 / 中位 / P95 每轮 | 1.942590 / 1.838751 / 2.423464 s | 每轮包含 10 个子集更新 |
| 进程总耗时 | 1969.547 s（32.83 min） | 含准备、重建、保存和退出 |
| 前 100 轮合计 | 178.660897 s | 与此前约 180 s 的量级一致 |
| it100 对历史同参数 it100 relL2 | 9.872965e-8 | 通过预先的 1e-5 数值门 |
| it1000 对 it100 relL2 | 13.163558% | 长迭代图像变化，不是误差率或分辨率 |
| it1000 / it100 总强度 | 1.000086568 | 总量变化约 +0.00866% |
| 最后 50 轮相对变化 | 0.184957% | 描述性收敛观察，未作为质量验收阈值 |

日志旧字段 `iteration_core_s=1491.687` 是 forward/ratio/backward、MLEM 更新、正则梯度和 Butterworth 等已计时阶段的累加，**不是完整迭代 wall time**；应以上表 1942.590 s 回答“1000 轮实际迭代用了多久”。其他 CPU 任务占用均值约 24.26%，当前结果不是隔离环境吞吐保证。

补充的大矩阵联合路径检查通过：真实 SC60 投影无损拆为两个 `p/2` 窗，分别使用相同原生 ROI 矩阵和权重 0.5；模型代数上应等于单窗。两窗总 host cache 限 8 GiB（各 4 GiB），N10/PSF1.5/1 轮，上限 600 s，与既有同参数单窗图像 relL2 **5.043511e-8**。新 BIN 写入独立目录并记录来源 hash；这是大矩阵多通道调度/缓存/数值检查，不是实际双能量采集或能窗标定。此次冷 sensitivity 构建进程耗时 187.292 s、完整迭代 60.793816 s；不能与全缓存单窗 1.94 s/轮直接解释为双窗算法开销。

### 图像与证据

![历史100轮、当前100轮与当前1000轮的同尺度切片](../recon_out/csrb_native_20260921/verified/paired_slices.png)

各行共用强度标尺，不做配准或强度拟合。轴位 z=+0.2 mm、矢状 x=+0.2 mm；冠状 y=-2.2 mm 由历史 it100 的轴位行和最大位置确定，避免按候选挑选切面。历史与当前 100 轮视觉一致；1000 轮峰结构更集中、杆间背景更低，但无独立真值或已校准分辨率判据，不能据此宣称绝对影像质量提升或最佳迭代数。

![SC60逐轮计时与PSC五床ABBA](../recon_out/csrb_native_20260921/verified/timing.png)

已完成记录集中在 `recon_out/csrb_native_20260921/verified/`：`real-PASS.json`、`real-identity.json`、`psc-results.json`、`sc60-checkpoints.json`、`analysis.json`、`joint-split-result.json`，以及每次运行的 command/result/log/occupancy。复现入口为 `run_real.py`、`check_real_joint.py`、`analyze.py`；重跑须选新输出目录，不覆盖已有证据。根目录旧 `psc_B1` 为修复 lease 前主动中止的未完成运行，由 `real-aborted.json` 标明，不能替代 `verified/` 的结果。

### 收尾修复与边界

长跑后的审查发现 `MATRIX_IO` 仅报告第一矩阵的读取计数，已在迭代器现有统计边界合计两份独立 directory；图像计算不变。数值/性能主验证 EXE `72E557...B3861D` 已保留为候选 bin 目录和证据目录的 `SPECT_recon.numeric-validated.exe`。最终带日志修复的 EXE SHA256 为 `560895755F162D06D212C0375235702717B04A6A6F7FB142F6E8D809CDDD99D7`；重新构建 EXE 和测试、319 项中 315 通过 / 4 既有跳过，记录 `build-io-total.log`、`tests-io-total.log|json`。不把先前 1000 轮归到这个新 hash，也不为日志改动重复昂贵长跑。

最终日志复测通过，脚本 `check_io_total.py`：四个全缓存小块的独立字节 oracle 为 `4*(9*8+8*4+8*4)=544 B`，旧程序只报第一窗 272 B/2 次加载，新程序正确报 544 B/4 次。最初用“零缓存两窗计数应相同”的夹具假设不成立，因为预取/存活引用复用不对称；改为上述全缓存可精确计数检查。旧 EXE 单独复制到证据目录缺少依赖 DLL 的启动失败也已保留，最终在原候选 bin 目录使用归档 EXE 完成对照，未改数值实现。新 EXE 重跑实际 ROI 拆窗任务，图像对日志修复前 relL2 **4.562366e-8**，总计迭代应用读取 **50,981,462,544 B**，证据 `verified/io-total-result.json`、`entrypoint_io_total_verified/`。早先 `joint-split-result.json` 中 25,193,843,920 B 是旧日志的第一窗值，不能作为双窗合计。四个受保护历史 EXE 的 SHA256 均复核不变；`git diff --check` 通过（只有既有 CRLF 提示），没有活动重建任务。

本阶段原生 CSRB 软件兼容、ROI 长跑和本机 PSC 五床回归已完成，仍为未部署候选。尚未完成或不属于本阶段：SC60 实采多床/机械步进、真实独立能窗物理标定、原始未裁剪矩阵 1000 轮、任意新对称格式、SMMD v4/TLV、CSRQ 扩展、现场 GPU 签收和发布替换。新对称能力必须有明确生成端格式和样例；现有几何校验不等于未知格式已兼容。

## 16. 标准 OSEM Release 打包（2026-09-21）

用户要求先打包 Release、默认标准 OSEM N=10，并进一步确认 Lu177 默认也改为标准 OSEM且保持双窗模型。本轮只改默认配方、发布说明和打包文档，不改重建数值实现。通用/Tc 沿用 `04_standard_osem50.conf`；Lu 默认改为 `06_standard_joint_osem50.conf`：均为50:10、无正则、最终保存。Lu保留208/113顺序和0.1036/0.0617权重，明确使用两份原生CSRB；历史05 BSREM/RDP/CSRQ配方保持显式可选，不再默认启用。SC60独立1000:10/PSF1.5配方保持原样，不将PSC几何当作SC60默认。

交付包 `dist/release_candidates/SPECT_recon_20260921_osem_n10.zip`，369,994,304 B，34个校验清单内文件；附包外 `.zip.sha256`。ZIP SHA256 `f46aaec638f07b0908e5e5813dec3e9578d81f8c88ef69e28d11289b8d9f9b82`；包内EXE SHA256 `1d48b48f5ccf1a3b1173b02f6b7032533a6c429fc1b400cf8756677e480ccb94`。版本 `20260921+635dec6-dirty` 如实保留未提交状态；无新提交或外部机器部署。带CUDA/MSVC和现有MAT读取运行库，不带矩阵、投影、旧输出、灵敏度缓存、测试DLL或NVRTC。

构建复用独立 `build/csrb-native-20260921`，更新嵌入版本说明并重新链接；原数值版和日志修复版EXE均归档。首轮测试发现CT测试INI还绑定旧Lu默认，已同步该一行，最终319项中315通过、4项既有跳过，见 `tests-release-osem-final.log|json`。没有通过修改测试断言掩盖默认值不一致。

验证脚本 `recon_out/csrb_native_20260921/verify_release.py` 对ZIP做CRC与全部34文件SHA校验，再解压独立副本；移除CUDA_PATH和开发工具链PATH，仅保留Windows目录作为运行库隔离检查。Tc/Lu默认dump确认MLEM50、subsets10、None、beta0、final-only，Lu共同观测合同保持；未部署矩阵的裸dump会按既有合同失败，因此验证显式传入矩阵目录，并未新增假矩阵或静默fallback。解压EXE通过既有双窗两床合成数值/异常入口检查，以及SC60 ROI32.8×12、N10/PSF1.5、单完整迭代实际数据重建，对既有同参数it1图像relL2 **5.143389e-8**；这是发布入口smoke，不是重跑1000轮或真实Lu50轮质量验证。

最终EXE中三份fatbin的完整字节与build产物匹配，逐份 `cuobjdump --list-elf` 均包含sm_86/89/120；EXE用普通字节数组嵌入fatbin，不能以直接扫描EXE无CUDA section误判为缺少kernel。证据 `release_verified/PASS.json`、`fatbins.json`、配置dump、smoke日志及 `tracked-source.patch`；未跟踪的两个新配方已随包交付。现有9月13日和14日冻结ZIP分别保持 `7ca6fa...71b86`、`6f0797...95ffd`；`git diff --check`通过，无活动重建。打包脚本重放须使用新PackageName/StagingRoot，不覆盖本包。
