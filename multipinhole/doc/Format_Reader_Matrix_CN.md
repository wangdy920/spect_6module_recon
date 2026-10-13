# 系统矩阵读取与多能窗合同

> CSRB 生产读取入口仍为 `-chunkCSRDir`。PSC 生成器名称中的 v4 不是 SMMD metadata version 4。当前读取器接受下表三个 metadata 布局；新增 SC60 `ROT90` 支持的验证状态见 `reports/2026-09-18-spect-matrix-adaptation-implementation-plan.md` 的实施记录。CSRQ 仍是独立的实验派生格式。

## 磁盘布局

```text
<matrix_dir>/
  metadata.bin        ; 88B / 92B legacy，或 124B SMMD v3
  metadata.txt        ; 文本版，仅供人工检查
  chunk_000.csrb      ; 按角度分块的 compact CSR（对称压缩，不展开）
  chunk_001.csrb
  ...
```

## metadata.bin

| 大小 | 名称 | 语义 |
|---|---|---|
| 88B | ExtendedV1 | 260601 canonical：几何（dim/vox/detN/ang）+ `use45` / `symCode` / `symmetryOrder` |
| 92B | legacy float64 voxel | 88B 布局中只有 voxelSize 从 float32 换为 float64 |
| 124B | SMMD v3 | 小端 magic `0x444D4D53`、version=3；原 voxelSize 换为 32B NUL 终止的十进制文本 |

当前 6module 矩阵对称码为 **SYM_90Z**（90° 旋转 + Z 对称）。程序按文件实际大小解析，进入 compact 路径后用 metadata 几何覆盖 CLI/INI 的 `dim` / `vox` / `detN` / `ang`（见 `ReconDataLoader.cpp::applyCompactMetadataGeometry`）。

SC60 使用 `ROT90`，即 4 个 XY 分支、无 Z 镜像，要求 `detNz == compactDetZ`；PSC `ROT90_Z` 要求 `detNz == 2*compactDetZ`。两者要求方形 XY 网格和四分之一 compact angles。其他 profile 与 SMMD v4/TLV 尚不支持；不能仅根据 branchCount 决定变换。CSRQ 共用 4/8 分支实现；2026-10-09 已通过两种 profile 的 CUDA 算子、驻留/流式、多子集 CTAC 回归，仍不是生产采用。

离线 `tools/convert_csrb_to_csrq.py` 支持 88/92/124B metadata，转换后原样保留 metadata。2026-09-28 的已验证转换版本仍在运行时拒绝 CSRQ ROT90；2026-10-08 候选补齐分支步长、缓冲和算子参数，不能将这次编译通过等同于真实重建通过。不能靠开启 Z 对称或改写 metadata 改变矩阵物理语义。

SMMD v3 十进制按 locale-independent 规则严格解析，再转为现有计算使用的 float32 voxelSize，与 SC60 生成端的 float32 计算约定一致。文件不足、未知版本、非有限或非正 voxel、索引维度溢出直接报错。

## Compact Chunk CSRB v1

实际 chunk 头为 28B：`u32 magic=0x42525343, u32 version=1, i32 rows, i32 angle_offset, i32 angle_count, u64 nnz`，随后是 `i64 row_ptr[rows+1]`、`i32 col_idx[nnz]`、`f32 values[nnz]`，均为小端。列维数来自 metadata；这里没有 n_cols。文件长度为 `28 + 8*(rows+1) + 8*nnz`。

不要将此布局与历史 24B flat CSRB 头混用：后者包含 cols，magic 为 `0x43535242`，并非 `chunk_*.csrb` 的头；`scripts/eval_csrb_crc_var.py` 和 `scripts/validate_6module_compact_inputs.py` 仍独立读取该历史格式。当前 compact host row_ptr 转为 u32，CUDA 每个上传 chunk/slice 的 nnz 还必须能表示为 i32。

## SC60 网格输出

使用 `config/profiles/sc60_osem1000.conf`，显式指定实际矩阵目录或配置匹配的 matrix root。默认 OSEM 1000:10、PSF 1.5 mm、每 50 轮保存；短验证可用 CLI 覆盖迭代数。坏探测器/norm 清空，支持区为 36x96 mm，关闭 PSC 输出滤波和轴翻转。

`[InputOutput] matrix_grid_output=true` 保留完整矩阵网格，不执行 PSC 固定中心裁剪或固定 ROI 覆盖。默认 false，保持既有 PSC 行为。对称 profile 与输出裁剪分别配置，不把对称性当作系统型号。

支持 1..16 个连续床号（PSC 非网格模式仍为 1..5）。多床须显式配置有限正 `bed_step_mm`，与整数体素层对齐（绝对误差 <= 1e-4 mm），范围为 1..baseNz 层；暂不支持非整数、非等距或带间隙的步进。2026-10-11 起多床为**联合重建**：各床矩阵网格按 `bed*step` 放入同一全局体 `Nx × Ny × [baseNz + (bedCount-1)*step]`，重叠体素的灵敏度与反投影相加后统一更新；保存的就是该全局体，不再做每床独立重建后的 cosine 融合，不做 gain-fit，不允许 seam XY alignment。输出尺寸与旧版相同，但体素内容语义改变（旧版为各床独立重建的加权平均）。圆柱支持区为各床局部圆柱的并集。联合模式暂不支持 CTAC、`-ctacAutoMu`、`-maskFromAtt`（这些 mu 图按每床局部 Z 堆叠）。旧的独立+融合行为保留在 2026-10-11 之前的 EXE 中，可作对照基线。PSC 默认输出及其端点权重不变。

`-beds <N|A:B>` 可替代冗长的 `-dfList`：给 `-df`（或单项 `-dfList`）一个目录或目录内任一 `BedK.bin`，程序在该目录搜索 `Bed1..BedN` 或 `BedA..BedB`（文件名须为 `BedK.bin`，大小写不敏感）；范围内任一床缺失即报错，范围外的床被忽略。第二能窗 `-df2` 同规则。与 `merge_time_segments` 同用时只能选中目录中的全部床。

早期 SC60 recipe 的 `bed_step_mm=0` 表示未指定，单床 manifest 保留 0，多床必须显式覆盖；测试夹具步进不是硬件推荐。现有实采记录需逐设备核对，不能自动改写通用配方或套用 PSC 步进。多床物理偏移以第一张选中床为零点，床号递增对应 +Z；轴翻转仍由配置声明，软件测试不能确定采集端的实际朝向。

## 子集读取与计量

`loadSubset` 在加载前由 angle range 跳过空切片。切片及 GPU 缓存持有源数据所有权，防止部分 host cache 下的预取悬空；弱引用只复用仍由调用者持有的 chunk，不扩展永久缓存预算。

2026-09-21 候选增加 CSRB 磁盘子集分段读取：已有完整 host chunk 优先复用；整个 chunk 能纳入剩余 host 预算，或请求覆盖其全部角度时，仍走整块加载。否则读取并校验完整 row_ptr，仅 seek/read 所选角度的 col_idx/float32 权值区间，相邻区间合并。保留全局列编号、绝对角度、子集行顺序和原权值，不重排或量化；临时存储只通过 subset view 暴露，不能被 loadChunk 当作完整矩阵。弱引用复用仍存活的预取切片，不新增永久子集缓存预算。此路径不改变 CSRQ。

分段读取校验完整 row_ptr 及实际读取的列索引，沿用原有 strict/sample/off 设置；未读取的 payload 不在当次校验范围。正常打开目录仍检查全部 chunk 的头、文件长度及角度覆盖。`MATRIX_IO` 日志区分应用层 payload 读取字节（包含 row_ptr）、payload 加载次数、仍存活的 host 数据复用字节、跳过的空切片和 CSRB H2D 字节。分段读取的 bytes 仅计实际读入的区间；应用层读取字节不是物理磁盘流量，host 复用字节也不代表同量 memcpy。计量 scope=iterations 不含 sensitivity 准备。N1/full-cache 路径不因本改动自动获得提速。

2026-10-09 配对计时后，删除此前部分 host cache 的逐块 async 前读候选，所有 CSRB 分子计算复用已有双槽预取；SC60 2 GiB/3×N10 的 core 两次由 65.573/83.013 s 降至 58.129/59.942 s，读取量不变，数值 relL2<=1e-5。`loader_work_s` 为累计读取和校验工作时间，不应加到迭代墙钟上。未增加第三个 GPU 矩阵槽、整矩阵锁页策略或新 CLI 开关；收益仅限已测条件。CSRQ 的 X/Y 按实际 K 分配，比值工作区按最大 chunk 行数分配，普通/CTAC 打包复用同一内核，前投影/Poisson 比值继续融合。

## 读取端

| 读取端 | 入口 | 说明 |
|---|---|---|
| `SPECT_recon` | `-chunkCSRDir <dir>`；或 `[SystemMatrix] default_chunk_csr_dir`；或 exe-local `bin/matrix/*` 唯一候选自动发现 | **唯一消费路径** |

解析与校验见 `src/io/CompactChunkCSR.cpp`。CSRB 列索引默认 sample，`SPECT_CHUNK_VALIDATE=strict` 检查全部已读取索引，`off` 仅关闭列索引范围检查并打印 NOTE；文件结构、几何、行指针和已读取权值的有限非负检查仍执行。CSRQ header 的 `cols` 必须与 metadata 的 `totalVoxels` 一致，在分配载荷前拒绝不匹配输入。

## 现行合同与职责（2026-09-20 审查）

本文是当前读取/执行合同；`../../reports/2026-09-18-csrb-universal-matrix-contract-design.md` 的 v4/TLV 是历史提案，不是读取器已经支持的格式。不新增并行 `MatrixContract`：已存在的 `CompactSymmetryMetadata` 负责归一几何，`CompactChunkCSRDirectory` 负责负载和缓存，recipe 负责运行参数。

| 层 | 负责 | 不负责 |
|---|---|---|
| metadata reader | 88/92/124B 解析、float32 几何归一、封闭 symmetry profile | 根据目录名猜核素/能量、补全未记录的物理参数 |
| chunk directory | 格式、连续角度覆盖、行数、索引/数值校验、数据生命周期与 host cache | 推断另一通道的 metadata 或权重 |
| pipeline 输入边界 | 独立读取两窗 metadata、核对共享几何、投影床位/行数、加载校正 | 在不相容网格间隐式重采样 |
| projector | 给定通道及 angle subset 的 q/S、格式能力拒绝、CUDA 传输与运算 | 核素名称解释、PSF 输出策略、优化器更新 |
| reconstruction loop | 子集调度、PSF/ROI 放置、更新和保存 | 改写原矩阵、修补输入物理语义 |

### 索引与共享坐标

- 图像列：`x + Nx*(y + Ny*z)`，X 最快，长度 `Nx*Ny*Nz`；voxelSize 为各向同性 mm。
- compact 行：`compactAngle*compactRaysPerAngle + izCompact*detNx + ix`；full 行：`fullAngle*(detNx*detNz) + izFull*detNx + ix`。
- XY 分支使用整数索引置换；full angle 为 `compactAngle + symId*compactNumAngles`。Z 分支同时映射体素 Z 和 detector Z；ROT90 不允许额外 Z 镜像。
- energy channel 不是 angle subset，也不是对称分支。一次子更新先累计全部测量通道，再更新同一幅图像；一轮包含 N 次这样的子更新。
- 双窗必须先分别通过 reader，再逐字段匹配共享图像尺寸、体素间距、投影/探测器维度、对称 profile、pixelSize、detectorY、detectorThickness。浮点按 reader 归一后的 float32 精确比较；不存在自动重采样或容差对齐。不同 metadata 编码版本、maxNnz、chunk 切分及数值允许不同。
- 上述校验只能覆盖已存储的坐标约定。角度零点、模块/探测器重排、物理原点等没有独立描述字段，必须由生成端与输入准备合同保证，不能仅凭行数相等证明。

### 测量和权重

当前 BIN 是分箱投影：头部两个 int32 为角度数、每角度射线数，后接 float64 计数。它不是逐事件 list-mode；窗内能量等丢失属性无法靠矩阵格式转换恢复。两窗投影必须按同一床位和 full-row 顺序一一对应，不允许事件重复计入两个独立 Poisson 通道；重叠能窗须由上游明确去重或另行建模。

令 `H_k = w_k A_k`，忽略此处未展开的 PSF/CTAC 组合；有效行的效率为 `e_k`（无校正时为 1）。测量与加性背景均使用原始探测计数空间：

```text
mean_k = e_k * H_k x + b_k
q_m = sum_k H_(k,m)^T [y_(k,m) / (H_(k,m) x + b_(k,m)/e_(k,m))]
S_m = sum_k H_(k,m)^T e_(k,m)
```

`w_k` 必须有限且 >0，不是对计数额外乘一次。正校正因子 `c_k` 的倒数为效率 `e_k=1/c_k`；detZ 行权重入口则直接提供效率，不取倒数。两入口互斥，均不代表效率已经物理标定。mask 行同时剔除测量、灵敏度和最终残差统计；背景不需要在调用前自行屏蔽。各窗 norm 独立，不得复用合窗 norm 冒充逐窗校正。

**归一化与背景：**2026-10-09 起单窗/双窗 CSRB、CSRQ 支持 norm 与背景共用，调用参数不变，双窗 norm 仍须成对提供。准备阶段只对有校正的背景计算一次 `b/e`，原背景留给最终残差；投影计数不预乘 c。既有 CUDA 比值内核继续计算 `H^T[y/max(Hx+b/e,1e-10)]`，不新增内核。上式与完整效率模型在不触发数值下限时代数等价；保留现有 Hx 单位下的数值下限，不承诺极小预测值时与另一种截断定义相同。`b/e` 非有限时报错。无背景路径及默认配方不变。

背景文件必须与 y 的床位、行序、采集时间和能窗相同，且为有限非负的预计探测计数；不是预减散射后的投影，也不是已除效率的背景。效率行权重与背景的组合现在同样使用 `b/e`，修正旧 sensitivity-only 组合的不一致。不把负的散射减后计数夹零作为输入修复。上游负责估计背景，重建只验证并消费它。

### 上游交接的最小物理约定

在现有 SN 配置说明或生成记录中给出下表内容，不新增二进制版本或历史身份注册库。未知项明确写未知，不由重建猜测。

| 项目 | 调用方/生成端应明确 |
|---|---|
| 坐标 | 角度零点与正方向、模块/像素顺序、原点、床号增加方向；输出旋转不是输入配准 |
| 两窗选择 | high/low 的实际能窗；完整分窗优先，全部缺失才走既定 merged/high 兼容模式，部分缺失失败 |
| 矩阵与权重 | A 是每光子/每衰变/何种归一量；是否已含发射谱、探测效率、准直器响应；避免 w 或 e 重复计算 |
| 计数与时间 | BIN 是计数，不是未经声明的计数率；曝光/有效时间差异由上游统一到矩阵/标定约定 |
| 背景与校正 | b 为同次采集、同能窗的探测计数，c 为对应窗的正校正因子；不先校正 y 再重复应用 c |
| 定量边界 | 未完成独立标定时输出相对强度；不自动宣称 Bq/ml，不以每图百分比色标比较摄取量 |

合窗模型只有在响应与权重相应组合时才与其计数空间匹配；合并 BIN 配单个 high 矩阵是保留的兼容配方，不与分窗联合声称物理等价。核素和矩阵选择仍在 wrapper/SN 层，核心不认证来源身份或计算全矩阵 SHA。

### Lu-177 实际矩阵来源（2026-09-20 续查）

`test_data/6module/Lu177_sysmat` 下 113/208 两目录的 92B metadata 完全相同，仅证明已记录的共享几何一致。两份文本均标记 `v7.9_pinhole_plane_sampling_zhalf_fold`；113 记录 `photon_energy_keV=113`、准直器/晶体衰减系数，208 未记录对应三字段，并追加 half-voxel 到 detector-Z-fold 的后处理公式。这些是生成端声明，不能替代生成程序与参数的可追溯记录。

仓库 `6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v4.cu` 是更早版本，其中确有准直器穿透和晶体吸收/DOI 的概率项，但未建立它与两份实际 v7.9 payload 的生成关系；不能据此认证实际矩阵已包含或未包含全部探测效率、发射分支比、时间或全局归一化。旧 binary metadata 不表达这些属性，现有文本也不足以闭合单位。

当前 joint recipe 的 `208/113 -> 0.1036/0.0617` 是冻结的运行系数。源码显示其在前向均值与相应反投/灵敏度中配套使用；这不是重复加权。是否与矩阵内含的系数重复，仍缺实际生成来源证据。`reports/2026-09-13-subset-release.md` 的 113 有效权重拟合只是同数据敏感性实验，不是独立标定，不能替换默认值。当前 joint recipe 两个 norm 文件均为空。

关闭该来源缺口最少需要：实际生成源码/可执行文件身份、完整生成参数（含全局 scale/发射谱/能窗响应/时间）、Z-fold 和 CSRQ 派生命令及输入身份，以及各窗独立标定因子的定义与数据来源。先补这些记录，不新增无人消费的协议字段，也不凭名称补物理默认值。

### 物理属性：已知与未表达

| 属性 | 当前承载位置 / 边界 |
|---|---|
| 核素、matrix ID、两窗次序 | recipe/CLI；名称校验是选择约束，不是矩阵物理内容证明 |
| 能窗边界、发射谱、跨窗散射/穿透贡献 | 旧 metadata 未表达；需要生成记录，不从 113/208 等文件名推定 |
| 每发射光子/每衰变/每单位活度、是否包含效率与时间 | 旧 metadata 未表达；权重与单位必须由生成记录和标定共同确认 |
| CTAC | 运行时独立因子；每窗能量缩放是现行近似模型，不代表矩阵已存储完整散射路径 |
| PSF | 运行时 image-domain 配置；应核查矩阵是否已包含同一物理模糊，避免重复建模 |
| ROI 派生 | 新目录与 provenance；不得冒充全支持区原矩阵，PSF 下须重新明确支持区合同 |

这些缺失属性是协议待明确项，不在本轮强行写入旧二进制或新增未经消费的 sidecar。新系统接入前，最少提供：行/列映射、物理单位与权重包含关系、能窗及响应组成、对称成立条件和派生来源。缺失时可做限定范围的执行/相对数值验证，不宣称物理等价或绝对定量。

### 当前执行能力

| 组合 | 当前状态 |
|---|---|
| 单窗 CSRB + ROT90_Z | 既有支持 |
| 单窗 CSRB + ROT90 | SC60 候选已做 ROI 1000 轮、N10/PSF1.5 验证；原始未裁剪矩阵仅短验证，非生产替换 |
| 单窗 / 双窗 CSRQ + ROT90_Z | 2026-10-09 共用内核 CUDA 回归通过，含驻留/流式、多床、多子集 CTAC；仍为实验格式 |
| 双窗 CSRB + ROT90 / ROT90_Z | 2026-09-21 原生候选；CUDA q/S oracle、合成双窗两床入口、实际 SC60 ROI 矩阵拆窗等价检查通过，不需转换 CSRQ；非物理校准或生产采用 |
| 混合 CSRB/CSRQ | 显式拒绝 |
| CSRQ + ROT90 | 2026-10-09 的 4 分支算子/子集/CTAC/BSREM CUDA 回归通过；真实数据/量化质量与部署仍分别验收 |
| 任意新角度/Z profile、SMMD v4/TLV、list-mode | 未实现；不得按相似维度降级套用现有 profile |

原生 CSRB 双窗复用同一 q/S 数据项：逐窗累加 `H_k^T[y_k/(H_k x+b_k/e_k)]` 和 `H_k^T e_k` 后更新，其中 H 含窗权重、PSF/ROI 与 CTAC；零效率行屏蔽，窗口权重不乘背景，逐窗效率与 CTAC 指数保持独立。最多两窗，共享坐标检查、成对 norm 规则保持，norm/background 可共用。CSRB 与 CSRQ 的 subset+CTAC 路径均已实现。2026-10-09 CSRQ 候选按矩阵/子集复用角度映射，只暂存选中角度的因子切片，复用原有 K4/K8 内核和双槽传输；N1 全量诊断仍独立于当前子集。无格式/CLI 变更；GPU 数值回归已通过，不可据此直接生产采用。

主机缓存先按两矩阵合计字节数计算一次预算，再按矩阵大小分配；`SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` 在双 CSRB 运行中是总上限，不是每窗上限。每窗沿用完整缓存/分段读取；GPU 沿用既有最多两块缓存。CSRB 投影/背景按 channel×bed 缓存，新任务或显式输入变化时刷新，静态切窗不再重复上传。矩阵计算量与读取量仍约为两窗之和，不承诺与单窗等速。CSRQ 编码不变；ROT90 候选状态见上表。

所有读入的 CSRB 权值必须有限且非负，subset 错误定位原文件 nnz；列索引检查仍由 `SPECT_CHUNK_VALIDATE` 控制，首次接入外部矩阵应设 `strict`。CASToR PSF 的 resident 路径支持局部 ROI mask，前向 PSF 后/反向 PSF 前执行，满视野圆柱也不假设矩阵角落为零。

### 运行期职责与缓存（2026-10-08）

重建核心只检查本次传入的 BIN、metadata、CSRB/CSRQ 是否可读、格式受支持、几何/行列/通道匹配，以及实际读取的索引与数值是否合法；不认证它们属于历史中的哪个矩阵或标定版本。设备与矩阵的选择由 wrapper/调用方负责，不要求建立历史矩阵库。上述校验不等于物理标定正确或定量准确。

运行期已移除全矩阵 SHA256、摘要字段和加密库依赖，不做独立载荷扫描，也不在正常读取时附带计算 SHA。分母缓存保留为计算加速，v6 键由规范化完整输入路径、大小、修改时间、已解析的 chunk 布局/metadata 和实际分母计算参数构成；小型 manifest 的文件名摘要不是矩阵载荷摘要。`[MATRIX_CACHE] key=path_stat payload_scan=none` 明确记录该策略。

这不是内容身份校验：同路径、同大小、保留修改时间的原地替换不会被缓存键发现，调用方此时须清除对应旧缓存或使用新缓存目录。运行期间输入必须保持不变。没有新增矩阵版本号、注册表、可信导入机制或 CLI 开关；数值合法性检查不依赖缓存键。

旧 v3/v4/v5 自动缓存不复用，首次生成 v6 缓存；若显式指定旧缓存文件，仍按既有规则报告 manifest 不匹配，需由调用方换缓存路径或有意删除旧缓存后重算。移动矩阵路径也会生成新缓存，不再承诺跨路径复用。原始矩阵和发布包不改动；当前验证见 `../../reports/2026-09-27-cuda-matrix-design-review.md` 第 9/10 节。

## 参考

- `6module_bak_src/README.md` — 生成端编译与产出
- `include/io/CompactSymmetryMetadata.h` — metadata 布局常量
- `test_data/6module/260601/CompactChunkCSRB_Reconstruction_README_CN.md` — 260601 数据包内说明
- `doc/Capability_Matrix_CN.md` — 能力边界
