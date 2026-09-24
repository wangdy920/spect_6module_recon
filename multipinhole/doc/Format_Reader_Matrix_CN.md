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

SC60 使用 `ROT90`，即 4 个 XY 分支、无 Z 镜像，要求 `detNz == compactDetZ`；PSC `ROT90_Z` 要求 `detNz == 2*compactDetZ`。两者要求方形 XY 网格和四分之一 compact angles。其他 profile 与 SMMD v4/TLV 尚不支持；不能仅根据 branchCount 决定变换。CSRQ 暂时只允许 `ROT90_Z`。

SMMD v3 十进制按 locale-independent 规则严格解析，再转为现有计算使用的 float32 voxelSize，与 SC60 生成端的 float32 计算约定一致。文件不足、未知版本、非有限或非正 voxel、索引维度溢出直接报错。

## Compact Chunk CSRB v1

实际 chunk 头为 28B：`u32 magic=0x42525343, u32 version=1, i32 rows, i32 angle_offset, i32 angle_count, u64 nnz`，随后是 `i64 row_ptr[rows+1]`、`i32 col_idx[nnz]`、`f32 values[nnz]`，均为小端。列维数来自 metadata；这里没有 n_cols。文件长度为 `28 + 8*(rows+1) + 8*nnz`。

不要将此布局与 `CSRTypes.h` 中历史 24B `CSRHeader` 混用：后者包含 cols，magic 也不同，并非 `chunk_*.csrb` 的头。当前 host row_ptr 转为 u32，CUDA 每个上传 chunk/slice 的 nnz 还必须能表示为 i32。

## SC60 网格输出

使用 `config/profiles/sc60_osem1000.conf`，显式指定实际矩阵目录或配置匹配的 matrix root。默认 OSEM 1000:10、PSF 1.5 mm、每 50 轮保存；短验证可用 CLI 覆盖迭代数。坏探测器/norm 清空，支持区为 36x96 mm，关闭 PSC 输出滤波和轴翻转。

`[InputOutput] matrix_grid_output=true` 保留完整矩阵网格，不执行 PSC 固定中心裁剪或固定 ROI 覆盖。默认 false，保持既有 PSC 行为。对称 profile 与输出裁剪分别配置，不把对称性当作系统型号。

2026-09-21 候选支持 1..5 个连续床号。多床须显式配置有限正 `bed_step_mm`，与整数体素层对齐（绝对误差 <= 1e-4 mm），范围为 1..baseNz 层；暂不支持非整数、非等距或带间隙的步进。每床重建仍独立存储，输出尺寸 `Nx × Ny × [baseNz + (bedCount-1)*step]`，保留原 ROI 配置。复用 cosine 累加：完整网格模式在体素中心采样渐变权重，按覆盖该体素的有效掩膜归一化；掩膜外值不参与平均。重叠可超过两床，不做 prototype gain-fit，不允许 seam XY alignment。PSC 默认输出及其端点权重不变。

用户确认 SC60 仍在组装、没有实采多床数据，机械步进待采集后确定。因此 SC60 recipe 的 `bed_step_mm=0` 表示未知，单床 manifest 保留 0，多床必须显式覆盖；测试夹具步进不是硬件推荐。多床物理偏移以第一张选中床为零点，床号递增对应 +Z；轴翻转仍由配置声明，软件测试不能确定未来采集端的实际朝向。

## 子集读取与计量

`loadSubset` 在加载前由 angle range 跳过空切片。切片及 GPU 缓存持有源数据所有权，防止部分 host cache 下的预取悬空；弱引用只复用仍由调用者持有的 chunk，不扩展永久缓存预算。

2026-09-21 候选增加 CSRB 磁盘子集分段读取：已有完整 host chunk 优先复用；整个 chunk 能纳入剩余 host 预算，或请求覆盖其全部角度时，仍走整块加载。否则读取并校验完整 row_ptr，仅 seek/read 所选角度的 col_idx/float32 权值区间，相邻区间合并。保留全局列编号、绝对角度、子集行顺序和原权值，不重排或量化；临时存储只通过 subset view 暴露，不能被 loadChunk 当作完整矩阵。弱引用复用仍存活的预取切片，不新增永久子集缓存预算。此路径不改变 CSRQ。

分段读取校验完整 row_ptr 及实际读取的列索引，沿用原有 strict/sample/off 设置；未读取的 payload 不在当次校验范围。正常打开目录仍检查全部 chunk 的头、文件长度及角度覆盖。`MATRIX_IO` 日志区分应用层 payload 读取字节（包含 row_ptr）、payload 加载次数、仍存活的 host 数据复用字节、跳过的空切片和 CSRB H2D 字节。分段读取的 bytes 仅计实际读入的区间；应用层读取字节不是物理磁盘流量，host 复用字节也不代表同量 memcpy。计量 scope=iterations 不含 sensitivity 准备。N1/full-cache 路径不因本改动自动获得提速。

## 读取端

| 读取端 | 入口 | 说明 |
|---|---|---|
| `SPECT_recon` | `-chunkCSRDir <dir>`；或 `[SystemMatrix] default_chunk_csr_dir`；或 exe-local `bin/matrix/*` 唯一候选自动发现 | **唯一消费路径** |

解析与校验见 `src/io/CompactChunkCSR.cpp`（chunk 校验可用 `SPECT_CHUNK_VALIDATE=off` 关闭，会打印 WARN）。

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

令 `H_k = w_k A_k`，忽略此处未展开的 PSF/CTAC 组合，当前代码计算：

```text
q_m = sum_k H_(k,m)^T [y_(k,m) / (H_(k,m) x + b_(k,m))]
S_m = sum_k H_(k,m)^T W_(k,m) 1
```

`w_k` 必须有限且 >0，在前向均值与相应灵敏度/反投语义中配套应用，不是对计数额外乘一次。`W_k` 是既有 sensitivity-only 行权重，包含 mask、detZ 权重及 norm 倒数；无权重时为单位阵。测量计数不直接乘 W；censor 的行在上游清零。各窗 norm 独立，不得复用合窗 norm 冒充逐窗校正。

**归一化边界：**生产 CLI 已禁止 norm 与背景投影同时使用（单窗的 `-bProj/-bProjList`、双窗任一窗的对应参数均拒绝），并要求双窗 norm 成对提供。因此它不是当前开放但未标定的组合。设正校正因子 `c_k` 的倒数为效率 `e_k=1/c_k`，无背景、有效行预测为正且不触发数值截断时，原始计数模型 `mean_k=e_k H_k x` 的 EM 分子中效率抵消：`(e_k H_k)^T[y_k/(e_k H_k x)] = H_k^T[y_k/(H_k x)]`，分母仍为 `H_k^T e_k`。这解释了现有 sensitivity-only norm 的实现，不表示可以预先把计数乘 c 后再次应用该 norm。mask 行须同步剔除测量和灵敏度。

普通 `detZRowWeights` 是另一入口，不应自动当成已标定 norm；它与背景可以进入现有通用 q/S 接口，但其物理解释仍需来源合同。将来若开放 norm+background，必须先明确 b 的计数空间并推导一致公式，不能仅删除 CLI 拒绝。上述代数与测试均不证明 c 的标定正确或绝对活度单位成立。

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
| 单窗 / 双窗 CSRQ + ROT90_Z | 已有执行及 toy 数值测试；CSRQ 为派生实验格式 |
| 双窗 CSRB + ROT90 / ROT90_Z | 2026-09-21 原生候选；CUDA q/S oracle、合成双窗两床入口、实际 SC60 ROI 矩阵拆窗等价检查通过，不需转换 CSRQ；非物理校准或生产采用 |
| 混合 CSRB/CSRQ | 显式拒绝 |
| CSRQ + ROT90 | 显式拒绝；不能只复制 metadata 后运行 |
| 任意新角度/Z profile、SMMD v4/TLV、list-mode | 未实现；不得按相似维度降级套用现有 profile |

原生 CSRB 双窗复用同一 q/S 数据项：逐窗累加 `w_k A_k^T[y_k/(w_k A_k x+b_k)]` 和 `w_k A_k^T W_k 1` 后更新；窗口权重不乘背景，逐窗 W 与 CTAC 指数保持独立。最多两窗，metadata 共享坐标检查和 norm/background 拒绝规则不变。projector 的 subset+CTAC 数值测试不代表 CLI 开放该组合，CLI 现有 N>1/CTAC 限制仍在。

主机缓存先按两矩阵合计字节数计算一次预算，再按矩阵大小分配；`SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` 在双 CSRB 运行中是总上限，不是每窗上限。每窗沿用完整缓存/分段读取；GPU 沿用既有最多两块缓存。CSRB 切窗时刷新 per-bed 投影/背景槽，矩阵计算量与读取量约为两窗之和，不承诺与单窗等速。CSRQ 调度与编码不变；CSRQ ROT90、ROI+PSF resident 优化仍是独立工作项。

## 参考

- `6module_bak_src/README.md` — 生成端编译与产出
- `include/io/CompactSymmetryMetadata.h` — metadata 布局常量
- `test_data/6module/260601/CompactChunkCSRB_Reconstruction_README_CN.md` — 260601 数据包内说明
- `doc/Capability_Matrix_CN.md` — 能力边界
