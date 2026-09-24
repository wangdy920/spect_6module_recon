# CSRB 通用矩阵契约设计草案（2026-09-18）

> 历史提案，非当前执行规格。2026-09-20 已实现 124B SMMD v3 / CSRB ROT90 候选；本文原始“不支持”判断针对 2026-09-14 受保护 EXE。独立 MatrixContract、SMMD v4/TLV 尚未采用，当前沿用 CompactSymmetryMetadata。实际合同以 [矩阵读取与多能窗合同](../multipinhole/doc/Format_Reader_Matrix_CN.md) 为准，验证记录见 [实施记录](2026-09-18-spect-matrix-adaptation-implementation-plan.md)。以下保留当时分析，不作为待全部实现的清单。

## 0. 结论和边界

本文件汇总当前 SPECT-recon 已有系统矩阵格式、主线支持边界、SC60 兼容性冲突，并给出一个后续可扩展的 CSRB metadata v4 契约草案。它不是已完成适配的验证报告；截至本文撰写时，主线 `multipinhole` 尚不能读取 SC60 的 124B metadata，也不能执行无 Z 对称的 `SYM_90` 分支。

设计目标：

1. 保留现有 `.csrb` chunk 负载格式，不在首轮适配中重写 62.5 GiB 级 SC60 矩阵。
2. 在读入边界把 88B、92B、124B metadata 归一成一个内部 `MatrixContract`。
3. 用显式 symmetry profile 描述角度对称和 Z 对称，禁止从多个布尔值猜语义。
4. 新格式采用固定 metadata core + TLV 扩展区；未知关键扩展必须失败，未知非关键扩展可跳过。
5. 把“metadata 可读”和“执行引擎支持该对称分支”分开验收，避免把格式兼容误报成重建兼容。

非目标：

- 不在本设计中替换 CSRB 生产路径；CSRQ 仍是实验派生格式。
- 不把 OSEM 子集数、PSF、滤波、bad detector、norm factor 等重建配置写入矩阵 metadata。
- 不改写任何既有冻结矩阵目录；旧目录继续原地只读。

## 1. 证据来源和当前事实

### 1.1 现有 metadata 布局

| 版本 | 识别方式 | 大小 | voxel 表示 | 现状 |
|---|---|---:|---|---|
| legacy float32 | 无 magic，按文件长度识别 | 88 B | `float32` | Tc99m PSC 矩阵使用；主线支持 |
| legacy float64 | 无 magic，按文件长度识别 | 92 B | `float64` | Lu177 113/208 矩阵使用；主线支持 |
| SC60 v3 | `SMMD` magic + version 3 | 124 B | 32 B 十进制字符串 | SC60 使用；主线不支持 |
| 本设计 v4 草案 | `SMMD` magic + version 4 + header/core/TLV | 变长 | core 中 float64 三元组 | 待实现 |

主线读取端常量只接受 88B 和 92B：`multipinhole/include/io/CompactSymmetryMetadata.h:14-15`。读取时先按大小分派：`multipinhole/src/io/CompactSymmetryMetadata.cpp:39-48`。SC60 v3 读取器存在于 `test_data/SC60_luzhou/spect_recon_bedmerge.cu:973-1055`，但它是原型分支源码，不在主线 `multipinhole` 生产包内。

注意：`multipinhole/doc/Format_Reader_Matrix_CN.md` 写的是 88B/96B，而当前实现常量是 88B/92B。协议文档更新时应以实现和实际矩阵文件为准，并修正这处文档漂移。

### 1.2 chunk CSRB v1

现有 `.csrb` chunk 头为 28 B：

```text
u32 magic    = 0x42525343  // "CSRB"
u32 version  = 1
i32 rows
i32 angle_offset
i32 angle_count
u64 nnz
```

负载：

```text
i64 row_ptr[rows + 1]
i32 col_idx[nnz]
f32 values[nnz]
```

这一层在三类 metadata 版本间保持一致，因此通用适配的首选策略是只扩展 metadata 契约，不改 chunk 负载。CSRQ 使用独立 `chunk_*.csrq` 派生格式，不应作为 CSRB 通用适配的中间格式。

### 1.3 当前实际矩阵

| 矩阵族 | 目录示例 | metadata | 几何 | compact 对称 | 分支 |
|---|---|---:|---|---|---:|
| PSC-6M Tc99m 141 keV | `test_data/6module/Tc99m_sysmat/..._141keV_120x120x220_csrq`（本地 CSRQ 转换副本，保留 88B source metadata 契约） | 88 B | 120x120x220, 0.3 mm, det 48x48, 90/360 angles, compactDetZ 24 | 90° + Z mirror | 8 |
| PSC-6M Lu177 113 keV | `test_data/6module/Lu177_sysmat/matrix_PSC_PIN26x28x20_113keV_120x120x220` | 92 B | 同上 | 90° + Z mirror | 8 |
| PSC-6M Lu177 208 keV | `test_data/6module/Lu177_sysmat/matrix_PSC_PIN26x28x20_208keV_120x120x220_ZFOLD` | 92 B | 同上 | 90° + Z mirror | 8 |
| SC-3M-60 Tc99m 141 keV | `test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV` | 124 B | 90x90x240, 0.4 mm, det 96x96, 90/360 angles, compactDetZ 96 | 90°，无 Z mirror | 4 |

SC60 目录含 18 个 `.csrb`，总 payload 67,106,070,664 B（约 62.48 GiB）。其 `metadata.txt` 明确：

- `metadata_binary_version=3`
- `metadata_binary_size_bytes=124`
- `full_angles=360`
- `det_nz=96`
- `compact_det_z=96`
- `sym_mode=90`
- `symmetry_order=4`
- `use_z_symmetry=0`

对应文件：`test_data/SC60_luzhou/matrix/matrix_3module_18pin_YSO_90x90x240_141keV/metadata.txt:2-31`。

两个 SC60 仿真投影 `121314derenzo_1200s.bin` 和 `sen_100MBq_1200s.bin` 的头部均为 `na=360, rpa=9216`，payload 为 float64，大小 26,542,088 B，符合 `8 + 360*9216*8`。投影几何与矩阵 full projection 形状一致；当前失败点在 metadata 读取，尚未进入投影和 CSR 数值验证。

## 2. 当前主线支持矩阵

### 2.1 主线 `multipinhole` 生产 EXE

参考 EXE：`build/clean-635dec6-20260914/bin/Release/SPECT_recon.exe`，版本 `20260914+635dec6`。

已支持：

- metadata：88B float32 voxel、92B float64 voxel。
- chunk：`.csrb` 和实验 `.csrq`。
- 对称：只接受 canonical `SYM_90Z`。
- 几何：metadata 覆盖 CLI/INI 的 dim、voxel、detN、angles。
- CSRB 执行：生产路径。
- CSRQ 执行：实验路径，kernel 固定 8 分支。
- OSEM：`-it A:B`，B 个 compact angle 子集；B 必须整除 compact angles。

硬性拒绝：

```text
use90 != 1
useZ != 1
use45 != 0
symCode != 90
symmetryOrder != 4
detNz != 2*compactDetZ
compactNumAngles != fullNumAngles/4
```

见 `multipinhole/src/io/CompactSymmetryMetadata.cpp:113-128`。

2026-09-18 实测指定 EXE 读取 SC60 目录时立即失败：

```text
[FATAL] Compact metadata size mismatch (124 bytes, supported SYM_90Z metadata
layouts are canonical float32 88 or legacy float64-voxelSize 92)
```

日志：`recon_out/sc60_multipinhole_compat_it1000/compat_osem10_psf1p5_it1000_20260918_160958.log`。因此 1000 次 OSEM、N=10、PSF 1.5 mm 的兼容性测试没有开始迭代。

### 2.2 CSRB 执行分支的现状

主线分支枚举函数按 `meta.xySymmetryCount() * meta.zBranchCount()` 动态构造分支，见 `multipinhole/src/sparse/cuda/CompactSymmetricProjectorCUDA.cpp:913-918`。这是一个有利条件：CSRB 执行层并不是在每个 kernel 调用点手写 8。

但当前 `CompactSymmetryMetadata` 内部把能力固定为：

```cpp
xySymmetryCount() { return 4; }
zBranchCount()    { return 2; }
```

所以即使打通 124B metadata，若不把 metadata 能力改成 profile 驱动，仍会继续把 SC60 解释成错误的 8 分支或在校验层失败。

### 2.3 CSRQ 的边界

CSRQ 是从 CSRB 离线转换的实验格式，布局由 `tools/convert_csrb_to_csrq.py` 生成：

- run word：`start_col << 6 | (len - 1)`
- value：每行 rowmax + u16 sqrt companding
- chunk header 48 B
- 行内列按升序 run 编码

当前转换器只接受 88B/92B metadata：`tools/convert_csrb_to_csrq.py:215-221`。CSRQ kernel 也有编译期固定分支数：

```cpp
kCsrqBranchCount = 8
```

见 `multipinhole/include/cuda/CudaLaunchConfig.h:46` 和 ready 检查 `CudaSparseOpsState.h:3787-3799`。因此 CSRQ 不是绕过 SC60 no-Z 适配的捷径；先改 CSRQ 反而会同时碰 metadata、kernel 分支数、显存布局和量化数值验证。

## 3. 已知 symmetry 语义

SC60 原型读取器已经认识多类 symmetry，见 `test_data/SC60_luzhou/spect_recon_bedmerge.cu:208-215`：

| 原型 enum | 含义 | XY 分支 | Z 分支 | 总分支 |
|---|---|---:|---:|---:|
| `SYM_NONE` | 无角度对称、无 Z 对称 | 1 | 1 | 1 |
| `SYM_ZONLY` | 仅 Z mirror | 1 | 2 | 2 |
| `SYM_180` | 180° 旋转，无 Z | 2 | 1 | 2 |
| `SYM_Z180` | 180° + Z mirror | 2 | 2 | 4 |
| `SYM_90` | 90° 旋转，无 Z | 4 | 1 | 4 |
| `SYM_90Z` | 90° + Z mirror | 4 | 2 | 8 |
| `SYM_45` | 45° sector，无 Z | 8 | 1 | 8 |
| `SYM_45Z` | 45° sector + Z mirror | 8 | 2 | 16 |

主线生产包当前只支持 `SYM_90Z`。SC60 需要的是 `SYM_90`。未来若引入其它系统，不能根据 `use90/use45/use_zsym/symmetry_order` 的组合在各处重复推断，而应在 metadata reader 处归一成一个受支持的 profile。

## 4. 通用 Matrix Contract v4 设计

### 4.1 文件布局

```text
matrix_dir/
  metadata.bin      # 唯一权威 metadata，SMMD v4
  metadata.txt      # 人读审计和生成参数，不参与重建决策
  chunk_000.csrb
  chunk_001.csrb
  ...
```

规则：

1. `metadata.bin` 是唯一 authority。
2. `metadata.txt` 不用于修复或覆盖二进制字段；若显示不一致，审计工具报警，不影响以 `metadata.bin` 为准的解析。
3. 目录内 `.csrb` 和 `.csrq` 不允许混用。
4. 矩阵身份、生成器、系统族信息进入 metadata TLV；运行时 recipe 继续负责选择矩阵 ID。

### 4.2 metadata v4 header

全部小端：

```text
u32 magic       = 0x444D4D53  // "SMMD"
u32 version     = 4
u32 header_size = 32
u32 core_size
u32 tlv_size
u32 flags       = 0
u32 reserved[2] = 0
```

v4 不再通过文件长度识别布局。读取前 8 B 即可区分：

- `SMMD` + version 3：SC60 124B 原格式
- `SMMD` + version 4：新通用契约
- 无 `SMMD` + 88/92 B：legacy PSC

### 4.3 metadata v4 core

Core 只放解释 chunk 和执行对称所必需的字段：

```text
payload_kind       = CSRB
payload_version    = 1

stored_rows        = compact_angles * compact_rpa
expanded_rows      = full_angles * det_nx * det_nz
image_voxels       = Nx * Ny * Nz
total_nnz          = sum(chunk nnz)

full_angles
compact_angles
det_nx
det_nz
compact_det_z
compact_rpa

Nx
Ny
Nz
voxel_size_xyz[3]  // float64；当前引擎可继续要求三者相等

symmetry_profile

column_layout      = ZYX_X_FASTEST
row_layout         = ANGLE_DETZ_DETX
value_encoding     = FLOAT32_LE
index_encoding     = ROWPTR_INT64_COL_INT32
```

命名上区分 `stored_rows` 和 `expanded_rows`，避免旧 `total_rows` 在不同对称族里产生歧义。

### 4.4 TLV 扩展区

```text
repeat:
  u32 tag
  u32 length
  u8 value[length]
```

兼容规则：

1. tag 最高位为 critical bit。
2. unknown non-critical TLV：跳过，不影响重建。
3. unknown critical TLV：立即失败，不允许近似解释。
4. singleton tag 重复出现：失败。
5. 改变 chunk 解释方式的字段进入 core 或 bump chunk version，不放进 optional TLV。

建议初始 tag：

| Tag | 是否关键 | 内容 |
|---|---|---|
| `IDENTITY` | 非关键 | system family、matrix ID、isotope、energy |
| `GENERATOR` | 非关键 | generator version、命令摘要、源 metadata hash |
| `GEOMETRY_AUDIT` | 非关键 | module/pinhole/aperture 摘要、几何审计状态 |
| `LEGACY_SOURCE` | 非关键 | 原 88/92/124B 来源、转换时间、保留原因 |
| `ROI_INFO` | 非关键 | 矩阵生成阶段 ROI、体素采样和 FOV 摘要 |

这些 TLV 帮助防串矩阵和追溯来源，但不决定数学语义。未来如果某个 TLV 开始影响执行，必须提升为关键扩展或进入 core，并更新版本策略。

### 4.5 symmetry profile

v4 core 不存多个互相冗余的布尔值，只存一个 `symmetry_profile`。初始 registry：

| ID | Profile | compact angle 规则 | detector Z 规则 | branch 数 |
|---:|---|---|---|---:|
| 0 | `NONE` | compact = full | detNz = compactDetZ | 1 |
| 1 | `Z_MIRROR` | compact = full | detNz = 2*compactDetZ | 2 |
| 2 | `ROT180` | compact = full/2 | detNz = compactDetZ | 2 |
| 3 | `ROT180_Z` | compact = full/2 | detNz = 2*compactDetZ | 4 |
| 4 | `ROT90` | compact = full/4 | detNz = compactDetZ | 4 |
| 5 | `ROT90_Z` | compact = full/4 | detNz = 2*compactDetZ | 8 |
| 6 | `ROT45_SECTOR` | full = 8*(compact-1) | detNz = compactDetZ | 8 |
| 7 | `ROT45_SECTOR_Z` | full = 8*(compact-1) | detNz = 2*compactDetZ | 16 |

执行入口统一检查：

```text
if matrix_contract.symmetry_profile not in engine.supported_profiles:
    fail("unsupported symmetry profile: ...")
```

这能把当前错误从“metadata size mismatch”提前变成明确的能力错误：

```text
matrix requires ROT90 (4 branches), engine supports ROT90_Z (8 branches)
```

### 4.6 归一化后的不变量

对所有版本读入后统一验证：

```text
stored_rows == compact_angles * compact_rpa
expanded_rows == full_angles * det_nx * det_nz
compact_rpa == det_nx * compact_det_z
image_voxels == Nx * Ny * Nz
total_nnz == sum(chunk nnz)
chunk angle intervals continuous and non-overlap
chunk matrix cols == image_voxels
z relation matches symmetry_profile
angle rule matches symmetry_profile
voxel sizes positive and finite
```

当前主线几何覆盖逻辑位于 `multipinhole/src/pipeline/ReconDataLoader.cpp:179-190`，应继续保留“metadata 几何优先于 CLI/INI”的规则。

## 5. 版本和兼容策略

1. 新 reader 必须读取 88B、92B、SMMD v3、SMMD v4。
2. 旧 EXE 不要求读取 v4。前向兼容的目标是“新 reader 读全部历史格式”，不是“旧软件无修改读新格式”。
3. 增加 optional TLV 不 bump metadata version。
4. 增加 symmetry profile ID 不改 core layout；旧 v4 reader 遇到未知 profile 必须失败。
5. 修改 core 字段含义必须 bump metadata version。
6. 修改 chunk 负载编码必须 bump `.csrb` chunk version，不用 metadata 隐藏。
7. 转换或重写不修改原矩阵目录；新产物写入新目录并保留 provenance。

## 6. 适配工作分解

### 6.1 第一阶段：MatrixContract reader

输入：88B、92B、124B。

工作：

1. 新建统一 `MatrixContract` 结构。
2. 按首 4 B 和大小分派 legacy/v3/v4。
3. 将 legacy 布尔组合归一为 symmetry profile。
4. 执行全部 section 4.6 不变量。
5. 更新测试矩阵：三个 PSC 实目录 + 一个 SC60 实目录。

验收：

- PSC 88/92 归一结果与现 `CompactSymmetryMetadata` 相同。
- SC60 124B 归一为 `ROT90`、4 branches、compactDetZ 96。
- v4 unknown critical TLV 和未知 profile 均失败。

### 6.2 第二阶段：CSRB no-Z 执行

工作：

1. `CompactSymmetryMetadata` 从固定 `xy=4,z=2` 改为由 profile 驱动。
2. 保持 `ROT90_Z` 生产路径数值不变。
3. 启用 `ROT90` 的 4 分支前向、ratio、反投、transform-add。
4. OSEM 子集检查继续基于 compact angles；SC60 90 angles 支持 N=10。
5. 增加空 bad-detector 或 SC60 专用 bad-detector 配置，不能沿用 PSC detector 68。

验收：

- PSC 冻结 Lu177/Tc99m 回归，同输入 relL2 不超过既有 `1e-5` 变体门；纯 reader 重构应尽量做到输出不变。
- SC60 短迭代 smoke：矩阵/投影形状、投影计数、灵敏度、更新值全部有限。
- SC60 OSEM N=10、PSF 1.5 mm 可运行；每 50 次保存。

### 6.3 第三阶段：generator v4 输出

工作：

1. 生成端写 `SMMD v4` metadata。
2. 同步生成 `metadata.txt`。
3. 填写 `IDENTITY`、`GENERATOR`、`GEOMETRY_AUDIT` TLV。
4. 保留旧 generator 输出或提供显式 `--matrix-metadata-version 4` 开关，避免混入既有生产包。

验收：

- v4 writer/reader roundtrip。
- metadata hash 与 chunk inventory 固化。
- 新旧 reader 对 PSC `ROT90_Z` 几何归一一致。

### 6.4 第四阶段：CSRQ 泛化（延后）

只有在 CSRB no-Z 路径跑通并测出带宽瓶颈后再考虑。

工作：

1. converter 读取 MatrixContract。
2. CSRQ kernel 从固定 K=8 泛化到 K=4/8/16 或模板化。
3. 调整 packed branch 布局和显存预留。
4. 转换后与 CSRB 做同输入 relL2 对照。

验收：

- SC60 CSRB 与 CSRQ 图像 relL2 满足既有量化格式 `1e-5` 门。
- 明确记录 CSRQ 相对 CSRB 的运行时间、显存和主机内存变化。

## 7. 性能预期和风险

1. `ROT90` 从 8 分支降为 4 分支，图像变换和 scratch 数量减少。
2. 无 Z 对称使 compact 存储的 nnz 增大：PSC 208 keV CSRB 约 32.78 GiB，SC60 CSRB 约 62.48 GiB。
3. 物理完整投影规模由 `full_angles * det_nx * det_nz` 决定，不能简单说 no-Z 必然慢一倍。
4. 主要风险是 I/O、主机缓存覆盖率和设备上传带宽。SC60 62.48 GiB 在 16 GB GPU 上必须分块流式处理，在 64 GB 主机上也很可能无法完整缓存。
5. OSEM N=10 不增加每轮矩阵遍历量，只改变更新时机；其性能风险不在 metadata 适配。
6. CSRQ 的 u16/run 编码可能降低带宽，但当前固定 8 分支且为实验路径，不能作为首适配路径。

## 8. 留给后续设计评审的问题

后续可以让 GPT-6 Pro 重点评审以下点：

1. symmetry profile 是否继续使用封闭 enum，还是引入两轴 `angular_symmetry` + `detector_z_symmetry` 的组合描述。
2. 未来非均匀角度、非 90/45 sector、任意旋转对称是否需要 transform table；若引入，必须区分“描述性 table”和“执行能力”。
3. `ROT45_SECTOR` 的半轴角边界规则是否继续固定为 `full = 8*(compact-1)`，还是显式存 sector boundary 表。
4. anisotropic voxel 是否进入 core 三元组后立即开放，还是在 v4 初期继续要求 `vx=vy=vz` 并用 capability 位声明。
5. TLV critical bit 的分配策略是否足够，是否需要 per-tag schema version。
6. system family / matrix ID 应放在 metadata core、关键 TLV，还是仍由外部 recipe 绑定。
7. CSRQ 若泛化，K=4/8/16 应使用编译期模板、运行期分支，还是分别为独立 chunk version。
8. 是否为 62 GiB 级 SC60 矩阵引入生成端 ROI 裁剪，避免运行时重复读取无效列。

## 9. 推荐决策

首轮采用以下方案：

1. 实现 `SMMD v4`：固定 core + TLV。
2. symmetry 初期用封闭 profile enum，不支持任意 ad hoc transform。
3. 先做 CSRB no-Z 执行，不动 CSRQ。
4. PSC `ROT90_Z` 是受保护生产基线；任何共享代码修改必须通过同输入回归。
5. SC60 首个运行配置显式禁用 PSC bad detector，使用矩阵 metadata 几何和 SC60 专用 recipe。

这个路线的风险最小，也不会把未来 CSRQ、任意角度对称或生成端 ROI 决策提前锁死。
