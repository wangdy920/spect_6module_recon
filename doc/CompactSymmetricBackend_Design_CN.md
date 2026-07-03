# compact-symmetric 后端设计方案（6module SYM_45Z）

生成时间：2026-05-23T11:47:32+08:00
修订时间：2026-05-23T15:10:00+08:00（v2：Phase 1 加入 multi-bed 与标准 image-domain PSF；v3 已重新收敛 CLI）
再修订时间：2026-05-23T15:30:00+08:00（v3：对齐现有 `-dfList/-bedOverlap` 多床位语义；取消默认 auto 切换，改为显式 compact layout + 最小 metadata sidecar）
整合修订时间：2026-05-23T16:30:00+08:00（v4：并入 CLI 简化/优化计划；公开面移除 DistanceDependent PSF 与 OSMAPOSL；补充验证资料来源与验收清单入口）

## 1. 目标与当前事实

本方案目标是在当前 SPECT-recon 中新增一个可切换的 compact-symmetric 系统矩阵后端，使现有压缩 CSRB 可以直接服务 6module 数据重建，而不必先展开为 full-row CSRB。第一阶段只优先打通 MLEM，其他算法在链路通过后再规划。

当前已确认的数据：

```text
压缩 CSRB: gatedata/6module/matrix_6module.csrb
  rows = 52992
  cols = 933120
  nnz  = 470220768

待重建投影 BIN: gatedata/6module/PSC_6module_24pin_10MBq_600s_1.bin
  num_angles = 360
  rays_per_angle = 2304
  projection_length = 829440

metadata: gatedata/6module/metadata.bin
  compact_num_angles = 46
  det_nx = 48
  compact_det_z = 24
  compact_rpa = 1152
  full_num_angles = 360
  det_nz = 48
  image = 108 x 108 x 80 = 933120 voxels
```

因此：

```text
compact rows = 46 * 1152 = 52992
full rows    = 360 * 48 * 48 = 829440
```

这不是标准 full-row CSRB。`matrix_6module.csrb` 只是把旧 compact chunk 的行域转入 CSRB 容器，仍缺少 runtime symmetry 展开语义。当前主线若直接按 `-loadCSR matrix_6module.csrb` 跑，会把投影长度按 `matrix.rows` 校验或截断，导致 full 投影与 compact 矩阵错配。

## 2. 设计原则

1. 不改变标准 CSRB 的默认语义。普通 `.csrb` 继续表示显式 full-row CSR。
2. compact 对称不是 sparse backend 的普通 CPU/CUDA 选择，而是系统矩阵布局语义，应与 `-sparseBackend` 分离。
3. 第一阶段支持 `MLEM + subset=1 + multi-bed + 标准 image-domain PSF（forward/backward 同口径）`，显式排除 CTAC、内部 scatter solver、projection-domain PSF 与 OSEM/ADMM。2026-06-06 当前实现已扩展到外部 additive `-bProj/-bProjList`、`BSREM+RDP` 和 multi-bed global-volume Operator PSF；多床位必须复用现有 `-dfList/-bedOverlap` 语义：`-dim` 表示单床位尺寸，主流程由 overlap 推导全局 Z 和 bed 起点，compact 后端只替换单床矩阵投影算子。
4. 复用 `6module_bak_src/spect_recon_mlem.cu` 已验证的核心语义：SYM_45Z 角度折叠、Z mirror、图像旋转/反旋转、full projection 到 compact ratio 的映射。
5. 先以 `metadata.bin` 作为 compact sidecar；只在用户显式声明 compact layout 时读取，避免普通 CSRB 因同目录存在 `metadata.bin` 被误切换。
6. 所有 compact 路径必须有显式校验，失败时不静默回退到标准 CSR 路径。
7. Phase 1 的 image-domain PSF 必须是 XY 各向同性核（Z 任意），否则与 D4/Z 对称变换不可交换，必须显式拒绝。
8. 借 6module compact 接入机会收敛公开 CLI：只暴露当前主线和 Phase 1 必需能力，边缘功能先从文档/默认配置退场，再按验证计划删除代码路径。

## 3. 是否在 CSRB 中加入压缩字段

### 3.1 不建议在 Phase 1 修改 CSRB v1

当前 `io::CSRHeader` 只有：

```cpp
uint32 magic;    // CSRB
uint32 version;  // 1
uint32 rows;
uint32 cols;
uint64 nnz;
```

当前 `loadCSRBinary()` 和 `CSRMap::open()` 按固定 24 字节 header 和固定文件长度解析。若直接追加字段，旧 reader 会因为文件大小不匹配或 offset 不一致而失败。若复用现有字段保存压缩标志，又放不下 detector、角度、symmetry、ROI 等必要信息。

所以 Phase 1 推荐：只提供一个布局开关，metadata 作为 sidecar 读取，但不默认自动切换普通 CSRB 语义。

```text
-loadCSR gatedata/6module/matrix_6module.csrb
-matrixLayout compact-symmetric
```

当 `-matrixLayout compact-symmetric` 时，默认读取 `dirname(loadCSR)/metadata.bin`。仅在 sidecar 不在默认位置时才需 `-compactMeta <path>` 覆盖。

### 3.2 后续可考虑 CSRB v2

若 compact 后端稳定后，才考虑 CSRB v2：

```text
CSRHeaderV2:
  magic
  version = 2
  header_bytes
  rows
  cols
  nnz
  layout_flags
  metadata_bytes
  metadata payload 或 JSON/TLV
```

但这需要统一升级 `saveCSRBinary/loadCSRBinary/CSRMap/convert_6module_sysmat_to_csrb` 和测试，不应阻塞当前 MLEM 试验。

## 4. CLI 与切换方式

设计原则：最小新增参数，不重造多床位参数，不让普通 `.csrb` 因同目录存在 `metadata.bin` 被隐式改写语义。

本节已合并 `CLI_Simplification_Plan_CN.md` 与既有 `CLI_Optimization_Proposal.md` 的当前结论：公开 CLI 只保留当前主线需要的入口；历史别名和边缘模式不再作为推荐用法出现。

新增两个参数，其中只有第一个是常用入口：

```text
-matrixLayout standard|compact-symmetric   # 默认 standard
-compactMeta  <metadata.bin>               # 可选，仅 metadata 不在 loadCSR 同目录时使用
```

参数语义：

| 参数 | 作用 |
| --- | --- |
| `-matrixLayout standard`（默认） | 按 full-row CSR 解释，完全保持现有行为。 |
| `-matrixLayout compact-symmetric` | 按 compact-row CSR + metadata 解释；缺 metadata 直接报错。 |
| `-compactMeta` | 可选 sidecar 覆盖路径；未指定时默认 `dirname(loadCSR)/metadata.bin`。 |
| `-sparseBackend cuda` | compact Phase 1 必须为 CUDA；standard 仍支持 auto/cpu/ref/cuda。 |

symmetry 模式从 metadata 推断（当前只支持 `SYM_45Z`），不需要 CLI 暴露 `-compactSym`。

这比 `auto` 更保守：压缩矩阵由用户显式选择，非压缩 CSRB 不受任何 sidecar 文件影响。若后续实测使用体验需要，可再把 `auto` 作为非默认模式加入，而不是 Phase 1 直接改变默认行为。

### 4.1 公开 CLI 面收敛

当前公开文档建议只展示以下主入口：

```text
输入/矩阵: -df 或 -dfList, -loadCSR, -mapCSR
几何:     -dim, -vox, -ang, -detN
算法:     -alg mlem|bsrem|admm（MLEM/OSEM 由 -subsets 区分）
后端:     -sparseBackend cuda|auto|cpu|ref, -butterBackend auto|fftw|cufft
compact:  -matrixLayout compact-symmetric, -compactMeta
PSF:      -psf <sigma>, -psfMode Operator|PostSmooth|None
多床位:   -dfList, -bedOverlap, -bProjList
```

收敛规则：

- 文档统一使用 `-alg`，不再推荐 `-opti`。
- 文档统一使用 `-dfList` 表示多床位，不允许重复 `-df`。
- 文档统一使用 `-bedOverlap` 表示床位关系，不新增床位偏移 CLI。
- 文档统一使用 `-psf <sigma>`，不再展示 `-psfFWHM` 或 `-psf <sigma,kernel>`。
- `DistanceDependent` PSF 不再作为公开模式展示；保留代码入口只作为迁移缓冲，后续 P1/P2 删除。
- `OSMAPOSL` 不再作为公开算法展示；保留历史实现只作为兼容缓冲，compact-symmetric 后端不规划接入。

### 4.2 退场功能分期

| 功能 | 当前状态 | 后续动作 | 验证要求 |
| --- | --- | --- | --- |
| DistanceDependent PSF | 已从 `USAGE_CN.md` 与默认 `spect_config.ini` 退场 | P1: CLI/config 显式拒绝；P2: 删除 DD kernel 数据结构和 CPU/CUDA 分支 | `-psfMode DistanceDependent` 与 `-psfDist*` 均失败；`-psf <sigma> -psfMode Operator` 仍通过 |
| OSMAPOSL | 已从公开使用说明、README 和 smoke 脚本说明退场；配置文件标记 deprecated | 后续若确认无生产依赖，再删除 parser、optimizer、reconstructor 和测试 | `-alg osmaposl` 显式失败；MLEM/OSEM、BSREM、ADMM 回归不变 |
| `-opti` 算法别名 | 暂保留兼容，不再推荐 | 后续版本转为 warning，再移除 | `-alg` 全路径覆盖常用算法 |
| `-conv` legacy PSF alias | 暂保留兼容，不再推荐 | DD PSF 删除后单独评估 | `-psf` 行为稳定 |

典型命令（与现有 6module 数据一致）：

```powershell
.\build_cuda_scaffold\bin\SPECT_recon.exe `
  -df gatedata\6module\PSC_6module_24pin_10MBq_600s_1.bin `
  -loadCSR gatedata\6module\matrix_6module.csrb `
  -matrixLayout compact-symmetric `
  -alg mlem -subsets 1 -it 1 `
  -sparseBackend cuda `
  -detN 48,48 -dim 108,108,80 -vox 0.3 `
  -out recon_out\6module_compact_mlem_smoke
```

多床位沿用现有 `-dfList` 与 `-bedOverlap`，共享同一 compact 矩阵；不要新增床位偏移 CLI 参数：

```powershell
.\SPECT_recon.exe `
  -dfList "bed1.bin;bed2.bin" `
  -bedOverlap 0 `
  -loadCSR gatedata\6module\matrix_6module.csrb `
  -matrixLayout compact-symmetric `
  -alg mlem -subsets 1 -it 1 `
  -psf 1.27 -psfMode Operator `
  -sparseBackend cuda `
  -out recon_out\6module_compact_mlem_2bed
```

多床位中 `-dim` 仍表示单床位尺寸；全局输出 `Nz` 由 `prepareBedSetup()` 根据 bed 数与 `-bedOverlap` 计算，床位起点由 `makeBedZStarts()` 推导。compact 后端不得引入第二套床位 offset 参数。

非压缩 full-row CSRB 路径完全不受影响：

```powershell
.\SPECT_recon.exe -df proj.bin -loadCSR full_matrix.csrb -sparseBackend cuda
```

## 5. 新数据结构

新增文件建议：

```text
include/io/CompactSymmetryMetadata.h
src/io/CompactSymmetryMetadata.cpp
include/cuda/CompactSymmetricProjector.h
src/cuda/CompactSymmetricProjectorCUDA.cpp 或并入 src/cuda/SparseOpsCUDA.cpp 的独立 section
```

核心结构：

```cpp
enum class CompactSymmetryMode {
  None,
  ZOnly,
  Sym45,
  Sym45Z,
  Sym90,
  Sym180,
  SymZ180
};

struct CompactSymmetryMetadata {
  int totalRows;
  int totalVoxels;
  int compactNumAngles;
  int fullNumAngles;
  int detNx;
  int detNz;
  int compactDetZ;
  int compactRaysPerAngle;
  int Nx;
  int Ny;
  int Nz;
  float voxelSize;
  float pixelSize;
  float detectorY;
  float detectorThickness;
  bool useZSym;
  bool use45Sym;
  int symCode;
  int symmetryOrder;
  CompactSymmetryMode mode;

  int fullRaysPerAngle() const { return detNx * detNz; }
  int64_t fullRows() const { return int64_t(fullNumAngles) * fullRaysPerAngle(); }
  int64_t compactRows() const { return int64_t(compactNumAngles) * compactRaysPerAngle; }
};
```

校验必须覆盖：

```text
matrix.rows == meta.compactRows()
matrix.cols == meta.totalVoxels == meta.Nx * meta.Ny * meta.Nz
projection.size() == meta.fullRows()
args.detNz == meta.detNz, args.detNx == meta.detNx
args.numAngles == meta.fullNumAngles
args.Nx/Ny/Nz 与 metadata 一致
SYM_45Z 要求 Nx == Ny
compactNumAngles == fullNumAngles / 8 + 1
compactRaysPerAngle == detNx * compactDetZ
compactDetZ <= detNz
```

## 6. 数学模型

旧 compact 后端表达的不是 full matrix `A`，而是 compact matrix `Wc` 加 symmetry runtime 变换。

记 `K` 为 image-domain PSF 算子（标准 Gaussian，XY 各向同性，可分离），其在 Phase 1 中要求与所有 D4 XY 变换及 Z mirror 可交换：

```text
K · T_{s,z} = T_{s,z} · K   ∀ s,z
```

实现层面这要求 PSF kernel 在 XY 平面各向同性（圆对称），Z 方向任意；distance-dependent 投影域 PSF 不在 Phase 1 范围。

对床位 `b`、XY symmetry branch `s`、Z branch `z`，compact helper 只负责单床位局部体 `f_b`：

```text
f_b           : 从全局图像按 bedZStarts[b] 取出的单床位局部体
P_b           : bed b 的 full 投影
K_b f_b       : PSF-smoothed 图像
T_{s,z}(K_b f_b) : 旋转/翻转后的 smoothed 图像
y_c(s,z,b)    = Wc · T_{s,z}(K_b f_b)
r_c(row,s,z,b)= P_b(phi(row, s, z)) / max(y_c(row), eps)
bp_rot        = Wc^T · r_c
local_numer_b += K_b^T · T_{s,z}^T(bp_rot)       // accumulate over (s,z)
```

敏感度生成单床位局部结果，随后按现有多床位主线累加到全局分母：

```text
local_denom_b += K_b^T · T_{s,z}^T( Wc^T · 1_valid(row,s,z) )
```

其中 `1_valid` 在 §7 给出的 valid 规则下为 1，否则为 0；等价于对 full row 的 unit vector 做 backward。

然后沿用现有多床位主线，把单床位局部结果累加到全局体：

```text
global_numer[bedZStarts[b] + local_z, x, y] += local_numer_b[local_z, x, y]
global_denom[bedZStarts[b] + local_z, x, y] += local_denom_b[local_z, x, y]
```

MLEM 更新只对全局图像执行一次：

```text
f_new[j] = f_old[j] * global_numer[j] / max(global_denom[j], eps)
```

多床位说明：

- 同一 compact 矩阵 `Wc` 在所有床位共享；每个床位只更换局部图像窗口和该床位 full 投影。
- 床位起点由现有 `-dfList/-bedOverlap` 推导，不新增第二套床位偏移参数。
- 重叠区体素按现有主线语义累加 `global_numer/global_denom`，不是分别更新后再拼接。
- `local_denom_b` 通常与床位投影无关，可在 Phase 1 正确性通过后缓存复用；第一版可以按床位重算，先避免缓存语义引入额外变量。

Phase 1 不引入 CTAC、内部 scatter solver、projection-domain PSF，也不接入 OSMAPOSL。当前实现已允许外部 additive background projection 进入 ratio，并已支持 compact `BSREM+RDP`。

## 7. SYM_45Z 映射逻辑

### 7.1 full angle from compact

复用旧代码规则：

```cpp
int Q = full_num_angles / 8;
// compact_num_angles = Q + 1, base angle covers 0..45 deg inclusive.
if ((sym_id & 1) == 0) {
  if (ck >= Q) invalid;
  fa = sym_id * Q + ck;
} else {
  if (ck <= 0) invalid;
  fa = sym_id * Q + (Q - ck);
}
```

对当前数据：`full=360`, `Q=45`, `compact=46`。偶数 branch 使用 `ck=0..44`，奇数 branch 使用 `ck=1..45`，避免 0/45 度边界重复。

### 7.2 Z mirror

compact row 中：

```text
rr = row % compact_rpa
ix = rr % det_nx
izc = rr / det_nx
```

非 mirror：

```text
iz_full = izc
```

mirror：

```text
iz_full = det_nz - 1 - izc
if iz_full < compact_det_z: invalid
```

当前数据 detZ=48、compactDetZ=24，因此 mirror branch 映射到上半 Z，不重复下半 compact Z。

### 7.3 full projection index

```text
full_idx = full_angle * (det_nx * det_nz) + iz_full * det_nx + ix
```

## 8. 图像坐标变换

SYM_45Z 的 XY branch 使用 D4 八种变换，要求 `Nx == Ny`：

| sym_id | 变换 |
| --- | --- |
| 0 | identity |
| 1 | reflect x=y |
| 2 | rotate 90 CCW |
| 3 | flip X |
| 4 | rotate 180 |
| 5 | reflect anti-diagonal |
| 6 | rotate 270 CCW |
| 7 | flip Y |

Z branch：

```text
z_flip = 0: oz = iz
z_flip = 1: oz = Nz - 1 - iz
```

旧代码的 forward transform：

```text
img_rot[p] = img[g(p)]
```

旧代码的 backward add：

```text
bp_orig[g(p)] += bp_rot[p]
```

Phase 1 可直接移植这些 CUDA kernels，保留 float device 计算，与现有 CUDA sparse backend 的 float-on-device 策略一致。

## 9. 与现有 MLEM 链路的集成

当前 MLEM 路径是：

```text
MainProgram -> MLEMOptimizer -> OSEM_step(io::CSRView)
```

普通 `OSEM_step` 假定 `p.size() == A.rows * bedCount`。compact 模式下 `p.size() == meta.fullRows() * bedCount`，因此必须在该检查前进入 compact 分支。

推荐最小侵入方案：

1. `ReconArgs` 新增：
   ```cpp
  std::string matrixLayout = "standard";   // standard|compact-symmetric
   std::string compactMetaPath;         // 可选 sidecar override
   ```
2. `ReconOpts` 新增只读句柄：
   ```cpp
   const CompactSymmetryMetadata* compactMeta = nullptr;
   ```
3. `MainProgram` 在加载 CSR 后：
  - 仅当 `args.matrixLayout == "compact-symmetric"` 时进入 compact 模式；
  - metadata 路径优先取 `-compactMeta`，否则取 `dirname(loadCSR)/metadata.bin`；
   - 解析 metadata 后用 `meta.fullRows()` 校验每个床位的投影长度；
   - 多床位允许：bed 数量 ≥ 1，所有 bed 共享同一 `Wc`；
   - 仅允许 `subsets == 1`、`alg == mlem`；其他算法/选项显式报错；
   - 标准 image-domain PSF（高斯）允许通过，distance-dependent 或 projection-domain PSF 显式报错；
  - 对每床位调用 compact sensitivity helper 得到 `local_denom_b`，再用现有 bed 起点累加到全局 `denom`。
4. `OSEM_step(io::CSRView)` 顶部检查 `opts.compactMeta`：
   - 进入 compact 分支后跳过 standard 行域检查；
  - 从全局 image 中按 `bedZStarts[b]` 提取/引用单床位局部窗口；
  - 对每个床位调用 `runCompactSymmetricMlemNumerCUDA()` 得到 `local_numer_b`，再累加到全局 `numer`；
  - 复用现有 update loop 完成 `f *= global_numer / max(global_denom, eps)`；
   - mask / profile / 输出全部复用主线。

PSF 集成约束：

- compact helper 接收的 `image` 参数应为 **已经过 K 平滑** 的图像，或 helper 内部以回调方式调用主线 PSF；推荐前者，便于复用 `VolumeOps`/CUDA PSF。
- 返回的 `numer` / `denom` 在 helper 内部完成所有 16 个 branch 的累加，但 PSF^T 在 helper 外部对累加结果再施加一次（因为 K 与 T_{s,z} 可交换，等价但更省 kernel launch）。
- 等价路径：`numer_b = K^T · Σ_{s,z} T^T (Wc^T (P_b/φ ÷ Wc T(K f_b)))`；外层只调一次 `K f_b` 和一次 `K^T (·)`。

这样可以最大限度复用现有 MLEM 更新、输出、mask、profile、PSF、`bedZStarts`/overlap 与 `IOptimizer` 框架。

## 10. 新 CUDA helper API

建议新增（注意：PSF 由调用方在 helper 之外施加，helper 内部只跑 16-branch compact 流水）：

```cpp
namespace spect::cuda {

// 计算单床位 sensitivity（已应用 16-branch 累加，未施加 PSF^T）。
// 调用方需在外部对结果再做一次 K^T 平滑。
bool computeCompactSymmetricSensitivityCUDA(
    const io::CSRView& compactMatrix,
    const CompactSymmetryMetadata& meta,
    int threads,
    double* denom);

// 计算单床位 MLEM numerator（image 已是 K-smoothed；输出未做 K^T）。
// 调用方在外部完成 K f -> helper -> K^T numer 的串联。
bool runCompactSymmetricMlemNumerCUDA(
    const io::CSRView& compactMatrix,
    const CompactSymmetryMetadata& meta,
    const double* fullProjection,     // bed b 的 full 投影，长度 = meta.fullRows()
    const double* smoothedImage,      // bed b 的 K-smoothed 图像，长度 = meta.totalVoxels
    int threads,
    double* numer);

}
```

多床位调用约定：MainProgram 按床位串行调用上述两个 helper；compact 矩阵的 cuSPARSE 描述符与 device buffer 在多次调用之间复用（与现有 sparse cache 策略一致），仅 image / projection device buffer 按床位刷新。每个 helper 只产出单床位局部结果，调用方负责按 `bedZStarts[b]` 累加到全局 `numer/denom`，与当前 standard 多床位路径保持同一 overlap 语义。

Phase 1 内部流程：

```text
for sym_id in 0..7:
  for z_mirror in 0..1:
    img_rot = T_sym,z(smoothedImage)
    y       = Wc * img_rot
    ratio   = gather_full_projection_and_divide(fullProjection, y, sym_id, z_mirror)
    bp_rot  = Wc^T * ratio
    numer  += T_sym,z^T(bp_rot)
```

敏感度同理，只是 `ratio` 由 `unit_ratio` kernel 生成（valid row 处填 1.0）。

实现位置：初期并入 `src/cuda/SparseOpsCUDA.cpp`，复用现有 cuSPARSE/cache/float conversion 工具；稳定后拆到独立 `CompactSymmetricProjectorCUDA.cpp`。

## 11. 支持范围与显式拒绝

Phase 1 支持：

```text
algorithm        = MLEM
matrixLayout     = compact-symmetric（显式指定）
sparseBackend    = cuda
subset           = 1
bedCount         ≥ 1（multi-bed enabled，共享同一 compact 矩阵）
bedOverlap       任意合法值（沿用现有 -bedOverlap；bed 起点由主流程推导）
image-domain PSF = 标准 Gaussian（XY 各向同性，forward 与 backward 同口径）
CTAC             = off
bProj            = optional additive projection term（-bProj/-bProjList）
wProj/wImg       = off
projection-domain PSF = off
regularization   = BSREM+RDP only
detN             = 48,48
dim              = 108,108,80
```

Phase 1 显式拒绝并报错：

```text
BSREM / ADMM / 其他非 MLEM 算法
subset > 1
CTAC attenuation factors
内部 scatter solver；`wProj/wImg` 乘性校正
projection-domain PSF / distance-dependent PSF
非 XY 各向同性 image-domain PSF
non-SYM_45Z metadata
Nx != Ny
projection length != meta.fullRows() * bedCount
matrix.rows != meta.compactRows()
matrix.cols != meta.totalVoxels
显式 compact-symmetric 模式下找不到 metadata
```

## 12. 验证计划

### P0：文件与 metadata 单测

新增测试：

```text
CompactMetaTest.Reads6ModuleMetadata
CompactMetaTest.DetectsSym45ZFromLegacyMetadata
CompactMetaTest.ValidatesRowsAndProjectionLength
```

用当前文件验证：

```text
matrix rows = 52992
projection rows = 829440
meta compact rows = 52992
meta full rows = 829440
```

### P1：小型 synthetic parity

构造极小 `Nx=Ny`、`detZ=2*compactDetZ`、`fullAngles=8*Q` 的 toy compact matrix，CPU 参考直接按公式展开：

```text
compact numer == explicit full operator numer
compact denom == explicit full operator denom
```

该测试不依赖真实大文件，先锁定角度折叠、Z mirror 和图像 transform。

### P2：当前 6module 1 iter smoke

运行：

```text
SPECT_recon compact MLEM 1 iteration（显式 -matrixLayout compact-symmetric；metadata 默认取 loadCSR 同目录）
```

验收：

```text
打印 "compact-symmetric layout enabled via <metadata path>"
不截断投影
denom 非空且无 NaN/Inf
image 输出尺寸 108x108x80
profile 中出现 compact_cuda_numer / compact_cuda_sens
```

### P2b：multi-bed 与 PSF 等价性测试

构造 2 床位 toy 数据（或复用 6module 单床位投影做 bed1=bed2 的退化测试）：

```text
- 2 床位、共享 Wc，沿用 -dfList/-bedOverlap：
  · overlap=0 且 bed1=bed2 时，两个 local numer/denom 一致
  · overlap>0 时，全局重叠区 global_numer/global_denom 等于两个 local 结果相加
- 启用标准 image-domain Gaussian PSF：
  · K f -> compact-helper -> K^T 流程与 explicit-full + 同 PSF 路径在 P1 toy 上数值一致
  · 关 PSF 与开 σ=0 PSF 结果一致
- 拒绝路径：传入 distance-dependent PSF 或 projection-domain PSF -> CLI 直接报错
```

### P3：对齐旧 `spect_recon_mlem.cu`

同输入、同初始化、同 MLEM 1 iter 和少量 iter，对比输出：

```text
max_abs_diff
relative_l2
slice/profile sanity
```

由于当前主线 double host / float device 与旧程序 float device 可能存在微小差异，建议首轮容差按 `1e-5 ~ 1e-4` 相对量级评估，再根据实测收紧。

## 13. 后续算法扩展顺序

MLEM 通过后再扩展：

1. OSEM subset：需要定义 subset 是 full angle subset 还是 compact row subset。建议按 full angle subset，ratio kernel 中按 `fa` 过滤。
2. distance-dependent / projection-domain PSF：需要专用 kernel 与对称性兼容性分析；不能直接套用 §6 的 `K·T = T·K` 简化。
3. background/scatter：ratio kernel 改为 `P_full / max(y + b_full, eps)` 或按当前主线语义接入。
4. CTAC：需要明确 attenuation 是 full angle/full voxel 还是 compact branch 后应用，不能直接复用标准 CSR row angle 推断。
5. BSREM：复用 compact numerator，并补 subset/global denom、curvature/kappa 语义。
6. ADMM：等 MLEM/BSREM numerators 稳定后再评估是否接入。
7. OSMAPOSL：公开入口退场，compact-symmetric 后端不规划接入；如未来确有生产需求，需重新立项并补独立验收。
8. 跨床位 `denom` 缓存：当 PSF 与 bedZStarts/overlap 语义满足缓存可重用条件时复用，避免每床位重算。

## 14. 风险与规避

| 风险 | 规避 |
| --- | --- |
| full 投影被标准主线按 compact rows 截断 | compact 模式在投影长度校验前接管 expected rows = `meta.fullRows() * bedCount`。 |
| 45 度边界角重复或漏算 | 严格移植 `full_angle_from_compact()` 的 valid 规则。 |
| Z mirror 重复中心层 | 严格使用 `iz_full < compact_det_z` invalid 规则。 |
| 图像旋转方向错 | synthetic explicit-full parity 测试先行。 |
| CSRB v1 header 扩展破坏旧 reader | Phase 1 使用 `metadata.bin` sidecar。 |
| compact 后端被误用于普通 CSRB | 默认 `standard`；只有显式 `-matrixLayout compact-symmetric` 才读取 metadata。 |
| sidecar 误匹配（同目录恰好存在无关 metadata.bin） | compact 模式仍严格校验 `rows`、`cols`、`fullRows` 与 CSRB / 投影一致，不一致直接报错；记录 metadata 路径到日志。 |
| 多床位之间 sensitivity 误共享 | 默认按床位重算 denom；缓存策略留到 Phase 1 之后。 |
| 标准 image-domain PSF 在压缩对称下不可交换 | 限制 PSF 为 XY 各向同性 Gaussian；非对称核显式拒绝。 |
| 后续多算法过早扩张 | Phase 1 只开 MLEM；其他路径显式报错。 |

## 15. 推荐实施顺序

```text
1. 新增 CompactSymmetryMetadata 读写与校验。
2. CLI 增加 -matrixLayout/-compactMeta；默认 standard，compact 必须显式指定，metadata 默认取 loadCSR 同目录。
3. compact mode：用 meta.fullRows*bedCount 校验投影，禁止不支持选项；校验 PSF 类型与现有 multi-bed 参数。
4. CUDA helper：computeCompactSymmetricSensitivityCUDA（含 PSF^T 由调用方施加的约定）。
5. CUDA helper：runCompactSymmetricMlemNumerCUDA（输入已 K-smoothed）。
6. OSEM_step(io::CSRView) 顶部接 compact numerator 分支，按床位循环生成 local numer/denom，再用 bedZStarts 累加到全局 update。
7. 添加 synthetic parity tests（含 multi-bed overlap 累加等价、PSF 开关等价）与当前 6module smoke test。
8. 与 6module_bak_src/spect_recon_mlem.cu 做 1 iter 对齐。
```

## 16. 验证资料来源与新增验收入口

本设计的验证步骤来自以下既有资料，而不是重新发明一套测试口径：

| 资料 | 可复用结论 |
| --- | --- |
| `refactor_docs/SRC_RECOVERY_GUIDE.md` | 结构、构建、全量单测、行为、CUDA 分阶段验收的 DoD 结构。 |
| `refactor_docs/CUDA_SPARSEOPS_DESIGN.md` | 先做最小 CUDA 后端闭环，再做缓存/重复调用成本控制，最后做真实主流程端到端回归。 |
| `docs/Testing/COMPARISON_TEST_GUIDE.md` | 同输入、同参数、同输出对比的回归方法。 |
| `doc/SystemMatrix_6module_compact_vs_CSRB_CN.md` | compact chunk/CSRB 语义差异和“不能直接用 compact rows 跑 full projection”的风险依据。 |
| `refactor_docs/REFACTOR_AUDIT_LOG.md` | 已验证过的多床位、CTAC、PSF、CUDA VolumeOps、full-subset fused path 的历史回归事实。 |

新增执行目标指令保存到：`doc/CompactSymmetricBackend_Execution_Target_CN.md`，用于固定 6module 圆柱仿体单床位 smoke：`PSC_6module_24pin_10MBq_600s_1.bin` + `matrix_6module.csrb`，标准 PSF FWHM=1.5 mm，Butterworth `0.35,2`，并要求加载到 1 次 MLEM 迭代完成小于 20 s。

新增验收清单保存到：`doc/CompactSymmetricBackend_Acceptance_Checklist_CN.md`。

## 17. 当前结论

当前压缩 CSRB 可以作为 compact-symmetric 后端的矩阵输入，但不能作为标准 CSRB 直接重建 full 投影。Phase 1 通过显式 `-matrixLayout compact-symmetric` + `metadata.bin` sidecar 切换 compact 后端，覆盖 `MLEM + subset=1 + multi-bed + 标准 image-domain PSF`，多床位沿用现有 `-dfList/-bedOverlap` 与全局 overlap 累加语义。2026-06-06 当前实现已扩展到外部 additive `bProj`、compact `BSREM+RDP` 和 multi-bed global-volume Operator PSF；仍显式排除 CTAC、内部 scatter solver、projection-domain PSF、OSMAPOSL、ADMM compact 后端与未定义的乘性校正。CSRB v2 扩展应等 compact 主线验收稳定后再设计。
