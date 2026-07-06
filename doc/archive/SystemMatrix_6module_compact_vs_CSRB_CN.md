# 6module compact chunk 与 SPECT-recon CSRB 系统矩阵格式对比

生成时间：2026-05-23T09:48:15+08:00

## 结论先行

`6module_bak_src/spect_recon_mlem.cu` 依赖的系统矩阵不是当前 SPECT-recon 的 `.csrb` 单文件格式，而是 `metadata.bin + chunk_*.bin` 的目录型 compact chunk 格式。该格式由 `6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v2.cu` 生成，核心目标是利用 45 度旋转对称和 Z 镜像对称，只保存 compact 基础角度/探测器区域，再由 CUDA 重建端在每次 forward/backward 时用图像坐标变换和投影行映射恢复 full-angle 语义。

当前 SPECT-recon 的主线格式是 `.csrb`：单文件显式 CSR，包含 `CSRB` magic/version/rows/cols/nnz 头，以及连续的 `row_ptr(uint32) + col_idx(uint32) + val(float)`。它不在文件格式层保存几何对称语义，而是把系统矩阵视为完整的行压缩稀疏矩阵，由 CLI/投影数据/多床位参数解释 rows 与 image grid 的关系。

工程判断：

- 对当前 SPECT-recon 主项目、现有 CUDA sparse 后端、MLEM/OSEM/BSREM/OSMAPOSL/ADMM 统一流程，`.csrb` 更适合作为标准输入格式。
- 对固定 6 模块、固定对称规则、只做专用 CUDA MLEM/MAP-EM 的独立程序，旧 compact chunk 格式在磁盘/显存占用上更有针对性，可能更适合极限压缩系统矩阵。
- 不建议把旧 compact chunk 直接替代当前项目的 CSRB。若要复用旧生成器，应优先做“compact chunk -> 标准 CSRB”的转换器；若必须保留 compact 对称收益，则应作为一个新的专用矩阵后端接入，而不是让 `-loadCSR` 接受 chunk 目录。

## 代码依据

检查的主要文件：

- `6module_bak_src/spect_recon_mlem.cu`
- `6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v2.cu`
- `include/io/CSRTypes.h`
- `include/io/CSRBuilder.h`
- `src/io/CSRBuilder.cpp`
- `src/main/MainProgram.cpp`
- `src/app/OSEMReconstructor.cpp`
- `include/interfaces/ISparseOps.h`
- `src/cuda/SparseOpsCUDA.cpp`
- `docs/Design/Refactor_Consensus_Plan.md`
- `doc/RECON_FLOW_CN.md`

## 旧 6module compact chunk 格式

### 输出组织

`spect_sysmat_PSC_6module_24pin_YSO_v2.cu` 的默认输出是一个目录，例如：

```text
matrix_PSC_6modules_24pin_YSO_3X5_PEN_DOI_V2/
  metadata.bin
  metadata.txt
  chunk_000.bin
  chunk_001.bin
  ...
```

重建端 `spect_recon_mlem.cu` 通过 `--matrix-dir DIR` 读取该目录，不读取 `.csrb`。

### metadata.bin

`metadata.bin` 是几何和压缩语义描述，不是标准 CSR header。按代码读写顺序，它包含：

- `total_rows`
- `total_voxels`
- `max_nnz`
- `compact_num_angles`
- `det_nx`
- `compact_det_z`
- `compact_rpa`
- `full_num_angles`
- `Nx, Ny, Nz`
- `voxel_size`
- `det_nx_dup`
- `det_nz`
- `pixel_size`
- `detector_y`
- `detector_thickness`
- `use_90sym`
- `use_zsym`
- 可选新增字段：`use_45sym`, `sym_code`, `symmetry_order`, `roi_radius`, `roi_side`

这些字段使重建端能够判断 `45`, `90`, `180`, `zonly`, `45z` 等 symmetry mode，并在 CUDA kernel 中把 full-angle/full-detector 投影行映射回 compact row。

### chunk_*.bin

每个 chunk 文件是“按行变长列表”格式：

```text
int32 rt            # 本 chunk 行数 = angle_count * compact_rays_per_angle
int32 angle_offset  # compact angle 起点
int32 angle_count   # compact angle 数
for row in 0..rt-1:
    int32 n
    int32 col[n]    # 0-based voxel index
    float weight[n]
```

这不是磁盘上的连续 CSR，因为文件里没有全局 `row_ptr` 数组。`spect_recon_mlem.cu` 读取时先扫一遍每行 `n` 构造 `row_ptr`，再回到 payload 开头读取 `col_idx/values`。

### 生成端 direct/coo 的含义

生成器内部有两种输出模式：

- `direct`：GPU 常驻定长 row buffer，跳过 COO sort/reduce，速度通常更快但显存占用更高。
- `coo`：先写 COO key/value，再 sort/reduce，显存更可控，适合验证或回退。

无论内部用 `direct` 还是 `coo`，落盘的 chunk 文件仍保持上述旧版 row-wise chunk 格式。

### 重建端如何使用

`spect_recon_mlem.cu` 的流程是：

1. 读取 `metadata.bin`，确定 compact 行数、full angle 数、图像尺寸、对称模式。
2. 扫描并排序所有 `chunk_*.bin`。
3. 按 GPU 分配 compact angle 范围。
4. 对每张 GPU，把相关 chunk 的行拼成一个 `HostCSR64`：
   - `row_ptr(int64)`
   - `col_idx(int32)`
   - `values(float)`
5. 初始化 GPU worker 时把 `row_ptr` 转成 `int32`，并创建 cuSPARSE CSR descriptor。
6. 每轮迭代中，对每个 symmetry/z mirror：
   - 把图像变换到 compact 矩阵坐标系；
   - 对 compact CSR 做一次大 SpMV；
   - 在 GPU 上从 full projection 取对应观测值并形成 compact ratio；
   - 对同一 compact CSR 做 transpose SpMV；
   - 把反投影结果旋回原图像坐标并累加。

因此旧格式本身只保存 compact 矩阵；完整系统矩阵语义由“compact CSR + 对称变换 + full projection 行映射”共同定义。

## 当前 SPECT-recon CSRB 格式

### 文件结构

当前 `.csrb` 是单文件 CSRBinary：

```text
CSRHeader:
  uint32 magic = 0x43535242  # 'CSRB'
  uint32 version = 1
  uint32 rows
  uint32 cols
  uint64 nnz

uint32 row_ptr[rows + 1]
uint32 col_idx[nnz]
float  val[nnz]
```

`loadCSRBinary()` 会检查 magic 和文件长度；`CSRMap::open()` 会把同一文件映射成只读 `CSRView`，并把 `row_ptr/col_idx/val` 指针直接指向 mmap/MapViewOfFile 区域。

### 构建来源

当前项目可以从 MATLAB 风格的 `W_ind/W_dat` 构建 CSR：

- `W_ind` 必须是 `int32`，输入索引为 1-based；
- `W_dat` 可为 double 或 single；
- 构建时按阈值剪枝，并把列索引转换为 0-based `uint32`；
- 值落盘为 `float`；
- 可 `-saveCSR` 持久化为 `.csrb`，后续用 `-loadCSR` 复用。

### 重建端如何使用

`src/main/MainProgram.cpp` 的主线是：

1. 若传入 `-loadCSR`：
   - `-mapCSR on` 时走 `CSRMap::open()`，得到零拷贝只读 `io::CSRView`；
   - 否则走 `loadCSRBinary()`，读入内存 `io::CSR`。
2. 若未传入 `-loadCSR`，从 MAT 索引/权重文件构建 CSR，可选择 stream 或 in-memory 模式。
3. 校验 `matrix.cols == baseNvox`。
4. 投影长度按 `matrix.rows * bedCount` 检查。
5. 计算 `denom = A^T * 1`。
6. MLEM/OSEM/BSREM 等统一通过 `IOptimizer` 调用 `OSEM_step` 或对应 step。
7. 若矩阵是 `io::CSRView` 且 sparse backend 为 CUDA，full-subset 有专门的 fused CUDA numerator 路径；否则通过 `ISparseOps::spmvSubset/spmvTSubset` 或 CPU fallback 执行。

当前项目把矩阵格式、算法流程、CUDA 后端解耦：`.csrb` 只表达显式稀疏矩阵，CUDA 细节在 `SparseOpsCUDA.cpp` 中处理。

## 主要区别

| 维度 | 旧 6module compact chunk | 当前 SPECT-recon CSRB |
| --- | --- | --- |
| 文件形态 | 目录：`metadata.bin + chunk_*.bin` | 单文件：`.csrb` |
| 是否标准 CSR 落盘 | 否。chunk 内是 row-wise 变长列表，无全局 `row_ptr` | 是。header 后直接是 `row_ptr/col_idx/val` |
| 行语义 | compact angle / compact det-z 行，依赖 symmetry runtime 展开 | 显式 rows，通常直接对应投影射线行 |
| 几何信息 | metadata 中包含 detector、voxel、symmetry、ROI 等 | CSRB 只含 rows/cols/nnz；几何由 CLI/配置/投影数据解释 |
| 对称压缩 | 强依赖 45 度/90 度/Z 对称，能显著减少矩阵规模 | 默认不编码对称；如果要利用对称，需要在生成 CSRB 前展开或新增后端 |
| 索引和值类型 | chunk col 为 `int32`，host row_ptr 为 `int64`，GPU row_ptr 转 `int32`；值为 `float` | `row_ptr/col_idx` 为 `uint32`，header `nnz` 为 `uint64`，值为 `float` |
| 校验能力 | 无 magic/version；主要依赖读取顺序和 metadata 合法性 | 有 magic、version、文件长度、CSR 单调性等校验 |
| mmap 支持 | 不支持直接 mmap 成 CSRView；需先逐 chunk 拼 CSR | 支持 `CSRMap/CSRView` 只读内存映射 |
| CUDA 常驻方式 | 初始化时为每张 GPU 构造 compact CSR 并常驻设备 | 通过 generic CUDA sparse backend，支持 chunking、prepared host cache、device chunk cache |
| backward 结构 | 对 compact CSR 调 `CUSPARSE_OPERATION_TRANSPOSE`，再做坐标反变换 | 对显式 CSR 调 `spmvT/spmvTSubset`；CUDA full-subset 路径也以 CSR transpose 为核心 |
| 多算法适配 | 专用 MLEM/MAP-EM 程序，算法和矩阵格式耦合较强 | MLEM/OSEM/BSREM/OSMAPOSL/ADMM 共用同一矩阵接口 |
| 多床位/CTAC | 文件格式未提供当前项目的虚拟床位堆叠和动态 CTAC 接口 | 已支持单床矩阵虚拟堆叠、多床位 projection、动态 CTAC attenuation factors |
| 可维护性 | 适合固定几何专用程序；变更几何/投影顺序风险高 | 适合项目主线、测试、回归、跨算法复用 |

## 对 CUDA 加速的影响

### 旧 compact chunk 的 CUDA 优势

旧格式最强的优势是“对称压缩”。对于 6 模块、45 度对称、Z 对称成立的几何，它只保存 compact 基础矩阵。重建时用 GPU kernel 做图像旋转/翻转、投影行映射、反投影旋回。这样可以：

- 降低落盘矩阵大小；
- 降低初始化后设备端 CSR 的体量；
- 避免把 full-angle 系统矩阵完全展开；
- 把每张 GPU 的 compact angle 范围合并成大 CSR，减少 per-angle 小 SpMV 调度开销。

如果目标是一个固定 6module 几何的独立 CUDA MLEM 程序，且对称规则长期不变，这种格式是有针对性的。

### 旧 compact chunk 的 CUDA 代价

代价是格式和算法强耦合：

- 读取时仍要把 row-wise chunk 拼成 CSR，不能直接作为 `CSRView` 使用；
- full projection 与 compact row 的映射逻辑在 CUDA kernel 中，脱离该重建器不完整；
- projection 排布、symmetry 规则、det-z 镜像一旦变化，矩阵解释也必须同步修改；
- 只能自然服务于该 CUDA MLEM/MAP-EM 路径，不适合作为项目级通用 sparse backend 输入；
- 与当前 SPECT-recon 的 `-loadCSR/-mapCSR`、multi-bed、CTAC、统一 optimizer 不兼容。

### CSRB 的 CUDA 优势

当前 CSRB 的优势不是“最小矩阵体积”，而是“统一、可映射、可缓存、可回归”：

- 单文件可 mmap，避免大矩阵一次性复制到 host vector；
- `io::CSRView` 是当前 CPU/CUDA sparse backend 的共同入口；
- CUDA 后端已围绕 CSRB/CSRView 实现 chunking、prepared host cache、device chunk cache、full-subset fused numerator；
- MLEM、OSEM、BSREM、ADMM 等算法共享同一 forward/backward 接口；
- 多床位虚拟堆叠不复制系统矩阵，只通过 row/col offset 和 attenuation factors 在运行时处理。

### CSRB 的 CUDA 代价

CSRB 是 row-oriented CSR。对 MLEM 来说，forward `A*f` 与 backward `A^T*ratio` 都很频繁；其中 backward 在 CSR 上天然不如 forward 友好。当前项目已有记录表明，CUDA 瓶颈长期容易集中在 `cusparse_bwd` / transpose SpMV 和 chunk 供数上。

但这个问题不是 `.csrb` 扩展名本身造成的，而是 row-major CSR 对 transpose 访问不友好。旧 compact 格式最终在 GPU 上也创建 CSR descriptor，并同样调用 transpose SpMV；它只是通过对称压缩减少了矩阵规模和 full expansion 成本。

## 对 MLEM 迭代重建的影响

MLEM 的基本计算是：

```text
denom = A^T * 1
yhat  = A * f
ratio = projection / max(yhat, eps)
numer = A^T * ratio
f_new = f * numer / denom
```

当前 CSRB 直接表达 `A`，所以 MLEM/OSEM 子集、BSREM numerator、ADMM f-update 都能复用相同结构。`OSEM_step(io::CSRView)` 中，若 sparse backend 可用，会通过 `spmvSubset` 计算 forward，通过 `spmvTSubset` 计算 numerator；full-subset CUDA 路径则进一步把 forward、ratio、backward 融合。

旧 compact chunk 不直接表达 full `A`。它表达的是 `Wc`，完整模型更接近：

```text
Forward : y_compact = Wc * T_sym,z(f)
Ratio   : ratio_compact(row) = P_full(mapped_full_row) / y_compact(row)
Backward: bp += T_sym,z^T * (Wc^T * ratio_compact)
```

这对固定对称几何是高效的，但对通用 MLEM 主线不如显式 CSRB 直观。特别是多床位、CTAC、子集、投影域 PSF、不同优化器共用 numerator 时，CSRB 的显式行语义更容易验证和维护。

## 哪个更适合

### 当前 SPECT-recon 主项目

应选择 CSRB。

理由：

- 当前主流程、CLI、文档、测试、CUDA sparse backend 都以 `io::CSR/CSRView` 为接口。
- `.csrb + -mapCSR on` 已是大矩阵生产路径。
- 多床位虚拟堆叠、动态 CTAC、统一 optimizer、CUDA fused full-subset 都围绕 CSRB 实现。
- 旧 compact chunk 不能直接送入 `-loadCSR`，强行替换会绕开大量已验证逻辑。

### 固定 6module 专用 CUDA MLEM

可以选择旧 compact chunk。

条件是：

- 几何固定为该 6module/24 pinhole/YSO 对称结构；
- 投影排布与 `spect_recon_mlem.cu` 的 full-row mapping 完全一致；
- 主要目标是 MLEM/MAP-EM CUDA 重建；
- 更关心矩阵体积和 compact 对称收益，而不是项目级通用性。

### 如果要把旧生成器接入当前项目

推荐路径有两个：

1. 转换为标准 CSRB。
   - 读取 `metadata.bin + chunk_*.bin`。
   - 根据 symmetry 规则决定是否展开 full rows。
   - 生成 `row_ptr(uint32), col_idx(uint32), val(float)`。
   - 用 `saveCSRBinary()` 写成 `.csrb`。
   - 优点是可直接走当前 `-loadCSR/-mapCSR`。
   - 缺点是如果展开 full symmetry，磁盘和运行时矩阵规模会增大。

2. 新增 compact-symmetric 后端。
   - 保留旧 compact chunk 和 metadata。
   - 在 SPECT-recon 中新增矩阵类型与专用 CUDA projector/backend。
   - 把 symmetry transform、full projection mapping、z mirror mapping 明确封装。
   - 优点是保留压缩收益。
   - 缺点是实现范围大，必须重新覆盖 MLEM/OSEM/BSREM/CTAC/multi-bed 的一致性测试。

不推荐直接让 `-loadCSR` 同时接受 `.csrb` 文件和 chunk 目录。二者语义差异太大，会让入口校验和错误定位变差。

## 最终建议

当前项目继续以 CSRB 为标准系统矩阵格式。旧 6module compact chunk 可作为专用 CUDA MLEM 的参考实现或上游生成格式，但不应成为 SPECT-recon 的默认重建输入格式。

如果下一步目标是“让 6module 生成器服务当前 SPECT-recon”，最小可控方案是先写离线转换器，把旧 chunk 目录转换为 `.csrb`，用现有 `CSRMap/CSRView`、CUDA sparse backend 和 MLEM 回归测试验证数值一致性。只有当转换后的 full CSRB 体积或速度不可接受时，才值得投入 compact-symmetric 专用后端。

## 本次未做的验证

本次是源码级格式和数据流审查，没有运行系统矩阵生成、转换或重建数值对比。原因是用户问题聚焦在格式差异和适用性判断；若后续要决定是否迁移旧生成器，仍需要实际生成一个小矩阵样例，验证：

- chunk 目录能否完整解析；
- 转换后 CSRB 的 rows/cols/nnz 是否符合当前投影数据；
- `A*f` 与旧 compact-batched forward 在同一输入图像下是否一致；
- MLEM 1 iteration 输出是否与旧 CUDA 重建器一致。

## 2026-05-23 接入补充：BIN 投影与 chunk 转 CSRB 工具

本轮已在主工程加入两项最小兼容：

1. `-df/-dfList/-bProj/-bProjList` 现在可直接读取 `6module_bak_src/mat_convert_bin.m` 写出的投影 BIN。BIN 格式为：

```text
int32   num_angles
int32   rays_per_angle
float64 data[num_angles * rays_per_angle]
```

数据顺序与现有 MAT 三维投影重排一致，都是 angle-major，每个角度内按 Z 外层、X 内层展开。需要注意的是，BIN 头只保存 `rays_per_angle`，不保存 `detZ/detX`。因此主流程会优先保留命令行或配置中的 `-detN <detZ,detX>`；只有当 `detZ*detX` 与 `rays_per_angle` 不一致时，才退回 `1 x rays_per_angle` 并给出 warning。若使用投影域 PSF 或任何依赖探测器二维形状的功能，应显式传入正确 `-detN`。

2. 新增工具 `spect_convert_6module_sysmat_to_csrb`，用于把 `metadata.bin + chunk_*.bin` 转为当前项目的 `.csrb`：

```powershell
cmake --build build_cuda_scaffold --target spect_convert_6module_sysmat_to_csrb
.\build_cuda_scaffold\bin\spect_convert_6module_sysmat_to_csrb.exe `
  --matrix-dir ".\matrix_PSC_6modules_24pin_YSO_3X5_PEN_DOI_V2" `
  --out ".\gatedata\matrix_6module.csrb"
```

转换器按三遍流式处理，避免在 36GB 内存机器上把整矩阵展开到内存：

- 第 1 遍：只读每行 `n`，构造 `row_ptr(uint32)` 并累计 `nnz`；
- 第 2 遍：按 chunk/row 顺序写出所有 `col_idx(uint32)`；
- 第 3 遍：按同一顺序写出所有 `val(float)`。

内存占用主要是 `row_ptr`，约为 `(rows + 1) * 4 bytes`，不会持有完整 `col_idx/val`。输出仍受当前 CSRB 结构限制：`row_ptr` 是 `uint32`，因此总 `nnz` 必须不超过 `2^32-1`。

## 2026-07-03 补充：转换工具 vs 260601 chunk CSRB 生产路径

| 路径 | 输入 | 输出语义 | 用途 |
| --- | --- | --- | --- |
| `-chunkCSRDir`（主线） | `metadata.bin` + `chunk_*.csrb` | compact 行 + 运行时对称展开 | **260601 标准格式** |
| `spect_convert_6module_sysmat_to_csrb` | `metadata.bin` + **`chunk_*.bin`**（v2 row-wise） | compact 行 CSRB，**不展开对称** | legacy 离线转换；见 §2a 弃用清单 |

格式 README（`test_data/6module/260601/CompactChunkCSRB_Reconstruction_README_CN.md`）已按代码行为修正：该工具**不接受** `chunk_*.csrb`，**不会**生成 full-row standard CSRB。

**运行时对称验收**：当前仅 SYM_45Z / SYM_90Z 通过 `validateCompactSymmetricGeometry()` 签收；其余推断模式未生产验收。

**metadata.bin 大小**：76B（260525 legacy）或 88B（260601，含 use45/symCode/symmetryOrder）；读取失败或尾部截断将直接报错。

重要限制：该工具做的是 direct compact row-wise chunk 到 CSRB 的格式转换，不做 45/90/Z 对称的 full expansion。如果 `metadata.bin` 显示 `symmetry_order > 1` 或 `use_z_symmetry=1`，输出 CSRB 仍是 compact 行语义。要在标准 SPECT-recon 中直接配合 full-angle 投影 BIN 使用，应重新生成无对称压缩的系统矩阵；否则需要同步提供行数匹配的 compact 投影，或后续实现专用 compact-symmetric 后端。
