# B.4 设计：compact 路径 CTAC（per‑angle 衰减）— 2026‑07‑03

目标：使 `-chunkCSRDir`（CompactChunkCSRB）路径支持 `-ctac`（含 `-ctacAutoMu`），数学语义与标准 CSR 路径的 `computeNormAT_Atten` / `buildAttenuatedContiguousSubsetCSR` 完全一致：

y_i = Σ_j a_ij · att[a(i), j] · f_j，sensitivity_j = Σ_i a_ij · att[a(i), j]

其中 `att[a][j] ∈ (0,1]` 为角度 a 下体素 j 的衰减因子（uint16 量化，scale 1/65535），由 `computeAttenFactors`（line‑integral 或 per‑voxel 模型）在**未旋转的全角度坐标系**下预计算，360 角 × 933 120 体素 × 2 B ≈ 672 MB。

## 1. 对称折叠下的衰减语义（核心推导）

compact 矩阵行 r（compact 角 c）在 branch (symId, zMirror) 下代表 full 行 i，其 full 角为
`a = fullAngleFromCompactAngle(meta, c, symId)`（`src/io/CompactSymmetryMetadata.cpp`）。
矩阵作用于**旋转后**图像：`y_i = Σ_j A[r,j] · rot[j]`，其中 `rot[j] = f[M_b(j)]`（`transformForwardInto` 的 gather 映射 M_b）。

引入衰减后：

y_i = Σ_j a_ij′ att[a,j′] f_j′（j′ 为未旋转体素）
    = Σ_j A[r,j] · att[a, M_b(j)] · f[M_b(j)]

因 T_b 是纯置换（gather），有 `T_b(att_a ∘ f) = T_b(att_a) ∘ T_b(f)`，故：

**前向**：对 branch b、compact 角 c，spmv 输入 = `attRot_b,a[j] · imgRot_b[j]`，其中 `attRot_b,a[j] = att[a][M_b(j)]` 可在核内用现有 device transform params（`makeCompactDeviceTransformParams` 同款坐标映射）对未旋转 att 表做 gather，无需存 8 份旋转副本。

**反向/灵敏度**：`numer_j += Σ_c att[a][M_b(j)] · (A_c^T ratio_c)[j]`（旋转系内先乘 att 再累加，最后每 branch 一次 `transformBackwardAdd`）。数学上与标准路径 `computeSubsetSensitivity` 的 `v *= att[angle][col]` 逐元素一致。

关键点：衰减破坏了"一个 branch 一个输入向量"的批结构——**必须按 compact 角切片**，因为同一 chunk 内不同角的输入图像不同。

## 2. 执行结构（选型：device 端按 branch 烘焙矩阵值）

两个候选方案：

- **方案 A（per‑angle 行切片）**：per (chunk, branch, compact 角) 循环，输入图像逐角乘 att 后做 1152 行的小 spmv。总 nnz 工作量不变，但要重构 chunk 级批结构、引入 720 个切片描述子与 rebase row_ptr。
- **方案 B（值烘焙，选定）**：利用"chunk 行按 compact 角均匀分块（每角 rpa=1152 行）"的结构，在 device 上对每个 (chunk, branch) 做一次值烘焙核：

```
val'[t] = val[t] · att[ a(row(t), symId) ][ M_b(col(t)) ]
```

（一 block 一行，行号→compact 角为整除，角→full 角与列→未旋转体素的映射均复用既有 device 端 `compute_compact_ratio_chunk` / `compact_transform_backward_add` 的坐标逻辑），随后前向 spmv、ratio、反向 spmvT 全部**保持现有整 chunk 批结构不变**，只是 spmv/spmvT 换用烘焙后的 values 指针（row_ptr/col_idx 复用已上传缓存，无额外矩阵流量）。

选 B 的理由：投影器循环结构零重构（`runCompactChunkMlemNumerCUDA` 仅把 `spmvFloatCUDA`/`spmvTFloatCUDA` 换成 bake+baked‑spmv 三个调用）；无 720 描述子/切片管理；同一次 bake 同时服务前向与反向。代价：device 需一块 max‑chunk‑nnz 的 float 烘焙缓冲（~2.8 GB，按需增长、跨 chunk 复用）＋烘焙核 ~2×10¹⁰ 次乘加/迭代（估 1–2 s/迭代，v1 可接受）。

- att 表一次性常驻 device：uint16，360 角 × 933 120 = 672 MB（16 GB 卡可承受），按主机指针+长度作缓存键。
- 灵敏度同路径（unit ratio + 烘焙 spmvT）。
- CTAC 启用时**禁用** `SPECT_COMPACT_DEVICE_RATIO` / device‑backward 融合路径（这些核不含 att 语义），走通用路线；融合核补 att 支持留给后续性能轮（符合"正确性先行"分工）。

## 3. 接口与开关变更

| 位置 | 变更 |
|---|---|
| `ArgumentParser.h` L1661 | compact + `-ctac` 不再拒绝（保留 compact + `-attMap` Chang 拒绝）；单文件 `-loadCSR -compactMeta`（Phase 1 旧入口）维持拒绝 |
| `CompactSymmetricProjectorCUDA.h/.cpp` | `runCompactChunkMlemNumerCUDA` / `computeCompactChunkSensitivityCUDA` / multi‑bed 变体增加可选 `const uint16_t *attFactors, int fullNumAngles` 参数（nullptr = 现行为，二进制兼容） |
| `SparseOpsCUDA.cpp`（或新 `CompactAttenCUDA.cu`） | 新增：att 表常驻上传/释放；gather‑multiply 核（核 1/2，参数复用 `CompactDeviceTransformParams`）；per‑angle 行切片 spmv/spmvT 封装 |
| `MainProgram.cpp` compact 分支 | `-ctac` 时把 `opts.bedAttenFactors[bed]` 传入 compact runner；`compactDenomCache` manifest 加入 CTAC 模型/att 表哈希（避免误命中无衰减缓存） |

多床：per‑bed att 表逐床传入（与现 multi‑bed 循环结构一致）。

## 4. 验证计划

1. **零 mu 等价回归（gtest）**：合成小型 chunk 目录（复用 `test_cuda_scaffold` 的 toy compact 构造），att 全 1（mu=0）时 numer/denom 与不传 att 的现路径 bitwise/1e‑6 内一致。
2. **对称一致性（gtest）**：同一 toy 矩阵展开为 full CSR，用 `computeSubsetSensitivity`（带 att）作参考，compact+att 结果相对误差 <1e‑5。
3. **真实数据 smoke**：`test_data/share` 260601 + `-ctac -ctacAutoMu`，确认能跑通、无 NaN、sensitivity 统计合理；与标准路径同参数结果做差分（预期 float 量化差异量级）。
4. 现有 234+ 项全量单测无回归。

## 5. 已知边界

- Sym45Z 的无效角（`fullAngleFromCompactAngle` valid=false）在 per‑angle 循环直接跳过——与 unit‑ratio 现语义一致。
- att 量化（uint16）与标准路径一致，无额外精度损失。
- PSF：compact 路径 PSF 在图像域、att 在投影生成前的图像域逐点乘——顺序为先 PSF 模糊再乘 att（与标准路径 `OSEM_step` 中 blur→attenuated spmv 顺序一致）。

## 6. 实现状态（2026-07-03，已完成）

实现落点与设计一致，差异与验证结果如下：

- **核**：`compact_bake_atten_values`（NVRTC，`src/cuda/SparseOpsCUDA.cpp`），一 block 一 compact 行；Sym45Z 无效角行改为**复制原值**（其 ratio 恒 0，前向输出被忽略，语义等价于跳过）。
- **API**：`compactAttenSpmvFloatCUDA` / `compactAttenSpmvTFloatCUDA`（`SparseOpsBackend.h`），参数打包为 `CompactAttenBakeParams`；三个 compact runner 的 `attFactors` 追加为末位默认参数。
- **缓存**：att 表按主机指针+长度常驻 device；烘焙值按 (matrix key + device row/col 指针 + branch + rowBase + 表指针) 单槽缓存，同一 (chunk, branch) 的前向/反向共享一次烘焙。
- **CLI**：`-chunkCSRDir + -ctac` 放开，限单床（设备 att 表缓存单槽，多床会逐分支重传 672 MB 表）；单文件 compact 入口维持拒绝。denom cache manifest 追加 `ctac_model/ctac_rot_sign/mu-map 文件身份`（仅 CTAC 激活时写入，不影响既有缓存命中）。
- **验证**：
  1. gtest `CompactAttenBakedSpmvMatchesCpuWhenCudaAvailable`：4 个 Sym90 分支 vs CPU 参考（复用 `mapSym90ZVoxelCoord`）一致，单位表还原 plain SpMV；scaffold 26/26、主套件 241/241 通过。
  2. share 260601 零 mu 回归：recon_it3 vs 无 CTAC 基线 rel_l2=4.7e-8（uint16 表=65535 时烘焙为精确乘 1）。
  3. 水圆柱 mu（r=12 mm，0.0152 mm^-1）：sensitivity 68.99M→62.79M（-9%），中心/环强度比 1.121→1.288，衰减补偿方向正确。
- **性能现状**：CTAC 开启时 forward+backward 由 ~1.0 s/iter 增至 ~11 s/iter（烘焙核 + 融合路径禁用），符合"正确性先行"预期；融合核补 att 语义留给性能轮。
