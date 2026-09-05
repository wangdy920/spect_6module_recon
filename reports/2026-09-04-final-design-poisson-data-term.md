# 最终设计：Poisson 数据项引擎 + 分离的优化器/先验/物理修正（2026-09-04）

目标支持范围：`MLEM` 与 `BSREM-II + RDP`，subset = 1，CTAC，Lu-177 双窗（113/208 keV）联合重建，多床。
约束：**CSRB 是生产路径与受保护基线，CSRQ 是实验路径**（用户决定 2026-09-03）。任何一步都必须让 CSRB Derenzo MLEM 回归（`recon_out/u16sqrt_value_format_20260902/derenzo_fp32/recon_it50`，relL2 门限 1e-6，无背景/无 CTAC 时期望位一致）继续通过。

本文在 GPT 5.6 PRO 方案（用户 2026-09-04 提供）的基础上，用代码事实做了修正。第 1 节是采纳/修正清单，第 2–6 节保留设计和原验收要求；实际实现与验证状态见第 7 节，较晚的阶段记录覆盖较早的实现状态描述，原验收门不因此自动关闭。

## 1. 对 GPT 方案的采纳与修正

| GPT 建议 | 结论 | 依据（代码事实） |
|---|---|---|
| 统一"Poisson 数据项"(q, S)，优化器/先验从投影链拆出 | **采纳** | `runCompactChunkMlemNumerMultiBedCUDA` 对 MLEM/BSREM 已是同一调用，只差命名与 `residentMlemEligible` 把 BSREM 踢出常驻循环 |
| 对外两种模式 `MLEM` / `BSREM-II + RDP`，不叫 "MLEM+RDP" | **采纳** | 现有 `-opti mlem|bsrem` 与 `applyBsremUpdate` 已是 BSREM-II 相对更新（κ 图、曲率、clip[-0.9,1]、sMin） |
| channel 是测量通道不是 subset；所有窗/chunk 累加后只更新一次 | **采纳（已实现）** | `runCsrqMlemNumerMultiBed` 各 channel 顺序写同一设备累加器，最后统一更新 |
| `pack_branch8` 提到 channel/chunk 循环外 | **采纳，收益小** | 当前每 (chunk, bed) 打包一次，每次 ~0.4 ms，每迭代 ≤ 10 ms；无 CTAC 时提到 per-bed 一次即可。**有 CTAC 时打包必须按 (channel, chunk 角度组) 进行**（见 §4.2），不能只做一次 |
| CSRQ `forward + ratio` 融合成模板核 `<HasBackground, HasMask, HasRowNormalization, HasAttenuation>` | **部分采纳** | 融合 forward+ratio 采纳（warp 算完整行后直接出 ratio）；**不采纳四布尔模板**：背景/行权重用逐行空指针判断（不是逐 nnz），CTAC 放在 pack/收口阶段，SpMV 核只保留一个实例 |
| CSRB 背景并回 device branch-batch | **采纳（P0 首项）** | `compute_compact_ratio_chunk(_batch)` 加 b 即可删掉 host-ratio 分支 |
| RDP 梯度 + BSREM 更新 + 正值约束融合为一个 stencil 核，xCurrent/xNext 双缓冲 | **采纳** | 但公式以 `applyBsremUpdate` 为准（κ、曲率、clip、sMin、mlemEquivalenceMode），CPU 版保留为 oracle；设备图像/更新用 fp64 以保持与主机位级一致 |
| cache miss 时双 RHS 反投（q 与 S 同遍） | **不采纳** | S 已缓存；带 CTAC 时缓存被关闭是因为 manifest 缺表身份——修 manifest（加表哈希）比加双 RHS 核便宜且彻底 |
| 113/208 共享稀疏结构的双矩阵核 | **不适用** | 两矩阵 nnz 不同（CSRQ 6295 vs 9191 MiB），结构指纹必然不一致 |
| CTAC 需要 CSRQ 保存传播路径几何 / sidecar / on-the-fly 线积分 | **不需要** | 现有 CTAC 模型是 **`att[fullAngle, voxel]` u16 表**（`CompactAttenBakeParams`：`effective = val · att[fullAngle(row), unrotatedVoxel(col)]`），因子只依赖 (全角度, 未旋转体素)，可以精确地从 nnz 循环里因子分解出来（§4.2）。CSRQ 的 row/col/branch 已足以定位它 |
| 不把 8 个分支预展开成患者相关矩阵 | **采纳** | 同上：按分支用 `att[fullAngle(θ_c, k), ·]` 乘在图像域，μ 不需要任何对称性 |
| CSRB 与 CSRQ 核不合并，上层接口合并 | **采纳（已实现一半）** | `CompactMatrixChannel` + `isCsrqChannelSet` 分派已存在 |
| forward/backward mega-kernel | **不做** | 同 GPT 判断 |

新增的、GPT 未覆盖的事实：
- 行 mask / 归一化目前通过 `S = Aᵀ((A1·w)/A1)` 借分子路径合成（`CompactSymmetricReconLoop.cpp` ~L814–934），测量值在 MainProgram 置零。统一 ratio 核加可选行权重 w 后可直接算 `S = Hᵀw`，w≡1 时保持位一致。
- CSRB 单床 CTAC 走最慢的通用路径（`compactAttenSpmvFloatCUDA` + CPU ratio + CPU `transformBackwardAdd`），多床无背景才有 `compactDeviceRatioAttenSpmvTAccumulateFloatCUDA`；有背景 + CTAC 全部回 CPU。目标配置恰好落在这里。
- CTAC 表每床每能量 360 × 3 168 000 × 2 B = **2.28 GB**，按床缓存在设备；双窗需两张（μ 随能量变化）。
- 单床时 `residentBatchInput` 要求 `bedCount > 1`，所以单床 MLEM 也走非常驻路径（每迭代 CPU 旋转 8 分支 + 上传 101 MB）；CSRQ 单床 0.25 s/iter 中约 0.05 s 是这部分。

## 2. 测量算子与充分统计量（映射到现有实现）

第 k 个能窗、第 b 个床的测量算子写成

```
H_k = W_k · s_k · A_k(att_k)          期望  p̄_k = H_k x + b_k
```

| 符号 | 含义 | 现有实现位置 |
|---|---|---|
| `A_k` | 该能量的几何系统矩阵（K=8 对称压缩，CSRB 或 CSRQ） | `CompactChunkCSRDirectory` |
| `att_k` | `att_k[fullAngle, voxel]` u16 表（1/65535），未旋转帧，按床一张 | `opts.bedAttenFactors`、`compactAttFactorPtrs` |
| `s_k` | 窗计数尺度（能窗权重） | `CompactMatrixChannel::weight`，`-energyWeights` |
| `W_k` | 行权重：坏探测器/detZ 边缘 censoring 置 0，归一化效率 1/c | `detZSensWeights`（现只进分母） |
| `b_k` | 加性背景（TEW 散射等） | `-bProj/-bProj2` |

每迭代只算两组量（S 只算一次并缓存）：

```
q = Σ_k H_kᵀ ( p_k / (H_k x + b_k) )          反投比值
S = Σ_k H_kᵀ W_k 1                           灵敏度
g_data = q − S                               Poisson 对数似然梯度
```

无背景时 s_k 在 q 与 S 中同时出现并在 MLEM 比值里抵消；有背景时 s_k 决定 b_k 的相对尺度，不能省。

subset = 1 的一次迭代严格是：

```
for bed:  pack xCurrent（无 CTAC：每床一次；有 CTAC：每 (channel, chunk 角度组) 一次）
clear q
for channel k:  for chunk:  forward → ratio → backward-add → q      （流式矩阵每 chunk 每迭代过 PCIe 一次）
one image update:  MLEM  或  BSREM-II + RDP
swap(xCurrent, xNext)
```

113 keV 与 208 keV 不是两个 subset，6 个 chunk 也不是；RDP 每迭代恰好作用一次。当前 CSRQ 已是这个顺序，保持。

## 3. 分层与接口

```
PreparedReconPlan
├─ channels[k]: PreparedChannel
│    { matrix backend (CSRB|CSRQ, device slot / residency), projections[bed], background[bed] | null,
│      weight s_k, rowWeights W_k | null, attTable[bed] | null (u16, 能窗共享), attenPower, attResidency }
├─ sensitivity S（缓存 manifest = 目录身份 + dual_window + weights + W_k 哈希 + attTable 哈希 + 版本）
├─ optimizer:  MLEM | BSREM-II（α 调度、κ 图、曲率、clip、sMin —— 全部沿用 ReconArgs 现有字段）
└─ penalty:    None | RDP（β, γ, 邻域权重, mask）
```

投影层对外只有三个入口（channel 形式），命名去掉 MLEM 语义；单目录旧名保留为薄包装，CSRB 调用方不改：

```cpp
bool accumulatePoissonBackprojectedRatioCUDA(channels, meta, projections, backgrounds, images|null, bedCount, …);  // q  （现 runCompactChunkMlemNumerMultiBedCUDA）
bool computePoissonSensitivityCUDA(channels, meta, rowWeights|null, …);                                             // S  （现 computeCompactChunkSensitivityCUDA）
bool prepareChannelExecutionCUDA(channels);                                                                          //    （现 prepareCompactChunkExecutionCUDA）
```

图像更新层的原设计（设备，fp64 图像，xCurrent 只读 / xNext 只写；P1 实际采用独立梯度缓冲后原位更新，见第 7 节）：

```cpp
launchMlemUpdate(xCur, q, S, mask, nonneg, xNext);                         // 现 compact_mlem_update，语义不变
launchBsremRdpUpdate(xCur, q, S, mask, rdpParams, bsremParams, xNext);     // 新：∇R stencil + κ/曲率 + 相对步长 + clip + 非负，一核完成
```

两种更新是两个核、由 plan 在迭代前选定，不在核内按算法分支；`residentMlemEligible` 中"算法 == MLEM"这一条删除，butter/smooth 输出滤波与 debugStats 的例外保留。

分派点只有两个（与现在相同）：`CompactChunkCSRDirectory::open()` 决定格式；`isCsrqChannelSet(channels)` 在三个入口选后端。CSRB 后端与 CSRQ 后端各自实现一遍下面 §4 的语义，核不共享。

## 4. 核级设计

### 4.1 一种 ratio 语义，两处实现

```
r_i = p_i / (s·ŷ_i + b_i)              b 为可选指针，按行判空（每行一次，不在 nnz 循环内）
unit:  r_i = W_i                        用于 S = Hᵀ W 1
```

（执行时修正：W 只进入 S，不乘进 q 的 ratio——这是现有生产语义，坏探测器/censoring 的测量值在 MainProgram 已置零，归一化效率是 denominator-only；P0 保持该语义以便回归位级可比。）

- CSRQ：把 ratio 并进 `csrq_forward_k8` 的行尾（warp 归约出 8 个分支的 ŷ 后直接写 r），`csrq_ratio_k8` / `csrq_unit_ratio_k8` 退化为同一核的 `mode` 参数；需要输出前向投影（`computeCompactChunkForwardProjectionCUDA`、残差 CSV）时走 `storePrediction` 实例。收益主要是删掉一条调用链，不是速度（ratio 现总计 0.012 s / 50 迭代）。
- CSRB：`compute_compact_ratio_chunk` 与 `_batch` 版加 `background` 与 `rowWeights` 指针。之后 `useDeviceRatio = !hasBackground` 恒为真，`makeProjectionRatioChunkFloat` 及整条 host-ratio 分支删除；`spmvFloatCUDA → 下载 → CPU → 上传` 不再出现在 numer 路径。
- 分母：`S = Hᵀ W 1` 直接用 unit 模式 + 行权重，替代 `Aᵀ((A1·w)/A1)` 合成；`W ≡ 1` 时结果位一致，作为回归门。

### 4.2 CTAC：从 nnz 循环因子分解出来

现有模型 `h_ij = a_ij · att[θ(i), j]`（θ 为全角度，j 为未旋转体素）。对压缩角 θ_c 的分支 k，全角度 θ = fullAngle(θ_c, k)，且旋转图像满足 `x_k[j'] = x[unrotate_k(j')]`，于是

```
前向：  (H x)_i = Σ_j a_ij · (x ⊙ att[θ, ·])[unrotate_k(j)]      → 在 pack 阶段先乘再旋转，SpMV 核不变
反投：  (Hᵀ r)_j = att[θ, j] · Σ_{i∈θ} a_ij r_i                   → 按角度分开累加，收口时乘 att 再 transform-add
```

两侧用同一张表、同一索引，伴随关系精确成立（伴随测试为验收项）。μ 不需要任何对称性——每个分支查的是自己那个全角度的表。CSRQ 与 CSRB 两个后端都按此实现；`compact_bake_atten_values`、`compact_atten_fused_spmv/spmvt` 与 `compactDeviceRatioAttenSpmvTAccumulateFloatCUDA` 在 CSRB 侧被替换后删除。

实现形态（角度组大小 G 为 plan 参数）：

| 步骤 | 新核 | 暂存（G = 8，nvox = 3.168M） |
|---|---|---|
| pack | `pack_branch8_atten(x, attTiles, θ_c 组, k) → x̃[g][nvox][8]` | 8 × 101 MB = 811 MB |
| forward | 现核，x 基址按行所属角度 g 选 | — |
| backward | 现核，累加目标 `y[g][8][nvox]` | 811 MB |
| 收口 | `atten_transform_backward_add(y, attTiles) → acc` | — |
| att tiles | 每 chunk 需要 G×8 个全角度切片 × 6.3 MB | 405 MB / chunk / 能窗 |

原预算以 G = 8、显存紧张时 G = 4 估算；这些分组未实施。实际取 G = 1，暂存约 101 MB，每次 compact 角度 1152 行，以维持反投的 L2 驻留范围，见第 7 节。
表的驻留与矩阵一样由 plan 决定：常驻（2.28 GB / 床 / 能窗）或从 pinned 主机按 chunk 流式（每迭代每床每能窗整表过一次 PCIe，2.28 GB ≈ 0.14 s，可与计算重叠）。

### 4.3 图像更新核

- `compact_mlem_update`：保持不变（生产基线）。
- `bsrem_rdp_update`（新，一体素一线程，gather stencil，无原子）：
  1. 读 x_j 与 6/18/26 邻域（fp64），算 RDP 的 ∂R/∂x_j 及（`bsremUseCurvDenom` 时）曲率项，语义与 `tryComputeRegGradientCUDA` 一致；
  2. `λ_j = β·κ_j`（κ 图可空）；`u = α·(q_j − S_j − λ_j·r_j) / (S_j + λ_j·max(c_j,0) + ε)`；
  3. `ū = clip(u, −0.9, 1.0)`；`x_next = x·(1+ū)`；`S_j ≤ sMin` 置零；mask 外不动；非负夹紧；
  4. `mlemEquivalenceMode` 作为编译期/启动期开关复现主机语义。
  与 `applyBsremUpdate` 的差别只允许来自浮点求和顺序；CPU 版保留为 oracle，测试逐迭代比对。
- 双缓冲：xCurrent/xNext 各 25 MB（fp64），迭代末交换指针；sieve 保存点从 xCurrent 生成 PSF 卷积副本，逻辑不变。
- 单床也进入常驻循环：去掉 `residentBatchInput` 的 `bedCount > 1` 条件（旋转/打包本来就在设备上），CPU 侧只剩保存点与日志。

### 4.4 明确不做

- CSRB 与 CSRQ 核合并；forward/backward 大核；双矩阵共享结构核；双 RHS 反投。
- 逐 nnz 的运行时 `if (ctac)` / `if (background)`：前者不存在（CTAC 在 pack/收口），后者按行判一次。
- `CudaSparseOpsState.h` ~L3539–4424 的全矩阵/子集路径：无引用，subset=1 定案后删除（独立提交，与 CSRB compact 路径无关）。

## 5. 预算（RTX 4070 Ti SUPER / RTX 5060 Ti，16 GB，单床，CSRQ 双窗 + CTAC）

| 配置 | 设备显存 | 每迭代 PCIe | 预计 s/iter（50 迭代） |
|---|---|---|---|
| 两矩阵流式，表流式，G = 8 | 5.2 流槽 + 0.6 暂存 + 1.6 CTAC 暂存 + 0.8 表切片 ≈ **8.2 GB** | 15.5 GB 矩阵 + 4.6 GB 表 | 0.83（实测无 CTAC）→ ~1.0–1.1（≈ 55 s） |
| 两矩阵流式，两张表常驻 | 8.2 − 0.8 + 4.6 ≈ **12.0 GB** | 15.5 GB | ~0.95 |
| 208 常驻 + 113 流式，表流式，G = 4 | 9.2 + 2.1 + 0.6 + 0.8 + 0.8 ≈ **13.5 GB**（+2 GiB 保留） | 6.3 + 4.6 GB | ~0.75；需 GPU 独占 |
| 多床（5 床） | 同上（表按床流式；打包/累加暂存复用） | × 5 | 参考 Tc99m 5 床 1.13 s/iter 线性外推 |

`planCsrqStreaming` 需扩展为同时规划矩阵与表的驻留，并修正 WDDM 下 `cudaMemGetInfo` 偏乐观的问题（2026-09-04 复现：另一进程占 3.2 GB 时报 15.1 GB 空闲）：放不下时把常驻矩阵降级为流式重试，而不是 FATAL。

## 6. 验收门

| 项 | 方法 | 门限 |
|---|---|---|
| CSRB 生产基线 | Derenzo MLEM 50，同一命令，vs `derenzo_fp32/recon_it50` | 无背景/无 CTAC：位一致或 relL2 ≤ 1e-7（运行噪声 5e-8）；每个 P 步后跑 |
| CSRB 背景并回设备 | `csrb_lu113_bg_it5` 重跑 vs 现输出 | relL2 ≤ 1e-6 |
| ratio 行权重替代 A1 合成 | 有坏探测器/detZ censoring 的分母 vs 现缓存 | relL2 ≤ 1e-6（W≡1 位一致） |
| CTAC 伴随性 | toy 矩阵 + 随机 att 表：`<Hx, y> = <x, Hᵀy>`，CSRB/CSRQ 两后端 | 相对误差 ≤ 1e-5（fp32） |
| CTAC 数值 | 真实矩阵 + 真实 μ 表，CSRQ 新路径 vs CSRB 现有 CTAC 路径（作为参考实现保留到此项通过） | it20 relL2 ≤ 1e-5 |
| BSREM-II + RDP 设备核 | 每迭代 vs `applyBsremUpdate` + `tryComputeRegGradientCUDA` 主机 oracle（fp64） | relL2 ≤ 1e-12 / 迭代 |
| RDP 梯度 | 有限差分 vs 解析梯度（小体积） | 相对误差 ≤ 1e-6 |
| 双窗 penalty 只作用一次 | 两 channel 同一目录、权重 0.5/0.5 的 BSREM vs 单 channel | relL2 ≤ 1e-6（现有 `derenzo_dual_same` 的 BSREM 版） |
| 显存/时间 | 2 s 采样 + `[PERF]`，按 §5 表核对 | 不超预算 |

## 7. 执行顺序与改动范围

**P0 — 架构解耦，结果不变**（CSRB 回归位一致）
1. channel 形式入口改名（`accumulatePoissonBackprojectedRatioCUDA` / `computePoissonSensitivityCUDA` / `prepareChannelExecutionCUDA`），旧名薄包装；`PreparedChannel` 扩展 background/rowWeights/attTable 字段。文件：`CompactSymmetricProjector.h`、`CompactSymmetricProjectorCUDA.cpp`、`CompactSymmetricReconLoop.cpp`。
2. CSRB ratio 核加 background + rowWeights，删 host-ratio 分支与 `makeProjectionRatioChunkFloat`；CSRQ forward+ratio 融合。文件：`projdomain_chunk_ops.cu`、`CudaProjKernelLauncher.*`、`CudaSparseOpsState.h`。
3. 分母改 unit 模式 + 行权重；manifest 加 W_k 与 att 表哈希（带 CTAC 也能缓存）。
4. 删除无引用的全矩阵/子集路径。

**P1 — 优化器上设备**（已完成，提交 `80aabd2`）
5. ~~xCurrent/xNext 双缓冲~~ → 执行时改为两核方案：先用现有 RDP/TV 梯度核（保留 CT 引导权重逻辑）从设备图像算出 ∇R 到独立缓冲，再逐体素原位更新（`compact_bsrem_update`，逐项复刻 `applyBsremUpdate`），不需要 ping-pong；`residentUpdateEligible` 去掉算法条件，`residentBatchInput` 去掉 `bedCount > 1` 与背景项限制。
6. 测试：BSREM oracle（单元 + Derenzo it1–5 relL2 ≤ 1.1e-8）。现有 `RegularizerTest` 含 RDP 有限差分检查，但不是第 6 节全部条件下的 1e-6 证明；双窗 BSREM penalty-once 专项仍未关闭。**发现：** `algo_bsrem.conf`（δ=0.01）的迭代对 fp32 累加噪声每迭代放大 2–3×，同一二进制两次运行 it20 相差 ~1e-3。早期迭代是当前已测证据，不能替代原 1e-12 门或长迭代稳定性验收。

**P2 — CTAC 因子分解**（CSRQ 侧已完成，2026-09-04）
7. ~~`pack_branch8_atten`、按角度组的 x̃/y 布局、`atten_transform_backward_add`~~ → 执行时取 **G = 1**（按 compact 角度逐个处理）：`csrq_pack_branch8_atten` 在 pack 时乘 `att[fullAngle(θ_c,k), M_k(v)]`，`compact_transform_backward_add_batch_atten` 在收口时乘同一因子；forward/backward 核只是按行子区间（1152 行）启动，核本体不变。选 G = 1 的原因：backward 的两分支 L2 驻留论证要求 y 目标 ≤ 25 MB，G > 1 会把它放大 G 倍；pack / 收口的总量与 G 无关；启动数 720/迭代/床可忽略。表驻留：每 (chunk, 床) 只暂存该 chunk 触及的 60 个全角度切片（≈0.38 GB，同步 `cudaMemcpy`，每床每迭代整表过一次 PCIe ≈ 0.25 s），未进 plan；与计算重叠留给 P3。**CSRB 侧未改**：CSRB 是受保护生产路径，其现有 CTAC 核作为参考实现保留（§6 门已用它通过）；是否用因子分解形式替换（预计 44 s/iter → ~3 s/iter，两床）留作用户决定。真实 μ 表数据：MDP 小鼠 2h（Tc-99m，CT/SPECT DICOM `dcmdata/mouse_samples/20260729_SPECT_mouse_MDP/20260729_162015`，投影 `E:\...\mdp_seam_ab_20260730\preproc_2h`），`prepare_ct_mu_map.py --profile recon220 --beds 2,3 --z-bed-flip`；Lu-177 在本机只有 CT 原始投影（`data_platform1.mhd` 是 1536×1536×720 的投影栈，不是重建体），等厂商 CT DICOM。
8. ~~伴随测试 + CSRQ-vs-CSRB CTAC 数值门~~ 通过：单元测试（随机 24 角矩阵、2 chunk、两张逐床表）CSRQ vs CSRB 全部 ≤ 1e-5、伴随 ≤ 1e-5；真实数据 it20 relL2 1.4e-7（分母 1.1e-7，零模式一致），BSREM-II+RDP it1–5 ≤ 1.7e-7；两床 1.02 s/iter（核 0.48 s + 表上传 0.5 s）vs CSRB CTAC 44 s/iter。

**P3 — 性能收尾**（2026-09-04 完成本机可做的部分）
9. ~~WDDM 规划修正与降级重试~~ 做法：不改估计（WDDM 下 `cudaMemGetInfo` 没有可靠替代，NVML 另算依赖），而是把"放不下"变成可恢复错误 `CsrqDeviceMemoryError`：`warmCompactChunks` 捕获后把本次调用已常驻的矩阵降级为流式（清 lease、重读进 pinned 主机）再重试失败的矩阵。用 4 GiB 的 torch 进程复现今早的场景：208 常驻 → 113 "even streamed" 放不下（空闲 1197 MiB）→ 208 降级 → 113 常驻 → 跑完（原来 FATAL）。~~表流式与计算重叠~~ 做法：`cudaHostRegister` 页锁定调用方的逐床表（2.2 GB 用 0.08 s，失败则退回同步拷贝），切片经上传流进两个设备槽，与上一 (chunk, 床) 的核用 ready/free 事件双缓冲；逐矩阵的切片索引计划只建一次，遍内无同步拷贝。两床 CTAC 1.02 → **0.60 s/iter**，数值不变。多床时打包/累加暂存本来就共享（`dCsrqX_/dCsrqY_/dCsrqRatio_` 与两个切片槽），无需再改。**未做**：5060 Ti 上 `red.global.v4.f32` 向量原子（需要现场 GPU）；CTAC 切片槽（2 × 0.38 GB）未计入 `beginCsrqWarmup` 的 scratch 估计，现由 2 GiB 保留吸收。

每个 P 步单独提交、单独跑 §6 对应门；P0 完成前不开始 P1，P2 依赖 P0 的 ratio/行权重契约。

## 7.1 后续（2026-09-04 晚，用户："1、2、3 都可继续"）

- **双窗 CTAC（原开放项 2）— 完成。** 不做逐光电峰 μ 表：现有两条 μ 图分支（auto-mu 水膜、`prepare_ct_mu_map.py` 的 HU 转换）都是 μ = μ_water(E)·body 的单材料模型，能量只以标量进入，故 att(E) = att(E_map)^(μ_w(E)/μ_w(E_map))。一张逐床表服务所有能窗（5 床双窗省下 22.8 GB 主机表与第二次线积分），指数在 `csrq_pack_branch8_atten` / `compact_transform_backward_add_batch_atten` 内用 `__powf` 施加（指数为 1 时跳过）。NIST XCOM 液态水表进 `core/AttenuationCoefficients.h`（核素预设本就是它的插值）。CLI `-ctacWindowEnergies`、`-ctacMapEnergy`；分母缓存与 run manifest 记录指数。局限：骨按水缩放；精确到骨需第二张"骨过量"线积分表（att = exp(−a_k L_w − b_k L_b)），主机内存与两套表相同，只在 ≥3 窗时才划算，未做。
- **CSRB 侧移植（原开放项 1）— 完成。** §4.2 在 CSRB 上的形态：分支批（`[branch][voxel]`，cuSPARSE 的列主 B）用 `compact_branch_batch_atten_scale` 乘因子；每个 compact 角度的行作为子 CSR 处理——自定义 packed-K 行核直接用 chunk 数组 + 行指针偏移（绝对 nnz 下标，保持 16 B 对齐；cuSPARSE CSR_ALG2 在 1152 行 × 30k nnz 的子矩阵上要 20 ms/次，行核 0.3 ms），转置用逐 RHS cuSPARSE 在零基行指针副本上；收口复用 `compact_transform_backward_add_batch_atten`。删除 fused/baked 核、`CompactAttenBakeParams`、逐床驻留表、`SPECT_COMPACT_CTAC_BAKED_REFERENCE`，以及 CudaChunkCache 自引用的 prepared/attenuated 缓存（净 −550 行）。门：单元测试改为两格式各对独立主机 oracle；MDP 两床 vs 旧 fused 路径 it20 relL2 1.2e-7；CSRB 守护 1.15e-7；两床 44 → 2.35 s/iter。§6 表中"CSRB 现有 CTAC 路径作为参考实现保留到此项通过"至此闭合。
- **sm_120 向量原子（原开放项 3）— 代码就位，性能待现场。** `csrq_backward_k8_packed`：一遍矩阵把 8 个分支全部散射进 `[voxel][8]`，每 nnz 两个 `float4` 原子（`csrq_atomic_add_float4`：`__CUDA_ARCH__ >= 900` 用 `atomicAdd(float4*)`，SASS 确认 sm_120 为 `REDG.E.ADD.F32x4`，sm_86/89 退化为 4 个标量 `RED.E.ADD.F32`）；消费者 `compact_transform_backward_add_packed8`（可带 CTAC 因子与能量指数）。开关 `SPECT_CSRQ_BACKWARD_PACKED=1` / `setCsrqPackedBackwardCUDA`，默认关。本机（sm_89）数值与默认分支主序路径一致（单元测试两条路径、Derenzo it10 relL2 4.3e-8），但标量退化路径每遍 730 ms vs 175 ms（与此前 715 ms 的 `[nvox][8]` 结论一致）。在 RTX 5060 Ti 上要比的是：一遍 × 2 个 16 B 向量原子 vs 四遍 × 2 个标量原子。

## 7.2 2026-09-05 收敛与 release 准备

契约：以 `a3239d7` 为本轮源码基线，保留上述数学模型、默认配置、CSRB 生产路径、CSRQ 实验路径和默认关闭的 sm_120 trial。只删除已拒变体和无调用代码、修正文档，不改变精度、迭代数或图像处理顺序。可用端点为现有 Tc99m Derenzo MLEM it50、MDP 两床真实 μ CTAC it20，以及 CSRQ packed it10。清理前后同输入 Derenzo relL2 门为 1e-7；历史 Derenzo 顶层守护门 1e-6、CTAC 门 1e-5。性能只记录，不据单次测量宣称提速。

- **明确收敛项**：8 月 Phase-B 已拒的 `SPECT_COMPACT_T_SPMM` 全部变体仍在源码中。现删除开关/解析器、SpMM/preprocess/CSC/真转置旁路、真转置的主机缓存与 pin、设备半块槽、上传流、warmup API，以及无调用的 prepared-host 字节估算。保留默认 packed-K 正向与 CSR_ALG2 反投；独立 `transpose_bench` 和历史报告作为复现证据保留，旧实验代码定位到 `a3239d7`。
- **资源与调用复查**：CTAC 的两个切片槽仍包含在 2 GiB reserve 内，没有新增一套驻留规划；共享表与 ready/free 事件路径保持不变。没有本轮实测支持的新内核优化，不扩大架构改动。
- **正式性能门缺口**：本机当前缺失 `test_data/6module/Sampling_Approach`；现有 120×120×220 矩阵不替代它。126.261 s 是此前保护记录，本轮不更新 120 s/180 s 判定。现场 RTX 5060 Ti packed 性能和 CSRQ 生产采纳仍未关闭。
- **验证证据**：`recon_out/release_convergence_20260905/` 保存构建/CTest 日志、冻结命令、可执行文件及输入哈希、逐项比较结果；`run_validation.py` 重放已有日志中的完整命令，仅替换 exe 与输出目录，独立设置 packed trial 环境变量。

清理净减少 629 行源码/头文件/实验驱动脚本，`CudaSparseOpsState.h` 从 4514 行降到 3941 行；没有修改 CUDA kernel 源码和默认配置。删除过程一次构建发现误删了相邻 `stageInputH2D`，已恢复该函数及原 `private` 边界，之后全量构建与测试通过。

| 验证 | 本轮结果 | 证据 |
|---|---|---|
| 清理前、清理后全量构建 + CTest | 均 283 通过、4 跳过、0 失败；原有 LNK4098 链接告警仍在 | `build_baseline.log`、`build_candidate_fixed.log`、`ctest_baseline.log`、`ctest_candidate.log` |
| CSRB Derenzo it50 | 对本轮 a3239d7 基线 relL2 4.9903e-8；对历史 P1 输出 4.9580e-8 | `candidate/csrb_mlem50/validation.json` |
| CSRB CTAC 两床 it20 | 对 09-04 因子分解输出 relL2 5.1716e-8；48.613 s / 20 it | `candidate/csrb_ctac20/validation.json` |
| CSRQ CTAC 两床 it20 | 对 09-04 P3 输出 relL2 4.5466e-8；11.953 s / 20 it | `candidate/csrq_ctac20/validation.json` |
| CSRQ packed it10 | 对 09-04 packed 输出 relL2 4.3319e-8（仅 sm_89 数值守护） | `candidate/csrq_packed10/validation.json` |

上述相对路径均在 `recon_out/release_convergence_20260905/` 下。4 项跳过分别是需要显式启用的五数据集输出过滤测试、缺少本地 `260601` 的 3 项 IO 测试；不是运行失败。Derenzo 单次核心时间清理前 83.030 s、清理后 70.986 s，上传时间也从 53.546 s 变为 42.316 s；此次删除默认未启用的分支，不将该波动解释为算法或内核加速。

本地候选包固定位置为 `dist/release_candidates/SPECT_recon_rc_20260905_convergence/` 和同名 zip；以 `bin/SPECT_recon.exe --version`、`SHA256SUMS.txt` 及 `recon_out/release_convergence_20260905/release_verification.json` 绑定干净提交、包内文件与最终可执行文件 smoke。发行说明 `multipinhole/doc/RELEASE_20260905_CN.md` 随包保存未关闭的门；本轮不作 CSRQ 生产晋升或现场性能承诺。

## 7.3 2026-09-05 第二轮：残留缓存与主机缓冲收敛

契约：以已验证 RC `e5d376a` 为源码对照，沿现有 CLI → 准备 → 投影／更新 → 输出链审查；允许删除无调用代码、合并相同数据的生命周期，不改变数学核、精度、默认配置或 CSRB／CSRQ 地位。MLEM 同输入 relL2 ≤ 1e-7；BSREM 前五次结构回归 ≤ 1e-6，明确不替代 §6 的 1e-12 门；CTAC 沿用 ≤ 1e-5。完成同输入回归和本地 RC 后停止，不用缺失 Sampling 数据的替代测量作性能采纳。

- 删除 P0 全矩阵／subset 入口撤除后留下的 `HostCSRData`、转换器、子集／行区间缓存、瞬态上传队列和自引用 helper，连同无引用的 `CudaCsrBuilder.cpp` 构建单元。实际 compact 预取仍调用 `uploadIoCsrViewAsync`，使用独立的 `compactUploadStream_`；CSRB 缓存与输入校验保留。
- 投影／背景指针表在预上传时一次建立；主机全局／逐床分子容量跨迭代复用，逐床分子由后端写满，去掉调用层重复清零；这些暂存退出迭代作用域后释放。主机正则梯度改为空向量，首次计算由既有 CUDA wrapper 调整容量；关闭 κ 图时不再创建全 1 体积。
- 代码及构建定义净减少 **744 行**；`CudaSparseOpsState.h` **3941 → 3678 行**。驻留路径少保留 `8 × globalNvox` 字节的主机梯度，单床 120×120×220 为 **24.17 MiB**；BSREM 关闭 κ 时另省同样大小的体积。主机路径从第二次迭代起少做 `bedCount + 1` 次分子体积申请及每床一次清零；这是源码分配／写入量分析，没有声称端到端加速或实测峰值减少。

首次构建发现生产预取仍引用旧上传流数组，已保留该调用并收为一个独立流；修正后全量构建及 CTest **283 通过、4 跳过、0 失败**。4 项跳过与 §7.2 相同，LNK4098 为原有告警；CUDA 数学源码与默认配置无差异。

| 回归 | 保存点最大 relL2 | 对照 |
|---|---|---|
| CSRB 驻留 MLEM it50 | 4.9185e-8 | e5d376a 包内 exe 已冻结的 it50，复用其分母缓存 |
| CSRB 主机 MLEM it1–3（vb2） | 2.9718e-8 | e5d376a 同命令、同缓存 |
| CSRB 主机 BSREM it1–5 | 1.1113e-8 | e5d376a 同命令、同缓存 |
| CSRB 驻留 BSREM，κ off，it1–5 | 1.0469e-8 | e5d376a 同命令、同缓存 |
| CSRQ 双窗，仅第二窗有背景，主机更新 it1–3 | 3.1931e-8 | e5d376a 同命令；同矩阵／投影复制为两窗，以投影作合成背景，仅为接口回归 |
| CSRB CTAC，MDP 两床 it5/10/15/20 | 5.4068e-8 | 09-04 因子分解输出 |
| CSRQ CTAC，同两床 it5/10/15/20 | 6.2355e-8 | 09-04 P3 输出 |

证据在 `recon_out/convergence_buffers_20260905/`：`source.patch`、构建／CTest 日志、各组 `validation.json` 记录 exe SHA-256、配置／投影／μ 图哈希、矩阵文件大小与修改时间不变检查、完整命令及逐保存点数值。`replay.py --exe <exe> --label <新目录名> <case...>` 可复跑；`host_mlem host_bsrem resident_bsrem_no_kappa joint_background` 的基线位于 `baseline/`，其余对照路径在脚本中冻结。输入源保持只读，新的分母及输出均在隔离目录。

本轮本地 RC 为 `dist/release_candidates/SPECT_recon_rc_20260905_buffers/` 及同名 zip。干净源码提交、构建／staging／包内 exe 一致性、包内哈希、三架构 fatbin、包内默认配置 smoke 与包内 exe it50 回归由 `verify_release.py` / `release_verification.json` 记录。上一轮 RC 与证据保留。Sampling、sm_120 现场性能、CSRQ 采纳及严格 BSREM 门仍按 §7.2 保持开放。

## 7.4 2026-09-05 RC 重建内存与速度实测

用户要求验证实际占用和速度。本轮只测量，不改源码、配置或 RC：A=`20260905+e5d376a`，B=`20260905+0510e4f`；使用生产 `spect_config.ini` / `algo_mlem.conf`、Tc99m 120×120×220 CSRB（24.331 GiB）、Derenzo 单床、MLEM 50:50、vb1、PSF sieve/castor 1.3 mm、无 CTAC／正则／迭代内滤波，默认输出 100×100×66。各预热 5 次迭代一次，再按 **A-B、B-A、A-B** 顺序测三组，正式测量均为同一分母缓存 hit。GPU 为 RTX 4070 Ti SUPER 16 GB，驱动 610.62，主机 64 GB / 20 逻辑 CPU；另有 PET 作业共享机器。Sampling_Approach 仍缺失，本轮不关闭其性能门。

每秒通过 Windows 进程计数采样工作集、私有提交和进程专用显存，NVML 单独记录整卡占用；迭代典型值取第 5–45 次的样本中位数，主机峰值另用 OS 累计峰值计数。私有提交不等于常驻物理内存，进程专用显存与整卡占用也不混用。

| 指标（3 次中位数，括号为范围） | A 旧 RC | B 当前 RC |
|---|---|---|
| 50 次核心时间 / s | 96.378（83.123–96.536） | 97.609（71.706–97.638） |
| 整条命令 / s | 122.099（109.522–122.414） | 122.723（96.101–122.913） |
| 迭代期物理工作集 / GiB | 24.749 | 24.725 |
| 进程物理工作集峰值 / GiB | 24.749 | 24.725 |
| 迭代期私有提交 / GiB | 37.720 | 37.695 |
| 迭代期进程专用显存 / GiB | 12.644 | 12.644 |

**内存节省实测成立，但幅度小。** 三组迭代期工作集分别减少 **24.172 / 24.219 / 24.215 MiB**，约总工作集 **0.095%**，与移除单床 24.17 MiB 梯度缓冲吻合；24.331 GiB 的主机矩阵缓存仍是主要占用。当前 RC 整卡显存采样峰值最高 **14.141 GiB**（含其它进程）；进程专用显存未下降。

**没有可归因于本次收敛的加速证据。** 三组核心时间候选相对旧版分别 **+1.28% / +1.14% / −13.74%**，全部保留，未挑选最快运行。按执行顺序，系统 CPU 样本中位利用率为 **57.05 / 53.95 / 54.75 / 54.90 / 34.55 / 25.20%**；最后一组随共享负载变化明显变快，不能把它解释为代码加速，前两组也没有显示加速。当前环境下可报告的当前版范围为核心 **71.71–97.64 s / 50 次**（1.43–1.95 s/次）、整条命令 **96.10–122.91 s**。严格版本速度比较需要稳定的独占负载；正式门还需要原 Sampling 输入。

三组主输出 relL2 为 **4.9405e-8 / 4.9536e-8 / 4.9100e-8**，均通过 1e-7；矩阵／投影／算法 effective manifest 一致。输入哈希、矩阵文件大小／mtime／metadata 哈希及两个 RC 的全部 payload 哈希均保持不变。执行 completed；内存证据 validated；速度归因 inconclusive；生产采纳状态不变。

原命令 `python recon_out/perf_memory_20260905/measure.py`；脚本、冻结输入与顺序 `contract.json`、八次运行的 console/JSON/每秒 CSV、环境快照及统计 `summary.json` 均在该目录。复跑应将脚本置于 `recon_out` 下新的隔离目录，原目录拒绝覆盖。该记录补充 §7.3 的分配量推导，没有更新或重打包已验证 RC。

## 7.5 2026-09-05 CSRB 主机锁页上传局部试验

契约（用户：“继续，开始尝试吧”）：A 为已验证 RC `20260905+0510e4f`，候选基于 `df0bd45`（其相对 RC 仅文档变化），在 `build/csrb-pin-src/` 独立工作树试验。唯一机制是为完整 CSRB 主机缓存恢复有明确所有权的锁页注册：现有 accelerator lease 保持 chunk 所有者，上传结束后解锁；部分缓存不注册。保持文件格式、数值精度、数学核、默认配置、设备缓存槽和迭代顺序。

输入与 §7.4 相同，复用已冻结分母缓存；两次 5 迭代预热，三组 A-B / B-A / A-B 的 MLEM50 比较，记录核心／整命令／上传计时、RSS、私有提交、进程专用显存和共享负载。数值门 relL2 ≤ 1e-7；新增复杂度最低收益沿用 ≥5%，同时要求各组方向一致、内存无矩阵副本级增长。每次运行限 240 s，任一执行／所有权／数值错误即停止性能链并先修复。先检查完整／部分缓存、反复释放注册及现有数值回归；源码改动须完成构建和测试。Sampling_Approach 缺失，因此本轮仅决定局部候选是否值得保留，不能关闭正式性能或生产采纳门。旧 RC 不改；不成立则撤回候选，保留最小复现证据。

**结果：局部候选值得保留，三组核心时间均减少 35.6%–37.4%；主机内存基本不变。** 执行 `completed`；本机单床速度／数值证据 `validated`；生产采纳 `pending`，代码仅保留于独立工作树，当前 RC 未替换。

实现只扩展原 `CompactChunkResidentCacheLease`：完整缓存的 `colIdx`／`values` 直接注册，不复制矩阵；lease 持有对应 chunk 的 `shared_ptr`，释放时先同步再解锁。部分缓存不注册，维持原路径。同步修正了 State 中已不符合当前加载器行为的 pinned-prefetch 注释。生产源码净 +45 行，既有测试文件新增 44 行生命周期检查，没有新增配置开关、设备缓存槽或数学核。

| 本机同输入测量（3 次中位数，括号为范围） | A 当前 RC | B 锁页候选 |
|---|---|---|
| MLEM50 核心 / s | 70.941（70.664–71.369） | 45.432（44.207–45.955） |
| 日志 `full_command_s` / s | 93.168（92.804–93.354） | 72.933（67.700–73.016） |
| 外部观察至进程退出 / s | 94.088（94.088–95.087） | 75.071（69.072–75.073） |
| 准备阶段 / s | 18.606（17.516–19.067） | 22.621（19.094–23.308） |
| 进程物理工作集峰值 / GiB | 24.72539 | 24.72544 |
| 迭代期进程专用显存 / GiB | 12.64399 | 12.69172 |
| 上传调用的主机计时 / s | 42.397（42.034–42.461） | 0.171（0.165–0.179） |

三组按 A-B / B-A / A-B 配对的核心减少量分别为 **35.609% / 35.958% / 37.441%**，均超过事前 5% 局部收益要求；按中位数约省 **25.51 s / 50 次**。含进程退出的观察耗时中位数约减少 **20.2%**（采样轮询间隔 1 s，包含轮询检测延迟）；这把日志 `full_command_s` 之后的资源释放也包含在内。准备阶段中位数有所增加，因此不将核心收益直接等同于整条命令收益。主机峰值配对增量仅 **48 / 132 / 20 KiB**，没有矩阵副本级增长；专用显存计数增加 **48.875 MiB**，整卡采样峰值仍约 14.1 GiB。私有提交与物理工作集分开保存在 JSON 中。

机制证据：两版上传次数均为 151、矩阵复用计数均为 300；仅改变主机数组注册和生命周期后，上传调用的主机等待显著降低，迭代也稳定缩短。该计时包含 API 入队／主机阻塞，**不是 GPU 实际传输时长**；不能由 42.397 → 0.171 s 推断 PCIe 传输消失。带宽改善与计算重叠各自的贡献未用 GPU 时间线拆分。按运行顺序系统 CPU 样本中位利用率为 26.7 / 28.4 / 27.7 / 24.35 / 23.15 / 31.9%，共享环境仍记录在案；此次三个交错配对方向一致，且基线核心范围窄，支持本机该负载的局部加速结论。

验证：独立工作树完整构建通过，CTest 283 通过／5 跳过／0 失败；其中额外跳过的一项 Lu-177 metadata 测试在原工作区数据目录补跑通过，合计 **284 项通过、4 项原有跳过**。新测试反复创建／释放完整缓存 16 次，检查注册类型、目录销毁后 lease 保持内存、lease 释放后解锁与最终所有者释放；其余多床／伴随／CTAC 单元检查通过。真实矩阵 14 GiB（3/6 块缓存）两版各 3 次迭代通过，relL2 **2.9824e-8**，候选未启用锁页。三个 MLEM50 配对 relL2 最大 **4.9535e-8**，通过 1e-7；真实 MDP 两床 CSRB CTAC it5/10/15/20 对 §7.3 既有验证输出最大 **4.5912e-8**，通过 1e-5。无注册／解锁错误。有效 matrix／projection／algorithm／multibed manifest 一致，输入哈希与矩阵大小／mtime／metadata 哈希不变，当前 RC 全部 **30** 个 payload 哈希复核通过。原有 LNK4098 告警保留。

证据目录为 `recon_out/csrb_pin_20260905/`：`contract.json`、`candidate.patch`、构建／测试日志、两个预热、两个部分缓存检查、六次 MLEM50 的 console／JSON／每秒 CSV、CTAC 保存点及 `summary.json`。候选 exe 为 `build/csrb-pin-src/build/bin/Release/SPECT_recon.exe`，版本 `20260905+df0bd45-dirty`，SHA-256 **74d60d5dae77b5003fbbccce57b116bc051021593c5eba123aeeb6a218a55d0b**；相对 `df0bd45` 的补丁 SHA-256 **040c518154d4a774d2f432b36d64b8a2a33c50898b746f7b4255060750204161**。该身份明确为试验构建，未打包为 RC。

复现：现有目录只可用 `python recon_out/csrb_pin_20260905/experiment.py summarize` 重算统计与身份检查。重跑时将 `experiment.py`、`ctac_guard.py` 复制到 `recon_out` 下新的隔离目录，依次运行 `experiment.py warmups`、`experiment.py partial`、`experiment.py pairs`、`ctac_guard.py`、`experiment.py summarize`；每个运行目录拒绝覆盖。最小下一步是恢复原 Sampling_Approach 后执行正式五床 50 迭代门，或明确批准另行冻结的采纳范围；本轮不扩大到 CSRQ、现场设备或另一种格式。

## 8. 当前实现摘要

`Measurement operator (A, att, s, W, b) → Poisson statistics (q, S) → Optimizer (MLEM | BSREM-II)`。CTAC 在按角度的输入与收口阶段施加，背景进入 ratio，行权重只进入 S；BSREM 复用现有梯度核，再做逐体素更新。CSRB 与 CSRQ 各自保留必要的格式内核，不新增统一内核或算子类层次。
