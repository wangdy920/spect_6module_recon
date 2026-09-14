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

**结果：局部候选值得保留，三组核心时间均减少 35.6%–37.4%；主机内存基本不变。** 执行 `completed`；本机单床速度／数值证据 `validated`。截至本节试验结束，生产采纳 `pending`、代码仅保留于独立工作树；同日稍后用户批准合入主线并更换性能门，见 §7.6。旧 RC 仍未替换。

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

复现：现有目录只可用 `python recon_out/csrb_pin_20260905/experiment.py summarize` 重算统计与身份检查。重跑时将 `experiment.py`、`ctac_guard.py` 复制到 `recon_out` 下新的隔离目录，依次运行 `experiment.py warmups`、`experiment.py partial`、`experiment.py pairs`、`ctac_guard.py`、`experiment.py summarize`；每个运行目录拒绝覆盖。试验结束时提出的原 Sampling／另行冻结采纳范围下一步，已由 §7.6 用户决定解决；本轮不扩大到 CSRQ、现场设备或另一种格式。

## 7.6 2026-09-05 Lu-177 208 keV 五床新性能门与后续审查

用户决定：“利用现有 lu-177 208kev 作为新的性能门，并且以5床位数据为准，这部分提升我认为可以保留”。因此 §7.5 锁页实现获准保留并合入主线；正式性能门改用 `test_data/6module/Lu177_sysmat/matrix_PSC_PIN26x28x20_208keV_120x120x220_ZFOLD`。当前文件核对为 120×120×220、6 chunks、nnz **4,399,851,643**、CSRB 总大小 **35,199,642,800 B / 32.782222 GiB**、92 B metadata。旧 Sampling 126.261 s 与本节不可比较，其缺失不再阻塞新门。

五床输入冻结为 `E:/Xiang_Project/SPECT/6Module/experiments/recon_param_study/pre/mouse_normal_lu/Bed1..5.bin`（2026-08-27 正常小鼠，既有 2026-09-03 预处理）。来源记录显示这些计数合并了 113/208 两窗；用作计算性能和前后数值一致性负载，不解释为单 208 keV 窗定量验证。保持生产 INI／algo_mlem、MLEM50:50、vb1、PSF sieve/castor 1.3 mm、denom hit、无 CTAC／背景项／正则／迭代内滤波，床顺序及 16.8 mm 步长固定。延续核心目标 120 s／下限 180 s、后续新增复杂度最低 5% 收益和 relL2 1e-5；此次锁页移植另守护 1e-7。全量主机缓存必须由默认自动预算实际满足，记录空闲 RAM 与显存，不强行扩大预算。

计时边界保持源码原定义：`iteration_core_s` 是分子调用、更新、正则梯度与迭代 Butterworth 等主机计时区间的累加，并非整个迭代循环墙钟。输入准备调用和逐轮 `observeCompactMemory()` 在这些区间之外；异步 GPU 工作可能由后续同步承担，不能把差值直接解释成某个核的耗时。整命令和到进程退出的外部观察时间同时保留。

执行范围：合入已验证的唯一锁页实现，更新既有性能门驱动，完整构建和聚焦回归；用旧 RC `0510e4f` 与主线候选各 5 次预热后按 A-B / B-A / A-B 重复 MLEM50，记录主机／显存与阶段时间，每次限 600 s。预热失败则先修复，不能更换为单床或另一矩阵。分析 kernel 融合、路径合并和分支简化时，以新五床热路径和单独短程分析结果为依据，不把带分析开销的时间混入正式门，也不重新打开已拒算法／精度变体。

锁页实现与新门驱动已提交为 `6a3e59b`，主线 Release 身份为 `20260905+6a3e59b`。完整构建通过，CTest **284 通过、4 原有跳过、0 失败**（共 288）；保留原有 LNK4098 告警。初次 CMake 配置停在 vcpkg 安装阶段，停止本次自有配置进程后以已安装依赖 `-DVCPKG_MANIFEST_INSTALL=OFF` 完成配置与构建，没有更换依赖版本。

**执行 `completed`；同输入性能／数值证据 `validated`；锁页主线采纳 `merged`。新门目标 120 s 和下限 180 s 均未达到。** 初次预检发现另一个 CASToR 作业占用约 19 GiB、空闲约 28 GiB，未启动重建。默认完整缓存需矩阵 32.78 GiB + 8 GiB 保留；等待约 24.5 分钟至至少 42 GiB 空闲且无其它重建后自动接续，没有终止外部作业或扩大预算。两次预热通过（A/B 五次核心 28.923/20.980 s、relL2 5.650e-8），全部 B 日志确认 6 块锁页与 packed-K=2。生产配置五床在内部以 non-overlapping reconNz=1100 处理，16.8 mm 步长及 center66/cosine10 作用于最终输出拼接，primary 为 100×100×290、MET_DOUBLE。

### 正式配对结果与身份

| 配对执行顺序 | A：0510e4f 核心 / s | B：6a3e59b 核心 / s | 核心减少 | primary relL2 |
|---|---:|---:|---:|---:|
| A → B | 284.278 | 191.092 | 32.78% | 8.8022e-8 |
| B → A | 262.664 | 190.517 | 27.47% | 7.2588e-8 |
| A → B | 290.992 | 193.726 | 33.43% | 6.3136e-8 |

| 指标（三次中位数，括号为范围） | A：未锁页 | B：已锁页 |
|---|---:|---:|
| 50 次核心 / s | 284.278（262.664–290.992） | **191.092（190.517–193.726）** |
| 整命令 / s | 339.994（311.848–341.748） | **242.402（239.372–246.875）** |
| 到进程退出的观察时间 / s | 343.349 | 245.220 |
| 矩阵／分母准备 / s | 31.835 | 33.174 |
| OS 主机工作集峰值 / GiB | 34.157276 | 34.157070 |
| 迭代期私有提交 / GiB | 46.809700 | 46.808697 |
| 迭代期进程专用显存 / GiB | 12.296329 | 12.360626 |

锁页的五床核心时间减少 **27.5%–33.4%**；整命令中位数减少 **28.70%**。没有第二份矩阵主机副本：RSS 峰值基本相同；进程专用显存计数增加约 **65.84 MiB**。Windows RSS、私有提交与专用显存各自统计，整卡 NVML 值保留在原记录中，不当成本进程显存。系统 CPU 中位利用率按运行顺序为 31.5/11.4/11.8/11.7/31.5/28.9%，整卡 GPU 中位利用率均为 100%；三个配对结果全部保留，不以最快样本替代中位数，也未从三组样本估计稳定置信区间。

所有正式运行 denom hit，最大 primary relL2 **8.8022e-8 ≤ 1e-7**。配置／五床／预处理来源／分母哈希、矩阵文件身份、有效 matrix/projection/algorithm/multibed 输入字段一致，旧 RC 的 30 个 payload 哈希不变。首次汇总把 `multibed.fitted_bed_gains` 也当输入要求位相同而停止；源码 `RunManifest.cpp` 在最终 SaveVolume 上重新拟合该字段，它属于派生输出。保留首次日志，修正比较分类后只重算汇总，逐运行保留 gain，未重跑或更改图像容差。

证据目录 `recon_out/lu208_5bed_gate_20260905/`：`contract.json`、`summary.json`、逐运行 CSV／日志／manifest、`integration_verification.json`、`experiment.py` 与 `measurement.py`。可用 `python recon_out/lu208_5bed_gate_20260905/experiment.py summarize` 复核已完成结果；新的实际测量须复制脚本到新的隔离目录，运行目录拒绝覆盖。既有门驱动 `tools/run_220_mlem_perf_gate.py` 已更换为该矩阵和五床，文件名保留以兼容已有入口。

新基线 exe 为 `build/verify-vs/bin/Release/SPECT_recon.exe`，版本 **20260905+6a3e59b**，SHA-256 `ccca6af6ac216e88e48ad40f8c4721882221d3c603ce86cf20c7894eef88674a`；同哈希备份在证据目录 `frozen_binary/SPECT_recon.exe`。备份仅含 exe，使用时仍依赖经验证的 runtime／fatbin 布局；本轮未重新发布 RC，旧 RC 和 §7.5 独立候选保持原样。

### 独立五次 Nsight 时间线：主要成本是稀疏乘法

正式配对后，单独运行 B 的同配置五次诊断，profile primary 对预热 B 的 relL2 **3.1062e-8**。`profile_B/trace.nsys-rep`、`trace.sqlite`、`analysis.json` 与可复核脚本 `analyze.py` 保留原始时间线；这次运行含分析开销，不进入正式性能统计。

| GPU 核（五次诊断） | 启动次数 | 核执行总和 / s |
|---|---:|---:|
| cuSPARSE `csrmv_v3_transpose_kernel` | 1,200 | **11.38637** |
| packed-2 `compact_csr_spmv_xpair` | 600 | **6.15649** |
| 反变换 `compact_transform_backward_add_batch` | 150 | 0.08642 |
| pack `compact_pack_colpair_to_float2` | 600 | 0.02826 |
| cuSPARSE partition | 1,200 | 0.01609 |
| ratio `compute_compact_ratio_chunk_batch` | 150 | 0.00188 |

以每次首个 packed forward 到末个 transform-add 为分子窗口，五段合计墙钟 **18.29231 s**，GPU kernel 活跃 **17.67519 s**。H2D 活跃 **5.38127 s**，其中 **5.04667 s（93.78%）**与核重叠，只有 0.33460 s 不重叠；kernel 和 copy 均未运行的间隙为 **0.28252 s（1.54%）**。这个窗口不含迭代更新和最前面的输入准备，不能替换 `iteration_core_s`；连同四段迭代间隙的宽窗口为 20.52272 s，其中 GPU 空闲 2.24340 s。

因此 pin 的主机 upload-call 中位计时 **94.620 → 0.430 s** 不能说成“PCIe 传输消失”；实际传输仍在，短诊断证明大部分与计算重叠。GPU 核时间主要是反投约 **64%**、正向约 **35%**，ratio／pack／反变换三项合计仅约 **0.66%**。全 trace 的 `cudaEventSynchronize` 16.587 s 主要是等待上述 GPU 工作，不是可直接删掉的 CPU 开销。`cudaMemGetInfo` 9 次总共 **0.000176 s**，不支持此前“内存观测可能昂贵”的假设；移走这项查询没有性能依据。

### 已定位的路径问题：默认 ROI 浮点边界误触发 host 路径

源码原本允许此类无 CTAC、CASToR PSF 的单／多床驻留输入及更新；本次时间线却没有 `compact_mlem_update`、resident gather／forward-transform 核。实际运行是 **host 输入准备／更新 + 同一 GPU CSRB 分子入口**，不能按“默认五床必定 resident”解释性能。

触发点为 `CompactSymmetricReconLoop.cpp::compactRoiPrunesMatrix` 的固定 `1e-6 mm` 比较。manifest 的 spacing 是 float32 的 0.3 扩成 double，即 **0.30000001192092896 mm**；120／220 体素宽度分别成为 **36.000001430511475／66.00000262260437 mm**，均超出默认 ROI 的 `36+1e-6`／`66+1e-6`，于是 `residentBatchInput` 被禁用。这是本输入上的直接判断复现，不是由计时猜原因。

进一步只读扫描全部 **4,399,851,643** 个 colIdx：引用体素 **1,457,796**，默认 mask 外引用 **0**；nominal 0.3 mm、实际 spacing + 默认 36/66 mm、实际 spacing + 完整 extent 三种计算得到的离散 mask 完全相同（**2,486,880** 个体素）。证据 `resident_eligibility_diagnosis.json`、`check_roi_support.py`、`roi_support_check.json`，检查耗时 24.11 s、矩阵身份不变。该矩阵支持没有被默认 mask 额外裁剪，固定绝对容差使资格判断产生了错误的 host 分类。

最小下一步是仅在现有资格判断处采用与输入 float 精度一致的尺寸比较，保留实际 mask、显式小 ROI／偏心 ROI 的 host 语义和现有 GPU 路径。先做 full-FOV／缩小／偏心边界检查，再用本五床对照核对 resident 核出现、primary relL2 与重复性能；不能直接删除 ROI guard 或修改 voxel/ROI 配置来绕过它。本轮只完成诊断及支持验证，未改这个判断，未声称 resident 路径已经通过本矩阵的 MLEM50 或能使 180 s 达标。

### 原始投影零计数的描述性筛查

在修改任何新核前，只读五床投影与六个 CSRB 的 row pointer；复用既有 metadata／CSRB header 读取器。metadata 与文本独立核对为 360 角度、48×48 探测面、90 个 compact 角度、24 个 compact Z 行、SYM90Z；按四个角度象限与正反 Z 半面展开，检查完整投影行覆盖为双射，row nnz 合计与 4,399,851,643 一致。这里尚未应用生产坏探测器 68 屏蔽，因此统计是原始输入的描述性上界，不能直接当执行时间或采纳结果。

| 原始输入 | 零计数 bin | 非零 RHS 覆盖的 nnz 比例 | K=2 两路至少一路非零的 nnz 比例 |
|---|---:|---:|---:|
| Bed1 | 81.40% | 20.26% | 35.89% |
| Bed2 | 73.97% | 28.26% | 47.32% |
| Bed3 | 73.27% | 29.01% | 48.01% |
| Bed4 | 81.64% | 20.45% | 35.83% |
| Bed5 | 89.91% | 11.21% | 21.03% |

推导依据是现有 `compute_compact_ratio_chunk_batch`：有限合法输入下，`p_i=0` 则 `p_i/max((Ax)_i+b_i, eps)=0`，所以该行分子不需要算 forward；**分母 sensitivity 仍必须保留该行**。在当前 K=2 的相邻 Z 镜像配对下，平均约 **62.38%** 的正向 nnz 工作属于“两路 p 都为零”的行。最小候选是在既有 packed-2 行核读取 values／colIdx 前做一次静态投影检查并直接写零；不裁剪源矩阵、不建立五套稀疏矩阵、不改变 denominator。此优化只适用于生成 ratio 的分子调用，独立 forward 输出／分母及 CTAC 调用必须保留其原语义。数据稀疏性随采集而变，这不是通用加速比例。

记录 `projection_sparsity.py` / `projection_sparsity.json` 包含投影 SHA-256、原始计数、nnz 加权比例、矩阵文件身份和映射校验。该筛查只形成下一步假设；没有实现新核，也没有把 62.38% 推算成端到端收益。

### 剩余方向：按本次证据排序

当前实际调用链是 `CompactSymmetricReconLoop` 的 host 输入准备／PSF／更新，进入 `accumulatePoissonBackprojectedRatioCUDA` 的同一单／多床 CSRB 入口，再由 `executeCompactBedBranchBatchDeviceRatioSpmvTAccumulateFloat` 执行每 chunk 的 40 RHS。下表依据源码、完整矩阵支持检查、输入筛查和时间线排序；没有将未实施方案的潜力当成实测收益。

| 方向 | 已有证据与最小改动位置 | 边界／最小验证 |
|---|---|---|
| **优先：复用已有 resident 路径** | 固定 1e-6 mm 导致默认 ROI 被误分类；完整 mask／列支持检查已排除真实裁剪。只修现有资格判断，不建立另一条路径 | 先查 full-FOV／缩小／偏心边界，再验证新门 resident trace、MLEM50 relL2 和配对 core／整命令时间。原 host 分支仍服务实际 ROI、debug 和不支持的 PSF |
| **其次：零计数行提前退出** | `projdomain_chunk_ops.cu::compact_csr_spmv_xpair` 当前每行都遍历 nnz；五床平均 62.38% 的 K=2 加权行工作两路 p 均为零，正向约占 GPU 核时间 35%。在现有分子核入口检查；保留 K=2、原矩阵和完整分母 | 先测真实 chunk 与两路均零／单路非零／稠密计数对照，并维持非有限输入的拒绝；再跑同一五床 MLEM50 relL2 与配对性能。独立 forward 与 CTAC 分子不能被误用成相同语义 |
| forward + ratio 融合 | 行归约后可直接计算 ratio，省中间 yhat 的写后读和独立启动；CSRQ 已有此形式。但 ratio 本身五次仅 0.00188 s，不足以单独支持 ≥5% 项目 | 保持 fp32 行和 → fp64 加背景／除法 → fp32 ratio 的原舍入点及角度／镜像映射。若作为零行检查的配套变动，仍需分开量时以区分机制 |
| 合并五床小核启动、简化反变换 | 每 chunk ratio 五次、transform-add 五次；反变换为每 `(branch, voxel)` 一次 fp64 atomicAdd。但 transform 五次合计 0.08642 s、约 0.49% GPU 核时间 | 当前收益上限小，不新增床位 launch／gather 变体；gather 会改变求和顺序，不能当位相同重构 |
| 重复 pack 与数据布局 | 每轮 120 次 pack，但五次仅 0.02826 s、约 0.16%。40 RHS 为 483.40 MiB，缓存另一套 packed 副本会增加相近显存 | 当前不值得换布局或增加副本；不能把床/RHS 循环移到 chunk 外导致矩阵反复上传，也不重开已拒 pack-xquad 融合 |
| 主机提交、计时与同步 | 每 chunk 创建四组计时事件，反投逐 RHS 重建 dense-vector 描述符；分子窗口 kernel/copy 均空闲的比例仅 1.54%，大部分 EventSynchronize 是 GPU 等待 | 当前不以 Graph／事件池作为 ≥5% 项目。只在 resident 修正后的新时间线显示明显提交间隙时考虑现有 State 内的复用，必须保留上传 ready、缓冲复用和非法值检查依赖 |
| 观测开销 | `cudaMemGetInfo` 九次总共 0.000176 s，已不支持查询开销是瓶颈的假设 | 保留现有观测口径，不通过移动计时／查询定义制造 core 达标 |
| 路径合并、分支简化 | 单／多床与背景 ratio 已共用生产入口；CTAC 因子处理、host debug/ROI/非 CASToR PSF 回退、MLEM／BSREM 更新分别有实际语义。`hasAtten` 在 chunk 级、背景判断在 row 级，均不在逐 nnz 内层 | 小型重复参数／分支表可随触及处收敛；目前没有依据把它们作为 ≥5% 加速项目。删除活跃回退或制作多布尔模板组合会扩大验证范围 |
| 反投算法／格式 | 40 RHS 的 cuSPARSE 反投是 GPU 核主成本（约 64%）。但它对零 ratio 的内部处理不能由调用次数推断；本次尚无证据说明究竟受 scatter 原子冲突、访存还是库调度限制 | 先以真实 chunk／RHS 的硬件计数区分机制，若获准才形成新算法合同。本轮不重开宽 SpMM、旧自定义转置、三块常驻、FP16 或 CSRQ 采纳 |

CUDA Graph 只在提交间隙被证实显著时作为后续项：官方说明它能减少重复启动准备开销，同时捕获期间不能同步正在捕获的流／事件；当前每块同步及缓存槽换绑需要先理清生命周期，不能只在循环外包一次 capture。[NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) 对异步传输，锁页只是使主机传输具备重叠条件，实际重叠仍由 stream 和事件依赖决定。[NVIDIA Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

## 7.7 2026-09-05 默认全视野 ROI 精度修复与 resident 复验

用户在 §7.6 分析后要求“继续”。本轮先修已确认的 `compactRoiPrunesMatrix` 误判；仅在既有判断处将矩阵物理 extent 转回 ROI 参数的 float 精度，保留原 1e-6 mm 比较余量、实际 mask、缩小／偏心 ROI 的 host 语义。不扩展 resident 支持范围，不增加新核或配置；首轮日志记录实际 input/update 路径和 ROI 判断。

冻结对照 A 为已接受 pin 的 `20260905+6a3e59b`（exe SHA-256 `ccca6af6ac216e88e48ad40f8c4721882221d3c603ce86cf20c7894eef88674a`），B 为本次修复。沿用 §7.6 Lu208 CSRB、正常小鼠五床、全部配置和输入身份，原 primary relL2 ≤1e-5、核心目标120／下限180 s；1e-7 是此前纯锁页移植的额外守护，本次不把不同 host/device 数值路径强定为位等价。

验证顺序：完整构建／CTest；默认五床各五次预热，确认 B 路径；实际缩小直径／高度、相邻较小 float 尺寸及偏心 ROI 的短程行为检查；A-B／B-A／A-B 三组 MLEM50、每次600 s、自动完整缓存和42 GiB空闲预检，另记录整命令和内存。短程数值仅作诊断，最终图像门用50次保护端点；若50次数值失败，不采纳并保留最小失败证据。独立短 Nsight 仅用于核出现与成本变化，不进入性能门。该 guard 也服务 CTAC，另复用既有真实两床 CSRB CTAC20 保护端点（每5次保存比较、relL2≤1e-5、每次600 s），检查共享调用方，不作另一性能比较。证据目录 `recon_out/lu208_resident_roi_20260905/`。本机制完成后才开始§7.8的独立候选，结果如下。

修复代码 `0ddc6ab`，执行 `completed`、本机同输入证据 `validated`、采纳 `merged`。完整构建通过，CTest284通过／4原有跳过／0失败；实际运行的四项 ROI 边界（直径／高度各小一个 float 刻度、35.7/65.7 mm、偏心）全部保持 host 路径，对旧版最大 relL2 **4.2468e-8**。默认全视野为 device input/update，五次预热 relL2 **3.1021e-8**。双方邻接库和实际加载的 CUDA／CRT／OpenMP 库哈希一致。

| 三组 MLEM50 | A：pin6a3e59b / s | B：ROI修复0ddc6ab / s | 核心减少 | primary relL2 |
|---|---:|---:|---:|---:|
| A → B | 193.294 | 181.052 | 6.33% | 1.3645e-7 |
| B → A | 193.601 | 181.209 | 6.40% | 9.4534e-8 |
| A → B | 195.512 | 181.476 | 7.18% | 7.0668e-8 |

核心中位数 **193.601 → 181.209 s**，整命令中位数 **245.582 → 215.926 s（−12.08%）**，新版范围214.396–218.834 s；**120／180 s 均未通过**。主机工作集峰值 **34.15720 → 33.46844 GiB**（约少705 MiB），迭代期私有提交46.81232→46.23950 GiB；进程专用显存 **12.36063 → 12.73368 GiB（+382 MiB）**。这是复用既有 resident 图像/累加缓冲的资源变化，没有另存一份矩阵。所有正式运行输入／矩阵身份、有效输入 manifest、完整主机缓存与 denom hit 检查通过；旧 RC 30个payload不变。

五次独立 Nsight 确认 **5 次 `compact_mlem_update`**，以及各25次 gather／forward-transform；profile图像对预热B relL2 **3.1007e-8**。稀疏核成本基本不变（反投11.40624 s、正向6.20344 s）。按与§7.6相同窗口定义，五段分子合计18.38511 s，含四段迭代间隙的跨度18.42408 s；四段间隙从先前约2.23042 s缩至 **0.03897 s**。这是省掉主机往返的时间线证据，profile绝对耗时不进入配对统计。CTAC共享调用方的真实两床CSRB20次回归通过，it5/10/15/20最大relL2 **6.1106e-8≤1e-5**，实际也进入device input/update。

证据目录的 `summary.json`、`edge_checks.json`、`profile_B/analysis.json`、`csrb_ctac20/validation.json` 和 `runtime_hashes.json`／`candidate_loaded_runtime.json` 支持以上结论；原命令、输出和复核脚本保留。固定0ddc6ab exe在 `recon_out/lu208_resident_roi_20260905/frozen_binary/SPECT_recon.exe`，SHA-256 `0fc94c77828025b3d692c00331a016c9789b2b1afbcb642bd6e78ef9fa8d6fc4`，作为下一机制的独立对照；exe备份仍需配套已验证runtime布局。

## 7.8 2026-09-05 零计数 pair 分子正向优化与复验

用户在§7.7执行期间再次要求“继续”。ROI机制已完成独立配对及CTAC回归，以下只考察§7.6已量出的零计数工作：当前默认K=2，raw五床约62.38%的nnz加权pair两路p均为0；正向仍约占GPU核时间35%。对合法有限输入，p=0对应的分子ratio为0，不需要该行Ax；灵敏度S仍保留所有原始行。

最小候选复用 `compute_compact_ratio_chunk_batch` 的现有行映射，在同一 `dBatchOutput_` 中暂存0/1投影活跃标记；packed-2分子核仅在两路标记均为0时提前退出，其余计算／舍入／反投不变。原ratio随后覆盖该缓冲；无新增矩阵、长期mask缓存、CUDA模块或配置。独立forward、CTAC、sensitivity和其它K保持原语义，CTAC调用显式使用普通packed-2行为。现有非有限S／图像检查保留。

冻结基线A为§7.7的 `20260905+0ddc6ab`（SHA `0fc94c77828025b3d692c00331a016c9789b2b1afbcb642bd6e78ef9fa8d6fc4`），B为零pair候选；沿用相同Lu208五床／生产配置／MLEM50／denom hit／完整默认主机缓存，relL2≤1e-5，核心目标120 s／下限180 s，新增复杂度要求三组配对中位核心收益≥5%。每次600 s，不并行占用GPU；5次预热为诊断，50次为保护端点。先完整构建／CTest（密集、两路均零、仅一路零的独立dense oracle及既有forward／S／CTAC测试），再A-B／B-A／A-B三组；另用短Nsight观察正向成本与额外活跃标记成本，复跑现有CTAC20保护端点以核对共享launcher未改变该路径。达到数值与收益门才保留；不重新打开已拒转置／SpMM／精度变体。证据目录 `recon_out/lu208_zero_pair_20260905/`；本轮执行 `completed`、本机同输入证据 `validated`、源码采纳 `merged`。

候选源码 `cfffea5`，生产代码净增28行（新增36行、删除或替换8行），既有dense-oracle用例增加15行覆盖两路均零／任一路非零／稠密行，未增加测试框架或矩阵副本。Release及sm86/89/120 fatbin完整构建通过，CTest **284通过／4既有跳过／0失败**。测试EXE为 `20260905+cfffea5`，SHA-256 `69fb2d2851854136fe2823cc1db085114807d1d114922ee3fdb40068a181a5ec`；11个配套runtime payload和10个实际加载模块与A哈希一致。预热B的device input/update检查通过，5次图像relL2为3.1033e-8。

| 三组 MLEM50 | A：ROI修复0ddc6ab / s | B：零pair cfffea5 / s | 核心减少 | primary relL2 |
|---|---:|---:|---:|---:|
| A → B | 181.066 | 144.843 | 20.01% | 6.9892e-8 |
| B → A | 180.897 | 144.821 | 19.94% | 5.4396e-8 |
| A → B | 181.016 | 145.079 | 19.85% | 5.1809e-8 |

核心中位数 **181.016 → 144.843 s**，配对收益中位数 **19.94%≥5%**；所有primary relL2≤1e-5。整命令中位数 **215.666 → 179.964 s（−16.55%）**，新版整命令179.877–180.820 s，process-exit观察中位数182.165 s另记。按既定的**核心**计时口径，**180 s门槛通过，120 s目标未通过**；整命令并非三次都小于180 s，不与核心口径混用。三次A核心范围180.897–181.066 s，B为144.821–145.079 s。

资源没有新增长期矩阵或mask：主机工作集峰值中位数 **33.468464 → 33.468494 GiB**（差32 KiB），迭代期RSS33.212963→33.212872 GiB、私有提交46.242027→46.234936 GiB；进程专用显存双方均 **12.733677 GiB**。六次均为完整默认主机缓存、denom hit；输入哈希、矩阵清单及有效输入manifest不变，拟合床增益仍作为输出量单独记录，旧RC30个payload哈希不变。系统CPU占用中位数A9.8%、B9.6%。

独立5次Nsight支持机制归因，profile绝对耗时不进入配对门：同为600次packed-2正向核，累计 **6.20344 → 2.40070 s（−61.30%）**；150次新增活跃标记仅 **0.000628 s**，150次实际ratio为0.001905 s。反投核仍1200次、11.59143 s（前一阶段profile为11.40624 s）；分母没有因零计数裁剪。按原“首个forward到最后transform-add”窗口，五段分子18.38511→15.15852 s；将新增标记准备计入后为 **15.15907 s**。此时kernel活跃14.12754 s、H2D5.43543 s，其中4.68886 s与kernel重叠；kernel/copy均空闲0.28496 s（1.88%）。不能将并行区间相加，也不能把正向的61.30%减少当成整轮收益。

profile仍有5次resident MLEM更新，图像对预热B relL2 **3.1061e-8**。共享launcher的真实两床CSRB CTAC20回归通过，it5/10/15/20最大relL2 **6.1629e-8≤1e-5**，输入与矩阵不变。证据为 `summary.json`、`integration_verification.json`、`profile_B/analysis.json`、`csrb_ctac20/validation.json`、`runtime_hashes.json` 和 `loaded_runtime_{A,B}.json`。固定EXE保存在同目录 `frozen_binary/SPECT_recon.exe`，仍需既有已验证runtime布局；CUDA helper fatbin内嵌EXE，本轮必要部署差量只有EXE，未重打包既有RC。

剩余方向按本次新时间线更新：**主反投核已占分子GPU核活跃时间约82.05%**；pack、标记/ratio、transform-add合计约0.84%，一般性融合和主机提交简化仍不足以支撑≥5%项目。H2D未与kernel重叠约0.747 s/五次（窗口的4.92%）是当前可见成本，但现有两槽与上传依赖仍须保留。要从144.843 s继续到120 s还需约24.843 s、17.15%的核心减少；最小下一步是对既有真实反投chunk/RHS采集硬件计数，区分原子冲突、访存与库调度，再定义有依据的候选。本轮止于已验证零pair优化，不重开已拒SpMM、旧转置、三块常驻、FP16或用CSRQ替代CSRB门。

复核现有配对数据：`python recon_out/lu208_zero_pair_20260905/experiment.py summarize`；复核时间线：`python recon_out/lu208_zero_pair_20260905/analyze_profile.py`。复跑使用新的隔离输出目录，复制该目录的测量/顺序驱动及gate_dry_run输入描述，冻结两版EXE身份后运行 `run_when_ready.py`；不要覆盖本轮原始输出。§7.7与§7.8各有独立A/B，不能跨轮相除声称严格配对累计收益。

## 7.9 2026-09-09 零计数行反投：pair 散射转置替代逐 RHS cuSPARSE

§7.8 之后分子 GPU 核时间约 82% 在 cuSPARSE `csrmv_v3_transpose_kernel`（1200 次/5 迭代、11.59 s）；它对每个 RHS 完整遍历 CSR(A)，不知道 ratio 中的零。而对合法输入 `p_i=0 ⇒ ratio_i=0`，五床原始输入 74–90% 的 bin 为零、K=2 配对约 62% 的 nnz 工作两路 ratio 均为零，这些行对反投贡献恒为 0。8 月 Phase-B 拒绝过"自定义原子 2-RHS 转置"，但那是在无零结构的 Sampling 数据上比较的（那里它 0.7× 慢于 cuSPARSE）；零行跳过是新的杠杆，因此重新以真实 chunk 做只读基准，再决定是否进生产。

**核**：`compact_csr_spmvt_pair(values, rowPtr, colIdx, x0, x1, y0, y1, rows)` 每行一个 block，先读两路 ratio，两路均为 0 立即返回（不读任何矩阵字节）；仅一路非零时只对该路做原子累加；其余与正向 pair 核相同的 float4/int4 对齐行遍历，`atomicAdd(y_k + col, a·r_k)`。输出为累加语义，调用方先 memset。`compact_csr_spmvt_quad` 为四路版本，仅供基准比较。两核使用绝对 nnz 下标，因此也可像 packed 正向核那样以 `rowPtr + angle·rpa` 偏移作用于按角度子区间（CTAC 路径未改，见下）。

**只读基准**（`tools/transpose_bench.cpp` 新增 `scatter` 模式，`spect_transpose_bench <matrix> 0 40 5 <zeroFrac> scatter`；真实 Lu-177 208 keV chunk 0，17280 行 × 3168000 列、730,587,994 nnz，40 RHS，RTX 4070 Ti SUPER；zeroFrac 为每个 (row, RHS) 独立置零概率，`relL2` 对 cuSPARSE 同输入结果）：

| zeroFrac（单 bin 零率） | 两路均零 pair 占 nnz | cuSPARSE 逐 RHS T / ms | pair 散射+跳过 / ms | 加速 | quad 散射+跳过 |
|---:|---:|---:|---:|---:|---:|
| 0（稠密） | 0% | 377.7 | 529.9 | **0.71×** | 0.73× |
| 0.15 | 2.2% | 373.7 | 446.2 | 0.84× | 0.87× |
| 0.30 | 9.0% | 370.3 | 369.2 | 1.00× | 1.04× |
| 0.40 | 16.0% | 379.1 | 326.0 | 1.16× | 1.21× |
| 0.55 | 30.2% | 380.2 | 250.4 | 1.52× | 1.59× |
| 0.70 | 49.0% | 370.2 | 168.3 | 2.20× | 2.28× |
| 0.80 | 63.9% | 378.3 | **118.9** | **3.18×** | 3.15× |
| 0.90 | 80.9% | 377.2 | 59.4 | 6.35× | 5.23× |

所有变体 relL2 ≤ 1.2e-7（fp32 原子求和顺序）。稠密时慢 0.71× 与 Phase-B 结论一致；盈亏平衡在单 bin 零率约 0.30；真实五床（74–90%）落在 3× 以上区间。pair 与 quad 在高零率下 pair 更优（跳过条件更松），低零率 quad 略优但那里本就应留在 cuSPARSE，故生产只用 pair。

**生产接入**（`executeColMajorBatchSpMVLocked` 新增 `zeroRowScatterTranspose`，仅 `executeCompactBedBranchBatchDeviceRatioSpmvTAccumulateFloat` 的分子反投传入）：投影上传时（`ensureCompactFullProjectionUploaded`，每床一次）统计精确零 bin 比例；每次分子调用取本 pass 各床均值，`compactScatterBackwardPreferred(zeroFrac, batchSize)` 在 ≥ **0.40**（`kCompactScatterBackwardZeroFractionMin`，留出 0.30 盈亏点以上 1.16× 的余量）时选散射：memset `dFullBatchOutput_` 后按 (b, b+1) 逐对启动，奇数余量仍走 cuSPARSE；否则整条路径与 cfffea5 完全相同。环境变量 `SPECT_CSRB_SCATTER_BACKWARD=0/1` 强制任一路径，用于同一 exe 的 A/B；首次调用打印 `[NOTE] compact numerator backward=… zero_bin_frac=…`。灵敏度、独立 forward、CTAC（按角度 cuSPARSE 转置）语义不变。跳过判据是 ratio 本身为 0，与零率估计无关，因此选择只影响速度不影响结果。

**单元门**：`CompactScatterBackward.ZeroFractionThresholdSelectsPath`（纯函数）；`MultiBedCompactNumeratorTest.ScatterBackwardMatchesCusparseAndDense`（70% 零 bin 的两床玩具投影，强制散射与强制 cuSPARSE 各跑一次，均对 dense oracle ≤1e-4，两者互差 ≤1e-6·max）。多床分子既有 8 项测试全部通过。

**五床 MLEM50 门**：冻结 A = `20260905+cfffea5`（`build/verify-vs`，SHA `69fb2d28…a5ec`），B = 本候选 `20260909+3b69b36-dirty`（`build/metadata-compat`，SHA `3684e4ea…d168d`；两树 CMake 配置一致：`/O2 /Ob2 /DNDEBUG`，sm 86/89/120）。协议同 §7.8：≥42 GiB 可用并稳定 60 s、无其它 SPECT_recon、GPU ≤2.5 GiB／≤20%（桌面本底 9–11%）且无外来计算进程，各一次 it3 预热（各树生成五床 denom cache `auto_338348f9b2178b4a`），再 A-B / B-A / A-B 三组 MLEM50，relL2 对首个 A primary ≤1e-5、三组配对核心收益 ≥5%；另加一次 `SPECT_CSRB_SCATTER_BACKWARD=0` 的 B 运行隔离机制。驱动与证据 `recon_out/lu208_scatter_backward_20260909/`（`run_gate.py run|summarize`，`summary.json`，逐运行 `console.log`／`result.json`／1 Hz `samples.csv`，内含本进程工作集、整卡显存、系统可用内存和外来 GPU 进程名）。

本机是共享主机：另一用户作业（PET 归一化 `gpu_px_direct.exe`、约 20 GiB 的 `compare_direct_cdf.py`，以及每隔数分钟启动约 2 s 的 `super_cuda.exe`）在测量期间间歇运行。驱动把"外来 GPU 计算进程出现 >5 个 1 Hz 采样，或系统可用内存 <4 GiB"的运行标记为污染并自动重跑；被污染的尝试全部保留在证据目录但不进入统计。最先一次 A 与 `gpu_px_direct` 同时运行得到 354.78 s（`A1_contaminated_gpu_px/`），直接说明为何需要这条规则；A2 首两次（144.438／147.708 s，各 2 s 外来采样）、A3 首次（146.190 s，7 s）、B3 前三次（76.045／75.333／75.198 s，8／1／1 s）和首次 `B_scatter_off`（147.089 s，11 s）也一并保留，其中 ≤5 s 者数值上与干净运行一致，取零外来采样的尝试进入正式表。

| 三组 MLEM50（执行顺序 A1 B1 B2 A2 A3 B3） | A：cfffea5 核心 / s | B：pair 散射 核心 / s | 核心减少 | primary relL2 vs A1 |
|---|---:|---:|---:|---:|
| A1 → B1 | 144.606 | **73.398** | **49.24%** | 7.7969e-8 |
| B2 ← A2（A2_r3） | 144.229 | **73.565** | **48.99%** | 1.3102e-7 |
| A3（A3_r2） → B3（B3_r4） | 147.816 | **74.760** | **49.42%** | 1.4366e-7 |

核心中位数 **144.606 → 73.565 s（−49.1%）**，整命令中位数 **190.310 → 117.423 s（−38.3%）**，B 整命令范围 115.969–118.343 s；A 三次 144.229–147.816 s 与 §7.8 的 144.8 s 基线一致。三组配对收益均 ≥5%；全部 primary relL2 ≤1.44e-7 ≤1e-5（fp32 原子求和顺序）。**核心 120 s 目标首次通过（中位 73.6 s），180 s 下限通过。** 同一 B exe 强制 `SPECT_CSRB_SCATTER_BACKWARD=0` 得 **143.162 s**（relL2 1.59e-7），与 A 同级，因此全部收益归于反投核替换，而非编译差异。六次正式运行均为完整默认主机缓存、denom hit、无污染；B 日志均为 `backward=scatter_pair_zero_skip zero_bin_frac=0.802396`（五床精确零 bin 均值 80.24%，与 §7.6 的 74–90% 描述性统计一致）。资源：本进程工作集峰值 A 33.213 GiB、B 33.207–33.411 GiB；整卡采样显存峰值双方 14.0–14.4 GiB，没有新增矩阵或掩码副本（散射仅 memset 既有 `dFullBatchOutput_`）。

三迭代预热（诊断，不入门）：A 10.005 s → B 5.403 s。CTAC 共享调用方守护：Lu-177 Derenzo 床 1 + Lu208 CSRB + 208.4 keV 水柱 μ 图，MLEM it5，A/B 主图 relL2 **3.1327e-8**，核心 10.851／10.906 s 相同，B 未出现散射日志（CTAC 分子仍走按角度 cuSPARSE 转置），见 `csrb_ctac_guard/validation.json`。单元套件 286 通过／4 既有跳过。

执行 `completed`、本机同输入证据 `validated`、源码采纳 `merged`（`e88b135`）。

### CSRB CTAC 逐角度转置复用同一核（2026-09-09，用户："全零行本身就应该跳过，继续将同类核应用于 CSRB CTAC 逐角度转置"）

`csrbAngleSpMVLocked(..., zeroRowScatterTranspose)`：op=T 且启用时先 memset 该角度的 `[branch][nvox]` 输出（8×3.168M×4 B ≈ 101 MB／角度），再以 `rowPtr + angle·rpa`、`rows = rpa`（1152）按 (b, b+1) 启动 `compact_csr_spmvt_pair`——与 packed 正向核使用子区间的方式相同，无需零基 row pointer 副本。只有 `executeCompactBedBranchBatchAttenDeviceRatioSpmvTAccumulateFloat` 的分子反投传入该标志，判据仍是 `compactScatterBackwardLocked`（各床精确零 bin 均值 ≥0.40，`SPECT_CSRB_SCATTER_BACKWARD` 可强制）；CTAC 灵敏度（全 1／行权重，无零）继续逐 RHS cuSPARSE。单元门：`CtacFactorisationMatchesCsrbAndIsAdjoint` 新增 60% 零 bin 两床 CSRB CTAC 分子，强制散射／强制 cuSPARSE 各对 host oracle ≤1e-5、互差 ≤1e-6；全套 286 通过／4 既有跳过。

真实数据 A（cfffea5）／B（本候选）各一次、单床 MLEM it5、Lu208 CSRB、208.4 keV 水柱 μ 图（μ 图对小鼠无物理意义，只作数值与时间对照），`-cyl 30,21.6` 使之走 host input/update 路径：

| 输入 | 零 bin 比例 | B 所选路径 | 核心 it5 A → B | `cuda.cusparse_bwd` 总计（含灵敏度） | primary relL2 |
|---|---:|---|---:|---:|---:|
| `derenzo_lu_6s/Bed1` | 2.9% | cuSPARSE（自动） | 10.872 → 10.827 s | — | 3.1869e-8 |
| `mouse_normal_lu/Bed3` | 73.6% | pair 散射 | **10.823 → 9.617 s（−11.1%）** | 2.990 → 1.673 s | 3.4872e-8 |

正向（1.55／1.57 s）与 transform-add（0.175／0.166 s）不变，因此差值全部来自按角度反投；CTAC 路径的其余成本（host 输入准备／更新、逐角度缩放与 ratio、上传等待）不受影响，所以整体收益小于纯 CSRB 分子。每角度 memset 约 0.1 s／遍（54 GB/遍）已计入反投计时；若以后需要，可让 `compact_transform_backward_add_batch_atten` 读后置零来省掉这一独立 pass，当前不值得单独立项。证据 `recon_out/lu208_scatter_backward_20260909/csrb_ctac_scatter{,_mouse}/`（`run_guard.ps1`、A/B `stdout.log`、`validation.json`）。

剩余方向：反投现在约为原来一半，正向 packed-2（cfffea5 已跳零对）与反投量级接近，五床分子的下一步应重新以短 Nsight 划分；稠密投影（Tc-99m 高计数）保持 cuSPARSE，阈值 0.40 若要下调需在 0.30–0.40 区间补实测。不重开已拒 SpMM／CSC 真转置／FP16。

## 7.10 2026-09-09 子集（OSEM / 分块 BSREM）调研结论与设计决定

本节保留9月9日调研与实施前设计。当前实现和发布门以[9月13日验证报告](2026-09-13-subset-release.md)为准：已支持角度子集N1/N5/N10及递减调度，CLI为`-it 迭代次数:子集数`，RDP/TV逐子更新。下文原拟MLEM50形态门不再代表Derenzo收敛验收；用户补充其通常需约600–700次MLEM进展，现有子集收敛比较使用冻结MLEM1000参照。相同子更新数只是记账，不证明算法等价或加速倍数。

用户问题：投影数据如何划分子集、能否加速、子集矩阵如何加载、子集间如何"融合"；主要算法族 MLEM / OSEM / BSREM + RDP / TV。以下为文献与专利检索（PMB、IEEE TMI、PETRIC 2024、CASToR 文档、Google Patents）后的结论；专利部分只是工程认知，不是法律意见。

### 调研结论

1. **加速机制与上限。** OSEM 的加速比约等于子集数（Hudson & Larkin 1994；Branderhorst 2010 复述 "x 次 OSEM × y 子集 ≈ xy 次 MLEM"）。收益来自完整迭代数减少，不是每次迭代更快；本机一次完整 pass 仍要传 33.5 GB（≈1.1 s，已占核心 1.47 s 的大部分），所以子集不会把单次完整迭代压到 1.1 s 以下。
2. **子集单元。** Utrecht/MILabs 的 POSEM（Branderhorst, Vastenhouw, Beekman, PMB 55:2023, 2010）指出：对**静止聚焦多针孔**（U-SPECT-II，75 针孔、投影截断）按"视角"分子集时 30 个以上子集即出现严重伪影（骨扫描均值偏差 24%），原因是体素位置相关的灵敏度破坏了 Hudson–Larkin 的"子集平衡"；他们改为把探测器像素按规则图案分到子集（POSEM），128 子集仍接近 MLEM。同一文中，15 个视角子集（每子集 5 个针孔投影、跨 5 环、相邻环角距≈90°）与 MLEM 的均值偏差仅 1.1%。我们的系统是 **360 角旋转采集**，每个 compact 角经 4 重 xy 对称覆盖 4 个相隔 90° 的全角度，交错子集天然是全 360° 均匀采样：N=6/10/15 时每子集 60/36/24 个全角度，远高于 POSEM 中已表现良好的"每子集 5 视角"。因此首选**按 compact 角交错的视角子集**；像素子集作为后备（可通过一次离线行重排得到连续存储，不需要新核）。
3. **低计数与子集数。** 同一团队的 CR-OSEM（Vaissier 2012）与 SR-OSEM（Vaissier, Beekman, Goorden, PMB 61:4300, 2016）说明：子集多、投影计数少时会出现"活动被错误抹除"的伪影；他们按体素自适应地降低子集数。五床 Lu-177 有 74–90% 零 bin，属低计数，因此 N 先取 ≤ 10，并以 Derenzo/小鼠对 MLEM 的偏差指标（Branderhorst 表格式：剖面均值/最大偏差占 MLEM 峰值百分比）为门。
4. **收敛型子集算法。** Ahn & Fessler（IEEE TMI 22:613, 2003）证明 modified BSREM-II / relaxed OS-SPS 全局收敛：缩放函数 D_j = M·x_j/p_j（p 为全局灵敏度，M 子集数，M=1、α=1 退化为 EM），每步后投影到 [ε, U−ε]，松弛 α_n = α_0/(γn+1) 满足 Σα_n=∞、Σα_n²<∞；正则项按 1/M 计入每个子集目标（其式 (27)，比只在部分子集加正则更稳）。文中明确："早期子集多为好，后期子集少为好"，并提出"跑几次 OS 再切换到非 OS 算法 / 逐步减少子集数"作为等价的收敛手段。GE Q.Clear（Ahn et al., PMB 60:5733, 2015；Ross 白皮书）即 modified BSREM + RDP：28 子集、约 25 次迭代跑到实用收敛，2 次 OSEM 预热。
5. **"融合"类算法。** COSEM/E-COSEM（Hsiao, Rangarajan, Gindi, PMB 49, 2004）存每子集完全数据像后再合成，收敛且无需松弛计划，但每床多 N 个体积；PETRIC 2024（RDP 正则 PET）冠军 MaGeZ 用带先验预条件的 SVRG（Ehrhardt/Schramm/Kereta，Frontiers Nucl Med 2025），约 25 子集，步长 3→1→0.5，cofactor 确定性子集顺序，SVRG 需每 epoch 一次额外全梯度 pass（对我们是额外一次 33.5 GB 传输），SAGA 则需存 N 个子集梯度。结论：这类方法在传输受限的本机不是第一步；先用已有 BSREM-II 的松弛 + 子集递减实现收敛，SAGA 留作后续。
6. **工程实现参考。** CASToR 直方图模式为每个子集**现算**灵敏度并允许每次迭代不同子集数，list-mode 则用全局灵敏度/子集数（仅在子集平衡时正确），并警告直方图事件顺序需打乱以免伪影。Branderhorst 也指出截断投影下某些体素在某个子集中无采样，需处理 S_k=0。因此我们存 **N 份精确的 S_k**（不用 S/N），S_k=0 处跳过更新。
7. **专利。** OSEM（1994 论文）、RAMLA（1996）、BSREM（De Pierro 2001）、松弛 OS（2003）、RDP（Nuyts 2002）、POSEM（2010）、COSEM（2002–04）均为公开文献。检索到的 GE 专利是 PL 重建的特定附加项（如 US9256967B2 PET PL 部分容积校正），MILabs 专利为聚焦针孔硬件（US7145153、US8067741）；未发现覆盖"角度交错子集 + 松弛 BSREM + 子集递减"的有效权利要求。这不是 FTO 意见。

### 设计决定（据此实施）

- **子集**：compact 角交错，N | 90，先支持 N ∈ {6, 9, 10, 15, 18, 30}（N ≥ 6 使一个子集切片不超过现有设备槽容量）；顺序用位反转/最大角间隔的确定性序，写入 manifest。
- **加载**：流式单元从 chunk 改为 (chunk, 子集) 切片：每个 chunk 中该子集各角的 nnz 区间（每段 ≈65 MB）按顺序拷入设备槽，host 生成重定基 rowPtr；沿用双槽预取。要求完整主机锁页缓存。ratio / unit-ratio 核增加按 rpa 块的角度表（空指针保持原语义）；正向 packed 核、pair 散射反投核不变。
- **灵敏度**：`computeCompactChunkSensitivityCUDA` 加子集限制，N 份 S_k 进分母缓存（key 含 N 与顺序）；CTAC 同样按子集。
- **优化器**：统一为 Ahn–Fessler 形式 u_j = α_n [M(n_{k,j} − S_{k,j}) − λ_j r_j] / (S_j + λ_j max(c_j,0) + ε)，M=1 时即现有 BSREM-II；MLEM 类（β=0，α=1）用 `-subsets` 调度串（如 `10:3,5:2,1:5`，末段 1 子集即 MLEM）得到与 MLEM50 可比的终点；BSREM+RDP/TV 用固定 N 加现有 α_n 松弛。TV 仍走现有梯度/曲率接口，不引入 SPDHG。
- **门**：单元测试（子集分子对 host oracle；ΣS_k = S；N=1 与 MLEM 一致）；Derenzo OSEM 调度 vs MLEM50 的 Branderhorst 式偏差与杆对比度；五床墙钟；不与 MLEM50 做 1e-5 relL2 门。

## 7.11 2026-09-13 角度子集RC与Lu-177继续验证

OSEM/BSREM角度子集、零子集灵敏度保留规则、每子更新RDP/TV、CLI语义和默认输出已通过独立构建与真实数据门；完整套件300通过、4项外部或opt-in跳过。N1/N10同参数对冻结EXE最大primary relL2为5.635e-8（门1e-5），真实五床RDP/TV生效、有限非负，包内TV启动和29项payload哈希通过。最终包为`dist/release_candidates/SPECT_recon_rc_20260913_subsets_final.zip`；旧包、冻结基线及原收敛记录保留。数值/运行验证与不同对象的最佳画质选择分开。

默认50:10、末次primary与独立Butterworth、关闭Gaussian；生产配置恢复CLI优先，避免全局None覆盖BSREM正则。CSRQ双能窗/双矩阵仍为显式N1实验路径。Derenzo100N1四臂及同数据权重敏感性检查仅属早期证据，不能作为收敛后的物理标定。当前按用户要求优先老鼠SPECT观感和norm资料，CT处理/配准/解剖引导/CTAC留待后续；执行状态、原始命令和图像集中在[当前报告](2026-09-13-subset-release.md)，不在此重复实验台账。

9月13日mouse/tumor正则化后续已完成23臂/83个primary及配置身份核查。当前实用补充预设为joint RDP0.03、300N1和sum208 CSRB RDP0.1、100N10；两者不是同一测量模型，β不跨模型移用。正常鼠与tumor2支持弱RDP的观感取舍，早停MLEM仍有竞争力。Tc99m IQ完成描述性对照，但真实结构/填充ROI未核验，不能用于宣布物理最优参数；完整图像、适用范围与复现见[当前报告](2026-09-13-subset-release.md)。冻结RC与通用默认值未改。

## 8. 当前实现摘要

`Measurement operator (A, att, s, W, b) → Poisson statistics (q, S) → Optimizer (MLEM | BSREM-II)`。CTAC 在按角度的输入与收口阶段施加，背景进入 ratio，行权重只进入 S；BSREM 复用现有梯度核，再做逐体素更新。CSRB 与 CSRQ 各自保留必要的格式内核，不新增统一内核或算子类层次。
