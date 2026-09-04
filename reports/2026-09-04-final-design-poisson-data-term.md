# 最终设计：Poisson 数据项引擎 + 分离的优化器/先验/物理修正（2026-09-04）

目标支持范围：`MLEM` 与 `BSREM-II + RDP`，subset = 1，CTAC，Lu-177 双窗（113/208 keV）联合重建，多床。
约束：**CSRB 是生产路径与受保护基线，CSRQ 是实验路径**（用户决定 2026-09-03）。任何一步都必须让 CSRB Derenzo MLEM 回归（`recon_out/u16sqrt_value_format_20260902/derenzo_fp32/recon_it50`，relL2 门限 1e-6，无背景/无 CTAC 时期望位一致）继续通过。

本文在 GPT 5.6 PRO 方案（用户 2026-09-04 提供）的基础上，用代码事实做了修正。第 1 节是采纳/修正清单，第 2–7 节是最终方案。

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
│      weight s_k, rowWeights W_k | null, attTable[bed] | null (u16, 每能量一张), attResidency }
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

图像更新层（设备，fp64 图像，xCurrent 只读 / xNext 只写）：

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

G = 1 时暂存回到现有的 101 MB，但前向每次只有 2304 行（不到一个 wave），预计前向慢 ~1.5×；G = 8 是默认，显存紧张时降到 4。
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
6. 测试：BSREM oracle（单元 + Derenzo it1–5 relL2 ≤ 1.1e-8）。RDP 有限差分与双窗 penalty-once 测试留待 P2 一并补。**发现：** `algo_bsrem.conf`（δ=0.01）的迭代对 fp32 累加噪声每迭代放大 2–3×，同一二进制两次运行 it20 相差 ~1e-3，BSREM 门只能取早期迭代。

**P2 — CTAC 因子分解**（CSRQ 侧已完成，2026-09-04）
7. ~~`pack_branch8_atten`、按角度组的 x̃/y 布局、`atten_transform_backward_add`~~ → 执行时取 **G = 1**（按 compact 角度逐个处理）：`csrq_pack_branch8_atten` 在 pack 时乘 `att[fullAngle(θ_c,k), M_k(v)]`，`compact_transform_backward_add_batch_atten` 在收口时乘同一因子；forward/backward 核只是按行子区间（1152 行）启动，核本体不变。选 G = 1 的原因：backward 的两分支 L2 驻留论证要求 y 目标 ≤ 25 MB，G > 1 会把它放大 G 倍；pack / 收口的总量与 G 无关；启动数 720/迭代/床可忽略。表驻留：每 (chunk, 床) 只暂存该 chunk 触及的 60 个全角度切片（≈0.38 GB，同步 `cudaMemcpy`，每床每迭代整表过一次 PCIe ≈ 0.25 s），未进 plan；与计算重叠留给 P3。**CSRB 侧未改**：CSRB 是受保护生产路径，其现有 CTAC 核作为参考实现保留（§6 门已用它通过）；是否用因子分解形式替换（预计 44 s/iter → ~3 s/iter，两床）留作用户决定。真实 μ 表数据：MDP 小鼠 2h（Tc-99m，CT/SPECT DICOM `dcmdata/mouse_samples/20260729_SPECT_mouse_MDP/20260729_162015`，投影 `E:\...\mdp_seam_ab_20260730\preproc_2h`），`prepare_ct_mu_map.py --profile recon220 --beds 2,3 --z-bed-flip`；Lu-177 在本机只有 CT 原始投影（`data_platform1.mhd` 是 1536×1536×720 的投影栈，不是重建体），等厂商 CT DICOM。
8. ~~伴随测试 + CSRQ-vs-CSRB CTAC 数值门~~ 通过：单元测试（随机 24 角矩阵、2 chunk、两张逐床表）CSRQ vs CSRB 全部 ≤ 1e-5、伴随 ≤ 1e-5；真实数据 it20 relL2 1.4e-7（分母 1.1e-7，零模式一致），BSREM-II+RDP it1–5 ≤ 1.7e-7；两床 1.02 s/iter（核 0.48 s + 表上传 0.5 s）vs CSRB CTAC 44 s/iter。

**P3 — 性能收尾**（2026-09-04 完成本机可做的部分）
9. ~~WDDM 规划修正与降级重试~~ 做法：不改估计（WDDM 下 `cudaMemGetInfo` 没有可靠替代，NVML 另算依赖），而是把"放不下"变成可恢复错误 `CsrqDeviceMemoryError`：`warmCompactChunks` 捕获后把本次调用已常驻的矩阵降级为流式（清 lease、重读进 pinned 主机）再重试失败的矩阵。用 4 GiB 的 torch 进程复现今早的场景：208 常驻 → 113 "even streamed" 放不下（空闲 1197 MiB）→ 208 降级 → 113 常驻 → 跑完（原来 FATAL）。~~表流式与计算重叠~~ 做法：`cudaHostRegister` 页锁定调用方的逐床表（2.2 GB 用 0.08 s，失败则退回同步拷贝），切片经上传流进两个设备槽，与上一 (chunk, 床) 的核用 ready/free 事件双缓冲；逐矩阵的切片索引计划只建一次，遍内无同步拷贝。两床 CTAC 1.02 → **0.60 s/iter**，数值不变。多床时打包/累加暂存本来就共享（`dCsrqX_/dCsrqY_/dCsrqRatio_` 与两个切片槽），无需再改。**未做**：5060 Ti 上 `red.global.v4.f32` 向量原子（需要现场 GPU）；CTAC 切片槽（2 × 0.38 GB）未计入 `beginCsrqWarmup` 的 scratch 估计，现由 2 GiB 保留吸收。

每个 P 步单独提交、单独跑 §6 对应门；P0 完成前不开始 P1，P2 依赖 P0 的 ratio/行权重契约。

## 7.1 后续（2026-09-04 晚，用户："1、2、3 都可继续"）

- **双窗 CTAC（原开放项 2）— 完成。** 不做逐光电峰 μ 表：现有两条 μ 图分支（auto-mu 水膜、`prepare_ct_mu_map.py` 的 HU 转换）都是 μ = μ_water(E)·body 的单材料模型，能量只以标量进入，故 att(E) = att(E_map)^(μ_w(E)/μ_w(E_map))。一张逐床表服务所有能窗（5 床双窗省下 22.8 GB 主机表与第二次线积分），指数在 `csrq_pack_branch8_atten` / `compact_transform_backward_add_batch_atten` 内用 `__powf` 施加（指数为 1 时跳过）。NIST XCOM 液态水表进 `core/AttenuationCoefficients.h`（核素预设本就是它的插值）。CLI `-ctacWindowEnergies`、`-ctacMapEnergy`；分母缓存与 run manifest 记录指数。局限：骨按水缩放；精确到骨需第二张"骨过量"线积分表（att = exp(−a_k L_w − b_k L_b)），主机内存与两套表相同，只在 ≥3 窗时才划算，未做。
- **CSRB 侧移植（原开放项 1）与 sm_120 向量原子（原开放项 3）**：见下方各自条目。

## 8. 一句话

把 `Projector → MLEM numerator → MLEM update` 改成 `Measurement operator (A, att, s, W, b) → Poisson statistics (q, S) → Optimizer (MLEM | BSREM-II) → Penalty (None | RDP)`；投影核每种格式只留一个实例，CTAC 放到 pack 与收口两个小核里，背景与行权重放进 ratio，优化器与先验合成一个 stencil 核。CSRB 的 SpMV 核本体全程不动。
