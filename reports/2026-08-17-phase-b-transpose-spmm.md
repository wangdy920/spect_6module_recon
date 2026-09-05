# Phase B.1/B.3 — 转置算子实验记录（SpMM 分组 / preprocess / CSC 控制组 / 64-32 索引）

日期：2026-08-17。矩阵：`test_data/6module/Sampling_Approach`（6 chunk，nnz 3,464,432,327，108×108×160）。
数据：5 床 MDP（`recon_out/ctac_mdp_smoke_20260813/preproc/20260729_mdp/SPECT/Bed1..5.bin`）。
驱动：`tools/run_220_mlem_perf_gate.py --it 3`，denom cache hit，`-vb 1`。
GPU：16 GB（桌面占用约 1.4 GiB）。基线与全部变体同日同机同构建。

## 问题（任务契约）

在 relL2 ≤ 1e-5（同输入）门不变时，外部建议的转置优化（多 RHS SpMM、64/32 索引、
换 cuSPARSE 算法、SpMV/SpMM preprocess、BF16/FP16）能否使 `iteration_core_s` 下降 ≥5%？

## 实验开关（临时，收敛时删除或转正）

`CudaSparseOpsState.h` 中 `SPECT_COMPACT_T_SPMM` 环境变量，默认关闭时生产路径逐字节不变。
批缓冲本身就是按 RHS 连续的 COL-major 切片（ld=向量长度），SpMM 分组**零打包**。
玩具级奇偶校验：`MultiBedCompactNumeratorTest.*` 全部 6 项在每个变体下 PASSED。

## 结果（3-iter，denom hit）

| 变体 | cusparse_bwd (s) | cusparse_fwd (s) | iteration_core_s | relL2 vs 基线 it3 | 判定 |
|---|---|---|---|---|---|
| off（基线：逐 RHS SpMV CSR_ALG2） | 6.244 | 1.762 | 9.781 | — | **保留** |
| alg1:2 | 6.323 | 1.769 | 10.622 | 3.1e-08 | 拒绝 |
| alg1:4 | 5.919 | 1.631 | 9.529 | 5.3e-08 | 拒绝（bwd −5.2%，core −2.6% < 5%） |
| alg1:5 | 5.891 | 1.652 | 9.604 | 8.0e-08 | 拒绝（bwd 最优 −5.7%，core 仍 < 5%） |
| alg1:8 | 15.348 | 1.654 | 18.916 | 4.8e-08 | 拒绝（L2 崖：2.46×慢） |
| alg1:10 | 22.275 | 1.742 | 26.612 | 8.0e-08 | 拒绝（3.57×慢） |
| default:4 | 5.914 | 1.736 | 9.561 | 5.3e-08 | 拒绝（与 alg1:4 等价） |
| default:8 | 15.940 | 1.553 | 19.156 | 6.4e-08 | 拒绝 |
| alg1pre:4（+SpMM_preprocess） | 5.961 | 1.633 | 9.675 | 8.1e-08 | 拒绝（preprocess 无增益） |
| spmvpre（ALG2+SpMV_preprocess） | 6.072 | 1.597 | 9.379 | 1.3e-07 | 拒绝（bwd −2.8%，噪声级） |
| cscn（CSC(Aᵀ) op=N 控制组） | — | — | OOM×2 | — | 不可行（见下） |

relL2 全部 ≪ 1e-5 门；非零来源是原子浮点累加顺序的非确定性（同算法重复跑也非零）。

## 三条硬结论

1. **转置路径是原子散射写主导，不是矩阵读带宽主导。**
   假设"40 次 ALG2 SpMV 重复读 A 40 遍，SpMM 共享读取应大幅提速"被证伪：
   K=4/5 只省 bwd 5.2–5.7%。若 A 读为主导，K=4 应接近 −75% A 流量、bwd 大幅下降。
2. **L2 输出工作集崖在 K∈(5,8] 之间。** 每 RHS 输出切片 = 1,866,240 voxel × 4B ≈ 7.5 MB；
   K=5→37 MB 尚可，K=8→60 MB 后 bwd 从 5.9 s 崩到 15.3 s（2.6×）。与已拒绝的 40 宽
   forward SpMM（298 MiB 工作集）同一机制。
3. **CSC 重解释拿不到免费 gather 路径。** `cusparseCreateCsc`(同一套 dRowPtr/dColIdx/dValues)
   + op=N：`bufferSize` 仅 2.26 MB/chunk，但 Sampling 跑到第 5 个 chunk 访问必 OOM
   （两次复现，浮出点为 prefetch stream-wait；同日其余 9 个变体全部跑通）。
   cuSPARSE 在用户 workspace 之外为 CSC 描述符惰性物化内部转置结构，
   两个 4.6 GB 常驻 A 槽位之下没有余量。

## 逐项对照外部建议

| 建议 | 结论 |
|---|---|
| 多 RHS 合并 cusparseSpMM 转置，"对 70% 瓶颈有明显收益" | 已测，收益仅 bwd ~5% / core ~2.6%（50-iter 外推 ~4%），< 5% 采纳线，拒绝 |
| 64/32 索引省 1/3 流量 | 设备端本就 32/32（`CachedMatrix` int32 row/col），前提不成立，no-op 关闭 |
| 测试 ALG1/2/3/default | SpMM ALG1/DEFAULT 已测同级；SpMV 侧 CSR_ALG3 不支持 op=T；保留 CSR_ALG2 |
| SpMV/SpMM preprocess | 两者已测：alg1pre、spmvpre 均无增益 |
| BF16/FP16 | 维持既有拒绝（FP16 采样 relL2 3.677e-4 > 1e-5 门），未重开 |

## B.2 真 CSR(Aᵀ) 微基准（2026-08-17，`spect_transpose_bench`）

新工具 `tools/transpose_bench.cpp`（目标 `spect_transpose_bench`，只读矩阵，不碰生产路径）：
加载单个真实 chunk，主机计数排序转置成 CSR(Aᵀ)，双向对比。chunk 0 与 chunk 3 结果一致。

环境事实（本日核实）：GPU = RTX 4070 Ti SUPER（sm_89，16 GB，L2 48 MB，PCIe gen4 x16）；
主机内存 64 GB；chunk 行数 17,280（40 角 × 432 compact 射线），列数 1,866,240，nnz ≈ 577M。

### 反投影方向（rays→voxels，70% 瓶颈），chunk 0，40 RHS 整批，5 次重复取最优

| 变体 | ms/批 | 加速比 | relL2 |
|---|---|---|---|
| A·op=T SpMV CSR_ALG2（现产线语义） | 327.3 | 1.00× | ref |
| Aᵀ·op=N 逐 RHS SpMV（ALG2/ALG1） | 297–298 | 1.10× | 4.0e-07 |
| **Aᵀ·op=N SpMM DEFAULT K=8** | **208.0** | **1.57×** | 4.0e-07 |
| Aᵀ·op=N SpMM DEFAULT/ALG1 K=40 | 210.6 | 1.55× | 4.0e-07 |
| Aᵀ·op=N SpMM ALG3 K=40 | 281.0 | 1.16× | 4.9e-07 |

gather 侧大 K 不塌崖（输出流式单写，无输出工作集复用需求）；chunk 3 复测 1.57×。

### 正投影方向（voxels→rays）：单份 Aᵀ 替换方案被否决

| 变体 | ms/批 | 加速比 |
|---|---|---|
| A·op=N SpMV ALG2（gather，产线为更快的 packed-K 定制核） | 297.0 | 1.00× |
| Aᵀ·op=T SpMV ALG2 | 697.2 | 0.43× |
| Aᵀ·op=T SpMM K=8 / K=40 | 496.0 / 465.8 | 0.60× / 0.64× |

否决依据（含定制内核上限估算）：散射正投每 40-RHS 批需 nnz×40 ≈ 23G 次原子加，
即使 shared-memory 私有化（射线域仅 69 KB/RHS）也受原子吞吐下限 ~0.7 s+/chunk 制约，
高于现 gather 正投 ~0.1 s/chunk 一个量级。**正投必须留在 A（gather）**；
不再为此立项定制散射内核。

### 收敛路线 R1：双矩阵分工（下一个工作包）

- 正投：A，packed-K 定制核，不动。反投：Aᵀ，`cusparseSpMM` op=N K=8（1.57×）。
- Aᵀ 来源：**启动时从 A 在主机内存转置**（实测 5.2–5.4 s/chunk 单线程含原子游标并行散射，
  6 chunk ≈ 32 s 一次性），不产生磁盘产物、不改 A 的盘上契约；主机常驻 +26.4 GB
  （64 GB 机器：A 缓存 22.3 GB + Aᵀ 26.4 GB + 床数据/系统，紧但可行；
  必要时下调 A 缓存上限换 NVMe 重读）。
- 设备侧：每 chunk 访问需 A_i（正投）与 Aᵀ_i（反投）先后在卡上；H2D 每迭代
  约 +27.6 GB（gen4 x16 有效 ~20 GB/s ≈ +1.4 s/iter），必须藏进计算流；
  槽位由 2×A 改为 3×4.6 GB 轮转 {A_i, Aᵀ_i, prefetch}。
- 预期收益（若 H2D 完全隐藏）：3-iter bwd 6.244→~3.98 s，core −23%；
  50-iter core 126.3→~94 s（−25%）。即使隐藏一半开销仍 >10%，远超 5% 采纳线。
- GO/STOP：产线 A/B（实验开关，默认关）3-iter 后 50-iter，relL2 ≤ 1e-5 对 126.261
  keep 卷；`iteration_core_s` 净收益 ≥5% 采纳，否则拒绝并保留 B0。
- 复现：`build\verify-vs\bin\Release\spect_transpose_bench.exe test_data\6module\Sampling_Approach <chunk> 40 5`。

### R1 上传预算实测与集成设计要点（2026-08-17 补）

本机实测 H2D：**pinned 22.2 GiB/s，pageable 12.9 GiB/s**（bench 内 2.3 GB values 数组，3 次取最优）。

预算表（50-iter 尺度，提速后计算 ≈ 1.89 s/iter）：

| 方案 | H2D/iter | 拷贝耗时（pinned） | 迭代 ≈ max(计算, 拷贝) | 净收益 vs 2.53 s/iter |
|---|---|---|---|---|
| 现状（仅 A，部分驻留） | ~17 GiB | 0.77 s（藏住） | 2.53 s（计算限） | — |
| R1 全上传（无驻留对） | ~43–52 GiB | 1.9–2.3 s | 1.9–2.3 s（拷贝限） | 9–25% |
| R1 + 1–2 个（半）chunk 对驻留 | ~34–43 GiB | 1.5–1.9 s | ~1.9 s（计算限） | **~20–25%** |

设计决定：
1. **半 chunk 粒度轮转**（每半 2.15 GiB；槽位 A_h、Aᵀ_h、2×prefetch ≈ 8.6 GiB + 缓冲，
   VRAM 有余量）。全 chunk 三槽位方案 13.8 GiB + 缓冲压在 15.1 GiB 空闲的刀刃上，
   cscn 的 OOM 已实证无此余量，否决。
   CSR 行域连续切半零成本（现成 `buildContiguousSubsetCSR`/range 机械）；
   反投两半各一次 SpMM，第二次 beta=1 累加；ratio 批按行域切片即可。
2. Aᵀ 半 chunk 主机侧必须 pinned（22.2 vs 12.9 GiB/s 是藏住与藏不住的差别）。
3. 尽量保留 1–2 个（半）chunk 对的 GPU 驻留以把拷贝压回计算之下。
4. 一切在默认关的实验开关后面；采纳门槛不变（≥5% net，relL2 ≤ 1e-5）。

### R1 产线集成实测与最终裁决（2026-08-17，`SPECT_COMPACT_T_SPMM=csrt`）：**拒绝**

实现：主机首访构建 CSR(Aᵀ) 两半（指纹缓存、OpenMP 原子游标散射、≥1 MB 才 pin），
侧流 JIT 上传两半设备槽位，反投两半 `cusparseSpMM` op=N（第二半 beta=1），
正投/A 路径不动。数值门通过，但系统级性能被上传否决：

| 指标（Sampling 3-iter，denom hit） | 基线 off | csrt |
|---|---|---|
| `cuda.cusparse_bwd` 设备时间 | 6.07–6.28 s | **3.70–3.89 s（1.6×，微基准兑现）** |
| `cuda.compact_matrix_upload` | 0.24 s / 10 次 | **12.2 s / 45 次（阻塞主机）** |
| `iteration_core_s` | 9.91–10.04 s | 19.8–21.0 s（**2× 回退**） |
| prepare（Aᵀ 主机构建一次性） | 27.2 s | 55.2 s（+27.4 s） |
| relL2 vs 基线 it3（MET_DOUBLE 卷） | — | **4.4e-08 ✓** |

死因是物理的，不是实现缺陷：
1. **pinned 预算墙**：A 缓存 22.3 GB + Aᵀ 26.4 GB ≈ 49 GB 需全 pinned 才能达到
   22.2 GiB/s 并异步隐藏；64 GB 主机不可能全 pin，实测大半 `cudaHostRegister`
   失败回退 pageable，`cudaMemcpyAsync` 对 pageable 源同步阻塞主机线程，
   拷贝全部暴露在迭代路径上。
2. **PCIe 地板算术**：双矩阵全流动 ≈ 53.7 GB/iter ÷ 22.2 GiB/s ≈ 2.4 s/iter 纯拷贝
   地板，而基线 50-iter core 仅 2.53 s/iter——即使拷贝完全隐藏、驻留 0 对，
   理论上限也只剩 ~3.5%，低于 5% 采纳线。预算表中 9–25% 的估计错在假设
   "计算 1.89 s/iter 可藏 2.3 s/iter 拷贝"（max 运算写反）与半驻留可达性。
3. 50-iter 门无需再跑：3-iter 实测 + 上述算术已封死上限。

2026-09-05 收敛更新：`csrt` 及其余被拒的 `SPECT_COMPACT_T_SPMM` 变体、缓存与上传支路已删除。原代码可从 `a3239d7` 检出，以下测量和历史复现命令保留为拒绝证据。

当时保留物：
本轮顺手修复并保留两项生产路径改进：
- `uploadIoCsrViewAsync` 的 rowPtr 走 pinned 暂存缓冲（消除 pageable 异步拷贝
  阻塞；基线回归验证 core 10.04 s ≈ 9.91 s 噪声内，relL2 9.3e-08 为原子噪声）；
- `releaseCompactExecution` 清理 csrt 态；csrt 自有 Aᵀ 向量的 pin 加 1 MB 门槛。

**同日撤销的一处 csrt 附带改动**：`tryRegisterIoCsrViewHost`（对 loader 拥有的
io CSRView 数组做 ad-hoc `cudaHostRegister`）在 220 矩阵生产跑上触发 FATAL
`cudaMemcpy invalid argument`——host 缓存仅驻 3/6 chunk，loader 逐出并重分配
缓冲，遗留的页注册与新分配区间重叠。HEAD 从未在此文件用过 `cudaHostRegister`；
Sampling 全天未触发只是分配器地址复用的侥幸。该注册已整体删除（loader 自有
pinned 预取路径覆盖带宽需求），删除后全套件 267/271 通过。教训：不得注册
生命周期不属于 sparse state 的缓冲。

结论：**反投 1.57× 的设备收益真实存在，但被 PCIe 双矩阵流动成本吞没。
B 阶段算子路线全部收敛：保持 B0（A 单矩阵，op=T ALG2 反投）。**
后续算力优先级转向减少全矩阵扫描次数（P1 PNCG / P2 子集），而非单次扫描提速。

## 历史复现命令（须先检出保留实验代码的 a3239d7）

```powershell
# 基线（开关关闭）
python tools\run_220_mlem_perf_gate.py --it 3 --stage tspmm_base
# 变体（示例 alg1:4；relL2 对基线 it3 体积）
$env:SPECT_COMPACT_T_SPMM = "alg1:4"
python tools\run_220_mlem_perf_gate.py --it 3 --stage tspmm_alg1_4 `
  --baseline-mhd <基线目录>\recon_it3.mhd
# 玩具奇偶校验
$env:SPECT_COMPACT_T_SPMM = "alg1:4"
build\verify-vs\bin\Release\spect_tests.exe --gtest_filter=MultiBedCompactNumeratorTest.*
```

原始输出目录：`recon_out/perf_220_mlem_tspmm_*_20260817_*/{console.log,result.json}`。

## 验证清单

- 已验证：全部变体玩具级 6 项数值测试通过；Sampling 3-iter relL2 ≤ 1.3e-07（门 1e-5）；
  分段计时来自 `[PROFILE]` 逐段累计（720 次调用）。
- 未验证：50-iter 全程计时（外推 ~−4% core，低于采纳线，故省略）；
  BF16/FP16（维持既有拒绝，未新测）。
- 环境注记：基线第一次尝试与 cscn 均在 prefetch 等待点 OOM；基线重试通过而 cscn
  两次必现，故 cscn 判定为路径自身不可行，基线首失败判定为瞬时显存压力。
