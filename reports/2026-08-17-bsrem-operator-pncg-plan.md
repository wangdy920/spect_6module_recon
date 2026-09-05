# Local plan: full-scan BSREM-II, operator T, then optional PNCG

Date: 2026-08-17. Provenance: user-accepted priority from external review, rewritten against this repo’s measured path and `.agent/CONSTRAINTS.md`.  
Status: **plan only**. No code, no A/B, no Derenzo 500-iter run. This file is the living ledger until a later freeze.

**中文摘要**  
主线仍是全量（subset=1）BSREM-II + 可选 RDP。PNCG 不替换主线。子集是 P2。算子内部顺序改为：cuSPARSE **2-RHS 转置 SpMM** → 可选真实 **CSR(Aᵀ)** → 确认 64/32（本地很可能已无收益）→ BF16/FP16 继续后移。先把 MLEM / β=0 BSREM / RDP-BSREM 三条臂拆开，并补目标函数与钳位诊断。

## Contract

| Item | Value |
|---|---|
| Exact question | 在不把 PNCG/子集做成主线的前提下，本地下一阶段按什么顺序做、每步比什么、何时停 |
| In scope | 计划、门、与已测事实的冲突表、实验登记 |
| Out of scope | 实现、开 PR、改生产默认、重开 mmap / 第三块常驻 / FP16 / NVRTC |
| Protected operator baseline | Sampling_Approach, 5-bed, denom hit, `-opti mlem -it 50:50 -vb 1`, `iteration_core_s=126.261` |
| Protected numerical keep (operator) | same-input primary `relL2 <= 1e-5` vs frozen MLEM volume |
| Algorithm vs operator | 不得混在同一次改动里（已有约束） |
| Derenzo 500–600 | 设备端目视口径，`diagnostic`；本仓库没有 500 次完整扫描的 keep |
| Gate maturity | 下文 GO/STOP 数字均为 `provisional` 采用停点，不是已标定的科学 FAIL |

External Amdahl (hypothesis, using Sampling 3-iter stage split): T 70.4%, A 20.1%, H2D 2.2%. T×2 → ~1.54× `iteration_core_s`; T×3 → ~1.88×. Not a measured result.

## Adopted local priority

```text
P0  全量 BSREM-II 正确性 + 转置算子
      P0-A  三臂等价 / 诊断 / 钳位与 κ–曲率
      P0-B  转置：2-RHS cuSPARSE SpMM → CSR(Aᵀ) → 索引类型确认
P1  BSREM 预条件 projected-PNCG（shadow，可回退）
P2  M=2/3、按 compact chunk 对齐的物理子集 BSREM（全局 D）
```

算子子顺序（只动分子核，算法不变时）：

```text
2-RHS 转置 SpMM  →  真实 CSR(Aᵀ) op=N  →  确认 descriptor 已是 32/32  →  BF16/FP16 后移
```

PNCG 现在不是主线。单轮仍是一遍 A + 一遍 Aᵀ（线搜索额外一次 Ad）。收益只来自更少的完整扫描次数。

## Already true here (do not redo)

| Fact | Evidence | Plan implication |
|---|---|---|
| Production numerator = packed-K `{1,2,4}` forward + **per-RHS CSR_ALG2 T** | CONSTRAINTS 2026-08-15; `CudaSparseOpsState.h` | 不要先做 4-RHS 正投 SpMM |
| Custom 2-RHS T (atomic / private-Y) **rejected and deleted** | 3-iter 10.530 s / 47.137 s vs packed-4 keep 8.845 s | **禁止复活**；P0-B 只测 cuSPARSE SpMM |
| 8-wide / ROW-B **forward** SpMM rejected | 68–73 s vs SpMV 27.076 s | 与 2-RHS **转置** SpMM 不是同一实验 |
| GPU CSR descriptor is **INDEX_32I / INDEX_32I**, values FP32 | `CudaCsrBuilder.h` `cusparseCreateCsr` | 所谓 64/64→64/32 在设备端已是 32/32 |
| On-disk compact `row_ptr` is int64, loaded then narrowed to uint32; col is 32-bit; per-chunk nnz must fit uint32 | `CompactChunkCSR.cpp` | 磁盘已付 8 B/row 的 row_ptr；payload 热点是 col+value=8 B/nnz |
| Clamp is `u = α·(num/den)` **then** clip to `[-0.9, 1]` | `BsremUpdate.cpp` | 顺序已符合建议；补测试，不要改成先钳分式再乘 α |
| `S=1` BSREM passes `denom` twice | `applyBsremUpdate(image, numer, denom, denom)` | β=0, α=1 时应退化为 MLEM |
| κ 乘在惩罚梯度上，**未**乘进曲率分母 | `term_penalty = β κ ∇R`; `denom_eff += β max(D_reg,0)` | Phase A 必须核对；固定点可不变，有效步长会变 |
| Default `curv_denom=false`; `algo_bsrem.conf` has `kappa_map=true`, `gamma=0.05`, RDP β=0.3 | conf + `ReconArgs` | 生产 INI `Regularization type=None` + `recon_sections_override_cli=true` 会盖掉 algo RDP |
| FP16 / mmap / triple-resident / NVRTC still rejected | CONSTRAINTS | 本计划不重开；BF16 仍后移 |
| `SPECT_COMPACT_BRANCH_BATCH=0` disables resident MLEM | CONSTRAINTS | 不是本计划的 A/B |

## Phase A — BSREM correctness (before claiming “faster”)

**Question:** 当前 BSREM 在 β=0、α=1 时是否就是 MLEM？钳位、κ、曲率是否按公式工作？

Keep three **named** arms. Do not compare “MLEM 600” to “BSREM+RDP 50” and call it optimizer speedup.

| Arm | Flags (intent) | What it answers |
|---|---|---|
| A1 MLEM | `-opti mlem`, no reg | 500–600 目视参考的算法；短跑用 50-iter keep |
| A2 BSREM β=0 | `-opti bsrem`, `beta=0`, `α_k=1` (`gamma=0`), `curv_denom=off` | 纯步长/钳位/曲率路径，无目标函数更换 |
| A3 BSREM+RDP | `-opti bsrem`, RDP, `beta>0` | penalized-likelihood 任务质量；**不要**写成 MLEM 的等价快收敛 |

Local unlock: A3 必须绕开 `spect_config.ini` 把 Regularization 打回 None。使用 `spect_config.ct_guided.ini`（`override=false`）或显式 INI/CLI 写 RDP，并在 manifest 记录实际 `type/beta/kappa_map`。

### A.1 Equivalence gate (machine-checked)

On a **short** compact run (3–5 iter, same matrix/projections, denom hit):

1. A2 vs A1 primary volume `relL2 <= 1e-5` (same bar as operator keep).
2. If fail: dump clamp fractions, `α_k`, whether `curv_denom` leaked on, whether RDP leaked on.
3. Clamp unit test: `α>1` still clips the **full** `u_j`, and `x(1+u)>0` when `u>=-0.9`.

### A.2 Formula audit (already observed, not yet tested)

Current relative update:

```text
u_j = α_k * (n_j - s_j - β κ_j ∇R_j) / (s_j + β [D_reg_j]^+)
```

then clip `u` to `[-0.9, 1]`, then `x ← x (1+u)`.

Advice’s matched curvature denom is `s + β κ D_reg^+`. Local code omits `κ` in the denominator when `kappa_map=true`. Phase A records:

- `from_source`: mismatch exists when both `kappa_map` and `curv_denom` are on;
- `hypothesis`: putting `κ` in D changes step size, not the β=0 fixed point;
- do not “fix” into production until A2 equivalence is green and A3 has a before/after on Φ and clamp rates.

### A.3 Diagnostics (every logged iter, opt-in verbosity or sidecar)

Need Φ and projected gradient; compact loop does not emit them today.

```text
Φ(x) = Σ_i [μ_i - y_i log μ_i] + λ R(x),   μ = Ax + b
g    = s - n + λ ∇R(x)
g^P_j = 0 if x_j ≤ x_min and g_j > 0 else g_j
```

Per iter, at least:

- data term, RDP term, total Φ;
- `|g^P|_2 / √N` and `|g^P|_∞`;
- `|x^{k+1}-x^k|_2 / |x^k|_2`;
- ROI total activity;
- fraction of voxels hitting u=-0.9 and u=+1, split low-s vs ROI;
- `λ D_reg^+ / s` median / p95 / p99 when curvature is on.

For `y_i=0` use `μ_i` only (no `0 log 0`). Do not silently `max(μ,eps)` as the objective.

**A exit:** A2≡A1 on short run; diagnostics land in log/sidecar; clamp-order test exists; κ–curvature documented as keep or explicit follow-up. No performance claim yet.

## Phase B — Transpose operator (same algorithm as packed-K MLEM)

**Question:** 在 Sampling 5-bed、relL2 门不变时，能否把 `cuda.cusparse_bwd` 降下来并带动 `iteration_core_s` ≥5%？

Use the existing 50-iter gate driver. One change per variant. Time the **whole T path**, not `cusparseSpMM` alone: ratio layout → SpMM/SpMV → layout into the update → pack/unpack.

Amdahl reminder: even a perfect 2× T is ~1.54× end-to-end. Reject &lt;5% `iteration_core_s` unless it is a correctness fix.

### B.1 2-RHS transpose SpMM (cuSPARSE only) — **measured, rejected (2026-08-17)**

Hypothesis: 40 independent ALG2 SpMV T re-read A 40 times per chunk; K=2 T SpMM reads A 20 times.

**Result: hypothesis falsified — the T path is atomic-scatter-bound, not A-read-bound.** Sharing A reads across K RHS cut `cusparse_bwd` only ~5% at best; larger K crosses the L2 output-working-set cliff and collapses 2.6×. Full table and re-run commands in `reports/2026-08-17-phase-b-transpose-spmm.md`. Verdicts vs same-day baseline (3-iter hit, core 9.781 s / bwd 6.244 s):

| Variant (env `SPECT_COMPACT_T_SPMM`) | bwd s | core s | relL2 | Verdict |
|---|---|---|---|---|
| off (B0 keep, per-RHS SpMV CSR_ALG2) | 6.244 | 9.781 | ref | **kept** |
| `alg1:2` (SpMM CSR_ALG1 col-major) | 6.323 | 10.622 | 3.1e-08 | rejected |
| `alg1:4` | 5.919 | 9.529 | 5.3e-08 | rejected (−5.2% bwd, −2.6% core < 5%) |
| `alg1:5` | 5.891 | 9.604 | 8.0e-08 | rejected (best bwd, −5.7%, core < 5%) |
| `alg1:8` | 15.348 | 18.916 | 4.8e-08 | rejected (L2 cliff: 8×7.5 MB C slices) |
| `alg1:10` | 22.275 | 26.612 | 8.0e-08 | rejected |
| `default:4` (SpMM ALG_DEFAULT) | 5.914 | 9.561 | 5.3e-08 | rejected |
| `default:8` | 15.940 | 19.156 | 6.4e-08 | rejected |
| `alg1pre:4` (+`cusparseSpMM_preprocess`) | 5.961 | 9.675 | 8.1e-08 | rejected (preprocess adds nothing) |
| `spmvpre` (per-RHS ALG2 + `cusparseSpMV_preprocess`) | 6.072 | 9.379 | 1.3e-07 | rejected (−2.8% bwd, inside noise) |

Row-major B1/B3 were not run: col-major slices need zero packing and already bounded the family below the 5% bar; adding pack/unpack kernels can only be slower.

Notes kept from planning (still true): CSR_ALG3 cannot T; do not revive deleted custom T kernels; do not auto-enable wide forward SpMM.

**Keep bar (applied):** ≥5% `iteration_core_s` or reject → all rejected; B0 stays. 50-iter extrapolation of best variant (bwd ≈70% of core at 50-iter) ≈ −4% core, still below bar; 50-iter run deliberately skipped.

### B.2 Real CSR(Aᵀ) — **closed, REJECTED at production A/B (2026-08-17)**

Hypothesis: CSR(A) `op=T` is scatter/atomics; CSR(Aᵀ) `op=N` is gather/reduce.

Compare, same chunk / same x — all three arms measured (`tools/transpose_bench.cpp`, evidence in `reports/2026-08-17-phase-b-transpose-spmm.md`):

1. CSR(A), op=T (keep): 327 ms / 40-RHS batch on chunk 0.
2. **Control:** CSC(Aᵀ) reinterpretation, op=N — **infeasible.** `cscn` variant: workspace reports only 2.26 MB but cuSPARSE materializes an internal transposed structure outside it; Sampling run OOMs deterministically after 4 chunk visits (twice) while all 9 SpMM/preprocess variants complete. No free gather by reinterpretation.
3. **True CSR(Aᵀ), op=N: SpMM DEFAULT K=8 = 1.57×** on the backward stage (chunks 0 and 3, relL2 ~4e-07). Per-RHS SpMV on Aᵀ only 1.10× — the win requires K-grouping on the gather path.

**Direction swap is dead:** forward on Aᵀ (op=T) measured 0.43–0.68×, and the custom-kernel ceiling is atomic-bound (nnz×K ≈ 23G atomic adds per 40-RHS batch, ≥0.7 s/chunk even with 69 KB shared-memory privatization, vs ~0.1 s gather forward). Forward stays on A / packed-K.

**Adopted route R1 (dual-matrix split), pending production A/B:**

- Forward: A, packed-K custom kernels, untouched. Backward: Aᵀ, `cusparseSpMM` op=N K=8.
- Aᵀ derived **at startup from A in host RAM** (measured 5.2–5.4 s/chunk; ~32 s one-time for 6): no disk product, no on-disk contract change; host resident +26.4 GB on the 64 GB machine (option: lower the A host-cache cap and eat NVMe reread).
- Device: per visit both A_i and Aᵀ_i needed sequentially; H2D +27.6 GB/iter (~1.4 s/iter at gen4 x16 effective ~20 GB/s) must hide under compute; slot scheme 2×A → 3×4.6 GB rotating {A_i, Aᵀ_i, prefetch}.
- Measured H2D on this box: pinned 22.2 GiB/s, pageable 12.9 GiB/s. Copy stream becomes the critical path unless 1–2 (half-)chunk pairs stay GPU-resident; projected net 9–25% at 50-iter (copy-bound worst case still >5% bar). GO/STOP: 3-iter then 50-iter behind a default-off experiment switch, relL2 ≤ 1e-5, net `iteration_core_s` ≥5% or reject and keep B0.
- Integration shape (decided): **half-chunk rotation** (2.15 GiB units; slots A_h, Aᵀ_h, 2 prefetch ≈ 8.6 GiB + buffers). Full-chunk 3-slot (13.8 GiB + buffers) sits on the 15.1 GiB free-VRAM knife edge already disproven by the cscn OOM — rejected. Row-contiguous halving is free in CSR (existing `buildContiguousSubsetCSR`/range machinery); backward = two SpMMs, second with beta=1; ratio batch slices by row range. Aᵀ host copies must be pinned.
- Environment facts corrected: GPU is RTX 4070 Ti SUPER (sm_89, 16 GB, 48 MB L2, PCIe gen4 x16); host RAM 64 GB.

VRAM: 16 GB cannot hold full A and Aᵀ. Per chunk, sequential A then Aᵀ is allowed; **do not** require both resident. First-class risk: today’s 2 slots + prefetch assume two **A** chunks. Switching a slot to Aᵀ may stall next-A prefetch. Measure `cuda.compact_matrix_upload` together with T; H2D is only ~2% today, so extra upload can still net-win if T drops hard.

Do not change the on-disk A contract. Miss of transpose cache = hard fail or explicit fallback to op=T, never silent.

**Production A/B verdict (`SPECT_COMPACT_T_SPMM=csrt`, 2026-08-17): REJECTED.**
Numerics pass (Sampling 3-iter relL2 4.4e-08 on the MET_DOUBLE volume) and the device-side
bwd win is real (6.07→3.70 s, 1.6×), but `iteration_core_s` regressed 2× (9.91→19.8 s):
~49 GB of host A+Aᵀ cannot all be pinned on 64 GB, `cudaHostRegister` falls back to
pageable, and pageable `cudaMemcpyAsync` blocks the host thread (12.2 s upload vs 0.24 s
baseline). Even with perfect hiding the PCIe floor (53.7 GB/iter ÷ 22.2 GiB/s ≈ 2.4 s/iter)
sits just under the 2.53 s/iter baseline — theoretical ceiling ~3.5% < 5% bar. The earlier
9–25% projection mis-modeled max(compute, copy) and assumed unreachable residency.
50-iter gate skipped as moot. `csrt` code was retained behind the default-off switch until the 2026-09-05 convergence cleanup; the rejected variants are now removed (historical source: `a3239d7`); two
production-path fixes are kept (pinned rowPtr staging in `uploadIoCsrViewAsync`;
1 MB `cudaHostRegister` threshold + csrt state cleanup in `releaseCompactExecution`,
baseline regression-checked at 10.04 s ≈ 9.91 s).

**Phase B conclusion: all operator variants (B.1 SpMM grouping, B.2 dual-matrix, B.3
64/32, B.4 low precision) rejected or no-op — B0 (single A, op=T ALG2 backward) is
frozen as the production operator. Wall-clock gains must now come from fewer full-matrix
scans (P1 PNCG, P2 subsets), not faster single scans.**

### B.3 64/32 index — **confirmed no-op, closed (2026-08-17)**

Device is already 32/32: `CachedMatrix` holds `DeviceArray<int32_t> dRowPtr` and `DeviceArray<int32_t> dColIdx` (`CudaSparseOpsState.h`), and per-chunk nnz > uint32 is rejected at load. The external suggestion "row offset int64 + col int32 saves 1/3 traffic" assumed a 64-bit column index we never had; int64 row offsets would *increase* traffic. No code change; closed as no-op.

### B.4 BF16/FP16 — still postponed

Still rejected by CONSTRAINTS (FP16 sampled relL2 `3.6770e-4`). Additional blockers from the review: mixed CSR datatypes may be unsupported; mixed precision may pick ALG3 which cannot T. After 32/32, value FP32→BF16 is 8→6 B/nnz (−25% value+col only), smaller end-to-end after SpMM amortizes reads. Any future trial must recompute `s=Aᵀ1` from the **same** quantized A. Not in P0.

**B exit:** either a kept T path with measured ≥5% on the 50-iter gate, or a decision table that B1–B4 and/or CSR(Aᵀ) lost. Freeze the T operator before Phase C claims wall-clock from α/β.

## Phase C — BSREM parameters (algorithm; operator frozen)

### C.0 BSREM pipeline profiling — **no optimization target; closed (2026-08-17; corrected same day)**

**Correction:** the runs below were intended as BSREM+RDP but the production
`spect_config.ini` sets `recon_sections_override_cli=true`, which silently restored
`[Regularization] type=None` over the CLI `-reg rdp,...` after parsing (manifest:
`regularization_enabled=false`). They therefore measure **bare BSREM** (α schedule +
clamp + kappa machinery, non-resident path) vs MLEM. True-RDP profiling re-measured
below. The 0.175 relL2 between the arms is the α-schedule difference, not RDP.

Same-binary, same-session Sampling A/B (denom hit):

| run | 3-iter core | 50-iter core | fwd 50-iter | bwd 50-iter |
|---|---|---|---|---|
| MLEM | 9.91–10.04 s | 137.997 s | 27.998 s | 107.343 s |
| bare BSREM (reg silently off) | 10.071 s | **136.262 s** | 27.439 s | 105.330 s |

Bare-BSREM glue (per-bed numer D2H + host accumulate, host BSREM update,
per-iteration image H2D) is at **parity with MLEM within run noise**: the non-resident
path costs ≈2% over 50 iters, fully shadowed by the frozen B0 operator.
Note: same-session MLEM 137.997 ≠ protected keep 126.261 (cross-day thermal/driver
drift); keep record unchanged.

The INI trap is now guarded in `tools/run_bsrem_rdp_convergence.py` (experiment INI
copy with the override off + post-run manifest assertion on
`regularization_enabled/type/beta`). Any future RDP run must verify the manifest,
not the command line.

**Question:** 在 Φ 下降且钳位已冷的前提下，全量 BSREM 能否用更少完整扫描达到任务质量？

### C.1 α 步长扫描（derenzo，2026-08-17 深夜）— **α=1.5 恒定为干净加速点，~1.45×**

用户定位（2026-08-17）：BSREM 的主要目的是**同迭代数下更快收敛**。以 MLEM it500 为参考 x*，
β=0、曲率关、钳位 [-0.9,1]、100 迭代，逐迭代诊断（`iteration_metrics.csv`）：

| 臂 | it100 relL2→x* | ≈MLEM 等效迭代 | 逐迭代 relChange（it100） | 判定 |
|---|---|---|---|---|
| MLEM（定义） | 0.3321 | 100 | — | 参考 |
| α₀=1, γ=0.05（历史标定） | 0.5057 | <50 | 1.8e-3 | **减速**，α_k 衰减致残 |
| α=1.5, γ=0 | **0.2665** | **~145** | 5.8e-3（平滑） | **干净 ~1.45×，采纳候选** |
| α=2, γ=0 | 0.2291 | ~165 | **1.1e-1 持续不降** | 2 周期乒乓振荡（10-it 净变化仅 0.027），迭代像被 ±11% 污染，拒绝 |
| α=3, γ=0 | 0.2480（非单调，it80=0.79） | — | 混沌钳位循环（clamp± 数万级往返） | 拒绝 |
| α₀=2, γ=0.01 | 0.2875 | ~130 | 4.2e-3（平滑） | 稳但后程退化，次选 |

要点：
1. 单子集 BSREM 的加速上限由步长放大给出；**无振荡的稳定上限在 α≈1.5（~1.4–1.5×）**。
   α=2 名义更快但恒定步长在乘性不动点附近过冲，形成周期-2 振荡——这正是
   Ahn–Fessler 要求 α_k→0 的工程体现。
2. 历史 α₀=1/γ=0.05 标定是为正则化重建的稳定性选的，用于加速目的是错的方向。
3. 想超过 ~1.5× 需要跨迭代信息：这就是 P1 projected-PNCG 的价值空间
   （共轭方向 + 线搜索天然消振荡）。
4. RDP 臂（β=0.3 历史标定）快速稳定到**更平滑的惩罚最优**（it100 自身 relChange
   5.9e-3、钳位近冷），但 derenzo 小杆分离明显差于 MLEM x*——迭代数减少主要来自
   RDP 压高频，不是优化器加速；产品叙述必须称 penalized-likelihood，
   与外部建议的警告一致。β 需按任务质量重标定（C.2）。

证据：`recon_out/bsrem_rdp_convergence_20260817/`（曲线与切片 PNG、
`convergence_summary.json`、各臂 `iteration_metrics.csv`）。

**推广验证（2026-08-18，五组数据全部完成）：α=1.5 γ=0 无需按数据集调参，
it100 等效 MLEM it142–148（1.42–1.48×），it50 处同样成立（非暂态）。**
历史 α₀=1/γ=0.05 在全部五组上减速。多床数据的床边缘低灵敏度体素在**所有
无惩罚 ML 家族臂**上发散（b0 也一样，α=1.5 只是更快）；center66 拼接使保存
卷不受影响；RDP 臂完全免疫（roiSum 恒定、image_max ~4e2 vs 1e12）。
生产加固项 C.3：冻结 `s_j < s_min` 体素（与 PNCG 计划的 floor 同源）。
完整表格与判读：`reports/2026-08-18-bsrem-alpha15-five-datasets.md`。

### C.1b α=1.5 × RDP 组合（2026-08-18）— **全因素矩阵拒绝；惩罚臂保持 α₀=1/γ=0.05**

derenzo 上 9 臂因素矩阵（β∈{0.03,0.1,0.3} × 日程{恒定,γ=0.02,γ=0.05} ×
曲率{关,冻结,活} × κ{开,关}）：**任何 α≥1.5 的惩罚臂要么进入持续极限环
（relChange 卡 0.14–0.7、数千至数十万体素持续钳位），要么输给历史日程**
（α₀=1.5/γ=0.05 的 it100=0.661 差于 α₀=1 的 0.619——早期过冲永不回本）。
判别方法：relL2-to-x* 平坦 ≠ 收敛（a15g0_b01 卡 0.673 是围绕污染态振荡，
对已收敛的 rdp_b01 距离反而在升）；真收敛以自身 relChange→8e-4、钳位归零
为准（rdp_b01 β=0.1 历史日程 ~it100 达成）。
机理：Nuyts RDP 梯度逐对有界 [−3,1]，但成形边缘处 ψ''→0——曲率分母在
不稳定发生的地方恰好没有阻尼可加（活曲率把钳位人口从 5 万压到 8 千，
但环不消）；失稳出现在成像形成期（~it20），warmup 混合日程躲不开。
**结论：α=1.5 只用于 β=0；RDP 生产维持 α₀=1/γ=0.05（其本身 50–100 迭代
即达惩罚最优，经济性来自压高频）。超越此天花板（两类目标）都指向 P1
PNCG 的线搜索——这是 PNCG GO 的直接证据。**
**采纳（修订）：α=1.5 γ=0 仅为无惩罚 BSREM 加速日程。**

真 RDP 成本补测（2026-08-18，同会话 Sampling 50 迭代，manifest 断言正则开）：
MLEM core 208.285 s vs BSREM+RDP 205.090 s——**真 RDP 管线与 MLEM 平价**
（主机 OpenMP RDP 梯度完全隐藏于 GPU 工作下），§C.0 更正债务关闭。
本会话整体比昨日慢 ~50%（跨会话漂移；保护 keep 126.261 不动）。

### C.2 β 任务质量标定 — **已定量（2026-08-18，用户指标：杆分离度 + 每棒轴向均匀性）**

指标管线 `rod_metrics.py`（x* 上一次性检测 16 杆/16 相邻对，全臂同坐标采样）。
derenzo it100 定量结论：
1. **α=1.5 β=0 支配**：分离度 0.379（=MLEM it~145，两个指标同时给出任务级
   1.45× 确认）、轴向 CV 0.136（与 MLEM 持平）。
2. **分离度的瓶颈是历史 γ=0.05 日程而非 β**：历史日程 β=0→0.3 分离度仅
   0.106→0.071（it100 时 α_k≈0.17 准冻结），而 β 对均匀性的改善弱且单调
   （0.128→0.112；x* 也只有 0.145——轴向 CV 本质是噪声指标，跨度 1.3× vs
   分离度跨度 9×）。
3. **建议**：杆类任务定量预算下用 α=1.5 β=0；每一档 β 用 −35%…−81% 分离度
   换 ≤+18% 均匀性，不划算；需更多均匀性应走后置滤波或在"均匀性真正重要
   的任务类"（器官摄取）上重测 β。历史 α₀=1/γ=0.05 配置对杆类任务任何 β
   都应退役。
证据：`derenzo_182222_rod_metrics.json` / `_rod_detection.png` /
`_metric_tradeoff.png`，新增稳定日程臂 rdp_b005/rdp_b003。
**Phase C 出口达成：推荐配置 (α₀, γ, curv, λ) = (1.5, 0, off, 0)（杆类任务）；
惩罚重建按任务类另标定 β 且维持 (1, 0.05)。PNCG GO 剩余输入全部就绪。**

### C.3 低灵敏度体素冻结 — **已实现并验证（2026-08-18）**

`-bsremSensFloor <rel>`（INI `[BSREM] sens_floor_rel`，manifest 记录；默认
0=关，所有冻结基线不动）：`s_j < rel·max(s)` 的体素冻结为零。mouse 验证
（α=1.5，rel=1e-4 vs 无地板）：内部 image_max 1.8e12→3.6e4、roiSum
2.1e13→3.7e7、clamp− 23220→4253；保存卷 it100 差异 relL2=8.4e-5，
relL2→x* 曲线每个 ckpt 一致到 4 位小数。套件 270 通过/4 跳过。
地板上方体素仍慢速增长（~1.04×/iter）；>100 迭代长跑建议 rel=1e-3。
**PNCG 前置条件 3（ROI 钳位分数 ≲1e-3）由此满足（~5e-4）。**

Sweep separately (one factor at a time):

- curvature off / on (and κ-in-D if A.2 opened it);
- 3 values of `α0`;
- 2–3 decay `γ` (include `γ=0` as the A2-like control);
- 3 effective `λ=βκ` (including 0 = A2).

Compare on **complete matrix scans**, wall-clock, and task metrics — not “iterations” with mixed subset definitions.

Task metrics (all `provisional` until calibrated on failure cases): FWHM, Derenzo rod visual/separability, uniform CV, RC, total-activity bias. Do not promote because Φ dropped if rods/RC regress.

α schedule: `k` increments **per outer scan**, not per future subset. Production `γ=0.05` already shrinks α to ~0.04 by scan 500; that is not MLEM-equivalent late behavior.

**C exit:** a recommended `(α0, γ, curv, λ)` for full-scan BSREM+RDP, with three-arm plots of Φ and a task metric vs scan count. If quality still needs ≳80–100 full scans after α/γ are exhausted, that is the PNCG GO input — not an automatic implement order.

### C.4 MLEM 1000 baseline + three 1000-iter recipes (2026-08-19)

**Question:** per-dataset MLEM I* from a 1000-iter ladder (A volume self-delta AND B task plateau); then three BSREM arms to 1000 (`α=1.5 β=0`, RDP `β=0.3`, RDP `β=0.03`) vs that I*.

Driver arms: `mlem1000`, `a15g0_1000`, `rdp_b03_1000`, `rdp_b003_1000` (+ fallback `rdp_b01_1000` / `rdp_b005_1000`). Do not overwrite `mlem500` / `bsrem_a15g0_500`. α=1.5+RDP stays rejected. Ledger: `reports/2026-08-19-mlem1000-baseline-bsrem-recipes.md`. Recipe axis treats `epguan` as the sole EP representative (`epguan_19p2mm` = bed-step confirmation only). Mouse axis: `mouse_ch6o6p2` (`mouse` BSREM 1000 not filled, 2026-08-20).

**C.4 measured (2026-08-20, existing volumes):** I* derenzo 700 / epguan 250 / mouse_ch6o6p2 300. α=1.5 J = 450 / 150 (mouse touches I* at 200 only). Fallback β=0.1/0.05 **not triggered**. Recipes: best resolution = α=1.5 it1000 (J=450); best mouse = RDP β=0.03 it1000; balance **unattained** (no arm hits 80% sep AND +10% CV). Scoped use, not a single production default. PNCG still blocked (α=1.5 still needs 450 derenzo scans to match I*).

## Phase D — PNCG shadow (P1; not mainline)

Only after:

1. T operator frozen (Phase B done or explicitly deferred);
2. three arms exist (Phase A);
3. ROI clamp fraction after the first few scans ≲ 10⁻³ (`provisional`);
4. Φ still decreases but target quality still needs ≳80–100 full scans;
5. α schedule cannot cut scans further;
6. RDP strength has a recorded calibration, not a conf default.

Implement **only**:

```text
BSREM diagonal M  +  projected PR+  +  fixed 8-candidate Armijo  +  auto-fallback BSREM
```

No subsets, no Wolfe, no matrix precision change in the same step.

Direction (minimize Φ):

```text
M_j = max(x_j, x_floor) / (s_j + λ D_reg,j^+ + ε)
d0  = -M g^P
```

Nonnegativity: bound `α ≤ α_x` (and optional `α_μ`) so `μ(α)=μ+α q` stays valid; do not `max(0, x+αd)` after the fact. Freeze `s_j < s_min`. Restart on active-set change ≳1%, line-search fail, or every 10–20 steps. `η` clip `[0,5]`. Consecutive Armijo failures → BSREM and stay there after 2 fails.

Cost model (hypothesis): extra `q=Ad` is ~+20% per scan (A share). GO of 60–70% scans × 1.2 work/scan ≈ 0.72–0.84 wall-clock; the 20% wall-clock GO is the same statement, not a second miracle.

**PNCG GO (adoption stop, `provisional`):** same RDP Φ or same task quality; scans ≤ 60–70% of full BSREM; wall-clock ≤ −20%; Armijo total-fail &lt;1%; extra restarts (beyond scheduled) &lt;20%; activity / FWHM / Derenzo / RC not materially worse. Else keep full-scan BSREM.

Fail-closed: production path remains BSREM; PNCG is a logged accelerator.

## Phase P2 — physical subsets (not now)

Only if PNCG is skipped or still scan-heavy.

- Partition by **compact angle chunks**; `M` **≤ chunk count**; first try **M=2 or 3**, not 6.
- Global denom stays the **preconditioner D** (v4 cache). Gradient still uses `s_m = A_mᵀ 1` (six or fewer extra volumes, not a v4 schema bump).
- Do not subset by bed or SYM_90Z branch (destroys the 8×beds pack).
- Do not expect `/M` scan counts from global-D BSREM without an explicit gradient scale (see 2026-08-17 chat). Classic OSEM `/M` is a different update.

## Experiment register

| ID | Purpose | Status | Baseline | Changed factor | Evidence |
|---|---|---|---|---|---|
| B0 | packed-K + per-RHS ALG2 T | **validated / merged** | — | — | 126.261 s keep |
| T-atomic-2 | custom 2-RHS T | **rejected** | packed-4 8.845 s / 3-iter | atomic T | 10.530 s |
| T-privateY | custom 2-RHS T | **rejected** | same | private-Y | 47.137 s |
| F-SpMM-8 | wide forward SpMM | **rejected** | SpMV 27.076 s / 3-iter | 8-wide SpMM | 68.362 s |
| A1/A2/A3 | three-arm split | exploring | — | algorithm | Phase A toy A0/A1 **validated**; A3 RDP not on Sampling |
| B1–B4 | 2-RHS T SpMM | exploring | B0 | cuSPARSE T layout/alg | blocked until Phase A Sampling adjoint or explicit defer |
| B-At | CSR(Aᵀ) op=N | exploring | B0 | explicit Aᵀ product | not run |
| B-idx | 64/32 descriptor | likely **no-op** | already 32/32 | none | source |
| C-* | α/γ/curv/λ | exploring | A2/A3 | BSREM params | not run |
| C.4 | MLEM1000 I* + 3 recipes | **measured / scoped use** | mlem1000 | α=1.5 / RDP 0.3 / RDP 0.03 | `reports/2026-08-19-mlem1000-baseline-bsrem-recipes.md`; balance unattained; fallback not run |
| D-pncg | projected PR+ shadow | exploring | frozen C | extra Ad + Armijo | blocked on GO |
| P2-M2 | chunk subsets M=2/3 | exploring | frozen C/D | physical subsets | blocked |

## Stop / resume

- **Stop by plan** after each phase exit table; do not start D or P2 because they are on the poster.
- **Blocked** if Sampling matrix, 5-bed BINs, or denom v4 hit is missing for an operator claim.
- Resume Phase B with `python tools/run_220_mlem_perf_gate.py` (default Sampling).
- Resume Phase A with a 3–5 iter A1 vs A2 pair and a new diagnostic sidecar, not a 500-iter Derenzo.

## What this plan does not freeze

This document is **not** a CONSTRAINTS update. Existing rejections (mmap, triple-resident, FP16, NVRTC, deleted T kernels, no folder-name forks, no auto wide SpMM) stay. User must explicitly reopen FP16 or custom T before those lines move.

## Verification (this writing)

| Check | Evidence |
|---|---|
| Source: T path, index types, clamp order, κ vs curvature | `CudaSparseOpsState.h`, `CudaCsrBuilder.h`, `CompactChunkCSR.cpp`, `BsremUpdate.cpp`, `CompactSymmetricReconLoop.cpp` |
| Source: protected numbers | `.agent/CONSTRAINTS.md`, `reports/2026-08-14-220-mlem-perf-goal.md` |
| Not run | no build, no gate, no Derenzo, no SpMM T A/B |
| NVIDIA catalog | listed 2026-08-17; no cuSPARSE-SpMM skill to install for this plan |
