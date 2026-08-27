# System-matrix dispatch design context

Date: 2026-08-15. Audience: human + GPT 5.6 PRO.  
Purpose: design a **discrimination / scheduling policy** for compact v4 CSRB matrices, or decide that one **parameterized** policy is enough.  
This file is self-contained. Do not invent stages. Numbers below are tagged `measured` / `from_source` / `unverified`.

**中文摘要（给同步咨询用）**  
三套矩阵共享同一份磁盘合同（`metadata.bin` 88B SYM_90Z + `chunk_*.csrb`）和同一条主线重建路径，但 CSR 访问模式不同，导致 40 路 COL SpMM / 逐 RHS ALG2 / ROW-packed 2-RHS 自定义核的相对快慢不同。当前生产路径对所有矩阵一视同仁。请在本文件事实之上，判断：按名字分叉、按 CSR 特征调度、还是用 L2 工作集算出的统一 K 就够。不要用目录名当身份。不要重开 mmap / 第三块常驻 CSR / FP16 / NVRTC。

Companion machine facts: `reports/2026-08-15-sysmat-dispatch-design-context.json`.

## Adopted policy (2026-08-15, after GPT 5.6 PRO)

Hybrid: **unified L2-budget packed-K is the production path**; `col_span/z_fill` is only a wide-SpMM guardrail (allow/forbid A/B, never auto-switch). Transpose stays per-RHS ALG2. Prefetch stays on.

```text
x_l2_budget = 0.60 * L2_bytes
K_raw = floor(x_l2_budget / (4 * n_cols))
K_mem = largest k in {1,2,4} with k <= K_raw and k <= batchSize

if K_mem < 2:          forward = per-RHS ALG2
elif K_mem >= 4
     and packed4_release_gate
     and sorted_hint:  forward = packed-4
else:                  forward = packed-2
```

`0.60` is a starting fraction, not a measured theorem. packed-2 does not require sorted columns. packed-4 must not depend on order for correctness; first auto-enable uses `sorted_hint = sorted_rows/sampled >= 0.95` as a performance hint. No runtime CSR sort. No folder-name fork.

Runtime sample (first loaded chunk, 16 nonempty stratified rows; gray zone expands to 32):

```text
full_footprint:     median(span)>=0.80 OR median(zfill)>=0.80  → forbid wide SpMM
compact_candidate:  p75(span)<=0.35 AND p75(zfill)<=0.35      → allow SpMM A/B only
else:               unknown → packed-K / ALG2
```

Measured on this host after the first production sample (`measured` 2026-08-15, 16 rows, chunk 0):

| Matrix | class | sorted_hint | span_med / p75 | zfill_med / p75 | K_mem / tile |
|---|---|---|---|---|---|
| Sampling_Approach | full_footprint | 1 | 0.99551 / 0.99766 | 0.98125 / 1.0 | 4 / 4 |
| 120×120×220 | compact_candidate | 0 | 0.15592 / 0.25137 | 0.16364 / 0.25909 | 2 / 2 |

220 `p75<=0.35` is now **measured**, not inferred from the earlier 96-row median. Wide SpMM is still not enabled. 260601 remains unclassified until sampled; recon may run on the safe default packed-2.

Protected Sampling hit after packed-4: `iteration_core_s=126.261` (`measured`). Target 120 s open. 2-RHS atomic transpose rejected (`measured` 3-iter 10.530 vs 8.845). Non-atomic private-Y T rejected (`measured` 3-iter 47.137). 2026-08-15 convergence deleted those T kernels and the `SPECT_CUDA_SPMV_*` env fork. Post-cleanup 50-iter `132.462` is not a new keep.

---

## 1. Question for GPT 5.6 PRO

Design a scheduling policy for the compact MLEM numerator on 16 GB consumer GPUs (`sm_86/89/120`) that:

1. Stays inside float32 compact v4, two resident CSR chunks, 6 GiB device reserve.
2. Does not branch on folder names.
3. Works for at least these three matrix *families*:
   - **G** — matrices emitted by `6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v4.cu` (field / 260601 class).
   - **S** — `test_data/6module/Sampling_Approach` (constructed 48 mm FOV, outer-Z source mirror).
   - **C** — `test_data/6module/matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220` (cylinder FOV + half-voxel Z fold).
4. Either (A) one unified parameterized policy, or (B) a small feature-based dispatcher with explicit predicates.
5. Rejects added complexity for &lt;5% `iteration_core_s` unless it fixes correctness.

Return: recommended policy, predicates / formulas, what to measure next, and what **not** to do.

---

## 2. Non-negotiable constraints

Provenance: `.agent/CONSTRAINTS.md` (user, 2026-08-14/15).

| Rule | Value |
|---|---|
| Production format | v4 compact CSRB directory only |
| Helper kernels | build-time fatbin for `sm_86/89/120`; no NVRTC, no disk PTX cache |
| Gate matrix | **S** only. **C** and 260601 are diagnostic |
| Gate metric | `[PERF] iteration_core_s` for 50 compact MLEM iters, denom **hit**, 5-bed mouse BINs, `-vb 1` |
| Target / floor | ≤120 s / ≤180 s |
| Numerical keep | same-input primary `relL2 <= 1e-5` |
| Complexity keep | ≥5% `iteration_core_s` or a correctness fix |
| Still rejected | mmap payloads; third persistent CSR chunk; FP16 values; runtime NVRTC |
| `SPECT_COMPACT_BRANCH_BATCH=0` | not a product switch (also disables resident MLEM) |
| Hardware this host | RTX 4070 Ti SUPER, 16 GB, 64 GB RAM, L2 **48 MiB** (`measured` device class) |
| Field GPU | RTX 5060 Ti 16 GB must be measured separately; do not stitch seconds |

Protected Sampling hit (`measured` 2026-08-15 packed-4): `iteration_core_s=126.261`, `floor_met`. Target 120 s open. Previous packed-2 hit 148.051 remains the packed-2 record.

---

## 3. Shared contract (all three families)

### 3.1 Disk

```text
<matrix_dir>/
  metadata.bin     ; exactly 88 bytes, canonical v4 SYM_90Z
  metadata.txt     ; human sidecar; NOT loaded by SPECT_recon
  chunk_000.csrb …
```

`metadata.bin` fields (`from_source` `multipinhole/include/io/CompactSymmetryMetadata.h`, `kCompactMetadataBytes=88`):

`totalRows, totalVoxels, maxNnz, compactNumAngles, detNx, compactDetZ, compactRaysPerAngle, fullNumAngles, Nx, Ny, Nz, voxelSize, detNxDup, detNz, pixelSize, detectorY, detectorThickness, use90, useZ, use45, symCode, symmetryOrder`.

Runtime loader (`multipinhole/src/io/CompactSymmetryMetadata.cpp`) **rejects** anything that is not SYM_90Z (`use90=1, useZ=1, use45=0, symCode=90, symmetryOrder=4`). It does **not** read `metadata.txt`.

CSRB header (`from_source` v4.cu `save_chunk_csr_binary` and `CompactChunkCSR.cpp`):

`uint32 magic='CSRB', uint32 version=1, int32 rows, int32 angle_offset, int32 angle_count, int64 nnz, int64 row_ptr[rows+1], int32 col_idx[nnz], float values[nnz]`.

Linear voxel index (`from_source` prototype header and production): `z*(Nx*Ny)+y*Nx+x`, x fastest.

### 3.2 Runtime recon path (production)

Entry: `SPECT_recon -chunkCSRDir <dir>`.  
Numerator: `runCompactChunkMlemNumerMultiBedCUDA` in `multipinhole/src/sparse/cuda/CompactSymmetricProjectorCUDA.cpp`.

For no-CTAC MLEM with resident batch (`useBranchBatch`, default on):

- SYM_90Z → `xySymmetryCount()=4` × `zBranchCount()=2` = **8 branches**.
- `batchSize = branchCount * bedCount` → 5-bed = **40 RHS**, 1-bed = **8 RHS**.
- Layout COL-major: `[bed][branch][voxel]`.
- Per chunk: forward SpMV/SpMM → ratio → transpose SpMV → transform-add.
- Chunk walk reverses every MLEM iter.
- Device CSR: **2 persistent slots + 1 overflow `cache_`**. Host preload join + pinned async prefetch of the next chunk (`measured` keep).
- **Current compute (all matrices, no switch):**
  - Forward: ROW-packed **2-RHS** fatbin `compact_csr_spmv_xpair` + `compact_pack_colpair_to_float2` (`projdomain_chunk_ops.cu`). Scratch = unused `dFullBatchOutput_`.
  - Transpose / odd remainder: cuSPARSE `CUSPARSE_SPMV_CSR_ALG2`.
  - Wire: `CudaSparseOpsState.h` `executeColMajorBatchSpMVLocked` (`usePackedPair` when `NON_TRANSPOSE && scratch && batchSize>=2`).

Prototype recon `6module_bak_src/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu` is **reference-only**. It does **single-RHS** `cusparseSpMV` with `CUSPARSE_SPMV_ALG_DEFAULT`, no 40-wide batch, possibly multi-GPU. It is **not** the production scheduler.

### 3.3 What metadata.bin cannot tell you

Construction tags live only in `metadata.txt`:

- S: `postprocess=outer_z_layers_source_mirror_x2`, `source_layout=24mm_center_z0_plus_12mm_center_z18_with_zfold`
- C: `voxel_fov_membership=center_inside_cylinder`, `zsym_sample_voxel_layers=110`, `voxel_z_expansion_formula=...`
- G: v4 writes `version=v7.6_csrb_csr_adaptive_rect3_roi`, `roi_radius`, `csr_agg_mode` — **no FOV / no Z-mirror postprocess fields**

`maxNnz` is 30000 on S and C (`measured` metadata.bin) — useless.

---

## 4. Three matrix families

### 4.1 G — v4.cu generator / field 260601 class

**Source (active generator, not in CMake):**  
`6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v4.cu`

Default `RunConfig` (`from_source` lines 3540–3568):

| Field | Default |
|---|---|
| Nx, Ny, Nz | **104, 104, 74** |
| voxel_size | 0.3 mm |
| num_angles / sym | 360 / `"90"` → compact 90 |
| use_zsym | true → `compact_det_z = (det_nz+1)/2` |
| chunk_angles | **23** → **4 chunks** for 90 compact angles |
| output_mode / csr_agg | `"csr"` / `"adaptive"` |
| roi_radius | 12 |
| coll_modules | 6 |

CLI can override `--nx --ny --nz --chunk-angles`. Field 260601 is a **CLI instance**, not the source default.

CSR-EDGE write path (`from_source` `save_chunk_from_edges`, comment at 3426–3427):

> 为了避免一次性排序，这里不保证 row 内 col 有序。重建 SpMV/自写 forward/backward 通常不要求列有序。

So **G is expected unsorted** unless a later postprocess sorts.

No cylinder FOV and no outer-Z source-mirror in this generator. Columns are physical voxels only.

**260601 on this host:** `test_data/6module/260601` is **absent** (`unverified` geometry). Docs conflict:

- Constraints text: Sampling is “same size class as 260601”.
- `scripts/run_260601_regression.ps1` uses `-bedOverlap 164` (if `step=56`, that implies `baseNz=220`).
- Older performance notes: ~18 GiB, **4 chunks**, 108×108×80, 1-bed Derenzo 50-iter median ~95 s (old path; **1-bed, not the 5-bed gate**).

Treat 260601 as **G-family, not measured this week**. Do not use 95 s against Sampling 148 s.

### 4.2 S — Sampling_Approach (`measured` on disk)

Path: `test_data/6module/Sampling_Approach`  
`metadata.txt` version: `MCmatrix_PSC_48h_cpp_linux_v1` — **not** v4.cu’s `v7.6_...`. Different generator + postprocess.

| Item | Value | Tag |
|---|---|---|
| Grid | 108×108×160, 1 866 240 voxels | measured |
| Compact rows | 103680 = 90×1152 | measured |
| Chunks | 6 × 15 compact angles | measured |
| nnz / payload | 3 464 432 327 / 25.813 GiB | measured |
| Construction | unscaled Z 40..119 (80-layer core); scaled Z 0..39 and 120..159 via `outer_z_layers_source_mirror_x2` | metadata.txt |
| axial_fov_mm | 48 | metadata.txt |
| CSR columns (96-row sample, chunks 0 and 3) | **strictly sorted, unique** | measured 2026-08-15 |
| median col_span / n_cols | **0.995** | measured |
| median Z layers / Nz | **157 / 160** | measured |
| median row nnz | ~33700 | measured |
| One image vector | 7.12 MiB | computed |
| 40-RHS COL B | 285 MiB | computed |

### 4.3 C — 120×120×220 (`measured` on disk)

Path: `test_data/6module/matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220`  
`metadata.txt` version: `v7.9_pinhole_plane_sampling_zhalf_fold` — later generator than v4.cu, plus `expand_half_voxel_csrb`.

| Item | Value | Tag |
|---|---|---|
| Grid | 120×120×220, 3 168 000 voxels | measured |
| Compact rows | 103680 (same detector compact) | measured |
| Chunks | 6 × 15 compact angles | measured |
| nnz / payload | 3 265 605 595 / 24.331 GiB | measured |
| FOV | cylinder, diameter 36 mm, height 66 mm, `center_inside_cylinder`, sample voxels 1 243 440 | metadata.txt |
| Z fold | `zsym_sample_voxel_layers=110`; expand `out(a,z,x)=in(a,z,x)+mirror_col(in(a,Ndetz-1-z,x))` | metadata.txt |
| CSR columns (96-row sample, chunks 0 and 3) | **unsorted, unique** | measured 2026-08-15 |
| median col_span / n_cols | **0.206–0.207** | measured |
| median Z layers / Nz | **45 / 220** | measured |
| median row nnz | ~29400 | measured |
| One image vector | 12.08 MiB | computed |
| 40-RHS COL B | 483 MiB | computed |

Earlier chat cited “16 / 220 Z layers” for a single median row. The 96-row sample is **45 / 220**. Use 45 unless a nnz-weighted resample is done.

### 4.4 Side-by-side

| Feature | G (v4 default / 260601) | S Sampling | C 220 |
|---|---|---|---|
| Generator | v4.cu CSR-EDGE | MCmatrix_PSC_48h + Z-mirror | v7.9 + half-voxel expand |
| On this host | 260601 **missing** | present | present |
| Nx×Ny×Nz | 104×104×74 default; 260601 unverified | 108×108×160 | 120×120×220 |
| Chunks | 4 at `chunk_angles=23` | 6 × 15 | 6 × 15 |
| Col sorted | **not guaranteed** (source comment) | yes | no |
| col_span / n_cols | unverified (expect mid/low if physical FOV) | 0.995 | 0.21 |
| Z fill | physical only | ~0.98 | ~0.20 |
| Packed-X sequential walk | weak if unsorted | **strong** | weak |
| Wide SpMM L2 | unknown; 1-bed 8-RHS is smaller | miss (285 MiB) | hot band may fit; full 40-RHS does not |

Shared: SYM_90Z, 48×48 det, compact 90×1152, voxel 0.3 mm, 8 recon branches.

---

## 5. Measured performance (this host, 5-bed mouse, denom hit, `-vb 1`)

Same beds: `recon_out/ctac_mdp_smoke_20260813/preproc/20260729_mdp/SPECT/Bed{1..5}.bin`.

### 5.1 Sampling (gate)

| Variant | 3-iter core s | 50-iter core s | vs-ref relL2 | Verdict |
|---|---|---|---|---|
| 40-wide COL SpMM | 90.809 | warmup 1048.981 | — | rejected (L2 thrash) |
| per-RHS SpMV ALG_DEFAULT | 33.450 | 452.767 | 8.89e-8 | intermediate |
| 8-wide SpMM tiles | 68.362 | — | 9.92e-8 | rejected |
| per-RHS SpMV CSR_ALG2 | 27.076 | 250.672 | 2.02e-7 | previous |
| ROW-major-B cuSPARSE SpMM DEFAULT | 48.790 | — | 9.33e-8 | rejected |
| ROW-major-B cuSPARSE SpMM ALG2 | 72.715 | — | 9.12e-8 | rejected |
| naive 4-RHS COL fatbin fwd + ALG2 T | 35.433 | — | 9.93e-8 | rejected (not the current kernel) |
| ALG2 + preload join + prefetch | 13.847 | 204.440 | 9.14e-8 | previous adopted |
| **ROW-packed 2-RHS fwd + ALG2 T** | **10.943** | **148.051** | **1.73e-7** | **adopted, floor_met** |

Evidence: `reports/2026-08-14-220-mlem-perf-goal.md`, `recon_out/perf_sampling_mlem_hit_xpair_20260815/result.json`.

Steady ~2.96 s/iter. Target 120 s needs ~2.4 s/iter (~1.23×).

### 5.2 220 (diagnostic)

| Variant | 3-iter core s | 50-iter core s | relL2 | Note |
|---|---|---|---|---|
| 40-wide COL SpMM | — | **154.607** | 7.69e-8 vs warmup | old protected 220 hit |
| per-RHS ALG2, no prefetch | **25.716** | — | — | transfer still in core |
| **current xpair exe** | **10.499** | **157.460** | 1.00e-7 vs ALG2 it3; 1.47e-7 vs SpMM it50 | +1.8% vs old SpMM |

Prefetch + 2-RHS recover the ALG2-only 220 regression. They do **not** beat old 40-wide SpMM. +1.8% is below the 5% switch bar.

### 5.3 Why S and C diverge under the same code

Not a different symmetry branch. Both are 8-branch × 5-bed.

S: one CSR row touches almost every Z. 40-wide COL B is 285 MiB and misses 48 MiB L2. Re-reading A 40 times with one 7.12 MiB X in L2 beat library SpMM. Packed 2-RHS then cut forward A traffic and walked sorted columns through a 14 MiB `float2` tile.

C: one row touches ~45/220 Z. Wide SpMM’s *hot* working set is much smaller than 483 MiB, so 40-wide SpMM was the previous winner. Current K=2 packed forward is still correct and only 1.8% slower.

G: expected closer to C if FOV is compact and columns unsorted; closer to a milder S if the physical grid is densely filled. **Not measured.**

---

## 6. Current scheduler (exact)

Always-on (all matrices):

1. Host cache auto (leave ≥8 GiB RAM).  
2. GPU: 2 resident chunks + overflow + prefetch.  
3. Forward packed-2 if `batchSize>=2`.  
4. Transpose ALG2.

No `Nx/Ny/Nz` test. No CSR sample. No `metadata.txt` parse.

Rejected ideas (do not revive without new evidence):

- 40-wide / 8-wide / ROW-B **library** SpMM on Sampling  
- naive COL-major 4-RHS fatbin (35.4 s vs ALG2 27.1 s on S 3-iter)  
- `SPECT_CUDA_SPMV_ALG` env fork  
- dual-stream SpMV as a 1.14× candidate  
- mmap / triple-resident / FP16 / NVRTC  

---

## 7. Discriminators (ranked)

Do **not** use the directory name.

| Rank | Signal | Runtime? | Distinguishes S vs C? | Survives a new grid? |
|---|---|---|---|---|
| 1 | First-chunk CSR sample: `col_span/n_cols`, `sorted_strict_inc`, `z_layers/Nz` | yes, milliseconds | yes (0.995 vs 0.21; sorted vs not) | **yes** — this is the L2 cause |
| 2 | `K_l2 = floor(L2_bytes / (4*Nx*Ny*Nz))` | yes, from metadata.bin + device L2 | S→6, C→3, 80-layer→13 | yes — working-set, not name |
| 3 | `Nx,Ny,Nz` from metadata.bin | already loaded | yes for *these two* products | **no** — a 108×108×160 cylinder would look like S |
| 4 | `metadata.txt` tags | not loaded | yes, explicit construction | only if we add a parser |
| 5 | chunk count / payload | yes | both S and C are 6×~4.3 GiB | weak |
| 6 | folder / `version=` string | no / sidecar | yes today | brittle |

96-row sample of chunks 0 and 3 took ~3 s for both matrices. Production can sample 8–16 rows after the first `loadChunk`.

---

## 8. Policy options (for GPT to pick)

### Option A — Unified parameterized (preferred hypothesis to attack)

One path, no matrix enum:

```text
prefetch always
vec_bytes = 4 * Nx * Ny * Nz
K = clip(floor(L2_bytes / vec_bytes), 1, batchSize)
if K >= 2 and (optional) columns_sorted:
    packed-K forward   # today only K=2 exists
else:
    per-RHS CSR_ALG2
transpose: CSR_ALG2 until a fused-T kernel wins
if col_span/n_cols <= T_span and (col_span * vec_bytes * batchSize) <= L2:
    allow 40-wide SpMM as an alternate forward/backward
```

This treats S and C as different *numbers*, not different *products*.  
Today K is hard-coded 2. C’s 12.08 MiB vector ⇒ K_max=3 on 48 MiB L2. S ⇒ K_max=6.

Open: should `columns_sorted` gate packed-K? C is unsorted and still gained vs ALG2-only (25.7 → 10.5 s 3-iter) because prefetch + A-reuse matter even without sequential X.

### Option B — Feature dispatcher (small enum)

```text
sample first chunk
if col_span/n_cols >= 0.8:   # S-like full-Z
    packed-K forward, ALG2 T; never wide SpMM
elif col_span/n_cols <= 0.35: # C-like / likely G
    try wide SpMM if it beats packed-K by >=5% on a 3-iter
else:
    packed-K or ALG2 by K_l2
```

Needs one-time 3-iter A/B per new matrix class, then freeze the predicate.

### Option C — Named product fork

`if Nz==160 && Nx==108 → S; if Nz==220 → C; else G`.  
**Discouraged.** Breaks the next 108×108×160 cylinder or 120×120×160 mirror.

### Option D — Generator writes a locality flag

Extend `metadata.bin` (breaks 88B v4) **or** add a tiny sidecar `locality.json` at generation time: `{sorted, col_span, z_fill, nnz}`. Cleanest long-term, requires generator + reader change. Not needed to ship a first dispatcher.

---

## 9. Suggested unified policy (author hypothesis, not adopted)

1. **Keep one code path** until a feature predicate wins ≥5% on a matrix we actually ship.  
2. **Always prefetch** (transfer is solved; 0.25 s / 50-iter).  
3. **Choose K from L2 / vector size**, not from the folder. Implement K=2 today; add K=4 only if S 3-iter beats 10.943 by ≥5%.  
4. **Do not restore 40-wide SpMM for C** unless a prefetch+SpMM 50-iter beats 157.460 by ≥5% (need ≤149.6 s). Old SpMM 154.6 is only −1.8%.  
5. **If a dispatcher is added**, predicate = first-chunk `col_span/n_cols` (and maybe sorted), never `Nx/Ny/Nz` alone.  
6. **G/260601**: measure 3-iter + 50-iter on the current exe before inventing a third arm. Expect unsorted + moderate span.  
7. **Prototype recon** stays single-RHS ALG_DEFAULT; do not copy its scheduler into production.

---

## 10. What we want GPT 5.6 PRO to decide

1. A vs B vs hybrid: is `K = f(L2, n_cols)` enough, or must `col_span` select SpMM vs packed-K?  
2. Numeric thresholds (`T_span`, min rows to sample, L2 fraction to leave for A streaming).  
3. Whether packed-K should require sorted columns.  
4. How to treat G/260601 when the directory is missing: infer from v4.cu comments, or refuse to schedule until sampled.  
5. Whether to add a generation-time locality sidecar vs runtime sample.  
6. Next measurement order toward Sampling 120 s (2-RHS transpose vs 4-RHS packed forward) given C must not regress &gt;5% vs 157.460.

Please cite the tables in this file. Do not recommend mmap / triple-resident / FP16 / NVRTC / folder-name switches.

---

## 11. Reproduction

```text
# CSR locality sample (both on-disk matrices)
# (inline Python used 2026-08-15; 96 rows × chunks 0 and 3)

# Sampling 50-iter gate
python tools/run_220_mlem_perf_gate.py --it 50 --stage hit_xpair --baseline-mhd recon_out/perf_sampling_mlem_warmup_20260814/recon_it50.mhd --out-dir recon_out/perf_sampling_mlem_hit_xpair_20260815

# 220 diagnostic (gate script refuses non-Sampling)
SPECT_recon.exe -config multipinhole/config/spect_config.ini -opti mlem -it 50:50 -vb 1 -chunkCSRDir test_data/6module/matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220 -dfList Bed1.bin;...;Bed5.bin
```

---

## 12. Source pointers

| Topic | Path |
|---|---|
| v4 generator defaults / CSR-EDGE unsorted | `6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v4.cu` ~3540, ~3410 |
| Prototype single-RHS SpMV | `6module_bak_src/spect_recon_PSC_6module_24pin_YSO_bedmerge.cu` ~4119 |
| metadata.bin contract | `multipinhole/include/io/CompactSymmetryMetadata.h` |
| Production batch numerator | `CompactSymmetricProjectorCUDA.cpp` `runCompactChunkMlemNumerMultiBedCUDA` |
| Packed-K / ALG2 dispatch | `CudaSparseOpsState.h` `executeColMajorBatchSpMVLocked` |
| Kernels | `projdomain_chunk_ops.cu` `compact_csr_spmv_xpair`, `compact_csr_spmv_xquad` |
| Memory policy | `CudaLaunchConfig.h` `kCompactChunkResidentDeviceReserveBytes=6GiB` |
| Gate / constraints | `.agent/CONSTRAINTS.md`, `tools/run_220_mlem_perf_gate.py` |
| Sampling vs 220 perf record | `reports/2026-08-14-220-mlem-perf-goal.md` |

---

## 13. Verification of this brief

| Check | Status |
|---|---|
| Sampling and 220 metadata.txt / metadata.bin / CSRB headers read | done |
| 96-row CSR sample chunks 0 and 3 | done 2026-08-15 |
| Sampling 50-iter 148.051 / 126.261 and 220 50-iter 157.460 / 143.287 | done, logs in `recon_out/` |
| Runtime 16-row sample + K dispatch | Sampling K=4, 220 K=2 |
| v4.cu defaults and `save_chunk_from_edges` unsorted comment | read |
| 260601 directory on this host | **absent**; geometry unverified |
| nnz-weighted Z histogram (vs unique Z count) | not redone |
| Field 5060 | not measured |
| Prefetch+SpMM A/B on 220 | not run |

Unknowns must stay unknown. Do not treat 260601 as 108×108×80 or 120×120×220 without the files.
