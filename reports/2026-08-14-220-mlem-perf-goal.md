# Sampling_Approach MLEM 50-iteration performance goal

Date: 2026-08-14. Provenance: user constraint (supersedes the 220-matrix gate).
Updated: 2026-08-15 (ROW-packed 4-RHS forward on L2-budget K; 180 s floor met).

## Contract

Subsequent performance tests use `test_data/6module/Sampling_Approach`
(108×108×160, 6 CSRB chunks, nnz 3,464,432,327, 25.813 GiB).

Grid is smaller than 120×120×220; on-disk payload is larger (25.813 vs 24.331 GiB).
Do not treat Sampling 108×108×160 as `260601` geometry (`unverified`; directory absent).

| Item | Value |
|---|---|
| Primary metric | `[PERF] iteration_core_s` for 50 compact MLEM iterations |
| Target | ≤ 120 s |
| Floor | ≤ 180 s |
| Numerical gate | same-input primary `relL2 <= 1e-5` |
| Workload | production INI, MLEM, no CTAC, denom hit, 5-bed mouse MDP BINs, `-vb 1` |
| Still rejected | mmap, triple-resident, FP16, runtime NVRTC |

Driver: `python tools/run_220_mlem_perf_gate.py`.

The 220-matrix five-bed hit `iteration_core_s=154.607` is diagnostic only and not this baseline.

## Measured 50-iter

| Run | denom | `iteration_core_s` | `full_command_s` | relL2 vs SpMM warmup it50 | Gate |
|---|---|---|---|---|---|
| SpMM warmup | miss | 1048.981 | 1115.687 | — | warmup_not_gate |
| SpMV ALG_DEFAULT hit | hit | 452.767 | 462.543 | 8.89e-8 | below_floor |
| SpMV CSR_ALG2 hit | hit | 250.672 | 260.413 | 2.02e-7 | below_floor |
| SpMV ALG2 + preload join + pinned prefetch | hit | 204.440 | 230.995 | 9.14e-8 | below_floor |
| ALG2 transpose + ROW-packed 2-RHS forward | hit | 148.051 | 176.920 | 1.73e-7 | floor_met |
| **ALG2 transpose + L2-budget packed-K (K=4 on Sampling)** | **hit** | **126.261** | **152.416** | **1.57e-7** | **floor_met** |
| 220-matrix hit (diagnostic, old SpMM exe) | hit | 154.607 | 183.077 | — | not this gate |
| 220-matrix hit (diagnostic, xpair exe) | hit | 157.460 | 208.153 | 1.47e-7 vs old SpMM it50 | not this gate |
| 220-matrix hit (diagnostic, packed-K exe, K=2) | hit | 143.287 | 185.905 | 4.64e-8 vs xpair it50 | not this gate |

Adopted vs SpMM warmup core: **−88.0%** (1049 → 126 s). vs previous packed-2 hit: **−14.7%** (148.051 → 126.261). Steady ~2.53 s/iter (need 2.4 s/iter for the 120 s target). Evidence: `recon_out/perf_sampling_mlem_hit_xquad_20260815/result.json`.

Prepare (`denom_and_matrix_prepare_s`) is 27.666 s on the adopted hit. That is host+GPU matrix warmup, outside the gate. Compact matrix upload inside core is 0.247 s / 151 calls.

Diagnostic 220 on the packed-K exe (not the gate): runtime sample `compact_candidate` (`span_p75=0.251`, `zfill_p75=0.259`, `sorted_hint=0`), so `K_mem=2` and packed-4 is not selected. 3-iter `iteration_core_s=9.746` vs xpair 10.499, relL2 `2.81e-8`. 50-iter `iteration_core_s=143.287` vs xpair 157.460 (−9.0%), relL2 `4.64e-8`. The 220 50-iter drop is the same packed-2 path, not a packed-4 effect; do not treat it as a new algorithm arm. Wide SpMM stays off (would still need ≤149.6 s vs the older 157.460 packed-2 bar). Evidence: `recon_out/perf_220_diag_xquad_3it_20260815/`, `recon_out/perf_220_hit_xquad_20260815/`.

## Diagnosis

Runtime path is the same SYM_90Z / 8-branch / 5-bed route. The original 7× gap vs 220 is the Sampling CSR access pattern under 40-wide COL-major SpMM, not a missing symmetry fold, not host-cache miss, and not `compactRoiPrunesMatrix`.

| | Sampling_Approach | 120×120×220 |
|---|---|---|
| Grid / nnz | 108×108×160, 3.464e9 | 120×120×220, 3.266e9 |
| Chunk | ~4.30 GiB, 17 280 rows | ~4.04 GiB, 17 280 rows |
| CSR columns (96-row sample, chunks 0 and 3) | strictly sorted, unique | unsorted, unique |
| Median col span / n_cols | 0.995 | 0.206–0.207 |
| Median Z layers touched | 157 / 160 | 45 / 220 |
| metadata.bin (runtime) | 108×108×160, 1 866 240 voxels | 120×120×220, 3 168 000 voxels |
| metadata.txt (sidecar, not loaded) | `postprocess=outer_z_layers_source_mirror_x2` | `voxel_fov_membership=center_inside_cylinder` |
| 40-RHS SpMM dense B | 298 MiB (`cols*40*4`) | hot band still much smaller than Sampling |

How to tell the two matrices apart (if a later split is needed): do not use the folder name. `metadata.bin` already has `Nx,Ny,Nz` (108×108×160 vs 120×120×220). That identifies these two products, but a future 108×108×160 cylinder matrix would be misclassified. The kernel-relevant test is a few-row sample of the first CSRB: Sampling is strictly sorted with `col_span/n_cols ≈ 0.99`; 220 is unsorted with `col_span/n_cols ≈ 0.21`. `metadata.txt` tags (`postprocess=outer_z_layers_source_mirror_x2` vs `voxel_fov_membership=center_inside_cylinder`) are explicit construction labels but are not in the v4 binary contract. No switch was added: 220’s +1.8% vs old SpMM is below the 5% bar.

`SPECT_COMPACT_BRANCH_BATCH=0` was not used: it also disables resident MLEM. The production change keeps resident batch input and replaces only the 40-wide SpMM with per-RHS SpMV, then fuses forward pairs.

After SpMV ALG2, transfer was exhausted (upload 0.305 s of 204 s). Remaining time was per-chunk CSR_ALG2 SpMV. A naive COL-major 4-RHS fatbin lost because it gathered four B columns at stride `n_cols`. The kept kernel packs each 2-RHS tile to `float2` (14 MiB, fits L2), walks sorted CSR columns sequentially through that tile, vectorizes A as float4/int4, and leaves transpose on CSR_ALG2.

## 3-iter denom-hit A/B (same command, five-bed, exe rebuilt only for the candidate)

Numerical reference: SpMM 3-iter `recon_it3.mhd`. Focused tests: `MultiBedCompactNumeratorTest.TwoBedsMatchDenseReference` and three siblings passed.

| Variant | `iteration_core_s` | vs SpMM | vs-SpMM relL2 | Verdict |
|---|---|---|---|---|
| 40-wide COL-major SpMM (protected before) | 90.809 | — | — | baseline |
| per-RHS SpMV, ALG_DEFAULT | 33.450 | −63.2% | 1.61e-7 | intermediate |
| 8-wide SpMM tiles | 68.362 | −24.7% | 9.92e-8 | **rejected** (2.0× slower than SpMV) |
| per-RHS SpMV, CSR_ALG2 | 27.076 | −70.2% | 1.87e-7 | previous adopted |
| ROW-major-B SpMM, DEFAULT | 48.790 | −46.3% | 9.33e-8 | **rejected** (1.80× slower than SpMV) |
| ROW-major-B SpMM, CSR_ALG2 | 72.715 | −19.9% | 9.12e-8 | **rejected** (2.69× slower than SpMV) |
| fused 4-RHS COL forward + SpMV transpose | 35.433 | −61.0% | 9.93e-8 | **rejected** (1.31× slower than SpMV) |
| ALG2 + preload join + pinned prefetch | 13.847 | −84.8% | 1.87e-7 | previous adopted (−48.9% vs 27.076) |
| ROW-packed 2-RHS forward + ALG2 transpose | 10.943 | −88.0% | 1.31e-7 | previous adopted (−21.0% vs 13.847) |
| **ROW-packed 4-RHS forward + ALG2 transpose** | **8.845** | **−90.3%** | **8.58e-8** | **adopted** (−19.2% vs 10.943) |
| packed-4 forward + 2-RHS atomic transpose | 10.530 | −88.4% | 2.05e-7 | **rejected** (+19.1% vs 8.845; bar 8.403) |

ROW-major B + COL-major C was VRAM-safe (reuse `dFullBatchOutput_`, no extra 298 MiB). Numerics passed. cuSPARSE reading A once was still slower than 40 CSR_ALG2 SpMVs, so “A re-read dominates” is falsified for the library SpMM path. A naive fatbin 4-RHS CSR forward (COL-major B tile) also lost to ALG2 SpMV.

The transfer change does not add a third persistent CSR chunk. It joins the existing host preload before the first numerator (so that I/O is in `denom_and_matrix_prepare_s`) and prefetches the next chunk into an idle resident slot or overflow `cache_` on a non-blocking stream after `cudaHostRegister` of the host-cached CSR arrays.

The 2-RHS forward kernel reuses `dFullBatchOutput_` as pack scratch during forward (backward has not started). It is not a fourth CSR buffer.

## Gap

Protected hit is **`iteration_core_s=126.261`**. Floor 180 s is met. Target 120 s needs ~1.05×. Steady ~2.53 s/iter; target needs 2.4 s/iter.

Transfer remains exhausted: 50-iter compact matrix upload is 0.292 s (0.23% of core). Remaining time is mixed custom-forward + CSR_ALG2-transpose SpMV.

| Candidate | Result |
|---|---|
| per-RHS SpMV CSR_ALG2 | kept (251 s / 50-iter, then 204 s with transfer work) |
| host-preload join + pinned async prefetch | kept (204.440 s / 50-iter, −18.4%) |
| ROW-packed 2-RHS forward fatbin | kept as K=2 (148.051 s / 50-iter, −27.6%; floor met) |
| ROW-packed 4-RHS forward fatbin (xpair-derived) | **adopted** on Sampling via L2 K (126.261 s / 50-iter, −14.7% vs packed-2) |
| 40-wide / 8-wide / ROW-B cuSPARSE SpMM | rejected, slower than SpMV |
| fused 4-RHS COL fatbin forward | rejected, 35.4 s vs 27.1 s on 3-iter |
| 2-RHS atomic CSR transpose | rejected, 10.530 s vs 8.845 s on 3-iter |
| non-atomic private-Y + reduce T | rejected, 47.137 s vs 8.845 s on 3-iter |
| dual-stream SpMV | not run: launch overlap cannot supply 1.14× of the old 204 s gap |
| mmap / triple-resident / FP16 / NVRTC | still rejected; not reopened |

2-RHS atomic CSR transpose was **rejected** (3-iter 10.530 s vs 8.845 s). Production stays packed-4 forward + ALG2 transpose at 126.261 s.

Stage split on that path (Sampling 3-iter, denom hit, K=4, `-vb 1`):

| Stage | 3-iter s | % of `iteration_core_s` | Scaled to 126.261 s |
|---|---:|---:|---:|
| `cuda.cusparse_bwd` (ALG2 T) | 9.008 | 70.4% | ~88.9 |
| `cuda.cusparse_fwd` (packed-4) | 2.570 | 20.1% | ~25.4 |
| `cuda.compact_matrix_upload` | 0.284 | 2.2% | ~2.8 |
| `cuda.transform_bwd` | 0.040 | 0.3% | ~0.4 |
| `cuda.ratio` | 0.004 | 0.0% | ~0.0 |
| unaccounted (host / prefetch join / MLEM update) | 0.883 | 6.9% | ~8.7 |

Pack vs CSR-forward (one-shot events, ratio only; those walls were inflated and are **not** an A/B bar): pack 0.016 s / csr-fwd 2.518 s (pack is 0.6% of forward).

Reject pack-into-xquad fuse and K>4 as next steps. The 120 s gap is 6.261 s (~5% of 126 s) and sits inside transpose. Both designed 2-RHS T candidates lost: atomic 10.530 s, non-atomic private-Y 47.137 s (`cusparse_bwd=40.068`, n=17). Production stays packed-4+ALG2. Do not retry low-occupancy private-Y, 4-RHS atomic T, COL-major tile4, or auto-enable wide SpMM.

The same-session coarse 3-iter wall was 12.789 s vs the keep 8.845 s. Do **not** replace the protected 8.845 / 126.261 records with 12.789. Evidence: `recon_out/perf_sampling_mlem_diag_stages_3it_coarse_20260815/`.

## Convergence (2026-08-15)

Optimization-phase cleanup of the adopted path. No algorithm change.

Removed: rejected `compact_csr_spmvt_xpair` / `compact_csr_spmvt_xpair_partial` / reduce kernels and launchers; `SPECT_CUDA_SPMV_{,FWD_,BWD_}ALG`; packed-4 capability flag; unused pack/csr PROFILE counters; private-Y scratch.

Production numerator is one path: L2 packed-K forward `{1,2,4}` + per-RHS CSR_ALG2. `SPECT_COMPACT_BRANCH_BATCH=0` still exists and still disables resident MLEM.

Post-cleanup Sampling 3-iter `iteration_core_s=9.385` (relL2 `1.10e-7`). 50-iter `iteration_core_s=132.462` (relL2 `1.68e-7` vs warmup it50). Do not replace the 126.261 keep with 132.462. Evidence: `recon_out/perf_sampling_mlem_diag_converge_3it_20260815/`, `recon_out/perf_sampling_mlem_hit_converge_20260815/`.

## Verification (2026-08-15)

| Check | Evidence |
|---|---|
| Focused numerator tests | 5/5 passed after packed-4 + K dispatch (4 MultiBed + K-budget) |
| Prefetch 3-iter vs SpMM `recon_it3.mhd` | relL2 1.87e-7 ≤ 1e-5; `iteration_core_s=13.847` vs 27.076 (−48.9%) |
| Prefetch 50-iter vs SpMM warmup `recon_it50.mhd` | relL2 9.14e-8 ≤ 1e-5; `iteration_core_s=204.440` vs 250.672 (−18.4%) |
| 2-RHS forward 3-iter vs SpMM `recon_it3.mhd` | relL2 1.31e-7 ≤ 1e-5; `iteration_core_s=10.943` vs 13.847 (−21.0%) |
| 2-RHS forward 50-iter vs SpMM warmup `recon_it50.mhd` | relL2 1.73e-7 ≤ 1e-5; `iteration_core_s=148.051` vs 204.440 (−27.6%) |
| packed-4 3-iter vs SpMM `recon_it3.mhd` | relL2 8.58e-8 ≤ 1e-5; `iteration_core_s=8.845` vs 10.943 (−19.2%; bar 10.396) |
| packed-4 50-iter vs SpMM warmup `recon_it50.mhd` | relL2 1.57e-7 ≤ 1e-5; `iteration_core_s=126.261` vs 148.051 (−14.7%; bar 140.648) |
| 2-RHS atomic T 3-iter vs SpMM `recon_it3.mhd` | relL2 2.05e-7 ≤ 1e-5; `iteration_core_s=10.530` vs 8.845 (**rejected**) |
| Stage-split 3-iter vs SpMM `recon_it3.mhd` | relL2 1.10e-7 ≤ 1e-5; coarse PROFILE above; wall 12.789 is **not** a new keep |
| Non-atomic private-Y T 3-iter vs SpMM `recon_it3.mhd` | relL2 2.47e-7 ≤ 1e-5; `iteration_core_s=47.137` vs 8.845 (**rejected**) |
| Convergence focused tests | 5/5 (`MultiBedCompactNumeratorTest.*` + `CompactPackedForwardK.*`) |
| Convergence Release CTest | 253 total, 0 failed, 4 skipped |
| Convergence 3-iter vs SpMM `recon_it3.mhd` | relL2 1.10e-7 ≤ 1e-5; `iteration_core_s=9.385` |
| Convergence 50-iter vs warmup `recon_it50.mhd` | relL2 1.68e-7 ≤ 1e-5; `iteration_core_s=132.462` (`floor_met`; keep remains 126.261) |
| 220 dispatch | `K=2 k_mem=2 sorted_hint=0 compact_candidate`; 3-iter 9.746 / 50-iter 143.287 |
| Focused tests after timing | 5/5 (`MultiBedCompactNumeratorTest.*` + `CompactPackedForwardK.*`) |
| Full Release CTest | 253 total, 249 passed, 4 skipped (frozen external / absent 260601), 0 failed |

Reproduction:

```text
python tools/run_220_mlem_perf_gate.py --it 3 --stage diag_xquad --baseline-mhd recon_out/perf_sampling_mlem_diag_spmm_3it_20260814/recon_it3.mhd --out-dir recon_out/perf_sampling_mlem_diag_xquad_3it_20260815
python tools/run_220_mlem_perf_gate.py --it 50 --stage hit_xquad --baseline-mhd recon_out/perf_sampling_mlem_warmup_20260814/recon_it50.mhd --out-dir recon_out/perf_sampling_mlem_hit_xquad_20260815
python tools/run_220_mlem_perf_gate.py --it 3 --stage candidate --baseline-mhd recon_out/perf_sampling_mlem_diag_spmm_3it_20260814/recon_it3.mhd --out-dir recon_out/perf_sampling_mlem_diag_stages_3it_coarse_20260815
```

220 packed-K exe 3-iter vs xpair it3: relL2 2.81e-8; `iteration_core_s=9.746` vs 10.499.
220 packed-K exe 50-iter vs xpair it50: relL2 4.64e-8; `iteration_core_s=143.287` vs 157.460.

Not verified this turn: field RTX 5060. Wide SpMM is still not a production branch.
