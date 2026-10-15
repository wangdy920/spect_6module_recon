# SPECT_recon Changelog

`SPECT_recon.exe -v` prints the exact build identity followed by this complete
history. `--version` remains a one-line, machine-readable build identity.

Only validated release packages and the current build are listed. Intermediate
experiment commits are intentionally omitted.

## Current build

- 2026-10-15: streamed CSRQ (matrix larger than free GPU memory, e.g. SC60
  18.3 GiB on 16 GB cards) uploads only the chunks holding rows of the current
  subset, so uploads alternate the two device slots and overlap the previous
  chunk's kernels; without CTAC each bed's packed input and backward target live
  for the whole subset pass (one pack and one symmetry transform per bed instead
  of per chunk). SC60 linxinguan 8 beds, OSEM 10 subsets, RTX 4070 Ti SUPER:
  2.72 -> 1.79 s/iteration (ABBA), 50-iteration image relL2 6.3e-7 vs the
  previous CSRQ build and 7.4e-7 vs CSRB. Page-locked host memory for a
  streamed CSRQ matrix now follows `SPECT_CSRQ_HOST_PIN_BYTES` (auto: the CSRB
  host-cache headroom rule); chunks beyond it, or refused by the driver, are
  re-read from their unchanged files every pass through two pinned staging
  buffers instead of failing, reported by `[MATRIX_IO] csrq_file_reread_bytes`.
  SC60 scanner binding defaults to the CSRQ matrix directory; CSRB remains the
  documented fallback. Formats, kernels' arithmetic and public CLI unchanged.
  Convergence: a driver refusal of page-locked memory hands back already pinned
  CSRQ payloads so row pointers and staging still fit (previously the staging
  request came last and could abort); CSRQ ignores
  `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` (it only duplicated chunks in pageable
  RAM); per-(chunk, bed) CUDA event timers of the CSRQ passes and their
  `-vb 2` `cuda.csrq_fwd/bwd` entries are removed (wall-clock stage times stay);
  CSRQ warmup prints one NOTE instead of two and CSRQ runs no longer print the
  CSRB-only `[MATRIX_IO] scope=iterations` zero lines.
- 2026-10-15: with `recon_sections_override_cli=true`, blank configured values
  (for example a recipe's `[Normalization] factor_file =`) no longer erase the
  matching CLI option; only filled reconstruction fields override the CLI, as
  documented. Fixes `-normFactorFile` being silently dropped by the packaged
  recipes. Filled values keep their priority; formats and defaults unchanged.
- 2026-10-13: numerator and sensitivity share one branch-metadata cache update,
  fixing stale GPU symmetry tables when ROT90 and ROT90Z calls are interleaved.
  Removed unused internal paths, redundant preparation/copies and covered event
  waits; public CLI/config, matrix formats, kernels and defaults are unchanged.
- 2026-10-12: CSRB numerator queues current-chunk GPU work before loading and
  prefetching the next slice, allowing reads and uploads to overlap computation.
  Plain and CTAC passes share the scheduling change and retain existing slot,
  ownership and completion checks. No reader thread, cache policy, kernel,
  precision, format or public CLI/config change; CSRQ remains unchanged.
- 2026-10-12: CSRB payload validation uses a vectorizable valid-data pass;
  invalid weights still fail with the original source index and diagnostic.
  Matrix values, cache policy, reconstruction kernels and defaults are unchanged.
- 2026-10-12: normal logs include the selected profile and effective settings
  after configuration/metadata overrides; the original CLI is explicitly labeled
  as requested input. Console and normal log progress use full iterations only.
  Subset progress, per-subset cache detail and kernel timing require `-vb 2`.
  Successful completion and `[WARNING]` lines reach the console even at `-vb 0`;
  completion is still emitted only after reconstruction outputs finish.
- 2026-10-12: synchronous CSRB uploads retain source storage until device-cache
  eviction, preventing false pointer-key hits with partial host caches. Output
  XY rotation is independent of the legacy X flip; configured scanner binding
  takes priority over a conflicting scanner CLI argument. Native input validity,
  CTAC subset support and the compact residual/normalization model are covered
  by the matrix-adaptation regression suite. CSRQ remains experimental.
- 2026-10-12: RC packaging rejects a dirty worktree or a staged executable whose
  embedded version does not match the current clean Git commit. Commit,
  reconfigure and rebuild before release; changing a package name is not enough.
- 2026-10-11: `matrix_grid_output` multi-bed (SC60) is now one joint
  reconstruction of a global volume `Nz = baseNz + (N-1)*step`: overlapping
  beds add sensitivity and backprojection in every update; the output is that
  volume, with no cosine blend or gain fit. Output size/orientation are as
  before, the overlap content is not. CTAC/AutoMu/`-maskFromAtt` are rejected
  in this mode. Matrix-grid runs accept up to 16 beds (PSC stays at 5).
  New `-beds N|A:B` picks `Bed<k>.bin` from one `-df`/`-df2` directory and
  fails on a missing bed. Interface change: `-dfList` limit is mode-dependent.
- 2026-10-11: a refused page lock of the full CSRB host cache no longer
  aborts with `cudaHostRegister ... out of memory`; chunks locked before the
  refusal stay locked, the rest upload from the same pageable host cache, and
  one `[WARNING] compact host cache page-lock refused` line records the split.
  Shipped configs default to `verbosity = 0`: the console shows only
  `YYYY-MM-DD HH:MM:SS [ITER] n/N compact` and error lines, while the per-run
  `.log` remains complete; `-vb 1` restores the previous console output.
  Reconstruction math, CLI and file formats are unchanged.
- 2026-09-24: preserve one legacy `[ITER] n/N compact` progress line per
  full iteration; subset details use `[SUBSET]` instead of `[ITER]`.
  Quiet mode retains full-iteration progress on stdout and subset details
  in the native log. Reconstruction math and CLI are unchanged.
- 2026-09-21: native SC60 SMMD v3/ROT90 CSRB compatibility, uncached
  subset-range reads, native dual-CSRB weighted reconstruction, and explicit
  full-grid multi-bed output with voxel-aligned steps and valid-mask blending.
  Joint host-cache budgets and I/O statistics cover both matrices. SC60 real
  multi-bed acquisition/step calibration remains pending; CSRQ is experimental.
- Default Tc and Lu recipes now use standard OSEM50 x 10 without regularization.
  Lu retains independent 208/113 observations and weights0.1036/0.0617, using
  native CSRB. Previous BSREM/RDP recipes remain explicit alternatives. SC60
  has a separate 1000 x 10/PSF1.5mm validation recipe, not a PSC geometry default.

## 2026-09-14 configuration baseline

- 2026-09-14: shared angle subsets for CSRB OSEM/BSREM and CSRQ joint
  dual-window reconstruction; retain per-subset RDP/TV and remove the earlier
  POSEM pixel branch. `-it A:B` is full iterations:subsets; `-save-every` is
  separate, default final-only. PSF remains enabled; output Gaussian is off
  and Butterworth is an independent output copy.
- One isotope-selected recipe overlays global defaults. Explicit recipes
  replace defaults; CLI priority follows `recon_sections_override_cli` for
  all configured reconstruction fields. Five profiles bind their observation
  model and matrix IDs; Lu joint norm uses two independent correction files
  or neither. Norm remains disabled pending matching calibration.
- CLI help and deployment/usage guides describe the current profile and
  joint N10 interfaces. Prior real-data parity and recipe scope are recorded
  in `reports/2026-09-13-subset-release.md`; this does not promote the presets
  to universal quality or quantitative calibration claims.

- CSRB numerator backward: zero-row scatter transpose (2026-09-09). The
  numerator ratio is exactly 0 for zero-count bins (74-90 % of the five-bed
  Lu-177 input), and the per-RHS cuSPARSE transpose still walked every row
  (~82 % of numerator GPU time after cfffea5). `compact_csr_spmvt_pair` now
  scatters two RHS per matrix pass with atomics over CSR(A) and returns before
  reading any matrix byte when both ratios of a row are zero (single-branch
  atomics when only one is). The path is chosen per pass from the zero-bin
  fraction of the beds' projections (`compactScatterBackwardPreferred`,
  threshold 0.40, recorded at projection upload); below it the per-RHS
  cuSPARSE CSR_ALG2 transpose is unchanged, so dense Tc-99m data never pays
  the 0.71x dense penalty. `SPECT_CSRB_SCATTER_BACKWARD=0/1` forces either
  path for A/B on one binary. Sensitivity, independent forward and CTAC
  transposes are untouched. Bench on the real Lu-177 208 keV chunk 0
  (`spect_transpose_bench ... scatter`, 40 RHS): 378 -> 119 ms at 80 % zero
  bins (3.18x), break-even at ~30 %. Unit test: scatter vs cuSPARSE vs dense
  oracle on a 70 %-zero toy projection. Frozen Lu-177 208 keV five-bed MLEM50
  gate (A = cfffea5, B = this build, A-B/B-A/A-B, full host cache, denom hit,
  runs shared with foreign GPU jobs repeated): core 144.606 -> 73.565 s
  median (-49.1 %, pairs 49.24/48.99/49.42 %), full command 190.3 -> 117.4 s,
  max primary relL2 1.44e-7; the same B exe forced to cuSPARSE gives 143.2 s.
  The 120 s core target is met for the first time. Design report §7.9.
- CSRB CTAC numerator backward uses the same zero-row pair scatter on each
  compact angle's sub-range (`csrbAngleSpMVLocked`, row pointers offset by
  the angle, per-angle memset of the `[branch][voxel]` output), under the same
  zero-fraction rule and env override; the CTAC sensitivity keeps per-RHS
  cuSPARSE. Mouse bed (73.6 % zero bins), water-cylinder mu-map, MLEM it5:
  core 10.82 -> 9.62 s (-11 %), per-angle backward timer 2.99 -> 1.67 s,
  relL2 vs cfffea5 3.5e-8; dense Derenzo bed (2.9 % zeros) auto-keeps
  cuSPARSE, relL2 3.2e-8, time unchanged. CTAC unit test now also checks the
  forced scatter vs forced cuSPARSE vs host oracle on a 60 %-zero projection.
- Buffer and dead-path convergence (2026-09-05, after e5d376a): removed the
  uncalled full/subset/range CSR conversion and upload-cache cluster. Compact
  CSRB prefetch keeps its own single upload stream. Host gradients and disabled
  kappa maps are allocated only when used; projection/background views are built
  once and host numerator capacity is reused without redundant clearing.
  Numerical kernels, production defaults and format/adoption boundaries are unchanged.

- Convergence cleanup (2026-09-05): removed the rejected Phase-B transpose
  experiments and their `SPECT_COMPACT_T_SPMM` switch (`alg1`, `alg1pre`,
  `default`, `spmvpre`, `cscn`, `csrt`), including the host transpose cache,
  its page registrations, device half slots, upload stream and warmup API.
  The default packed-K forward / per-RHS CSR_ALG2 backward is unchanged;
  the historical rejection evidence remains in the Phase-B report and VCS.
  CSRB stays the production path; CSRQ and the default-off packed backward
  trial stay experimental. No mathematical or default-configuration change.
- Experimental packed CSRQ backward for the sm_120 vector-atomic trial
  (`SPECT_CSRQ_BACKWARD_PACKED=1`, default off): `csrq_backward_k8_packed`
  scatters all eight branches in one pass over the matrix into a `[voxel][8]`
  buffer with two `float4` atomics per nnz (native `REDG.E.ADD.F32x4` on
  sm_90+, four scalar atomics on sm_86/89), and
  `compact_transform_backward_add_packed8` consumes it (with the CTAC factor
  and energy exponent when active). Numerically consistent with the branch-major
  default (unit tests on both the plain and the attenuated path; Derenzo CSRQ
  MLEM it10 relL2 4.3e-8). On this RTX 4070 Ti SUPER (sm_89) the scalar
  fallback is 4.2x slower per matrix pass (730 vs 175 ms), as measured for
  this layout before; the sm_120 timing needs the field RTX 5060 Ti.
- CSRB CTAC moved to the factorised path and the fused/baked attenuation
  kernels are gone. Both matrix formats now apply the u16 factor in the image
  domain per compact angle: the CSRB branch batch is scaled by
  `compact_branch_batch_atten_scale` before the forward, each angle's rows
  are addressed as a sub-CSR (production packed-K row kernels on the chunk
  arrays with the row pointer offset; per-RHS cuSPARSE CSR_ALG2 for the
  transpose), and `compact_transform_backward_add_batch_atten` scales the
  scattered outputs; only the chunk's factor slices are staged (shared
  double-buffered slots), so no 2.3 GB table per bed stays resident. CTAC
  runs of both formats use the resident branch-batch path (device rotation /
  PSF / update) and the per-window energy exponent. Removed:
  `compact_bake_atten_values`, `compact_atten_fused_spmv/spmvt`,
  `compactAttenSpmv[T]FloatCUDA`,
  `compactDeviceRatioAttenSpmvTAccumulateFloatCUDA`, `CompactAttenBakeParams`,
  `SPECT_COMPACT_CTAC_BAKED_REFERENCE`, the per-bed device tables, and the
  self-referential prepared/attenuated host chunk cache in CudaChunkCache
  (net -550 lines). Gates: the CTAC unit test now checks CSRB and CSRQ
  against an independent host oracle of the per-nnz model (<= 1e-5, plus the
  adjoint identity); MDP mouse beds 2-3 (real CT mu-map) new CSRB path vs the
  old fused path MLEM it20 relL2 1.2e-7 with the denominator recomputed;
  CSRB Derenzo it50 guard relL2 1.15e-7. Time for two beds: 44 -> 2.35 s/iter
  (CSRQ 0.60). The embedded changelog is now emitted as adjacent raw literals
  so MSVC's 16 KB single-literal limit no longer breaks the build.
- Dual-window CTAC via per-window energy scaling of one factor table. Under
  the water-scaled mu model shared by auto-mu and the HU->mu pipeline
  (mu = mu_water(E) * body), the attenuation factor at another photon energy
  is att(E) = att(E_map)^(mu_w(E)/mu_w(E_map)), so one line-integral table per
  bed serves every window: the CSRQ pack / transform-add kernels raise the
  u16 factor to the window's exponent (`CompactMatrixChannel::attenPower`),
  and no second table set is generated or held (5 beds x 2 windows would have
  been 22.8 GB). `core/AttenuationCoefficients.h` gained the NIST XCOM liquid
  water table with log-log interpolation (`waterMuPerMMAtEnergyKeV`,
  `waterMuEnergyRatio`; the existing nuclide presets are its values at their
  photopeaks). New options `-ctacWindowEnergies <keV,...>` (INI `[CTAC]
  window_energies_kev`, one per window; required for dual-window `-ctac`,
  e.g. Lu-177 `208.4,112.9` -> exponents 1 and 1.2176) and `-ctacMapEnergy
  <keV>` (INI `mu_map_energy_kev`, default: the `auto_mu_nuclide` preset
  energy). Exponents enter the denominator cache identity and the run
  manifest (`ctac_mu_map_energy_kev`, `ctac_window_energies_kev`,
  `ctac_atten_powers`). Gates: unit tests (lookup reproduces the presets to
  1e-9; a channel with exponent s equals the unscaled channel on the
  pre-powered table to 2e-4, single and joint); Lu-177 pair with a water
  cylinder mu-map runs end to end (dual same-matrix window equals the single
  window exactly up to the expected 2x scale, 1-NCC 1e-15). Bone is scaled
  like water in this model (a two-material variant would need a second
  line-integral table). The later CSRB factorisation port also supports the
  per-window exponent; joint dual-window reconstruction still requires CSRQ.
- CSRQ residency and CTAC streaming (P3 of the Poisson data-term design).
  The per-bed CTAC factor tables are page-locked once (`cudaHostRegister`,
  0.08 s for 2.2 GB; synchronous fallback when registration is refused) and
  the slices a chunk needs are copied on the upload stream into two device
  slots, double-buffered against the previous (chunk, bed) pair's kernels
  with ready/free events, and the per-matrix slice-index plan is built and
  uploaded once, so no synchronous copy sits between launches. MDP mouse
  CTAC (two beds) goes from 1.02 to 0.60 s/iter with identical numerics
  (it20 relL2 1.4e-7 vs the CSRB CTAC path). The residency planner no longer
  aborts when a later matrix does not fit even streamed (the free-memory
  estimate under WDDM ignores other processes' allocations): the matrices
  the same call made resident are demoted to streamed (re-read into pinned
  memory) and the failed one is retried. Reproduced with a 4 GiB co-tenant
  on the Lu-177 pair: 208 keV planned resident, 113 keV failed with 1.2 GB
  free, 208 keV demoted, 113 keV became resident, run completed (previously
  FATAL). `CsrqDeviceMemoryError` marks the fit failure;
  `setCsrqWarmupFitFailuresForTestsCUDA` exercises the retry in the unit
  test. Suite 281 pass.
- CTAC on the CSRQ path (P2 of the Poisson data-term design): the attenuation
  model `h_ij = a_ij * att[fullAngle(i), unrotatedVoxel(j)]` is factorised out
  of the nnz loop. A chunk is walked one compact angle at a time;
  `csrq_pack_branch8_atten` multiplies the rotated image of each branch by
  its full angle's factor slice before the forward, and
  `compact_transform_backward_add_batch_atten` multiplies the scattered
  branch outputs by the same slice at the unrotated voxel after the backward,
  so forward and backward stay exact adjoints and the SpMV kernels are the
  ones used without CTAC. Only the slices a chunk reaches (60 full angles,
  ~0.4 GB for the 120x120x220 grid) are staged per (chunk, bed) with a
  synchronous copy, so the table crosses PCIe once per bed per iteration
  instead of living on the device per bed (the CSRB path keeps one 2.3 GB
  table per bed resident, which pages under WDDM with five beds). Numerator,
  sensitivity and forward projection accept per-bed tables; CTAC runs now use
  the resident device path (rotation, PSF, update on the device) for both
  MLEM and BSREM-II. Gates: unit test (random 24-angle matrix, two chunks,
  two per-bed tables) CSRQ vs CSRB relL2 ≤ 1e-5 on sensitivity / numerator /
  forward / two-bed host and resident numerators, plus the adjoint identity
  <Hx, y> = <x, H^T y> ≤ 1e-5; real data (MDP mouse Tc99m, beds 2–3, CT
  μ-map from `prepare_ct_mu_map.py --profile recon220`) CSRQ vs CSRB CTAC
  MLEM it5/10/15/20 relL2 7.9e-8 / 1.2e-7 / 1.2e-7 / 1.4e-7, sensitivity
  relL2 1.1e-7 with identical zero pattern, BSREM-II+RDP it1–5 relL2 ≤
  1.7e-7. Time per iteration for two beds: 1.02 s (kernels 0.48 s, table
  upload ~0.5 s) vs 44 s on the CSRB CTAC path. Joint dual-window runs still
  reject `-ctac` (they need one μ table per photopeak). Suite 280 pass.
- BSREM-II + RDP now runs on the resident device image (P1 of the Poisson
  data-term design): the regularizer gradient is formed from the device image
  (`computeRegGradientDeviceCUDA`, RDP/TV with the CT-guided weights), the
  curvature surrogate and kappa map live on the device, and
  `compact_bsrem_update` applies the same relative update as the host
  `applyBsremUpdate` (clip [-0.9, 1], sensitivity floor, non-negativity). The
  host update remains for `-bsremDiag`, the MLEM-equivalence debug mode and
  in-iteration output filters. Single-bed runs (MLEM and BSREM, CSRB and CSRQ,
  with or without background) use the resident branch-batch path as well, so
  no per-iteration host rotation / PSF / numerator download remains on the
  default path. Device vs host BSREM Derenzo it1–5 relL2 ≤ 1.1e-8; MLEM it50
  relL2 5.0e-8 vs the host single-bed run. Note for regression design: with
  `algo_bsrem.conf` (RDP delta 0.01, kappa map, clipping) the iteration
  amplifies fp32 accumulation noise by ~2–3x per iteration, so two runs of the
  same binary differ by ~1e-3 relL2 at it20 (5e-8 at it1); BSREM gates must
  compare early iterations.
- Projector contract refactor (P0 of the Poisson data-term design,
  `reports/2026-09-04-final-design-poisson-data-term.md`): the channel-form
  entry points are `accumulatePoissonBackprojectedRatioCUDA` (q) and
  `computePoissonSensitivityCUDA` (S); single-directory names remain as thin
  wrappers. CSRB: the additive background (`-bProj`) is now folded into the
  device ratio kernels, so background runs stay on the device branch-batch
  path (the host ratio round trip is gone; 113 keV `-bProj` output relL2
  3.9e-8 vs the old path) and one device numerator serves any bed count
  (single-bed CTAC now uses the fused attenuation kernels). CSRQ: the Poisson
  ratio is computed in the forward kernel's row epilogue (`csrq_ratio_k8`
  removed). Sensitivity with detector masks / detZ censoring / efficiency
  weights is computed directly as `A^T W 1` through weighted unit-ratio kernels
  in both formats instead of the synthesized `A^T((A1*W)/A1)` route
  (denominator relL2 2.4e-8 vs the old route, same zero pattern); the
  denominator cache manifest hashes the final row weights and the CTAC tables,
  so CTAC runs use the auto cache too. Unreferenced full-matrix / subset CUDA
  paths (chunked whole-matrix SpMV, full-subset EM/BSREM, generic ratio kernel)
  are removed (~1200 lines). Derenzo CSRB MLEM it50 relL2 1.2e-7 vs the
  2026-09-02 fp32 reference; suite 278 pass.
- Experimental (CSRB stays the production path): joint dual-window MLEM
  (MB-JR, Phys. Med. Biol. 2025 doi 10.1088/1361-6560/ae2ce0) on the CSRQ
  path: `-chunkCSRDir2` + `-df2/-dfList2`
  (+ `-bProj2/-bProjList2`) + `-energyWeights w1,w2` add a second energy
  window with its own system matrix and projections to one Poisson likelihood
  (Lu-177 113 + 208 keV). Up to two CSRQ matrices are loaded; each stays
  resident when it fits next to the 2 GiB reserve and is otherwise streamed
  from pinned host memory through two device chunk slots once per iteration
  (`-csrqResidency auto|streamed`), so the 9.0 + 6.2 GiB Lu-177 pair runs on
  a 16 GB card. The CSRQ ratio gained a window scale and per-window
  background; the denominator cache identity and run manifest record the
  second matrix, weights and residency. The CSRQ backward now scatters
  branch-major, two branches per pass (150 ms per 120x120x220 matrix pass,
  from 715 ms).
- Experimental: new GPU-resident compact matrix format `chunk_*.csrq` (run-length sorted
  columns + u16 sqrt-companded values, 2.2 B/nnz; Tc99m 24.3 GiB → 6.74 GiB),
  produced offline by `tools/convert_csrb_to_csrq.py` from the canonical CSRB.
  `-chunkCSRDir` accepts either format (never mixed). CSRQ uploads the whole
  matrix once and runs custom K=8 forward/ratio/backward fatbin kernels
  (`cuda.csrq_fwd` / `cuda.csrq_bwd`); no CTAC or background projection on
  this path, hard failure when the matrix plus a 2 GiB reserve does not fit in
  VRAM. Frozen Derenzo MLEM it200 image relL2 vs fp32 CSRB `1.5e-6`, five-bed
  mouse it50 `6.2e-7`; `iteration_core_s` 0.25 s/iter single-bed and
  1.13 s/iter five-bed on the 120×120×220 matrix (CSRB 2.6 / 2.87 s).
- Single-bed output now defaults to the same `bed_step + 10` center crop used
  by each multi-bed segment (`66` layers at `16.8/0.3 mm`); use
  `-singleBedStepAligned 0` or `[MultiBed] single_bed_step_aligned=false` for
  the legacy fixed 72-layer output.
- BSREM gains production diagnostics and step control: `-bsremDiag` /
  `-bsremDiagInterval` write per-iteration Poisson objective, projected
  gradient and clamp/step quantiles to `iteration_metrics.csv`;
  `-bsremSensFloor` freezes low-sensitivity voxels; `-bsremEquiv` provides an
  exact MLEM-equivalence mode for regression anchoring.
- CUDA sparse path moves to packed-K forward and CSR_ALG2 transpose with the
  reworked chunk cache and launch configuration; `spect_transpose_bench`
  measures the transpose path in isolation. Regularizer energy is also
  reported unweighted for diagnostics.
- Legacy aliases `-iter(s)` / `-verbose` / `-v <n>` now fail hard instead of
  being silently ignored; recon log names drop the millisecond suffix and
  `[ITER]` progress prints at `-vb 0`.
- `spect_config.experiment.ini` is tracked for experiment use but excluded
  from installed/deployed packages.
- Helper kernels (`volume_ops`, `projdomain`, `butterworth_mask`) ship as
  build-time fatbins for sm_86/89/120. Runtime NVRTC is disabled; a GPU
  outside that set is a hard failure.
- Recon log files use `<outputName>_YYYYMMDD_HHMMSS_mmm.log` again (no PID).
  Default `-vb 1` keeps version, CUDA device, iteration progress, cache
  hit/miss and completion; fatbin loads, CSR/CUDA cache internals, per-bed
  mapping, run-configuration dump and center66 gain traces move to `-vb 2`.
- Device repeat diagnostics now leave denominator-cache selection to the INI
  unless `-DenomCachePath` is explicitly requested; CLI iteration/save and
  per-run output overrides remain authoritative.
- Replaced the fixed 25 GiB compact host cache with a matrix-size and
  available-RAM-aware default. A 64 GB host can retain the deployed 25.81 GiB
  sampling matrix, while constrained hosts keep explicit safety headroom;
  `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` remains the manual override.
- Device diagnostics now identify collector schema 2 and sample whole-system
  disk read/write throughput, active time and queue length alongside process I/O.
- Release configuration now discards stale CMake indices, and top-level CTest
  discovery follows the moved `multipinhole/` test target after repository cleanup.
- Split the reconstruction source tree into the current `multipinhole/`
  implementation and an explicit, not-yet-implemented `parallel_hole/`
  boundary while preserving root build commands and deployed package layout.
- Release builds now reset the shared runtime output directory before linking,
  preventing stale test-only DLLs from entering a clean deployment package.
- Device diagnostics can run up to nine reconstructions serially and record
  per-run cache/resident state plus full-command median in the one return file;
  it also records free RAM, process I/O/workset and matrix/output disk bus type.
- Development staging no longer junctions its matrix directory to protected
  source data; self-contained field packages still copy it with `-IncludeMatrix`.
- Made automatic compact-denominator caches stable under deployment: INI cache
  paths are exe-relative, manifest-hashed variants live under
  `bin/cache/compact_denom`, and explicit `-compactDenomCache` still wins.
- Compact-denominator schema v4 removes absolute source/package paths from the
  identity while retaining matrix profile, file stats and parsed detector
  policy, so the same fixed-profile cache survives a package-root move.
- Added a per-GPU Windows process lock so an accidental second
  `SPECT_recon.exe` exits before loading the matrix.
- Kept GPU resident-matrix warmup enabled when a reduced host-cache budget is
  used, while skipping background host preload that could not remain cached.
- Added an installed PowerShell collector that records process ancestry, GPU
  samples, config/cache identity and reconstruction logs into one text file.
- Multi-bed cosine overlap now fades along output Z under reversed storage:
  the bed already shown before a joint stays dominant at the blend entry.
  The previous polarity invert matched a packing mismatch and produced paired
  seam jumps; it is no longer the production stitch.
- Gain-fit overlap windows still follow reversed reconstruction storage.
  Optional `-seamXyAlign` / `[MultiBed] seam_xy_align` remains default off.

## 20260812+1a26fc5

- Added opt-in global-INI authority for Regularization, PSF, Filtering and
  Postprocess fields; the shipped config enables it to protect the validated
  prototype policy from upstream CLI overrides.
- Unknown and retired CLI arguments are now noted and ignored while invalid
  values for supported arguments still fail validation.
- Added `[InputOutput] output_filters_overwrite_primary`; when enabled, the
  unsuffixed primary volume contains the selected output-stage Smooth then
  Butterworth chain while independent suffixed audit copies remain available.
- Added this repository-level changelog as the single source for console and
  package release history.
- Embedded the changelog in `SPECT_recon.exe -v`; the executable does not depend
  on the working directory or an external file to show it.
- Added `CHANGELOG.md` to the installed package root while preserving the compact
  one-line `--version` output for scripts.

## 20260811+949f766

- Converged reconstruction to the protected `sieve + CASToR + FWHM 1.3 mm +
  3 sigma` prototype route, including output Smooth and Butterworth copies.
- Matched normfactor behavior by pruning cylinder support before forward and
  backward PSF operations; Derenzo 50-iteration relative L2 was `2.9892e-7`.
- Added opt-in in-memory merging of matching `BedN.bin` inputs across sibling
  `YYYYMMDD_HHMMSS/SPECT` time segments.
- Kept production nonnegative output clamping enabled and verified logs remain in
  the requested output directory.

## 20260811+6f3e950

- Kept the compact MLEM image, accumulators, denominator, mask, and update on the
  GPU across iterations.
- Reduced repeated projection and branch-metadata transfers and overlapped matrix
  preload with device warmup.
- Recorded 50-iteration medians of `94.9981 s` for one-bed Derenzo and `141.3719 s`
  for four-bed EPguan on the validated 64 GB plus NVMe setup.

## 20260811+7ed8259

- Removed rejected concurrent-bed CUDA and SpMM algorithm override paths.
- Preserved the scalar/branch-batch control path and packaged only the converged
  CUDA route.

## 20260811+17bd22c

- Restored independent output-stage CASToR Smooth `0.30 mm / 3 sigma` and
  Butterworth `5 cycles/cm / order 8` copies from the unfiltered primary volume.
- Added explicit linear parity and production nonnegative-clamp validation.

## 20260811+eadbe24

- Converged the main reconstruction path to the protected prototype contract.
- Corrected detector 68 mapping, first-iteration ROI application, reversed-bed
  gain windows, and cosine blend direction.
- Removed superseded object-specific output and legacy matrix/reconstruction
  branches.

## 20260716+9e4bdd5

- Derived multi-bed seam width from the actual overlap for supported bed steps.
- Defined `cropEnd` as post-flip zeroing without changing output geometry.
- Produced the validated clean variable-overlap package with 288 passing tests.

## v2026.07.15 (d7a4169)

- Converged the multi-bed stitching and CUDA reconstruction pipeline.
- Added the initial clean release tag used by subsequent field packages.
