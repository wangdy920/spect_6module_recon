# SPECT_recon Changelog

`SPECT_recon.exe -v` prints the exact build identity followed by this complete
history. `--version` remains a one-line, machine-readable build identity.

Only validated release packages and the current build are listed. Intermediate
experiment commits are intentionally omitted.

## Current build

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
