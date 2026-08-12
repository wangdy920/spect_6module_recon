# SPECT_recon Changelog

`SPECT_recon.exe -v` prints the exact build identity followed by this complete
history. `--version` remains a one-line, machine-readable build identity.

Only validated release packages and the current build are listed. Intermediate
experiment commits are intentionally omitted.

## Current build

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
