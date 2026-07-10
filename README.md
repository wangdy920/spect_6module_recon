# SPECT-recon

High-performance 3D SPECT reconstruction pipeline with C++17 modular architecture and CUDA acceleration. The project provides:

- **C++ recon engine**: consumes projection data (MAT/BIN) + system matrix (standard CSRB or compact chunk CSRB) and runs MLEM/OSEM, BSREM-II, or ADMM with PSF, TV/RDP regularisation, masks, Butterworth, and Joseph-like smoothing.
- **Compact-symmetric backend**: **default matrix 260601** — `-chunkCSRDir` loads `metadata.bin` + `chunk_*.csrb` (`test_data/6module/260601`, 88B metadata, Sym90Z); legacy 260525 single-file CSRB (`-loadCSR -compactMeta`) remains for reference-only parity.
- **CUDA GPU acceleration**: cuSPARSE sparse SpMV, cuFFT Butterworth, NVRTC kernels for PSF/RDP/TV gradients and TV proximal (Chambolle-Pock).
- **6-module matrix generation reference**: `6module_bak_src/` holds CUDA v2/v4 system-matrix generators and standalone MLEM reference (legacy).

## Project Status (Snapshot: 2026-07-08)

- **Current release**: `doc/RELEASE_20260708_CN.md` (`SPECT_recon_rc_20260708_*`).
- **Default system matrix**: 260601 compact chunk (`-chunkCSRDir` / `bin/matrix/260601` auto-discovery). B1 plain 50-iter vs `test_data/sys_260601_recon_ref/recon_iter050.img` signed off (`rmse/ref_mean ≈ 1.06×10⁻⁵`).
- **260525 / B2**: legacy compatibility / reference-only; not a release hard gate.
- **Target GPUs**: RTX 4070 (sm_89), RTX 5060/5070 (sm_120).
- **Public backends**: CUDA-only sparse ops; cuFFT Butterworth; PSF Operator/None; regularizers TV/RDP.
- **Optimization algorithms**: MLEM/OSEM, BSREM-II, ADMM (TV proximal via Chambolle-Pock). Compact path: MLEM/BSREM, subsets=1 only.
- See `doc/Capability_Matrix_CN.md` and `doc/Legacy_MLEM_B1B2_Parity_CN.md`.

**Project Structure**
- **Core Code**: [include/](include/), [src/](src/) — modular C++17 implementation
- **Tests**: [tests/](tests/) — GoogleTest unit tests
- **Documentation**: [doc/](doc/) — design, usage, acceptance checklists (中文)
- **Scripts**: [scripts/](scripts/) — validation, QA, and benchmark helpers
- **Matrix generators (reference)**: [6module_bak_src/](6module_bak_src/) — CUDA v2/v4 generators, legacy MLEM reference

## Typical workflow

1. **MATLAB**: tune projector/reg settings →

   ```matlab
   [row_ptr,col_idx,val] = pair_to_csr_mex(W_ind,W_dat,Nvox);
   save_csr_binary_file(row_ptr, col_idx, val, rows, cols, 'out/sysmat.csrb');
   ```

2. **C++**: reuse the CSR cache – no MAT loading required:

   ```powershell
   .\SPECT_recon.exe -df proj.mat -loadCSR out/sysmat.csrb -mapCSR on `
       -dim 200,200,200 -detN 40,40 -ang 72 -it 50:5 -th 8 `
       -psf 1.5,4 -cyl 30,30 -vox 0.4,0.4,0.4 `
       -fout recon_output/run1
   ```

   Each run writes a per-reconstruction log file alongside the outputs (`<outputName>_YYYYMMDD_HHMMSS_mmm.log`).

## Key switches

| Option | Description |
| --- | --- |
| `-loadCSR`/`-mapCSR` | Load cached CSR and (optionally) memory-map it; `-indexMat/-weightMat` become optional. |
| `-chunkCSRDir` / `bin/matrix/<name>` | Compact chunk matrix directory containing `metadata.bin` and `chunk_*.csrb`. Resolution priority is: CLI `-chunkCSRDir`, then `[SystemMatrix] default_chunk_csr_dir` in `bin/config/spect_config.ini`, then auto-use the single `bin/matrix/*` directory that contains `metadata.bin`. If multiple candidates exist, pass `-chunkCSRDir` explicitly. |
| `-Astream on|off`    | Build CSR directly from MAT files (default **off**, enable when RAM is tight). |
| `-df <file>` | Projection input can be MAT or the BIN layout written by `6module_bak_src/mat_convert_bin.m`; BIN stores only `num_angles` and `rays_per_angle`, so keep `-detN <detZ,detX>` when detector shape matters. |
| Sparse backend | Fixed to CUDA. The old `-sparseBackend` / `-spBackend` CLI switch is rejected. |
| `-psf [fwhm_mm[,N]]` | Image-domain Operator PSF; FWHM is in mm and N truncates the Gaussian at N*sigma. |
| `-psfMode Operator|None` | Force PSF on/off; `-psf` defaults to `Operator`. |
| `-joe <n>` | Joseph-like (B-spline) smoothing passes on denom/image. Default 0 = disabled. |
| `-subsets <n>` | OSEM subsets (1 = MLEM). |
| `-opti mlem|bsrem|admm` | Select optimisation algorithm (overrides config). |
| `-opti admm` | Select ADMM optimizer (with `-admmRho`, `-admmTvInner`, `-admmAdaptive`). |
| `-bsrem <alpha0[,gamma]>` | BSREM‑II step parameters (overrides config). |
| `-reg <type,beta,...>` | Regularisation. Public types: `None`, `TV`/`TVIsotropic`, `RDP`. For TV use `-reg TV,beta,delta`; for RDP use `-reg RDP,beta,gamma,delta`. `beta` is independent of `-bsrem` alpha/gamma. |
| `-bProj <file>` | Background/scatter projection term (MAT or BIN); added to forward model `yhat`. |
| `-dfList <a;b;...>` | Multi-bed projection list (MAT/BIN, comma/semicolon separated); paths containing `BedN` are sorted by numeric bed order; max 5 beds. |
| `-bedOverlap <b1,b2,b3,b4,b5>` | Fixed Bed1..Bed5 overlap table. `b1` maps `Bed1-Bed2`, `b2` maps `Bed2-Bed3`, etc.; one value is replicated across all fixed slots. |
| `-bProjList <a;b;...>` | Multi-bed background projections (MAT/BIN), sorted by `BedN` when present; list length must match beds. |
| `-cropEnd <top[,end]>` | Crop final output Z top/end slices after output flips. |
| `-zbedflip <0|1>` / `-zoutflip <0|1>` / `-xoutflip <0|1>` | Per-bed Z / whole-volume Z / whole-volume X flips. Pair with preprocess: `-chunkCSRDir` + `identity` → keep package `[MultiBed] z_bed_flip` default; single-file `-loadCSR`+`-compactMeta` + `xflip` → **`-zbedflip 1`**. |
| `-maskFromAtt <th>` | Generate body mask from attenuation map and zero outside mask each iteration. |
| `-config <file>` | Load INI config. Default search order: exe-local `config/spect_config.ini`, exe-local `spect_config.ini`, parent `config/`, cwd `config/`, cwd `spect_config.ini`. CLI args override config. |
| `-vb 0/1/2` | Verbosity (0=quiet, 1=info default, 2=debug). |

## Multi-bed reconstruction

Use `-dfList` to load multiple bed projections at once. The list may contain 1 to 5 beds. If paths contain `BedN`, `-dfList` and `-bProjList` are sorted by numeric bed order before loading. `-bedOverlap` is a fixed Bed1..Bed5 table: slot `b1` applies to `Bed1-Bed2`, `b2` to `Bed2-Bed3`, `b3` to `Bed3-Bed4`, and `b4` to `Bed4-Bed5`; `b5` is kept as the fixed five-bed placeholder. For example, `Bed2,Bed3,Bed4` uses `b2,b3`. The reconstruction dimensions in `-dim` are **per-bed** sizes; the combined output Z dimension becomes `Nz * beds` when `-bedOverlap=0`, or `Nz * beds - sum(overlaps)` when overlaps are enabled. Beds are processed using **Virtual Bed Stacking**, meaning system matrix memory usage remains identical to a single bed. Background projections can be supplied via `-bProjList`. Memory mapping (`-mapCSR on`) is supported for multi-bed; when `-ctac` is enabled, the recon uses dynamic on-the-fly attenuation factors so the mapped CSR remains read-only.

The shipped multi-bed default is `bed_overlap=16,16,16,0,0` with `crop_end=0,0`; use the same values for the mouse and spiral-hose validation datasets. `crop_end` remains zero so valid signal is not removed at the global ends.

Output orientation controls are explicit. `-zbedflip 1` flips each `-dfList` bed segment in Z before writing; `-zoutflip 1` then flips the final stitched output in Z; `-xoutflip 1` flips the final stitched output in X. `-cropEnd top,end` is applied after those flips and is only responsible for final top/end trimming.

## Release package layout

The RC package layout should keep runtime files self-contained under `bin/`, with config files next to the executable and DLLs:

```text
SPECT_recon_<version>/
  bin/
    SPECT_recon.exe
    *.dll
    config/
      spect_config.ini
      algo_mlem.conf
      algo_bsrem.conf
      algo_admm.conf
    matrix/
      <matrix_name>/
        metadata.bin
        chunk_000.csrb
        chunk_001.csrb
        chunk_002.csrb
        chunk_003.csrb
        manifest.json
  README.md
  PARAMETERS_CN.md
  RELEASE_NOTES.md
  SHA256SUMS.txt
```

Matrix directories are optional in the RC package. When a package contains
exactly one `bin/matrix/<matrix_name>/metadata.bin`, the executable selects it
automatically; when multiple matrix directories are present, set
`default_chunk_csr_dir = matrix/<matrix_name>` in `bin/config/spect_config.ini`
or pass `-chunkCSRDir bin/matrix/<matrix_name>`.

## Documentation

### Quick Links
- **[Documentation Index](doc/README.md)** — maintained docs only
- **[Release 2026-07-08](doc/RELEASE_20260708_CN.md)** — current release boundary
- **[Usage (中文)](doc/USAGE_CN.md)** — CLI reference
- **[Capability Matrix (中文)](doc/Capability_Matrix_CN.md)** — support boundary and durable decisions
- **[Field recon migration](doc/Recon_Interface_Migration_Field_CN.md)** — ThreeModalities → current RC
- **[Legacy B1/B2 Parity](doc/Legacy_MLEM_B1B2_Parity_CN.md)** — 260601 release regression
- **[Compact Chunk CSRB Format](test_data/6module/260601/CompactChunkCSRB_Reconstruction_README_CN.md)** — metadata/chunk layout
- **[Scripts](scripts/README.md)** — validation and helper scripts

## Environment variables (compact / CUDA)

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPECT_CSR_MMAP_MIN_BYTES` | code default | Minimum standard CSR file size to mmap payload |
| `SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES` | off (`uint64_t::max`) | Minimum chunk file size to mmap payload; below threshold uses full copy load (2026-06-02 benchmark default) |
| `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` | host RAM–derived | Host-side chunk cache budget (e.g. `24g`) |
| `SPECT_COMPACT_CHUNK_DEVICE_CACHE_BYTES` | GPU–derived | Device resident chunk cache budget |
| `SPECT_CHUNK_VALIDATE` | on | Set to `off` to skip compact chunk validation warnings/checks |
| `SPECT_COMPACT_DEVICE_RATIO` | off | Enable compact device-side ratio helper path |
| `SPECT_COMPACT_DEVICE_BACKWARD` | off | Enable compact device-side backward accumulation helper path |
| `SPECT_CUDA_DEVICE` | unset | Select CUDA device at startup; `-cudaDevice <id>` overrides it |
| `SPECT_CUDA_SPMV_ALG` | cuSPARSE default | SpMV algorithm selection when supported |
| `SPECT_CUDA_SPMV_FWD_ALG` | `SPECT_CUDA_SPMV_ALG` | Forward SpMV algorithm override |
| `SPECT_CUDA_SPMV_BWD_ALG` | `SPECT_CUDA_SPMV_ALG` | Backward SpMV algorithm override |
| `SPECT_CUDA_SYNC_LEGACY` | off | Set to `1` to force synchronous H2D/compute/D2H in compact float SpMV (disables PR21 pinned/async staging) |

mmap 失败时会 stderr 输出 `[WARN] compact-chunk mmap failed ... falling back to full copy load`。

## Build & Run

### Windows

Recommended release build (CUDA compact backend):

```bat
powershell -ExecutionPolicy Bypass -File scripts\clean_extra_builds.ps1
build_release.bat
cd staging\SPECT_recon_rc_dev
.\bin\SPECT_recon.exe ...
```

**Directory roles** (not duplicate builds):

| Path | Role |
|------|------|
| `build/Release/` | CMake compile tree (`.obj`, `.lib`, logs); exe at `build/Release/bin/Release/SPECT_recon.exe` |
| `staging/SPECT_recon_rc_dev/` | `cmake --install` dev package — RC layout (`bin/`, `PARAMETERS_CN.md`); run `.\bin\SPECT_recon.exe` from package root |
| `dist/release_candidates/SPECT_recon_rc_*/` | Frozen RC archives + zip (not regenerated by `build_release.bat`) |

Legacy CPU-only path: `build.bat` (without `-DSPECT_ENABLE_CUDA_SPARSEOPS`). Both build and install trees place `config/` beside the exe; normal runs do not need `-config`.

If you prefer CMake presets (Ninja):

```powershell
cmake --preset x64-release
cmake --build --preset x64-release -j 16
cmake --install --preset x64-release
.\install\cmake-presets\x64-release\bin\SPECT_recon.exe ...
```

### Ubuntu / macOS (bash)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/SPECT_recon \
  -df spect_1101.mat -loadCSR out/sysmat.csrb -mapCSR on \
  -dim 200,200,200 -detN 40,40 -ang 360 -it 50:5 -th 8 \
  -psf 1.5,4 -vb 1 -fout recon_output/linux_sample
```
