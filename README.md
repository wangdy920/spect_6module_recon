# SPECT-recon

High-performance 3D SPECT reconstruction pipeline with C++17 modular architecture and CUDA acceleration. The project provides:

- **C++ recon engine**: consumes projection data (MAT/BIN) + compact chunk CSRB system matrix and runs MLEM or BSREM-II with PSF, TV/RDP regularisation, masks, and Butterworth filtering.
- **Compact-symmetric backend**: **default matrix 260601** — `-chunkCSRDir` loads `metadata.bin` + `chunk_*.csrb` (`test_data/6module/260601`, 88B/96B metadata, Sym90Z). This is the **only** matrix input path; single-file CSRB/MAT entries have been removed.
- **CUDA GPU acceleration**: cuSPARSE sparse SpMV, cuFFT Butterworth, NVRTC kernels for PSF/RDP/TV gradients. Single GPU.
- **6-module matrix generation reference**: `6module_bak_src/` holds the CUDA v4 system-matrix generator, the bedmerge reconstruction prototype, and the MAT→BIN projection tool.

## Project Status (Snapshot: 2026-08-11)

- **Canonical evidence**: `reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html` and local certificate `recon_out/prototype_sieve_castor_fwhm1p3_final_20260811/current_production_parity_metrics.json` record the protected `sieve + CASToR + FWHM 1.3 mm` comparison. All five frozen 50-iteration endpoints pass.
- **Default system matrix**: 260601 compact chunk (`-chunkCSRDir` / `bin/matrix/260601` auto-discovery). B1 plain 50-iter vs `test_data/sys_260601_recon_ref/recon_iter050.img` remains signed off (`rmse/ref_mean ≈ 1.06×10⁻⁵`).
- **Multi-bed output**: one canonical path: physical Bed order + `z_bed_flip=1`, center 66 layers per bed, prototype robust gain fit, fixed 56-layer step, and 10-layer normalized cosine blend. There are no selectable stitch or object-aware seam-repair modes.
- **Output filters**: release default preserves the unfiltered canonical volume and independently derives CASToR Smooth 0.30 mm/3σ and 2× zero-padded Butterworth 5 cycles/cm/order 8 copies. `[InputOutput] output_filters_overwrite_primary=true` instead writes enabled stages to the unsuffixed primary in Smooth→Butterworth order; suffixed copies stay independent and the manifest records the policy.
- **Target GPUs** (consumer GeForce RTX): Release builds include sm_86, sm_89, and sm_120; CUDA Toolkit ≥ 12.8 (prefer 13.x), matching runtime DLLs, and a CUDA 13.0+ capable driver are required.
- **Public backends**: CUDA-only sparse ops; cuFFT Butterworth; PSF `none|operator|sieve` with `adjoint|castor` kernels; TV/RDP regularizers.
- **Optimization algorithms**: MLEM / BSREM-II, single-subset full update. OSEM/ADMM and multi-GPU paths are removed.
- See `multipinhole/doc/Capability_Matrix_CN.md`.

**Reconstruction boundaries**

- **Current implementation**: [multipinhole/](multipinhole/) — production
  multipinhole source, headers, configuration, tests, docs, and CMake targets.
- **Future implementation**: [parallel_hole/](parallel_hole/) — deliberately
  contains no code until a parallel-hole reconstruction and correctness check
  are introduced.

**Project Structure**
- **Release history**: [CHANGELOG.md](CHANGELOG.md) — embedded in `SPECT_recon.exe -v` and copied to the package root; `--version` remains one-line for scripts
- **Core Code**: [multipinhole/include/](multipinhole/include/), [multipinhole/src/](multipinhole/src/) — current modular C++17 implementation
- **Tests**: [multipinhole/tests/](multipinhole/tests/) — GoogleTest unit tests
- **Runtime configuration**: [multipinhole/config/](multipinhole/config/) — source defaults copied into the unchanged deployed `bin/config/` layout
- **Documentation**: [multipinhole/doc/](multipinhole/doc/) — maintained usage, capability, format, and field-integration contracts
- **Scripts**: [scripts/](scripts/) — reusable validation, QA, and packaging helpers
- **Matrix generator & prototype (reference)**: [6module_bak_src/](6module_bak_src/) — CUDA v4 generator, bedmerge prototype, MAT→BIN tool

## Typical workflow

1. **Generate or deploy the matrix**: v4 generator (`6module_bak_src/spect_sysmat_PSC_6module_24pin_YSO_v4.cu`) produces `metadata.bin + chunk_*.csrb`; deploy as `bin/matrix/<name>/` next to the exe, or keep the test matrix at `test_data/6module/260601`.

2. **Reconstruct**:

   ```powershell
   .\SPECT_recon.exe -df proj.bin -chunkCSRDir bin\matrix\260601 `
       -dim 120,120,220 -detN 48,48 -ang 360 -it 50:50 -th 8 `
       -psf 1.3,3 -psfMode sieve -psfKernel castor -cyl 30,21.6 -vox 0.3,0.3,0.3 `
       -fout recon_output/run1
   ```

   Each run writes a per-reconstruction log file alongside the outputs (`<outputDir>/<outputName>_YYYYMMDD_HHMMSS_mmm.log`). Absolute `-fout`/`-o` (drive letter or UNC) keeps both the `.log` and MHD/RAW under that absolute directory; relative paths resolve against the process CWD. The console prints `Logging console output to: ...` and `Output directory: ...`.

## Key switches

| Option | Description |
| --- | --- |
| `-chunkCSRDir` / `bin/matrix/<name>` | Compact chunk matrix directory containing `metadata.bin` and `chunk_*.csrb` (**only matrix entry**). Resolution priority is: CLI `-chunkCSRDir`, then `[SystemMatrix] default_chunk_csr_dir` in `bin/config/spect_config.ini`, then auto-use the single `bin/matrix/*` directory that contains `metadata.bin`. If multiple candidates exist, pass `-chunkCSRDir` explicitly. |
| `-df <file>` | Projection input can be MAT or the BIN layout written by `6module_bak_src/mat_convert_bin.m`; BIN stores only `num_angles` and `rays_per_angle`, so keep `-detN <detZ,detX>` when detector shape matters. |
| `[InputOutput] merge_time_segments=true` | With `-df <case/YYYYMMDD_HHMMSS/SPECT>`, scan sibling timestamp directories, require the same `BedN.bin` set and geometry in each, then sum matching beds in memory before reconstruction. Default is `false`; no merged projection files are written. |
| `[InputOutput] output_filters_overwrite_primary=true` | Write enabled output-stage filters to `<fout>_it<N>.mhd/.raw` in Smooth→Butterworth order. Default `false` preserves the unfiltered primary; `_smooth` and `_butter` remain independent audit copies in both modes. |
| `-badDetectorFile <file>` | Skip hardware blocks in both measured projection rows and sensitivity. Entries may be preferred global numbers `0..95` or detailed `link-<3|4>-FPGA-<0..5>-dector-<0..7>` IDs. The shipped list contains detector 68; this protected-prototype exception maps to angles 0..59, z 24..35, x 24..35. |
| `-detZRowWeightsFile <file>` / `-normFactorFile <file>` | Optional normalization inputs. `-detZRowWeightsFile` contains detector efficiency weights directly. `-normFactorFile` contains full projection correction factors `c` in `<uint32 angles><uint32 rays><float64 values>` BIN form; the background-free MLEM path applies reciprocal efficiency `1/c` to sensitivity. Default is empty/off. |
| `-psf [fwhm_mm[,N]]` | Image-domain PSF; FWHM is in mm and N truncates the kernel at N*sigma. Release default: `1.3,3`. |
| `-psfMode none\|operator\|sieve` / `-psfKernel adjoint\|castor` | `operator` applies PSF in-loop; `sieve` also filters save copies without changing the trajectory. `castor` is the release default and uses voxel-integrated erf taps, zero boundaries, Z-only normalization, and forward-reuse backward. |
| `-opti mlem|bsrem` | Select optimisation algorithm (overrides config). Single-subset full update; there is no `-subsets` switch. |
| `-bsrem <alpha0[,gamma]>` | BSREM‑II step parameters (overrides config). |
| `-reg <type,beta,...>` | Regularisation. Public types: `None`, `TV`/`TVIsotropic`, `RDP`. For TV use `-reg TV,beta,delta`; for RDP use `-reg RDP,beta,gamma,delta`. `beta` is independent of `-bsrem` alpha/gamma. |
| `-bProj <file>` | Background/scatter projection term (MAT or BIN); added to forward model `yhat`. |
| `-dfList <a;b;...>` | Multi-bed projection list (MAT/BIN, comma/semicolon separated); paths containing `BedN` are sorted by numeric bed order; max 5 beds. |
| `-bedOverlap <b1,b2,b3,b4,b5>` | Fixed Bed1..Bed5 overlap table. `b1` maps `Bed1-Bed2`, `b2` maps `Bed2-Bed3`, etc.; one value is replicated across all fixed slots. |
| `-bProjList <a;b;...>` | Multi-bed background projections (MAT/BIN), sorted by `BedN` when present; list length must match beds. |
| `-cropEnd <top[,end]>` | Zero final output Z top/end slices after output flips; Z size is unchanged. |
| `-zbedflip <0|1>` / `-zoutflip <0|1>` / `-xoutflip <0|1>` | Per-bed Z / whole-volume Z / whole-volume X flips. Pair with preprocess: `-chunkCSRDir` + `identity` → keep package `[MultiBed] z_bed_flip` default. |
| `-maskFromAtt <th>` | Generate body mask from attenuation map and zero outside mask each iteration. |
| `-config <file>` | Load INI config. Default search order: exe-local `config/spect_config.ini`, exe-local `spect_config.ini`, parent `config/`, cwd `config/`, cwd `spect_config.ini`. CLI normally wins; `[Compatibility] recon_sections_override_cli=true` makes the global INI authoritative only for Regularization/PSF/Filtering/Postprocess. Unknown or retired CLI arguments are noted and ignored. |
| `-vb 0/1/2` | Verbosity (0=quiet, 1=info default, 2=debug). |
| `-v` / `--version` | `-v` prints the exact build identity and complete embedded changelog; `--version` prints only the machine-readable build identity. |


## Multi-bed reconstruction

`-dfList` accepts 1–5 projection files and sorts paths containing `BedN` by bed number. The physical contract is fixed: `Bed1` is the maximum-physical-Z bed; subsequent beds step by 16.8 mm toward lower physical Z. `[MultiBed] z_bed_flip=1` is the shipped default and reverses each local bed inside the compact multi-bed mapping; do not compensate by reversing the file list.

For repeated acquisitions of one case, set `[InputOutput] merge_time_segments=true` and pass one timestamp's `SPECT` directory, for example `-df D:\data\predata\260810_dedenzo_mdp_6\20260810_210556\SPECT`. Immediate sibling directories named `YYYYMMDD_HHMMSS` are processed chronologically. Every included segment must contain the identical `BedN.bin` set with matching BIN size/header/geometry; mismatches and non-finite sums fail before reconstruction. The run manifest records every source path per bed.

The shipped 220-layer 6module geometry uses `-bedOverlap 164,164,164,164,0`, but the overlap is a derived quantity: `step_layers = round(bed_step_mm / voxel_z)` (default `-bedStepMm 16.8`; INI `[MultiBed] bed_step_mm`) and `overlap = baseNz - step_layers`. Omitting `-bedOverlap` auto-derives it from the scanner bed step and the matrix geometry; a wrong value is corrected with a warning (`bed_step_mm` is authoritative). The per-bed center crop is `step + 10` layers (66 for the 260601 geometry). Reconstruction segments are non-overlapping in storage. At save time the only production stitch:

1. takes the central `step + 10` layers from each bed (66 at 0.3 mm);
2. estimates adjacent-bed scalar gains with the prototype log-ratio median/MAD rule, anchored at Bed1 (always applies the prototype log-ratio chain);
3. advances by `step_layers` and blends the shared 10 layers with normalized cosine weights;
4. writes `Z = crop + (bed_count - 1) × step` layers, then applies whole-output Z/X flips and `cropEnd` zeroing.

The fixed output crop is `100×100` in XY. Object-aware seam repair, automatic profile classification, output FOV masking, `sigmoid`, `direct`, and offline auto-stitch controls are not production paths and have been removed.

`-bProjList` follows the same BedN ordering and must match the projection list. `-zoutflip 1` and `-xoutflip 1` are whole-output transforms; the shipped defaults remain zero.

## Release package layout

The RC package layout should keep runtime files self-contained under `bin/`, with config files next to the executable and DLLs:

```text
SPECT_recon_<version>/
  bin/
    SPECT_recon.exe
    *.dll
    config/
      spect_config.ini
      bad_dector
      algo_mlem.conf
      algo_bsrem.conf
    matrix/
      <matrix_name>/
        metadata.bin
        chunk_*.csrb
        manifest.json
  README.md
  CHANGELOG.md
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
- **[Documentation Index](multipinhole/doc/README.md)** — maintained docs only
- **[Prototype convergence evidence](reports/2026-08-10-prototype-sieve-castor-fwhm1p3-convergence.html)** — current algorithm/output decision record
- **[Usage (中文)](multipinhole/doc/USAGE_CN.md)** — CLI reference
- **[Capability Matrix (中文)](multipinhole/doc/Capability_Matrix_CN.md)** — support boundary and durable decisions
- **[Field recon migration](multipinhole/doc/Recon_Interface_Migration_Field_CN.md)** — ThreeModalities → current canonical executable
- **[CTAC / auto-mu validation](multipinhole/doc/CTAC_AutoMu_Validation_CN.md)** — CTAC / auto-mu cylinder debug gate
- **[Compact Chunk CSRB Format](test_data/6module/260601/CompactChunkCSRB_Reconstruction_README_CN.md)** — metadata/chunk layout
- **[Matrix format (中文)](multipinhole/doc/Format_Reader_Matrix_CN.md)** — v4 chunk CSRB spec
- **[Scripts](scripts/README.md)** — reusable validation and helper scripts
- **[Tools](tools/README.md)** — standalone CLI tools

## Environment variables (compact / CUDA)

| Variable | Default | Purpose |
| --- | --- | --- |
| `SPECT_COMPACT_CHUNK_MMAP_MIN_BYTES` | off (`uint64_t::max`) | Minimum chunk file size to mmap payload; below threshold uses full copy load (2026-06-02 benchmark default) |
| `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` | `25g` | Host-side chunk cache budget; `25g` covers the current 26.1 GB six-chunk matrix |
| `SPECT_COMPACT_BRANCH_BATCH` | on | Batch symmetry branches and beds in cuSPARSE SpMM; set `0` for the protected scalar-branch performance baseline |
| `SPECT_CHUNK_VALIDATE` | on | Set to `off` to skip compact chunk validation warnings/checks |
| `SPECT_CUDA_DEVICE` | unset | Select CUDA device at startup; `-cudaDevice <id>` overrides it |
| `SPECT_CUDA_SPMV_ALG` | cuSPARSE default | SpMV algorithm selection when supported |
| `SPECT_CUDA_SPMV_FWD_ALG` | `SPECT_CUDA_SPMV_ALG` | Forward SpMV algorithm override |
| `SPECT_CUDA_SPMV_BWD_ALG` | `SPECT_CUDA_SPMV_ALG` | Backward SpMV algorithm override |

mmap 失败时会 stderr 输出 `[NOTE] compact-chunk mmap failed ... falling back to full copy load`。16 GB 显存部署保留 6 GiB 给临时矩阵、SpMM、PSF 和输出滤波；50 次迭代的低时延配置建议使用 64 GB 主存及 SSD/NVMe 矩阵。32 GB 主存可通过降低 `SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES` 运行，但无法保证相同吞吐。

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

Both build and install trees place `config/` beside the exe; normal runs do not need `-config`.

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
  -df proj.bin -chunkCSRDir bin/matrix/260601 \
  -dim 200,200,200 -detN 40,40 -ang 360 -it 50:5 -th 8 \
  -psf 1.3,3 -psfMode sieve -psfKernel castor -vb 1 -fout recon_output/linux_sample
```
