# SPECT_recon RC __PACKAGE_NAME__

Windows x64, CUDA only; EXE and app-local DLLs reside in `bin/`. This template describes the current configuration interface. A frozen RC retains its original executable, configuration and validation scope; see its `RELEASE_NOTES.md`.

## Layout

```text
__PACKAGE_NAME__/
  bin/
    SPECT_recon.exe
    *.dll
    config/
      spect_config.ini
      profiles/                 isotope/system reconstruction recipes
      algo_mlem.conf            legacy compatibility configurations
      algo_bsrem.conf
    matrix/<isotope>/<matrix_id>/  deploy required matrices separately
    cache/compact_denom/         generated sensitivity cache
  README.md
  PARAMETERS_CN.md
  RELEASE_NOTES.md
  SHA256SUMS.txt
  docs/SPECT_recon_help.txt
```

## Configuration and discovery

The EXE discovers its local `config/spect_config.ini`. The global INI selects **one** recipe via `default_config_Tc99m`, `default_config_Lu177` or `default_config`. `-algoConf <file>` replaces that recipe; missing fields inherit global defaults. `-config` accepts a global INI or a complete recipe containing `[Algorithm]` without a global `default_config`.

`recon_sections_override_cli=false` gives explicit CLI > recipe > global. With true, configured reconstruction fields use recipe > global > CLI, including algorithm, iterations, PSF, Mask, Filtering, Regularization and matrix. A blank value (`key =`) is never a configured field at any layer. Normalization is bound only in the global INI as `[Normalization] factor_file_<isotope>` (`factor_file2_Lu177` for joint); an explicit `-normFactorFile` replaces it. Runtime input/output, device, threads and logging remain caller-owned. Source defaults use false; the updated 6Module deployment preserves true. `-opti` changes the optimizer without switching the selected recipe. Legacy `algo_<alg>.conf` discovery applies only when no default recipe is configured.

Matrix IDs resolve exactly under `[SystemMatrix] root` as `root/id` or `root/isotope/id`. Missing or ambiguous IDs fail. Matrix/norm/cache config paths are EXE-relative; recipe paths in the global INI are global-config-relative. Explicit matrix and cache CLI flags follow the same priority policy.

## Examples

Deploy the selected recipe's matrices first. From the package root:

```powershell
# Default Tc OSEM50 x 10
.\bin\SPECT_recon.exe -isotope Tc99m -df "D:\scan\Bed1.bin" -o recon_tc
# Explicit Tc TV recipe
.\bin\SPECT_recon.exe -isotope Tc99m -df "D:\scan\Bed1.bin" -algoConf bin/config/profiles/uniformity_Tc99m_BSREM_TV.conf -o recon_tv
# Default Lu OSEM50 x 10, native CSRB; merged projection and one matrix
.\bin\SPECT_recon.exe -isotope Lu177 -df "D:\scan\Bed1.bin" -o recon_lu
# Multi-bed Tc: retain real consecutive BedN identities; default physical step 16.8 mm
.\bin\SPECT_recon.exe -isotope Tc99m -dfList "D:\scan\Bed1.bin;D:\scan\Bed2.bin" -o recon_multibed
# Inspect the selected Lu model without reconstruction
.\bin\SPECT_recon.exe -isotope Lu177 -dumpConf
```

`-it A:B` means A full iterations with B angle subsets. `-it A` retains inherited subsets. `-save-every N` controls intermediate saving; default0 saves only the final iteration. Tc and Lu defaults are standard OSEM50 x 10 without regularization, using native CSRB. OSEM and BSREM share angle subsets, including homogeneous dual CSRB and experimental CSRQ. Subsets with CTAC are not enabled. RDP/TV must be selected explicitly; `-opti bsrem` alone does not enable RDP.

PSF stays enabled (sieve/castor, FWHM1.3mm), with existing Mask/bed processing. Output Gaussian is disabled. Only final primary and a separate output-only Butterworth copy (5cycles/cm, order8) are saved by default. Primary includes the existing sieve/save processing and is not the internal iteration array.

Lu channel order is 208/113, weights0.1036/0.0617 applied once in the count model. Norm is disabled until matching calibration files are supplied; joint requires independent files for both windows or neither. Do not reuse a merged-window norm. The original merged-Lu recipe03 and experimental joint BSREM/CSRQ recipe05 remain explicit alternatives to default native-CSRB recipe06.

SC60 uses the same recipes with `config/scanners/SC60-MPH3.ini` and an SC60 global INI (full matrix grid, PSF1.5mm, no PSC mask or bed step; the device package uses `bed_step_mm=12`). The former `sc60_osem1000.conf` validation recipe is kept under `config/historical/`. Multi-bed geometry comes only from `bed_step_mm`; the retired `bed_overlap` / `-bedOverlap` are ignored.

See [Chinese parameters and recipes](PARAMETERS_CN.md). Matrices and observations are not included. These recipes retain their tested data/model scope; they are not universal optima or a completed activity calibration. CSRQ remains experimental.
