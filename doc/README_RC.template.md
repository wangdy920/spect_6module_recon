# SPECT_recon RC __PACKAGE_NAME__

This Windows x64 CUDA RC build is self-contained under `bin/`.

## Layout

```text
__PACKAGE_NAME__/
  bin/
    SPECT_recon.exe
    *.dll
    config/
      spect_config.ini
      algo_mlem.conf
      algo_bsrem.conf
      algo_admm.conf
    matrix/
      <matrix_name>/          optional, not bundled in this package
  README.md
  PARAMETERS_CN.md
  RELEASE_NOTES.md
  SHA256SUMS.txt
  docs/SPECT_recon_help.txt
```

## Default Discovery

Run `bin/SPECT_recon.exe` directly. The executable first searches its own
directory for `config/spect_config.ini` and `config/algo_<alg>.conf`, so normal
commands do not need `-config`.

Use `-config <path>` only when intentionally overriding the packaged config.
Use `-algoConf <path>` only when intentionally overriding algorithm defaults.

Compact chunk matrix discovery is also exe-local:

1. CLI `-chunkCSRDir <dir>` wins.
2. Otherwise `[SystemMatrix] default_chunk_csr_dir` in `bin/config/spect_config.ini`
   is resolved relative to `bin/`.
3. Otherwise, if `bin/matrix/` contains exactly one directory with
   `metadata.bin`, that matrix is used automatically.
4. If multiple matrix directories exist, the executable fails fast and lists
   the candidates.

## Minimal Example

If this package has exactly one matrix under `bin/matrix/<matrix_name>/`:

```powershell
.\bin\SPECT_recon.exe `
  -df D:\data\PSC_6module_24pin_10MBq_600s_test_1.bin `
  -o recon_output `
  -it 50:5
```

For multiple matrix folders, either set `default_chunk_csr_dir = matrix/<name>`
in `bin/config/spect_config.ini` or add `-chunkCSRDir .\bin\matrix\<name>`.

For multi-bed input:

```powershell
.\bin\SPECT_recon.exe `
  -dfList "D:\scan\Bed1.bin;D:\scan\Bed2.bin;D:\scan\Bed3.bin;D:\scan\Bed4.bin" `
  -bedOverlap 0,0,0,0,0 `
  -o recon_multibed `
  -it 50:5
```

`-dfList` and `-bProjList` are sorted by `BedN` when present. `bedOverlap` is a
fixed `Bed1..Bed5` table: `b1=Bed1-Bed2`, `b2=Bed2-Bed3`, `b3=Bed3-Bed4`,
`b4=Bed4-Bed5`, `b5` reserved.

Detailed Chinese parameter notes are in `PARAMETERS_CN.md`.
