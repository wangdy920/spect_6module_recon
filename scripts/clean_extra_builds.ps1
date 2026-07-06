param(
    [switch]$DryRun,
    [switch]$KeepLegacy
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$removeDirs = @(
    "build_codex",
    "build_codex_cuda",
    "build_codex_cuda_attr",
    "out"
)
if (-not $KeepLegacy) {
    $removeDirs += "build_legacy_6module"
}
foreach ($name in $removeDirs) {
    $path = Join-Path $repo $name
    if (-not (Test-Path -LiteralPath $path)) { continue }
    if ($DryRun) {
        Write-Host "[dry-run] would remove $path"
    } else {
        Write-Host "Removing $path ..."
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

Write-Host @"

Canonical directories:
  build\Release\              CMake compile tree (intermediate + built exe under bin\Release\)
  staging\SPECT_recon_rc_dev\   cmake --install dev package (RC layout: bin/, PARAMETERS_CN.md)
  dist\release_candidates\SPECT_recon_rc_*\  frozen RC archives + zip
  build\cmake-presets\        Ninja preset build trees (optional)
  install\cmake-presets\      Ninja preset install trees (optional)

staging\ is not a second compile; it copies Release artifacts out of build\ for deployment.
Run from package root: cd staging\SPECT_recon_rc_dev && .\bin\SPECT_recon.exe ...
"@
