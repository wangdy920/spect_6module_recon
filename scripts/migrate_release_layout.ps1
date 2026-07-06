param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$moves = @(
    @{ From = "release"; To = "dist\release_candidates"; Mode = "contents" },
    @{ From = "dist\rc"; To = "dist\release_candidates"; Mode = "dir" },
    @{ From = "install\Release"; To = "staging\SPECT_recon_rc_dev"; Mode = "dir" }
)

foreach ($m in $moves) {
    $from = Join-Path $repo $m.From
    $to = Join-Path $repo $m.To
    if (-not (Test-Path -LiteralPath $from)) {
        Write-Host "[skip] missing $from"
        continue
    }
    if ($DryRun) {
        Write-Host "[dry-run] $($m.Mode): $from -> $to"
        continue
    }
    $toParent = Split-Path $to -Parent
    if ($toParent -and -not (Test-Path -LiteralPath $toParent)) {
        New-Item -ItemType Directory -Force -Path $toParent | Out-Null
    }
    if ($m.Mode -eq "contents") {
        if (-not (Test-Path -LiteralPath $to)) {
            New-Item -ItemType Directory -Force -Path $to | Out-Null
        }
        Get-ChildItem -LiteralPath $from | ForEach-Object {
            Move-Item -LiteralPath $_.FullName -Destination $to -Force
        }
        Remove-Item -LiteralPath $from -Force
        Write-Host "[move] $from\* -> $to\"
    } else {
        if (Test-Path -LiteralPath $to) {
            Remove-Item -LiteralPath $to -Recurse -Force
        }
        Move-Item -LiteralPath $from -Destination $to -Force
        Write-Host "[move] $from -> $to"
    }
}

$installRoot = Join-Path $repo "install"
if ((Test-Path -LiteralPath $installRoot) -and -not (Get-ChildItem -LiteralPath $installRoot -ErrorAction SilentlyContinue)) {
    if (-not $DryRun) { Remove-Item -LiteralPath $installRoot -Force }
    Write-Host "[cleanup] removed empty install\"
}

Write-Host @"

New layout:
  staging\SPECT_recon_rc_dev\   dev/QA package (RC layout, from build_release.bat)
  dist\release_candidates\SPECT_recon_rc_*\  frozen RC archives + zip
  build\Release\                CMake compile tree only
"@
