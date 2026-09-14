param(
    [string]$ReleaseDate = "",
    [string]$PackageName = "",
    [string]$StagingRoot = "",
    [switch]$SkipBuild,
    [switch]$IncludeMatrix,
    [switch]$CleanPrevious,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$gitExe = "D:\DevToolchain\Git\cmd\git.exe"
if (-not (Test-Path -LiteralPath $gitExe)) {
    $gitExe = "git"
}

if ([string]::IsNullOrWhiteSpace($StagingRoot)) {
    $StagingRoot = Join-Path $repo "staging\SPECT_recon_rc_dev"
}
if ([string]::IsNullOrWhiteSpace($ReleaseDate)) {
    $ReleaseDate = Get-Date -Format "yyyyMMdd"
}

Push-Location $repo
try {
    $gitHashRaw = (& $gitExe rev-parse --short HEAD 2>$null)
    if (-not $gitHashRaw) { $gitHashRaw = "local" }
    $dirty = (& $gitExe status --porcelain --untracked-files=no 2>$null)
    $gitDirty = [bool]$dirty
    # Package directory/zip names stay clean (no -dirty). Build -v may still report dirty until the tree is committed.
    $gitHash = $gitHashRaw
} finally {
    Pop-Location
}

if ([string]::IsNullOrWhiteSpace($PackageName)) {
    $PackageName = "SPECT_recon_rc_${ReleaseDate}_${gitHash}"
}

$distRoot = Join-Path $repo "dist\release_candidates"
$packageRoot = Join-Path $distRoot $PackageName
$packageBin = Join-Path $packageRoot "bin"
$packageConfig = Join-Path $packageBin "config"
$packageDocs = Join-Path $packageRoot "docs"

$runtimeDlls = @(
    "cudart64_13.dll",
    "cusparse64_12.dll",
    "cufft64_12.dll",
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcomp140.dll"
)

function Write-Step([string]$Message) {
    Write-Host "[package] $Message"
}

if ($gitDirty) {
    Write-Step "WARNING: working tree is dirty; package name uses clean hash '$gitHash' (exe -v may still show -dirty until rebuild on a clean tree)"
}

function Resolve-RuntimeDllPath([string]$Name) {
    $cudaRoot = $env:CUDA_PATH
    if (-not [string]::IsNullOrWhiteSpace($cudaRoot)) {
        $cudaCandidate = Join-Path $cudaRoot "bin\x64\$Name"
        if (Test-Path -LiteralPath $cudaCandidate) { return $cudaCandidate }
    }
    $sysCandidate = Join-Path $env:SystemRoot "System32\$Name"
    if (Test-Path -LiteralPath $sysCandidate) { return $sysCandidate }
    throw "Missing runtime DLL: $Name (checked CUDA_PATH and System32)"
}

if (-not $SkipBuild) {
    Write-Step "building via build_release.bat ..."
    if ($DryRun) {
        Write-Step "[dry-run] would run build_release.bat"
    } else {
        $buildBat = Join-Path $repo "build_release.bat"
        if (-not (Test-Path -LiteralPath $buildBat)) {
            throw "Missing $buildBat"
        }
        & cmd /c "`"$buildBat`""
        if ($LASTEXITCODE -ne 0) {
            throw "build_release.bat failed with exit code $LASTEXITCODE"
        }
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $StagingRoot "bin\SPECT_recon.exe"))) {
    throw "Staging exe missing: $StagingRoot\bin\SPECT_recon.exe (run build_release.bat first)"
}

if ($DryRun) {
    Write-Step "[dry-run] would create $packageRoot"
    exit 0
}

if ($CleanPrevious -and (Test-Path -LiteralPath $distRoot)) {
    Write-Step "remove previous RC outputs under $distRoot"
    Get-ChildItem -LiteralPath $distRoot -Force | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
}

if (Test-Path -LiteralPath $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageBin, $packageConfig, $packageDocs | Out-Null

Write-Step "copy staging bin -> $packageBin"
Get-ChildItem -LiteralPath (Join-Path $StagingRoot "bin") -Force | ForEach-Object {
    if ($_.Name -like "gtest*.dll") { return }
    # Matrix is handled once below; never copy the development placeholder or
    # its workstation-local source path into a frozen package.
    if ($_.Name -eq "matrix") { return }
    if ($_.Name -eq "output") { return }
    if ($_.Name -eq "config") { return } # copied once from the authoritative source below
    if ($_.PSIsContainer) {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $packageBin $_.Name) -Recurse -Force
    } else {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $packageBin $_.Name) -Force
    }
}

if ($IncludeMatrix) {
    $matrixSrc = Join-Path $repo "test_data\6module\260601"
    $matrixDst = Join-Path $packageBin "matrix\260601"
    if (-not (Test-Path -LiteralPath $matrixSrc)) {
        throw "Matrix source missing: $matrixSrc"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $matrixDst -Parent) | Out-Null
    Write-Step "copy matrix 260601 -> bin\matrix\260601"
    robocopy $matrixSrc $matrixDst /E /NFL /NDL /NJH /NJS /NC /NS | Out-Null
}

Write-Step "copy CUDA/MSVC runtime DLLs"
foreach ($dll in $runtimeDlls) {
    $src = Resolve-RuntimeDllPath $dll
    Copy-Item -LiteralPath $src -Destination (Join-Path $packageBin $dll) -Force
}

Write-Step "copy authoritative reconstruction configs"
Get-ChildItem -LiteralPath (Join-Path $repo "multipinhole\config") -File | ForEach-Object {
    if ($_.Name -in @("spect_config.ct_guided.ini", "spect_config.experiment.ini")) { return }
    Copy-Item -LiteralPath $_.FullName -Destination $packageConfig -Force
}

Copy-Item -LiteralPath (Join-Path $repo "multipinhole/config/profiles") -Destination $packageConfig -Recurse -Force

Write-Step "generate docs/SPECT_recon_help.txt and version.txt"
$helpPath = Join-Path $packageDocs "SPECT_recon_help.txt"
$versionPath = Join-Path $packageDocs "SPECT_recon_version.txt"
Push-Location $packageRoot
try {
    & (Join-Path $packageBin "SPECT_recon.exe") -help 2>&1 | Out-File -LiteralPath $helpPath -Encoding utf8
    & (Join-Path $packageBin "SPECT_recon.exe") -v 2>&1 | Out-File -LiteralPath $versionPath -Encoding utf8
} finally {
    Pop-Location
}

$paramTemplate = Join-Path $repo "multipinhole\doc\PARAMETERS_CN.template.md"
Copy-Item -LiteralPath $paramTemplate -Destination (Join-Path $packageRoot "PARAMETERS_CN.md") -Force


$readmeTemplate = Join-Path $repo "multipinhole\doc\README_RC.template.md"
if (Test-Path -LiteralPath $readmeTemplate) {
    $readme = Get-Content -LiteralPath $readmeTemplate -Raw -Encoding UTF8
    $readme = $readme -replace "__PACKAGE_NAME__", $PackageName
    Set-Content -LiteralPath (Join-Path $packageRoot "README.md") -Value $readme -Encoding UTF8
}

$releaseNotesPath = Join-Path $repo "multipinhole\doc\RELEASE_${ReleaseDate}_CN.md"
if (Test-Path -LiteralPath $releaseNotesPath) {
    Copy-Item -LiteralPath $releaseNotesPath -Destination (Join-Path $packageRoot "RELEASE_NOTES.md") -Force
} else {
    @(
        "# Release Notes - $PackageName",
        "",
        "Release date: $(Get-Date -Format 'yyyy-MM-dd')",
        "Git: $gitHash",
        "",
        "Built from staging/SPECT_recon_rc_dev with bundled CUDA/MSVC runtime DLLs."
    ) | Set-Content -LiteralPath (Join-Path $packageRoot "RELEASE_NOTES.md") -Encoding UTF8
}

Write-Step "write SHA256SUMS.txt"
$hashFile = Join-Path $packageRoot "SHA256SUMS.txt"
$hashLines = @()
Get-ChildItem -LiteralPath $packageRoot -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($packageRoot.Length + 1) -replace '\\', '/'
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $hashLines += "$hash  $rel"
}
$hashLines | Sort-Object | Set-Content -LiteralPath $hashFile -Encoding ASCII

$zipPath = Join-Path $distRoot "$PackageName.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Write-Step "create zip $zipPath"
# Compress-Archive fails on long matrix paths; tar handles them.
Push-Location $distRoot
try {
    & tar -a -cf $zipPath $PackageName
    if ($LASTEXITCODE -ne 0) {
        throw "tar zip failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
if (-not (Test-Path -LiteralPath $zipPath) -or ((Get-Item -LiteralPath $zipPath).Length -le 0)) {
    throw "zip missing or empty: $zipPath"
}

Write-Step "done: $packageRoot"
Write-Step "zip:   $zipPath"
Write-Step "invoke:"
Write-Host "  cd `"$packageRoot`""
Write-Host "  .\bin\SPECT_recon.exe -df <proj.bin> -o recon_output -it 50:1"
