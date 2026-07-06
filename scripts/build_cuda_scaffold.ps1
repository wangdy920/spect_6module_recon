param(
    [string]$BuildDir = ".\build_cuda_scaffold",
    [switch]$Build,
    [switch]$EnableCudaSelection,
    [switch]$EnableTests,
    [string]$Target
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildDirAbs = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))

$vcvarsCandidates = @(
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    "D:\win_programFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

$vcvars = $vcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($vcvars)) {
    throw "vcvars64.bat not found in known locations."
}

if (-not $env:VCPKG_ROOT -or -not (Test-Path (Join-Path $env:VCPKG_ROOT "scripts\buildsystems\vcpkg.cmake"))) {
    $vcpkgCandidates = @(
        "D:\win_programFiles\cppDev\vcpkg",
        "D:\vcpkg",
        "C:\vcpkg"
    )
    $env:VCPKG_ROOT = $vcpkgCandidates | Where-Object {
        Test-Path (Join-Path $_ "scripts\buildsystems\vcpkg.cmake")
    } | Select-Object -First 1
}

if ([string]::IsNullOrWhiteSpace($env:VCPKG_ROOT)) {
    throw "VCPKG_ROOT is not set and no standalone vcpkg installation was found."
}

if (-not $env:VCPKG_INSTALLED_DIR) {
    $env:VCPKG_INSTALLED_DIR = Join-Path $env:VCPKG_ROOT "installed"
}

New-Item -ItemType Directory -Force -Path $buildDirAbs | Out-Null

$enableTestsValue = if ($EnableTests) { "ON" } else { "OFF" }
$enableCudaSelectionValue = if ($EnableCudaSelection) { "ON" } else { "OFF" }
$toolchainFile = "$($env:VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake"
$installPrefix = "$repoRoot/staging/SPECT_recon_rc_dev"

$batchLines = @(
    "@echo off",
    "setlocal",
    "chcp 65001 >nul 2>nul",
    ('call "{0}"' -f $vcvars),
    "if errorlevel 1 exit /b %errorlevel%",
    ('cmake -S "{0}" ^' -f $repoRoot),
    ('  -B "{0}" ^' -f $buildDirAbs),
    '  -G "NMake Makefiles" ^',
    "  -DCMAKE_BUILD_TYPE=Release ^",
    ('  -DCMAKE_TOOLCHAIN_FILE="{0}" ^' -f $toolchainFile),
    ('  -DVCPKG_INSTALLED_DIR="{0}" ^' -f $env:VCPKG_INSTALLED_DIR),
    "  -DVCPKG_TARGET_TRIPLET=x64-windows ^",
    ('  -DVCPKG_MANIFEST_DIR="{0}" ^' -f $repoRoot),
    "  -DVCPKG_MANIFEST_MODE=ON ^",
    "  -DVCPKG_APPLOCAL_DEPS=ON ^",
    "  -DSPECT_BUILD_CUDA_SCAFFOLD=ON ^",
    ('  -DSPECT_BUILD_TESTS={0} ^' -f $enableTestsValue),
    ('  -DSPECT_ENABLE_CUDA_SPARSEOPS={0} ^' -f $enableCudaSelectionValue),
    ('  -DCMAKE_INSTALL_PREFIX="{0}"' -f $installPrefix)
)

$tempBat = Join-Path $env:TEMP ("spect_cuda_scaffold_{0}.bat" -f ([System.Guid]::NewGuid().ToString("N")))
Set-Content -Path $tempBat -Value ($batchLines -join [Environment]::NewLine) -Encoding Ascii

Write-Host "[CUDA-SCAFFOLD] Configuring..." -ForegroundColor Cyan
Write-Host $tempBat -ForegroundColor DarkGray
cmd /c $tempBat
if ($LASTEXITCODE -ne 0) {
    Remove-Item -Path $tempBat -ErrorAction SilentlyContinue
    throw "CMake configure failed with exit code $LASTEXITCODE"
}

if ($Build) {
    $targetName = if (-not [string]::IsNullOrWhiteSpace($Target)) {
        $Target
    } elseif ($EnableTests -and $EnableCudaSelection) {
        "spect_cuda_scaffold_tests"
    } else {
        "spect_sparse_cuda"
    }
    $buildBatchLines = @(
        "@echo off",
        "setlocal",
        "chcp 65001 >nul 2>nul",
        ('call "{0}"' -f $vcvars),
        "if errorlevel 1 exit /b %errorlevel%",
        ('cmake --build "{0}" --config Release --target {1}' -f $buildDirAbs, $targetName)
    )
    $buildBat = Join-Path $env:TEMP ("spect_cuda_scaffold_build_{0}.bat" -f ([System.Guid]::NewGuid().ToString("N")))
    Set-Content -Path $buildBat -Value ($buildBatchLines -join [Environment]::NewLine) -Encoding Ascii
    Write-Host "[CUDA-SCAFFOLD] Building target $targetName ..." -ForegroundColor Cyan
    Write-Host $buildBat -ForegroundColor DarkGray
    cmd /c $buildBat
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -Path $buildBat -ErrorAction SilentlyContinue
        Remove-Item -Path $tempBat -ErrorAction SilentlyContinue
        throw "CMake build failed with exit code $LASTEXITCODE"
    }
    Remove-Item -Path $buildBat -ErrorAction SilentlyContinue
}

Remove-Item -Path $tempBat -ErrorAction SilentlyContinue

Write-Host "[CUDA-SCAFFOLD] Done. BuildDir=$buildDirAbs" -ForegroundColor Green