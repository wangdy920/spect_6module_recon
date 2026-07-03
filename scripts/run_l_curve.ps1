param(
    [string]$Exe = "",

    # Base arguments passed to SPECT_recon.exe (excluding -reg). Recommended to pass as tokens to preserve paths with spaces.
    [Parameter(Mandatory = $true)]
    [string[]]$BaseArg,

    [string]$RegType = "TVIsotropic",

    # Comma list "0.001,0.01,0.1" or logspace "start:end:steps" (e.g. "1e-4:1e-1:8")
    [Parameter(Mandatory = $true)]
    [string]$Betas,

    # Additional reg params after beta, comma-separated (depends on reg type)
    [string]$RegParams = "",

    [string]$Output = "l_curve_results.png"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot

function Resolve-SpectExe {
    param([string]$UserExe)
    if ($UserExe) {
        if (-not (Test-Path $UserExe)) {
            throw "找不到可执行文件: $UserExe"
        }
        return $UserExe
    }

    $candidates = @(
        (Join-Path $RootDir "out/install/Release/bin/SPECT_recon.exe"),
        (Join-Path $RootDir "out/install/x64-release/bin/SPECT_recon.exe"),
        (Join-Path $RootDir "build/bin/Release/SPECT_recon.exe"),
        (Join-Path $RootDir "build_refactored/bin/Release/SPECT_recon.exe")
    )

    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    throw "未找到 SPECT_recon.exe。请先运行 build.bat 或 build_comparison.ps1，或用 -Exe 指定路径。"
}

function Resolve-Python {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Cmd = "python"; Extra = @() }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Cmd = "py"; Extra = @("-3") }
    }
    throw "未找到 Python（python/py）。请安装 Python 或将其加入 PATH。"
}

$ExePath = Resolve-SpectExe -UserExe $Exe
$Py = Resolve-Python

$ScriptPath = Join-Path $RootDir "utils/run_l_curve.py"
if (-not (Test-Path $ScriptPath)) {
    throw "缺少脚本: $ScriptPath"
}

$PyArgs = @(
    $ScriptPath,
    "--exe", $ExePath,
    "--reg_type", $RegType,
    "--betas", $Betas,
    "--output", $Output
)

foreach ($a in $BaseArg) {
    $PyArgs += @("--base_arg", $a)
}

if ($RegParams) {
    $PyArgs += @("--reg_params", $RegParams)
}

Write-Host "[INFO] EXE: $ExePath" -ForegroundColor Gray
Write-Host "[INFO] Script: $ScriptPath" -ForegroundColor Gray
Write-Host "[INFO] Output: $Output" -ForegroundColor Gray

& $Py.Cmd @($Py.Extra) @PyArgs
exit $LASTEXITCODE

