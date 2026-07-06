param(
    [string]$PackageRoot = "",
    [string]$MatrixName = "260601"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
    $PackageRoot = Join-Path $repo "staging\SPECT_recon_rc_dev"
}
$PackageRoot = (Resolve-Path -LiteralPath $PackageRoot -ErrorAction Stop).Path

$exe = Join-Path $PackageRoot "bin\SPECT_recon.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    throw "Expected installed exe at $exe — run build_release.bat first."
}

$packageName = Split-Path $PackageRoot -Leaf
$template = Join-Path $repo "doc\PARAMETERS_CN.template.md"
$destDoc = Join-Path $PackageRoot "PARAMETERS_CN.md"
if (-not (Test-Path -LiteralPath $template)) {
    throw "Missing template: $template"
}
Copy-Item -LiteralPath $template -Destination $destDoc -Force
Write-Host "[stage] $destDoc"

$matrixRoot = Join-Path $PackageRoot "bin\matrix"
$matrixSrc = Join-Path $repo "test_data\6module\$MatrixName"
$matrixDst = Join-Path $matrixRoot $MatrixName
if (Test-Path -LiteralPath $matrixSrc) {
    New-Item -ItemType Directory -Force -Path $matrixRoot | Out-Null
    if (Test-Path -LiteralPath $matrixDst) {
        Remove-Item -LiteralPath $matrixDst -Force -Recurse
    }
    New-Item -ItemType Junction -Path $matrixDst -Target $matrixSrc | Out-Null
    Write-Host "[stage] matrix junction: bin\matrix\$MatrixName -> test_data\6module\$MatrixName"
} else {
    Write-Host "[stage] skip matrix junction (missing $matrixSrc)"
}

Write-Host "[stage] package root: $PackageRoot"
Write-Host "[stage] invoke from package root:"
Write-Host "  cd $PackageRoot"
Write-Host "  .\bin\SPECT_recon.exe -df <proj.bin> -o recon_output -it 50:5"
