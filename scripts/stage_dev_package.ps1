param(
    [string]$PackageRoot = "",
    [string]$MatrixName = "260601"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($MatrixName) -or
    $MatrixName -in @(".", "..") -or
    [IO.Path]::GetFileName($MatrixName) -ne $MatrixName) {
    throw "MatrixName must be one directory name, got: $MatrixName"
}
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
$template = Join-Path $repo "multipinhole\doc\PARAMETERS_CN.template.md"
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
        $existingMatrix = Get-Item -LiteralPath $matrixDst -Force
        if (($existingMatrix.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            # Remove only the old development junction; never recurse into its
            # read-only source matrix.
            [IO.Directory]::Delete($matrixDst)
        } elseif (-not $existingMatrix.PSIsContainer) {
            throw "Matrix package path is not a directory: $matrixDst"
        }
    }
    if (-not (Test-Path -LiteralPath $matrixDst)) {
        New-Item -ItemType Directory -Path $matrixDst | Out-Null
    }
    if (Test-Path -LiteralPath (Join-Path $matrixDst "metadata.bin")) {
        Write-Host "[stage] keep existing package matrix: bin\matrix\$MatrixName"
    } else {
        $matrixLink = Join-Path $matrixDst "SOURCE_MATRIX_DIRECTORY.txt"
        @(
            "Development package matrix source (not copied):",
            $matrixSrc,
            "",
            "Use scripts/package_rc.ps1 -IncludeMatrix for a self-contained field package."
        ) | Set-Content -LiteralPath $matrixLink -Encoding UTF8
        Write-Host "[stage] matrix placeholder: bin\matrix\$MatrixName (source matrix remains read-only)"
    }
} else {
    Write-Host "[stage] skip matrix placeholder (missing $matrixSrc)"
}

Write-Host "[stage] package root: $PackageRoot"
Write-Host "[stage] invoke from package root:"
Write-Host "  cd $PackageRoot"
Write-Host "  .\bin\SPECT_recon.exe -df <proj.bin> -chunkCSRDir `"$matrixSrc`" -o recon_output -it 50:5"
