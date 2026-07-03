param(
    [string]$SpectExe = ".\build_cuda_scaffold\bin\SPECT_recon.exe",
    [string]$ProjectionBin = ".\gatedata\6module\PSC_6module_24pin_10MBq_600s_1.bin",
    [string]$CompactCSR = ".\gatedata\6module\matrix_6module.csrb",
    [string]$OutputPrefix = ".\recon_out\6module_compact_cylinder_mlem_psf15_butter",
    [string]$LegacyReference = ".\recon_out\legacy_6module_compact_mlem_psf15_butter\recon_iter001.raw",
    [double]$MaxRmseOverMean = 0.02,
    [double]$MaxSumRelDiff = 0.005,
    [switch]$SkipRecon,
    [switch]$SkipCompare
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path (Get-Location) $Path)
}

function Assert-Exists {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

$spectExePath = Resolve-RepoPath $SpectExe
$projectionPath = Resolve-RepoPath $ProjectionBin
$csrPath = Resolve-RepoPath $CompactCSR
$outputPrefixPath = Resolve-RepoPath $OutputPrefix
$legacyReferencePath = Resolve-RepoPath $LegacyReference

Assert-Exists $spectExePath "SPECT_recon.exe"
Assert-Exists $projectionPath "Projection BIN"
Assert-Exists $csrPath "Compact CSRB"

$outputDir = Split-Path -Parent $outputPrefixPath
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

if (-not $SkipRecon) {
    $reconArgs = @(
        "-df", $projectionPath,
        "-loadCSR", $csrPath,
        "-matrixLayout", "compact-symmetric",
        "-alg", "mlem",
        "-subsets", "1",
        "-it", "1",
        "-sparseBackend", "cuda",
        "-butter", "0.35,2",
        "-butterBackend", "cufft",
        "-psf", "0.637",
        "-psfMode", "Operator",
        "-detN", "48,48",
        "-ang", "360",
        "-dim", "108,108,80",
        "-vox", "0.3",
        "-fout", $outputPrefixPath
    )
    & $spectExePath @reconArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SPECT_recon failed with exit code $LASTEXITCODE"
    }
}

$mhdPath = "${outputPrefixPath}_it1.mhd"
Assert-Exists $mhdPath "Candidate MHD"

$qaDir = Resolve-RepoPath ".\recon_out\qa"
if (-not (Test-Path -LiteralPath $qaDir)) {
    New-Item -ItemType Directory -Path $qaDir | Out-Null
}

$stem = [System.IO.Path]::GetFileName($outputPrefixPath)
$qaJson = Join-Path $qaDir "${stem}_it1_qa.json"
$qaPng = Join-Path $qaDir "${stem}_it1_views.png"

python .\scripts\qa_compact_cylinder.py `
    --image $mhdPath `
    --out-json $qaJson `
    --out-png $qaPng
if ($LASTEXITCODE -ne 0) {
    throw "qa_compact_cylinder.py failed with exit code $LASTEXITCODE"
}
$qaMetrics = Get-Content -LiteralPath $qaJson -Raw | ConvertFrom-Json
if (-not $qaMetrics.calibrated_1iter_passed) {
    throw "Calibrated 1-iteration QA gate failed; see $qaJson"
}

$compareJson = Join-Path $qaDir "${stem}_vs_legacy_cuda_iter001_compare.json"
if (-not $SkipCompare) {
    if (Test-Path -LiteralPath $legacyReferencePath) {
        python .\scripts\compare_compact_legacy.py `
            --candidate $mhdPath `
            --reference $legacyReferencePath `
            --out-json $compareJson `
            --max-rmse-over-mean $MaxRmseOverMean `
            --max-sum-rel-diff $MaxSumRelDiff
        if ($LASTEXITCODE -ne 0) {
            throw "compare_compact_legacy.py failed with exit code $LASTEXITCODE"
        }
        $compareMetrics = Get-Content -LiteralPath $compareJson -Raw | ConvertFrom-Json
        if (-not $compareMetrics.passed) {
            throw "Legacy CUDA comparison failed; see $compareJson"
        }
    } else {
        Write-Warning "Legacy reference not found; skipped legacy comparison: $legacyReferencePath"
    }
}

Write-Host "6module compact acceptance artifacts:"
Write-Host "  candidate: $mhdPath"
Write-Host "  qa_json:   $qaJson"
Write-Host "  qa_png:    $qaPng"
Write-Host "  qa_gate:   calibrated_1iter_passed=$($qaMetrics.calibrated_1iter_passed)"
if ((-not $SkipCompare) -and (Test-Path -LiteralPath $compareJson)) {
    Write-Host "  compare:   $compareJson"
}
