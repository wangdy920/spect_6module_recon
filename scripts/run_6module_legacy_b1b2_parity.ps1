param(
    [string]$ProjectionBin = "",
    [string]$Matrix260525Dir = "test_data\6module\260525",
    [string]$Matrix260601Dir = "test_data\6module\260601",
    [string]$Matrix260601LegacyDir = "",
    [int]$Iterations = 50,
    [int]$Threads = 4,
    [string]$SpectExe = "staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe",
    [string]$LegacyExe = "build_legacy_6module\spect_recon_mlem_legacy.exe",
    [string]$LegacySource = "6module_bak_src\spect_recon_mlem.cu",
    [string]$CudaArch = "sm_86",
    [string]$OutputRoot = "",
    [string]$B1ReferenceImg = "test_data\sys_260601_recon_ref\recon_iter050.img",
    [string]$B2ReferenceImg = "test_data\legacy_ref\recon_iter050.img",
    [double]$MaxRmseOverMean = 0.02,
    [double]$MaxSumRelDiff = 0.005,
    [switch]$SkipCompile,
    [switch]$SkipButter,
    [switch]$SkipB1,
    [switch]$SkipB2,
    [switch]$SkipLegacy,
    [switch]$SkipCompare
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $repo $Path)
}

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Resolve-ProjectionBin {
    $defaultProj = "test_data\sys_260601_recon_ref\PSC_6module_24pin_10MBq_600s_test_1.bin"
    $candidates = @(
        $ProjectionBin,
        $defaultProj,
        "test_data\6module\PSC_6module_24pin_10MBq_600s_1.bin",
        "gatedata\6module\PSC_6module_24pin_10MBq_600s_test_1.bin"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    foreach ($c in $candidates) {
        $p = Resolve-RepoPath $c
        if (Test-Path -LiteralPath $p) { return $p }
    }
    throw "Projection BIN not found. Tried: $($candidates -join ', ')"
}

function Test-LegacyMatrixDir([string]$Dir) {
    if (-not (Test-Path -LiteralPath $Dir)) { return $false }
    $meta = Join-Path $Dir "metadata.bin"
    if (-not (Test-Path -LiteralPath $meta)) { return $false }
    return @(Get-ChildItem -LiteralPath $Dir -Filter "chunk_*.bin" -File).Count -gt 0
}

function Invoke-Logged([string]$Label, [string]$Exe, [string[]]$ArgList, [string]$LogPath) {
    Ensure-Directory (Split-Path -Parent $LogPath)
    $cmd = "$Exe $($ArgList -join ' ')"
    "`$ $cmd" | Set-Content -Encoding UTF8 -LiteralPath $LogPath
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $oldEap = $ErrorActionPreference
    $oldNativeEap = $PSNativeCommandUseErrorActionPreference
    $ErrorActionPreference = "Continue"
    $PSNativeCommandUseErrorActionPreference = $false
    $tmpOut = "$LogPath.stdout.tmp"
    try {
        & $Exe @ArgList > $tmpOut 2> $null
        $code = $LASTEXITCODE
        if (Test-Path -LiteralPath $tmpOut) {
            $text = (Get-Content -LiteralPath $tmpOut -Raw) -replace "`0", ""
            Add-Content -Encoding UTF8 -LiteralPath $LogPath -Value $text
        }
    } finally {
        Remove-Item -LiteralPath $tmpOut -ErrorAction SilentlyContinue
        $PSNativeCommandUseErrorActionPreference = $oldNativeEap
        $ErrorActionPreference = $oldEap
    }
    $sw.Stop()
    if ($code -ne 0) {
        throw "$Label failed (exit $code). Log: $LogPath"
    }
    return [pscustomobject]@{
        label = $Label
        exit_code = $code
        wall_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        log = $LogPath
    }
}

function Find-IterOutput([string]$Prefix, [int]$Iter) {
    $mhd = "${Prefix}_it${Iter}.mhd"
    if (Test-Path -LiteralPath $mhd) { return $mhd }
    $mhd2 = Join-Path (Split-Path -Parent $Prefix) ("$(Split-Path -Leaf $Prefix)_it${Iter}.mhd")
    if (Test-Path -LiteralPath $mhd2) { return $mhd2 }
    throw "Missing MHD for prefix $Prefix iter $Iter"
}

function Find-LegacyIterRaw([string]$LegacyOut, [int]$Iter) {
    $raw = Join-Path $LegacyOut ("recon_iter{0:D3}.raw" -f $Iter)
    if (Test-Path -LiteralPath $raw) { return $raw }
    $final = Join-Path $LegacyOut "recon_final.raw"
    if (Test-Path -LiteralPath $final) { return $final }
    throw "Missing legacy raw in $LegacyOut for iter $Iter"
}

function Compare-AgainstReference(
    [string]$Label,
    [string]$CandidateMhd,
    [string]$ReferencePath,
    [string]$MetricsDir,
    [string]$MetricsSuffix,
    [double]$MaxRmseOverMean,
    [double]$MaxSumRelDiff
) {
    $json = Join-Path $MetricsDir "${Label}_${MetricsSuffix}.json"
    & python (Resolve-RepoPath "scripts\compare_compact_legacy.py") `
        --candidate $CandidateMhd `
        --reference $ReferencePath `
        --out-json $json `
        --max-rmse-over-mean $MaxRmseOverMean `
        --max-sum-rel-diff $MaxSumRelDiff
    $passed = ($LASTEXITCODE -eq 0)
    if ($passed) {
        Write-Host "${Label} vs ${MetricsSuffix}: PASSED ($json)"
    } else {
        Write-Warning "${Label} vs ${MetricsSuffix}: threshold miss (see $json)"
    }
    return [pscustomobject]@{
        label = $Label
        metrics_json = $json
        passed = $passed
        reference = $ReferencePath
        candidate = $CandidateMhd
    }
}

function Compare-AgainstLegacyReference(
    [string]$Label,
    [string]$CandidateMhd,
    [string]$LegacyOut,
    [string]$MetricsDir,
    [double]$MaxRmseOverMean,
    [double]$MaxSumRelDiff,
    [int]$Iterations
) {
    $legacyRaw = Find-LegacyIterRaw $LegacyOut $Iterations
    return Compare-AgainstReference `
        $Label $CandidateMhd $legacyRaw $MetricsDir "vs_fresh_legacy" `
        $MaxRmseOverMean $MaxSumRelDiff
}

$projPath = Resolve-ProjectionBin
$matrix525 = Resolve-RepoPath $Matrix260525Dir
$matrix601 = Resolve-RepoPath $Matrix260601Dir
$matrix601Legacy = if ([string]::IsNullOrWhiteSpace($Matrix260601LegacyDir)) {
    $matrix601
} else {
    Resolve-RepoPath $Matrix260601LegacyDir
}
$csrb525 = Join-Path $matrix525 "matrix_6module.csrb"
$meta525 = Join-Path $matrix525 "metadata.bin"
$spectPath = Resolve-RepoPath $SpectExe
$legacyPath = Resolve-RepoPath $LegacyExe
$b1RefImg = Resolve-RepoPath $B1ReferenceImg
$b2RefImg = Resolve-RepoPath $B2ReferenceImg
$hasB1RefImg = Test-Path -LiteralPath $b1RefImg
$hasB2RefImg = Test-Path -LiteralPath $b2RefImg

$hasLegacy525 = Test-LegacyMatrixDir $matrix525
$hasLegacy601 = Test-LegacyMatrixDir $matrix601Legacy

foreach ($p in @($projPath, $matrix525, $matrix601, $spectPath)) {
    if (-not (Test-Path -LiteralPath $p)) {
        throw "Required path missing: $p"
    }
}
if (-not $SkipB2) {
    foreach ($p in @($csrb525, $meta525)) {
        if (-not (Test-Path -LiteralPath $p)) {
            throw "Required B2 path missing: $p"
        }
    }
}
if (-not $SkipLegacy -and -not $SkipB2 -and -not $hasLegacy525) {
    throw "B2/B2 requires legacy-readable matrix dir (metadata.bin + chunk_*.bin): $matrix525"
}
if (-not $SkipLegacy -and -not $SkipB1 -and -not $hasLegacy601) {
    Write-Warning "B1 fresh legacy skipped: no chunk_*.bin under $matrix601Legacy. B1 SPECT + archived compare still run if -SkipB1 not set."
    $SkipLegacyB1 = $true
} else {
    $SkipLegacyB1 = $SkipB1
}

if (-not $SkipCompile -and -not $SkipLegacy) {
    $legacyDir = Split-Path -Parent $legacyPath
    Ensure-Directory $legacyDir
    $src = Resolve-RepoPath $LegacySource
    Write-Host "Compiling legacy MLEM -> $legacyPath"
    $nvccArgs = @(
        "-O3", "-std=c++14", "-arch=$CudaArch",
        $src, "-o", $legacyPath,
        "-lcusparse", "-lcufft"
    )
    & nvcc @nvccArgs
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
}

if (-not $OutputRoot) {
    $OutputRoot = "recon_out\legacy_b1b2_parity_it$Iterations"
}
$outRoot = Resolve-RepoPath $OutputRoot
$metricsDir = Join-Path $outRoot "metrics"
Ensure-Directory $outRoot
Ensure-Directory (Join-Path $outRoot "logs")
Ensure-Directory $metricsDir

$commonSpect = @(
    "-config", "__missing_legacy_parity__.ini",
    "-matrixLayout", "compact-symmetric",
    "-opti", "mlem",
    "-subsets", "1",
    "-it", "$Iterations",
    "-th", "$Threads"
)
$geom525 = @(
    "-dim", "108,108,80",
    "-vox", "0.3,0.3,0.3",
    "-ang", "360",
    "-detN", "48,48"
)

$results = [System.Collections.Generic.List[object]]::new()
$comparisons = [System.Collections.Generic.List[object]]::new()
$cases = @("plain")
if (-not $SkipButter) { $cases += "butter" }

foreach ($case in $cases) {
    $legacyB2Out = Join-Path $outRoot "legacy_b2_$case"
    $legacyB1Out = Join-Path $outRoot "legacy_b1_$case"
    $b2Prefix = Join-Path $outRoot "b2_$case\recon"
    $b1Prefix = Join-Path $outRoot "b1_$case\recon"

    $legacyCommon = @(
        "--proj", $projPath,
        "--iters", "$Iterations",
        "--gpus", "1",
        "--save-every", "$Iterations",
        "--conv-mode", "none",
        "--sym", "auto"
    )
    if ($case -eq "butter") {
        $legacyCommon += @("--butter-cutoff", "0.35", "--butter-order", "2", "--butter-mode", "every")
    }

    if (-not $SkipLegacy -and -not $SkipB2) {
        Ensure-Directory $legacyB2Out
        $legacyB2Args = @("--matrix-dir", $matrix525, "--out", $legacyB2Out) + $legacyCommon
        $r = Invoke-Logged "legacy_b2_$case" $legacyPath $legacyB2Args (Join-Path $outRoot "logs\legacy_b2_$case.log")
        $results.Add($r) | Out-Null
    }

    if (-not $SkipLegacy -and -not $SkipLegacyB1) {
        Ensure-Directory $legacyB1Out
        $legacyB1Args = @("--matrix-dir", $matrix601Legacy, "--out", $legacyB1Out) + $legacyCommon
        $r = Invoke-Logged "legacy_b1_$case" $legacyPath $legacyB1Args (Join-Path $outRoot "logs\legacy_b1_$case.log")
        $results.Add($r) | Out-Null
    }

    if (-not $SkipB2) {
        Ensure-Directory (Split-Path -Parent $b2Prefix)
        $b2Args = @(
            "-df", $projPath,
            "-loadCSR", $csrb525,
            "-compactMeta", $meta525,
            "-fout", $b2Prefix
        ) + $geom525 + $commonSpect
        if ($case -eq "butter") {
            $b2Args += @("-butter", "0.35,2")
        }
        $r2 = Invoke-Logged "b2_$case" $spectPath $b2Args (Join-Path $outRoot "logs\b2_$case.log")
        $results.Add($r2) | Out-Null
    }

    if (-not $SkipB1) {
        Ensure-Directory (Split-Path -Parent $b1Prefix)
        $b1Args = @(
            "-df", $projPath,
            "-chunkCSRDir", $matrix601,
            "-fout", $b1Prefix
        ) + $commonSpect
        if ($case -eq "butter") {
            $b1Args += @("-butter", "0.35,2")
        }
        $r1 = Invoke-Logged "b1_$case" $spectPath $b1Args (Join-Path $outRoot "logs\b1_$case.log")
        $results.Add($r1) | Out-Null
    }

    if ($SkipCompare) { continue }

    if (-not $SkipB2) {
        $b2Mhd = Find-IterOutput $b2Prefix $Iterations
        if (-not $SkipLegacy) {
            $comparisons.Add((Compare-AgainstLegacyReference `
                "b2_$case" $b2Mhd $legacyB2Out $metricsDir `
                $MaxRmseOverMean $MaxSumRelDiff $Iterations)) | Out-Null
        }
        if ($case -eq "plain" -and $hasB2RefImg) {
            $comparisons.Add((Compare-AgainstReference `
                "b2_$case" $b2Mhd $b2RefImg $metricsDir "vs_archived_260525_img" `
                $MaxRmseOverMean $MaxSumRelDiff)) | Out-Null
        }
    }

    if (-not $SkipB1) {
        $b1Mhd = Find-IterOutput $b1Prefix $Iterations
        if (-not $SkipLegacy -and -not $SkipLegacyB1) {
            $comparisons.Add((Compare-AgainstLegacyReference `
                "b1_$case" $b1Mhd $legacyB1Out $metricsDir `
                $MaxRmseOverMean $MaxSumRelDiff $Iterations)) | Out-Null
        }
        if ($case -eq "plain" -and $hasB1RefImg) {
            $comparisons.Add((Compare-AgainstReference `
                "b1_$case" $b1Mhd $b1RefImg $metricsDir "vs_archived_260601_img" `
                $MaxRmseOverMean $MaxSumRelDiff)) | Out-Null
        }
    }
}

$summary = [pscustomobject]@{
    projection = $projPath
    projection_note = "B1/B2 默认共用 test_data/sys_260601_recon_ref/PSC_6module_24pin_10MBq_600s_test_1.bin"
    matrix_b2 = $matrix525
    matrix_b1 = $matrix601
    archived_reference_b1 = $(if ($hasB1RefImg) { $b1RefImg } else { $null })
    archived_reference_b2 = $(if ($hasB2RefImg) { $b2RefImg } else { $null })
    matrix_b1_legacy_bin_dir = $(if ($hasLegacy601) { $matrix601Legacy } else { $null })
    iterations = $Iterations
    cases = $cases
    output_root = $outRoot
    comparison_logic = "B1/B1 and B2/B2 only. Fresh legacy reference when chunk_*.bin available; plain archived img: sys_260601 (B1) / legacy_ref (B2)."
    runs = $results
    comparisons = $comparisons
}
$summary | ConvertTo-Json -Depth 8 |
    Set-Content -Encoding UTF8 -LiteralPath (Join-Path $outRoot "run_summary.json")
Write-Host "Done. Summary: $(Join-Path $outRoot 'run_summary.json')"
