param(
    [string]$ExePath = "",
    [string]$DataDir = ".\gatedata\SPECT_rectify_S_0105",
    [string]$CSRFile = ".\gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb",
    [string]$OutputDir = ".\recon_out\bsrem_kappa_sweep",
    [int]$Threads = 16,
    [string]$BedOverlap = "16",
    [int]$BaseNz = 100,
    [int]$SeamWindow = 2,
    [double]$SeamTolerance = 0.20,
    [switch]$EdgeCoverageAware = $true,
    [int]$CoarseIters = 6,
    [int]$ConfirmIters = 20,
    [string]$BsremAlpha = "1.0,0.05",
    [string]$RegSpec = "RDP,1e-4,2.0,0.01",
    [string]$Butter = "0.35,2",
    [string]$PriorMaskMhd = ".\recon_out\perf_check_fix\multibed_mlem_it20.mhd",
    [double]$MaskThreshold = 0.02,
    [switch]$SkipConfirm = $false
)

$ErrorActionPreference = "Stop"

function Resolve-ExePath {
    param([string]$InputPath)
    if ($InputPath -and (Test-Path $InputPath)) {
        return (Resolve-Path $InputPath).Path
    }
    $candidates = @(
        ".\out\install\Release\bin\SPECT_recon.exe",
        ".\build\bin\SPECT_recon.exe",
        ".\build\Release\SPECT_recon.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return (Resolve-Path $path).Path
        }
    }
    throw "SPECT_recon.exe not found. Run build.bat first or pass -ExePath."
}

function Assert-Exists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path $Path)) {
        throw "$Label not found: $Path"
    }
}

function Get-SeamZList {
    param(
        [int]$BaseNz,
        [int]$BedCount,
        [string]$BedOverlapSpec
    )
    if ($BedCount -le 1) {
        return ""
    }

    $parts = @($BedOverlapSpec -split "[,;]" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    if ($parts.Count -eq 0) {
        throw "Invalid -BedOverlap value: '$BedOverlapSpec'"
    }

    $overlaps = @()
    if ($parts.Count -eq 1) {
        $val = [int]$parts[0]
        for ($i = 0; $i -lt ($BedCount - 1); $i++) {
            $overlaps += $val
        }
    } elseif ($parts.Count -eq ($BedCount - 1)) {
        $overlaps = @($parts | ForEach-Object { [int]$_ })
    } else {
        throw "Bed overlap count must be 1 or bedCount-1. got=$($parts.Count), bedCount=$BedCount"
    }

    $seams = @()
    $zStart = 0
    for ($b = 1; $b -lt $BedCount; $b++) {
        $step = $BaseNz - $overlaps[$b - 1]
        if ($step -le 0) {
            throw "Invalid overlap: baseNz=$BaseNz overlap=$($overlaps[$b - 1])"
        }
        $zStart += $step
        $seams += $zStart
    }
    return ($seams -join ",")
}

function Parse-Metric {
    param([string]$LogPath)
    $line = Select-String -Path $LogPath -Pattern "^\[METRIC\]" | Select-Object -Last 1
    if (-not $line) {
        return [pscustomobject]@{
            beta = [double]::NaN
            likelihood = [double]::NaN
            penalty = [double]::NaN
            objective = [double]::NaN
        }
    }
    $m = [regex]::Match($line.Line, "Beta=([^\s]+)\s+Likelihood=([^\s]+)\s+Penalty=([^\s]+)\s+Objective=([^\s]+)")
    if (-not $m.Success) {
        return [pscustomobject]@{
            beta = [double]::NaN
            likelihood = [double]::NaN
            penalty = [double]::NaN
            objective = [double]::NaN
        }
    }
    return [pscustomobject]@{
        beta = [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        likelihood = [double]::Parse($m.Groups[2].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        penalty = [double]::Parse($m.Groups[3].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        objective = [double]::Parse($m.Groups[4].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    }
}

function Invoke-BSREMCase {
    param(
        [hashtable]$Case,
        [int]$Iters,
        [string]$Tag,
        [string]$Exe,
        [string[]]$CommonArgs,
        [string]$SeamZList,
        [string]$RefImage = ""
    )
    $name = $Case.Name
    $outPrefix = Join-Path $OutputDir "${Tag}_${name}"
    $args = @()
    $args += $CommonArgs
    $args += @("-it", ("{0}:{0}" -f $Iters))
    $args += @("-fout", $outPrefix)
    $args += @("-alg", "BSREM")
    $args += @("-subsets", "1")
    $args += @("-bsrem", $BsremAlpha)
    $args += @("-bsremCurv", "on")
    $args += @("-bsremCurvFreeze", "on")
    $args += @("-bsremCurvWarmup", "2")
    $args += @("-reg", $RegSpec)

    if ($Case.UseKappa) {
        $args += @("-bsremKappa", "on")
        $args += @("-bsremKappaClip", $Case.KappaClip)
    }
    else {
        $args += @("-bsremKappa", "off")
    }

    if ($Case.UseSoftMask) {
        $args += @("-bsremSoftMask", "on")
        $args += @("-bsremSoftBand", $Case.SoftBand.ToString())
        $args += @("-bsremSoftLambda", $Case.SoftLambda.ToString([System.Globalization.CultureInfo]::InvariantCulture))
        $args += @("-prior", $PriorMaskMhd)
        $args += @("-maskFromAtt", $MaskThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    }
    else {
        $args += @("-bsremSoftMask", "off")
    }

    Write-Host "--------------------------------------------------"
    Write-Host ("[{0}] Running {1}" -f $Tag, $name)
    Write-Host ("Output prefix: {0}" -f $outPrefix)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Exe @args
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    if ($exitCode -ne 0) {
        throw "Case '$name' failed with exit code $exitCode"
    }

    $mhdPath = "{0}_it{1}.mhd" -f $outPrefix, $Iters
    Assert-Exists -Path $mhdPath -Label "Output MHD"

    $logPath = Get-ChildItem -Path $OutputDir -Filter ("{0}_*.log" -f (Split-Path $outPrefix -Leaf)) |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
    Assert-Exists -Path $logPath -Label "Run log"

    $metric = Parse-Metric -LogPath $logPath

    $evalArgs = @("scripts/eval_mhd_stats.py", "--image", $mhdPath)
    if (Test-Path $PriorMaskMhd) {
        $evalArgs += @("--mask-source", $PriorMaskMhd, "--mask-threshold", $MaskThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    }
    if ($SeamZList) {
        $evalArgs += @("--seam-z-list", $SeamZList, "--seam-window", $SeamWindow.ToString())
    }
    if ($RefImage -and (Test-Path $RefImage)) {
        $evalArgs += @("--ref-image", $RefImage)
    }
    $statsJson = (& python @evalArgs)
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to evaluate MHD stats for '$name'"
    }
    $stats = $statsJson | ConvertFrom-Json

    $score = 0.0
    if ($null -ne $stats.inside_outside_contrast) {
        $score = [double]$stats.inside_outside_contrast
    }
    else {
        $score = [double]$stats.global_mean / [math]::Max([double]$stats.global_std, 1e-12)
    }

    return [pscustomobject]@{
        tag = $Tag
        name = $name
        iters = $Iters
        use_kappa = $Case.UseKappa
        kappa_clip = $Case.KappaClip
        use_softmask = $Case.UseSoftMask
        soft_band = $Case.SoftBand
        soft_lambda = $Case.SoftLambda
        runtime_sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        metric_beta = $metric.beta
        metric_likelihood = $metric.likelihood
        metric_penalty = $metric.penalty
        metric_objective = $metric.objective
        inside_mean = $stats.inside_mean
        outside_mean = $stats.outside_mean
        inside_p99 = $stats.inside_p99
        outside_p99 = $stats.outside_p99
        outside_sum_ratio = $stats.outside_sum_ratio
        inside_cv_pct = $stats.inside_cv_pct
        outside_cv_pct = $stats.outside_cv_pct
        contrast = $stats.inside_outside_contrast
        seam_score = $stats.seam_score
        seam_peak_max = $stats.seam_peak_max
        seam_baseline_mean = $stats.seam_baseline_mean
        rmse_ref = $stats.rmse_ref
        nrmse_ref_mean = $stats.nrmse_ref_mean
        nrmse_ref_std = $stats.nrmse_ref_std
        nrmse_ref_range = $stats.nrmse_ref_range
        score = $score
        output_mhd = $mhdPath
        log_path = $logPath
    }
}

function New-Case {
    param(
        [string]$Name,
        [bool]$UseKappa,
        [string]$KappaClip = "",
        [bool]$UseSoftMask = $false,
        [int]$SoftBand = 2,
        [double]$SoftLambda = 100.0
    )
    return @{
        Name = $Name
        UseKappa = $UseKappa
        KappaClip = $KappaClip
        UseSoftMask = $UseSoftMask
        SoftBand = $SoftBand
        SoftLambda = $SoftLambda
    }
}

$exe = Resolve-ExePath -InputPath $ExePath
Write-Host "Using executable: $exe"

$dataFiles = @(
    (Join-Path $DataDir "data_append_bed1.mat"),
    (Join-Path $DataDir "data_append_bed2.mat"),
    (Join-Path $DataDir "data_append_bed3.mat"),
    (Join-Path $DataDir "data_append_bed4.mat")
)

Assert-Exists -Path $CSRFile -Label "CSR file"
foreach ($df in $dataFiles) {
    Assert-Exists -Path $df -Label "Projection data"
}
if ($BaseNz -le 0) {
    throw "BaseNz must be positive."
}
$seamZList = Get-SeamZList -BaseNz $BaseNz -BedCount $dataFiles.Count -BedOverlapSpec $BedOverlap
if ($seamZList) {
    Write-Host ("Using seam z-list: {0} (window={1})" -f $seamZList, $SeamWindow)
}
if (-not (Test-Path $PriorMaskMhd)) {
    Write-Warning "Prior mask volume not found: $PriorMaskMhd"
    Write-Warning "Soft-mask cases will run without quality-mask evaluation support."
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$dataList = $dataFiles -join ","
$commonArgs = @(
    "-dfList", $dataList,
    "-dim", ("100,100,{0}" -f $BaseNz),
    "-vox", "0.3,0.3,0.3",
    "-ang", "360",
    "-detN", "40,40",
    "-th", $Threads.ToString(),
    "-loadCSR", $CSRFile,
    "-mapCSR", "on",
    "-butter", $Butter,
    "-bedOverlap", $BedOverlap,
    "-vb", "1"
)

$coarseCasesPhase1 = @(
    (New-Case -Name "baseline_no_kappa" -UseKappa:$false),
    (New-Case -Name "kappa_clip_0p10_10" -UseKappa:$true -KappaClip "0.1,10.0"),
    (New-Case -Name "kappa_clip_0p20_5" -UseKappa:$true -KappaClip "0.2,5.0"),
    (New-Case -Name "kappa_clip_0p05_20" -UseKappa:$true -KappaClip "0.05,20.0")
)

$results = @()
foreach ($case in $coarseCasesPhase1) {
    $results += Invoke-BSREMCase -Case $case -Iters $CoarseIters -Tag "coarse" -Exe $exe -CommonArgs $commonArgs -SeamZList $seamZList
}

$kappaOnly = $results | Where-Object { $_.tag -eq "coarse" -and $_.use_kappa -eq $true -and $_.use_softmask -eq $false } |
    Sort-Object score -Descending
if (-not $kappaOnly -or $kappaOnly.Count -eq 0) {
    throw "No kappa-only coarse results available."
}
$bestClip = $kappaOnly[0].kappa_clip
Write-Host ("Selected best coarse kappa clip: {0}" -f $bestClip)

$coarseCasesPhase2 = @(
    (New-Case -Name ("soft_b1_l50_{0}" -f $bestClip.Replace(",", "_")) -UseKappa:$true -KappaClip $bestClip -UseSoftMask:$true -SoftBand 1 -SoftLambda 50.0),
    (New-Case -Name ("soft_b2_l100_{0}" -f $bestClip.Replace(",", "_")) -UseKappa:$true -KappaClip $bestClip -UseSoftMask:$true -SoftBand 2 -SoftLambda 100.0),
    (New-Case -Name ("soft_b3_l100_{0}" -f $bestClip.Replace(",", "_")) -UseKappa:$true -KappaClip $bestClip -UseSoftMask:$true -SoftBand 3 -SoftLambda 100.0),
    (New-Case -Name ("soft_b2_l200_{0}" -f $bestClip.Replace(",", "_")) -UseKappa:$true -KappaClip $bestClip -UseSoftMask:$true -SoftBand 2 -SoftLambda 200.0)
)
foreach ($case in $coarseCasesPhase2) {
    $results += Invoke-BSREMCase -Case $case -Iters $CoarseIters -Tag "coarse" -Exe $exe -CommonArgs $commonArgs -SeamZList $seamZList
}

$coarseCsv = Join-Path $OutputDir "coarse_results.csv"
$results | Sort-Object score -Descending | Export-Csv -Path $coarseCsv -NoTypeInformation -Encoding UTF8
Write-Host "Coarse sweep results saved: $coarseCsv"

$confirmResults = @()
$confirmBaseline = $null
if (-not $SkipConfirm) {
    $coarseRanked = $results | Where-Object { $_.tag -eq "coarse" } | Sort-Object score -Descending
    $topCandidates = @()
    if ($coarseRanked.Count -gt 0) {
        $topCandidates += $coarseRanked[0]
    }
    if ($coarseRanked.Count -gt 1) {
        $topCandidates += $coarseRanked[1]
    }
    $baseline = $results | Where-Object { $_.name -eq "baseline_no_kappa" } | Select-Object -First 1

    $confirmDefs = @()
    if ($baseline) {
        $confirmDefs += (New-Case -Name "baseline_no_kappa" -UseKappa:$false)
    }
    foreach ($c in $topCandidates) {
        $confirmDefs += (New-Case -Name ("confirm_" + $c.name) -UseKappa:$c.use_kappa -KappaClip $c.kappa_clip -UseSoftMask:$c.use_softmask -SoftBand $c.soft_band -SoftLambda $c.soft_lambda)
    }

    foreach ($case in $confirmDefs) {
        $refImage = ""
        if ($null -ne $confirmBaseline -and $case.Name -ne "baseline_no_kappa") {
            $refImage = $confirmBaseline.output_mhd
        }
        $r = Invoke-BSREMCase -Case $case -Iters $ConfirmIters -Tag "confirm" -Exe $exe -CommonArgs $commonArgs -SeamZList $seamZList -RefImage $refImage
        $confirmResults += $r
        if ($case.Name -eq "baseline_no_kappa") {
            $confirmBaseline = $r
        }
    }

    if ($null -ne $confirmBaseline) {
        $baseSeam = [double]$confirmBaseline.seam_score
        foreach ($r in $confirmResults) {
            $ratio = [double]::NaN
            if ($null -ne $r.seam_score -and -not [double]::IsNaN($baseSeam) -and ($baseSeam -gt 0.0)) {
                $ratio = [double]$r.seam_score / $baseSeam
            }
            $r | Add-Member -NotePropertyName seam_ratio_vs_baseline -NotePropertyValue $ratio -Force
        }
    }
}

# Defensive cleanup: keep only case-result objects.
$confirmResults = @($confirmResults | Where-Object {
    $null -ne $_ -and
    $null -ne $_.PSObject -and
    $null -ne $_.PSObject.Properties["name"]
})

$confirmCsv = Join-Path $OutputDir "confirm_results.csv"
$confirmResults | Sort-Object score -Descending | Export-Csv -Path $confirmCsv -NoTypeInformation -Encoding UTF8
$confirmRankedCsv = Join-Path $OutputDir "confirm_results_ranked.csv"

$summaryPath = Join-Path $OutputDir "SUMMARY.md"
$bestFinal = $null
if ($confirmResults.Count -gt 0) {
    $rankPool = @($confirmResults)
    if ($EdgeCoverageAware -and $null -ne $confirmBaseline) {
        $threshold = 1.0 + [math]::Max(0.0, $SeamTolerance)
        $safePool = @()
        foreach ($r in $rankPool) {
            $ratio = [double]$r.seam_ratio_vs_baseline
            if ([double]::IsNaN($ratio) -or [double]::IsInfinity($ratio) -or $ratio -le $threshold) {
                $safePool += $r
            }
        }
        if ($safePool.Count -gt 0) {
            $rankPool = $safePool
        }
    }
    $ranked = $rankPool | Sort-Object `
        @{Expression = "score"; Descending = $true}, `
        @{Expression = { if ($_.use_softmask) { 1 } else { 0 } }; Descending = $false}, `
        @{Expression = "metric_penalty"; Descending = $false}, `
        @{Expression = "runtime_sec"; Descending = $false}
    $ranked | Export-Csv -Path $confirmRankedCsv -NoTypeInformation -Encoding UTF8
    $bestFinal = $ranked | Select-Object -First 1
}
else {
    $bestFinal = $results | Sort-Object score -Descending | Select-Object -First 1
}

$summary = @()
$summary += "# BSREM Kappa Sweep Summary"
$summary += ""
$summary += ("- Coarse iterations: {0}" -f $CoarseIters)
$summary += ("- Confirm iterations: {0}" -f $ConfirmIters)
$summary += ("- Regularization: {0}" -f $RegSpec)
$summary += ("- Prior mask: {0} (threshold {1})" -f $PriorMaskMhd, $MaskThreshold)
$summary += ("- EdgeCoverageAware ranking: {0} (seam tolerance={1})" -f ($(if ($EdgeCoverageAware) { "on" } else { "off" })), $SeamTolerance)
$summary += ""
if ($bestFinal) {
    $summary += "## Recommended Defaults"
    $summary += ("- -bsremKappa {0}" -f ($(if ($bestFinal.use_kappa) { "on" } else { "off" })))
    if ($bestFinal.use_kappa) {
        $summary += ("- -bsremKappaClip {0}" -f $bestFinal.kappa_clip)
    }
    $summary += ("- -bsremSoftMask {0}" -f ($(if ($bestFinal.use_softmask) { "on" } else { "off" })))
    if ($bestFinal.use_softmask) {
        $summary += ("- -bsremSoftBand {0}" -f $bestFinal.soft_band)
        $summary += ("- -bsremSoftLambda {0}" -f $bestFinal.soft_lambda)
    }
    $summary += ("- Score: {0} (contrast)" -f ([math]::Round($bestFinal.score, 6)))
    if ($null -ne $bestFinal.seam_score) {
        $summary += ("- Seam score: {0}" -f ([math]::Round([double]$bestFinal.seam_score, 6)))
    }
    if ($null -ne $bestFinal.seam_ratio_vs_baseline) {
        $summary += ("- Seam ratio vs baseline: {0}" -f ([math]::Round([double]$bestFinal.seam_ratio_vs_baseline, 6)))
    }
    if ($null -ne $bestFinal.nrmse_ref_mean) {
        $summary += ("- NRMSE vs baseline (mean): {0}" -f ([math]::Round([double]$bestFinal.nrmse_ref_mean, 6)))
    }
    $summary += ("- Runtime: {0}s" -f ([math]::Round($bestFinal.runtime_sec, 3)))
}
$summary += ""
$summary += "## Files"
$summary += ("- Coarse CSV: {0}" -f $coarseCsv)
$summary += ("- Confirm CSV: {0}" -f $confirmCsv)
if ($confirmResults.Count -gt 0) {
    $summary += ("- Confirm ranked CSV: {0}" -f $confirmRankedCsv)
}
$summary += ""
$summary += "## Notes"
$summary += "- This sweep uses a surrogate anatomical mask from prior reconstruction unless a CT-aligned attMap is available."
$summary += "- CSV includes NU4-style CV proxies (inside/outside CV%) and multi-bed seam score."
$summary += "- For edge-coverage phantoms (e.g., spiral S tube near full tangential FOV), recommendation uses seam ratio vs baseline instead of absolute seam score."
$summary += "- Tie-break policy prefers simpler configs (`-bsremSoftMask off`) when quality metrics are effectively equal."
$summary += "- Re-run with aligned CT prior/attMap before final clinical/production default freeze."

$summary -join "`n" | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "Summary saved: $summaryPath"
Write-Host "Sweep completed."
