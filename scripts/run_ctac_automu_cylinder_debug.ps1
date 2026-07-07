param(
    [string]$SpectExe = "build\Release\bin\Release\SPECT_recon.exe",
    [string]$ProjectionBin = "test_data\sys_260601_recon_ref\PSC_6module_24pin_10MBq_600s_test_1.bin",
    [string]$Matrix260601Dir = "test_data\6module\260601",
    [string]$ReferenceImg = "test_data\sys_260601_recon_ref\recon_iter050.img",
    [string]$OutputRoot = "recon_out\ctac_automu_cylinder_debug_260601",
    [int]$Iterations = 1,
    [string]$IterationList = "",
    [int]$Threads = 4,
    [double]$ThresholdFrac = 0.05,
    [double]$MeasureThresholdFrac = 0.35,
    [double]$WaterMu = 0.0152,
    [switch]$SkipRecon
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $repo $Path)
}

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Invoke-Logged([string]$Label, [string]$Exe, [string[]]$ArgList, [string]$LogPath) {
    Ensure-Directory (Split-Path -Parent $LogPath)
    "$ $Exe $($ArgList -join ' ')" | Set-Content -Encoding UTF8 -LiteralPath $LogPath
    $tmpBase = Join-Path (Split-Path -Parent $LogPath) ([System.IO.Path]::GetRandomFileName())
    $stdoutPath = "$tmpBase.stdout"
    $stderrPath = "$tmpBase.stderr"
    $argLine = ($ArgList | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " "
    try {
        $proc = Start-Process -FilePath $Exe -ArgumentList $argLine -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        foreach ($p in @($stdoutPath, $stderrPath)) {
            if ((Test-Path -LiteralPath $p) -and ((Get-Item -LiteralPath $p).Length -gt 0)) {
                Get-Content -Raw -LiteralPath $p | Add-Content -Encoding UTF8 -LiteralPath $LogPath
            }
        }
        $code = $proc.ExitCode
    } finally {
        Remove-Item -LiteralPath $stdoutPath,$stderrPath -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) {
        throw "$Label failed (exit $code). Log: $LogPath"
    }
}

$spectPath = Resolve-RepoPath $SpectExe
$projPath = Resolve-RepoPath $ProjectionBin
$matrixPath = Resolve-RepoPath $Matrix260601Dir
$refPath = Resolve-RepoPath $ReferenceImg
$outRoot = Resolve-RepoPath $OutputRoot
$mapDir = Join-Path $outRoot "maps"
$logsDir = Join-Path $outRoot "logs"
$metricsDir = Join-Path $outRoot "metrics"
$cacheDir = Join-Path $outRoot "denom_cache"
Ensure-Directory $mapDir
Ensure-Directory $logsDir
Ensure-Directory $metricsDir
Ensure-Directory $cacheDir

foreach ($p in @($spectPath, $projPath, $matrixPath, $refPath)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "Required path missing: $p" }
}

$prefix = "sys260601_ref"
& python (Resolve-RepoPath "scripts\qa_ctac_automu_cylinder.py") `
    --activity $refPath `
    --out-dir $mapDir `
    --prefix $prefix `
    --threshold-frac $ThresholdFrac `
    --measure-threshold-frac $MeasureThresholdFrac `
    --water-mu $WaterMu
if ($LASTEXITCODE -ne 0) { throw "qa_ctac_automu_cylinder.py failed" }

$trueMu = Join-Path $mapDir "${prefix}_true_mu.mhd"
$autoMu = Join-Path $mapDir "${prefix}_auto_mu_cylmasked.mhd"
$activityMhd = Join-Path $mapDir "${prefix}_activity.mhd"
$mapMetrics = Get-Content -Raw -LiteralPath (Join-Path $mapDir "${prefix}_metrics.json") | ConvertFrom-Json
$cyl = $mapMetrics.measured_cylinder
$cylDiameter = [double]$cyl.diameter_mm
$cylHeight = [double]$cyl.height_mm
$cylCenterX = [double]$cyl.center_x_vox + 1.0
$cylCenterY = [double]$cyl.center_y_vox + 1.0
$cylCenterZ = [double]$cyl.center_z_vox + 1.0
$runs = @()
$comparisons = @()
$iterationValues = if ([string]::IsNullOrWhiteSpace($IterationList)) {
    @($Iterations)
} else {
    $IterationList -split "," | ForEach-Object { [int]$_.Trim() }
}

if (-not $SkipRecon) {
    $commonBase = @(
        "-config", "__missing_ctac_automu_debug__.ini",
        "-df", $projPath,
        "-chunkCSRDir", $matrixPath,
        "-matrixLayout", "compact-symmetric",
        "-opti", "mlem",
        "-subsets", "1",
        "-th", "$Threads",
        "-ctac",
        "-ctacModel", "line-integral",
        "-ctacRotSign", "1",
        "-cyl", "$cylDiameter,$cylHeight",
        "-cylc", "$cylCenterX,$cylCenterY,$cylCenterZ"
    )

    foreach ($iter in $iterationValues) {
        $common = $commonBase + @("-it", "$iter")

        $truePrefix = Join-Path $outRoot "true_mu\recon"
        Ensure-Directory (Split-Path -Parent $truePrefix)
        $trueArgs = $common + @(
            "-attMap", $trueMu,
            "-compactDenomCache", (Join-Path $cacheDir "true_mu_global_denom.mhd"),
            "-fout", $truePrefix
        )
        Invoke-Logged "true_mu_ctac_it$iter" $spectPath $trueArgs (Join-Path $logsDir "true_mu_ctac_it$iter.log")
        $runs += [pscustomobject]@{ label = "true_mu_ctac_it$iter"; prefix = $truePrefix; iterations = $iter }

        $autoPrefix = Join-Path $outRoot "auto_mu\recon"
        Ensure-Directory (Split-Path -Parent $autoPrefix)
        $autoArgs = $common + @(
            "-ctacAutoMu", $activityMhd,
            "-ctacAutoMuThresh", "$ThresholdFrac",
            "-ctacAutoMuValue", "$WaterMu",
            "-compactDenomCache", (Join-Path $cacheDir "auto_mu_global_denom.mhd"),
            "-fout", $autoPrefix
        )
        Invoke-Logged "auto_mu_ctac_it$iter" $spectPath $autoArgs (Join-Path $logsDir "auto_mu_ctac_it$iter.log")
        $runs += [pscustomobject]@{ label = "auto_mu_ctac_it$iter"; prefix = $autoPrefix; iterations = $iter }

        $trueRecon = "${truePrefix}_it$iter.mhd"
        $autoRecon = "${autoPrefix}_it$iter.mhd"
        $outCompare = Join-Path $metricsDir "auto_vs_true_recon_it$iter.json"
        & python (Resolve-RepoPath "scripts\compare_recon_metrics.py") `
            --candidate $autoRecon `
            --reference $trueRecon `
            --out-json $outCompare `
            --positive-union
        if ($LASTEXITCODE -ne 0) { throw "auto vs true recon compare failed at $iter iterations" }
        $comparisons += [pscustomobject]@{
            label = "auto_vs_true_recon_it$iter"
            metrics_json = $outCompare
            candidate = $autoRecon
            reference = $trueRecon
            iterations = $iter
        }

        if ($iter -eq $iterationValues[0]) {
            $cxxAutoMu = Join-Path (Split-Path -Parent $autoPrefix) "recon_automu.mhd"
            if (Test-Path -LiteralPath $cxxAutoMu) {
                $outMuCompare = Join-Path $metricsDir "cxx_automu_vs_python_automu.json"
                & python (Resolve-RepoPath "scripts\compare_recon_metrics.py") `
                    --candidate $cxxAutoMu `
                    --reference $autoMu `
                    --out-json $outMuCompare `
                    --positive-union
                if ($LASTEXITCODE -ne 0) { throw "C++ auto-mu vs Python auto-mu compare failed" }
                $comparisons += [pscustomobject]@{
                    label = "cxx_automu_vs_python_automu"
                    metrics_json = $outMuCompare
                    candidate = $cxxAutoMu
                    reference = $autoMu
                }
            }
        }
    }
}

$summary = [pscustomobject]@{
    note = "Debug-only CTAC/auto-mu cylinder gate. Derenzo is intentionally excluded from the primary gate."
    projection = $projPath
    reference_activity = $refPath
    matrix_260601 = $matrixPath
    output_root = $outRoot
    map_metrics = (Join-Path $mapDir "${prefix}_metrics.json")
    true_mu = $trueMu
    auto_mu_python = $autoMu
    activity_mhd = $activityMhd
    iterations = $iterationValues
    runs = $runs
    comparisons = $comparisons
}
$summaryPath = Join-Path $outRoot "run_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $summaryPath
Write-Host "Done. Summary: $summaryPath"
