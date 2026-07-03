param(
    [string]$MatrixPath = ".\gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb",
    [int]$Threads = 16,
    [int]$Iterations = 2,
    [int]$Subsets = 1,
    [int]$BedCount = 1,
    [string[]]$BedOverlaps = @(),
    [string[]]$BaseDim = @(),
    [int]$WarmupRuns = 1,
    [int]$Repeats = 3,
    [int]$MaxRows = 80000,
    [string]$OutputDir = ".\recon_out\cuda_osem_compare",
    [switch]$Rebuild,
    [switch]$NoAutoFallback
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $MatrixPath)) {
    throw "Matrix file not found: $MatrixPath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildScript = Join-Path $repoRoot "scripts\build_cuda_scaffold.ps1"
$exePath = Join-Path $repoRoot "build_cuda_scaffold\bin\spect_cuda_osem_compare.exe"

$bedOverlapsArg = if ($BedOverlaps.Count -gt 0) { $BedOverlaps -join "," } else { "" }
$baseDimArg = if ($BaseDim.Count -gt 0) { $BaseDim -join "," } else { "" }

if ($Rebuild -or !(Test-Path $exePath)) {
    & $buildScript -Build -EnableCudaSelection -Target spect_cuda_osem_compare
}

function Invoke-Compare {
    param(
        [int]$Rows,
        [string]$JsonPath,
        [string]$LogPath
    )

    $args = @(
        "--matrix", $MatrixPath,
        "--threads", $Threads,
        "--iters", $Iterations,
        "--subsets", $Subsets,
        "--bed-count", $BedCount,
        "--warmup", $WarmupRuns,
        "--repeats", $Repeats,
        "--max-rows", $Rows,
        "--out", $JsonPath
    )
    if (-not [string]::IsNullOrWhiteSpace($bedOverlapsArg)) {
        $args += @("--bed-overlaps", $bedOverlapsArg)
    }
    if (-not [string]::IsNullOrWhiteSpace($baseDimArg)) {
        $args += @("--base-dim", $baseDimArg)
    }

    Write-Host "[CUDA-OSEM] Running compare (maxRows=$Rows, beds=$BedCount, subsets=$Subsets, iters=$Iterations)..." -ForegroundColor Cyan
    $output = & $exePath @args *>&1
    $output | Tee-Object -FilePath $LogPath | Out-Host
    return $LASTEXITCODE
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"

$candidateRows = [System.Collections.Generic.List[int]]::new()
$candidateRows.Add($MaxRows)
if (-not $NoAutoFallback) {
    foreach ($ratio in @(0.8, 0.67, 0.5, 0.4, 0.3, 0.2)) {
        $rows = [int][Math]::Floor($MaxRows * $ratio)
        if ($rows -ge 5000) { $candidateRows.Add($rows) }
    }
    $candidateRows.Add(20000)
    $candidateRows.Add(10000)
}
$candidateRows = $candidateRows | Sort-Object -Descending -Unique

$success = $false
$finalJson = ""
$finalLog = ""
$lastLog = $null

foreach ($rows in $candidateRows) {
    $jsonPath = Join-Path $OutputDir "cuda_osem_${timestamp}_r${rows}_s${Subsets}.json"
    $logPath = Join-Path $OutputDir "cuda_osem_${timestamp}_r${rows}_s${Subsets}.log"
    $exitCode = [int](Invoke-Compare -Rows $rows -JsonPath $jsonPath -LogPath $logPath)

    if ($exitCode -eq 0 -and (Test-Path $jsonPath)) {
        $success = $true
        $finalJson = $jsonPath
        $finalLog = $logPath
        break
    }

    $lastLog = $logPath
    if ($NoAutoFallback) {
        throw "Compare run failed with exit code $exitCode. log=$logPath"
    }

    $logText = if (Test-Path $logPath) { Get-Content $logPath -Raw } else { "" }
    if ($logText -notmatch "out of memory|CUDA") {
        throw "Compare run failed (non-fallback) with exit code $exitCode. log=$logPath"
    }

    Write-Host "[CUDA-OSEM] Compare failed at maxRows=$rows, trying smaller subset..." -ForegroundColor Yellow
}

if (-not $success) {
    throw "Compare failed after fallback attempts. last_log=$lastLog"
}

$res = Get-Content $finalJson -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "=== CUDA OSEM Compare Summary ===" -ForegroundColor Green
Write-Host ("rows={0}, cols={1}, nnz={2}, beds={3}, subsets={4}, iters={5}" -f $res.rows, $res.cols, $res.nnz, $res.bed_count, $res.subsets, $res.iterations)
Write-Host ("CPU avg: {0:N3} ms | GPU avg: {1:N3} ms | speedup: {2:N2}x" -f [double]$res.cpu.ms_avg, [double]$res.gpu.ms_avg, [double]$res.speedup)
Write-Host ("accuracy max_abs={0:E3}, rmse={1:E3}, l2_rel={2:E3}" -f [double]$res.accuracy.max_abs, [double]$res.accuracy.rmse, [double]$res.accuracy.l2_rel)
Write-Host ("host_bytes_estimate={0}, gpu_persistent_delta_bytes={1}" -f $res.host_bytes_estimate, $res.gpu.gpu_persistent_delta_bytes)
Write-Host ""
Write-Host "JSON: $finalJson"
Write-Host "LOG : $finalLog"