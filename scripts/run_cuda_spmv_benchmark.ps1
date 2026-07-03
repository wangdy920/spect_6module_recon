param(
    [string]$MatrixPath = ".\gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb",
    [int]$CpuThreads = 16,
    [int]$CpuIters = 5,
    [int]$GpuIters = 50,
    [int]$WarmupIters = 5,
    [int]$MaxRows = 80000,
    [ValidateSet("spmv", "spmvt", "both")]
    [string]$Op = "both",
    [string]$OutputDir = ".\recon_out\cuda_spmv_bench",
    [switch]$Rebuild,
    [switch]$NoAutoFallback
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $MatrixPath)) {
    throw "Matrix file not found: $MatrixPath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path ".\build\cuda_bench" | Out-Null

$exePath = ".\build\cuda_bench\cuda_spmv_benchmark.exe"

if ($Rebuild -or !(Test-Path $exePath)) {
    Write-Host "[CUDA-BENCH] Building benchmark binary..." -ForegroundColor Cyan
    $nvccCmd = @(
        "nvcc",
        "-O3",
        "-std=c++17",
        "-Xcompiler", '"/O2 /openmp /EHsc"',
        "-I.",
        "tools/cuda_spmv_benchmark.cu",
        "-lcusparse",
        "-o", $exePath
    ) -join " "
    Write-Host $nvccCmd -ForegroundColor DarkGray

    $clCmd = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($null -ne $clCmd) {
        Invoke-Expression $nvccCmd
        if ($LASTEXITCODE -ne 0) {
            throw "nvcc build failed with exit code $LASTEXITCODE"
        }
    }
    else {
        $vcvarsCandidates = @(
            "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
            "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
            "$env:ProgramFiles\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
            "D:\win_programFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        )
        $vcvars = $vcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($vcvars)) {
            throw "cl.exe not found in PATH, and vcvars64.bat not found in common locations."
        }
        $cmdLine = "`"$vcvars`" && $nvccCmd"
        cmd /c $cmdLine
        if ($LASTEXITCODE -ne 0) {
            throw "nvcc build failed with exit code $LASTEXITCODE"
        }
    }
}

function Invoke-Benchmark {
    param(
        [int]$Rows,
        [string]$JsonPath,
        [string]$LogPath
    )

    $runCmd = @(
        $exePath,
        "--matrix", "`"$MatrixPath`"",
        "--cpu-threads", $CpuThreads,
        "--cpu-iters", $CpuIters,
        "--gpu-iters", $GpuIters,
        "--warmup", $WarmupIters,
        "--op", $Op,
        "--max-rows", $Rows,
        "--out", "`"$JsonPath`""
    ) -join " "

    Write-Host "[CUDA-BENCH] Running benchmark (op=$Op, maxRows=$Rows)..." -ForegroundColor Cyan
    Write-Host $runCmd -ForegroundColor DarkGray

    $output = Invoke-Expression $runCmd *>&1
    $output | Tee-Object -FilePath $LogPath | Out-Host
    return $LASTEXITCODE
}

function Print-OpSummary {
    param(
        [string]$Name,
        [object]$OpNode
    )

    if ($null -eq $OpNode) {
        return
    }

    Write-Host ("[{0}] CPU avg: {1:N3} ms | GPU avg: {2:N3} ms | speedup: {3:N2}x" -f
        $Name, [double]$OpNode.cpu_ms_avg, [double]$OpNode.gpu_ms_avg, [double]$OpNode.speedup)
    Write-Host ("[{0}] accuracy max_abs={1:E3}, rmse={2:E3}, l2_rel={3:E3}" -f
        $Name,
        [double]$OpNode.accuracy.max_abs,
        [double]$OpNode.accuracy.rmse,
        [double]$OpNode.accuracy.l2_rel)
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

$lastErrorLog = $null
$success = $false
$finalJson = ""
$finalLog = ""

foreach ($rows in $candidateRows) {
    $jsonPath = Join-Path $OutputDir "cuda_spmv_${timestamp}_r${rows}.json"
    $logPath = Join-Path $OutputDir "cuda_spmv_${timestamp}_r${rows}.log"
    $exitCode = [int](Invoke-Benchmark -Rows $rows -JsonPath $jsonPath -LogPath $logPath)

    if ($exitCode -eq 0 -and (Test-Path $jsonPath)) {
        $success = $true
        $finalJson = $jsonPath
        $finalLog = $logPath
        break
    }

    $logText = ""
    if (Test-Path $logPath) {
        $logText = Get-Content $logPath -Raw
    }
    $lastErrorLog = $logPath

    if ($NoAutoFallback) {
        throw "Benchmark run failed with exit code $exitCode. log=$logPath"
    }

    if ($logText -notmatch "out of memory") {
        throw "Benchmark run failed (non-OOM) with exit code $exitCode. log=$logPath"
    }

    Write-Host "[CUDA-BENCH] OOM at maxRows=$rows, trying smaller subset..." -ForegroundColor Yellow
}

if (-not $success) {
    throw "Benchmark failed after fallback attempts. last_log=$lastErrorLog"
}

$res = Get-Content $finalJson -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "=== CUDA vs CPU Summary ===" -ForegroundColor Green
Write-Host ("rows={0}, cols={1}, nnz={2}, op_mode={3}" -f $res.rows, $res.cols, $res.nnz, $res.op_mode)

if ($null -ne $res.spmv) {
    Print-OpSummary -Name "SpMV" -OpNode $res.spmv
}
if ($null -ne $res.spmvt) {
    Print-OpSummary -Name "SpMV^T" -OpNode $res.spmvt
}

if ($null -eq $res.spmv -and $null -eq $res.spmvt) {
    Write-Host ("CPU avg: {0:N3} ms | GPU avg: {1:N3} ms | speedup: {2:N2}x" -f
        [double]$res.cpu_ms_avg, [double]$res.gpu_ms_avg, [double]$res.speedup)
    Write-Host ("accuracy max_abs={0:E3}, rmse={1:E3}, l2_rel={2:E3}" -f
        [double]$res.accuracy.max_abs,
        [double]$res.accuracy.rmse,
        [double]$res.accuracy.l2_rel)
}

Write-Host ("host_bytes={0}, gpu_alloc_bytes={1}, gpu_mem_delta_bytes={2}" -f
    $res.host_bytes, $res.gpu_alloc_bytes, $res.gpu_mem_delta_bytes)
Write-Host ""
Write-Host "JSON: $finalJson"
Write-Host "LOG : $finalLog"
