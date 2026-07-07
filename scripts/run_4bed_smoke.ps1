param(
    [string]$ExePath = "",
    [string]$DataDir = ".\gatedata\SPECT_rectify_S_0105",
    [string]$CSRFile = ".\gatedata\matrix_TDA_1mm_N100_vox030_cw.csrb",
    [string]$OutputDir = ".\recon_out\smoke_4bed",
    [int]$Threads = 8,
    [string]$BedOverlap = "16",
    [string]$CropEnd = "16,16",
    [int]$Iters = 1,
    [int]$Subsets = 5,
    [switch]$UseBackground = $false
)

$ErrorActionPreference = "Stop"

function Resolve-ExePath {
    param([string]$InputPath)
    if ($InputPath -and (Test-Path $InputPath)) {
        return (Resolve-Path $InputPath).Path
    }

    $candidates = @(
        ".\staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe",
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

function Run-Case {
    param(
        [string]$CaseName,
        [string[]]$CommonArgs,
        [string[]]$ExtraArgs,
        [string]$Exe
    )
    $outPrefix = Join-Path $OutputDir $CaseName
    $args = $CommonArgs + $ExtraArgs + @("-fout", $outPrefix)

    Write-Host "--------------------------------------------------"
    Write-Host "Running case: $CaseName"
    Write-Host "Output prefix: $outPrefix"
    $start = Get-Date
    & $Exe @args
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    if ($exitCode -ne 0) {
        throw "Case '$CaseName' failed with exit code $exitCode"
    }

    $outputs = Get-ChildItem "${outPrefix}*_it*.mhd" -ErrorAction SilentlyContinue
    if (-not $outputs) {
        throw "Case '$CaseName' finished but no *.mhd output was found for prefix '$outPrefix'"
    }
    Write-Host ("Case '{0}' OK, elapsed: {1:N2}s, outputs: {2}" -f $CaseName, $elapsed.TotalSeconds, $outputs.Count)
}

$Exe = Resolve-ExePath -InputPath $ExePath
Write-Host "Using executable: $Exe"

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

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$dataList = $dataFiles -join ","
$commonArgs = @(
    "-dfList", $dataList,
    "-dim", "100,100,100",
    "-vox", "0.3,0.3,0.3",
    "-ang", "360",
    "-detN", "40,40",
    "-it", ("{0}:1" -f $Iters),
    "-th", $Threads.ToString(),
    "-loadCSR", $CSRFile,
    "-mapCSR", "on",
    "-bedOverlap", $BedOverlap,
    "-cropEnd", $CropEnd,
    "-vb", "1"
)

if ($UseBackground) {
    $bg = ".\gatedata\SPECT_bproj.mat"
    Assert-Exists -Path $bg -Label "Background projection"
    $bgList = @($bg, $bg, $bg, $bg) -join ","
    $commonArgs += @("-bProjList", $bgList)
}

Run-Case -CaseName "mlem" -CommonArgs $commonArgs -ExtraArgs @(
    "-opti", "MLEM",
    "-subsets", "1"
) -Exe $Exe

Run-Case -CaseName "bsrem" -CommonArgs $commonArgs -ExtraArgs @(
    "-opti", "BSREM",
    "-subsets", $Subsets.ToString(),
    "-bsrem", "1.0,0.05",
    "-reg", "TVIsotropic,1e-6"
) -Exe $Exe

Write-Host "--------------------------------------------------"
Write-Host "All 4-bed smoke cases completed successfully."
Write-Host "Output directory: $OutputDir"
