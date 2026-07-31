param(
    [string]$SpectRoot = "F:\DATA\rawdata\SUPER_PSC_1_0\SPECT\20260709_spect_mouse",
    [string]$SpectExe = "",
    [string]$MatrixDir = "",
    [int]$Threshold = 20,
    [int]$Iterations = 30,
    [switch]$SkipPreprocess,
    [switch]$SkipRecon,
    [string]$OverlapCases = "12,16,20",
    [ValidateSet("none", "dicom", "png", "both")]
    [string]$ExportFormat = "both",
    [string]$ExportPngMode = "slices",
    [int]$CropEndVoxels = 12,
    [double]$FovDiameterMm = 30,
    [bool]$FovCropXY = $true,
    [string]$ExportCropZMm = ""
)

$ErrorActionPreference = "Stop"
$overlapList = @($OverlapCases -split '[,;]' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$py = "D:\DevToolchain\MiniConda\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "Python not found: $py" }

$spectData = Join-Path $SpectRoot "SPECT"
$preproc = Join-Path $SpectRoot "preproc"
$reconRoot = Join-Path $SpectRoot "recon"
New-Item -ItemType Directory -Force -Path $preproc, $reconRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($SpectExe)) {
    $candidates = @(
        (Join-Path $repo "staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe"),
        (Join-Path $repo "build\Release\bin\Release\SPECT_recon.exe")
    )
    $SpectExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $SpectExe -or -not (Test-Path -LiteralPath $SpectExe)) {
    throw "SPECT_recon.exe not found"
}
$SpectExe = (Resolve-Path -LiteralPath $SpectExe).Path

if ([string]::IsNullOrWhiteSpace($MatrixDir)) {
    $MatrixDir = Join-Path $repo "test_data\6module\260601"
}
if (-not (Test-Path -LiteralPath $MatrixDir)) {
    throw "Matrix dir missing: $MatrixDir"
}
$MatrixDir = (Resolve-Path -LiteralPath $MatrixDir).Path

function Ensure-RuntimeDlls([string]$BinDir) {
    $dlls = @(
        "cudart64_13.dll","cusparse64_12.dll","cufft64_12.dll",
        "nvrtc64_130_0.dll","nvrtc-builtins64_132.dll","nvJitLink_130_0.dll","nvfatbin_130_0.dll",
        "msvcp140.dll","vcruntime140.dll","vcruntime140_1.dll","vcomp140.dll","fftw3.dll"
    )
    $cuda = Join-Path $env:CUDA_PATH "bin\x64"
    $sys = Join-Path $env:SystemRoot "System32"
    $vcpkg = "D:\DevToolchain\vcpkg\installed\x64-windows\bin"
    foreach ($name in $dlls) {
        $dst = Join-Path $BinDir $name
        if (Test-Path -LiteralPath $dst) { continue }
        foreach ($root in @($cuda, $sys, $vcpkg, (Split-Path -Parent $SpectExe))) {
            $src = Join-Path $root $name
            if (Test-Path -LiteralPath $src) {
                Copy-Item -LiteralPath $src -Destination $dst -Force
                break
            }
        }
    }
}

Ensure-RuntimeDlls (Split-Path -Parent $SpectExe)

if (-not $SkipPreprocess) {
    Write-Host "[preproc] ENG -> BIN under $preproc"
    1..4 | ForEach-Object {
        $bed = "Bed$_"
        $bedDir = Join-Path $spectData $bed
        if (-not (Test-Path -LiteralPath $bedDir)) { throw "Missing $bedDir" }
        $outBin = Join-Path $preproc ("Bed{0}.bin" -f $_)
        $summary = Join-Path $preproc ("Bed{0}.json" -f $_)
        & $py (Join-Path $repo "scripts\eng_to_6module_projection_bin.py") `
            --bed-dir $bedDir `
            --out-bin $outBin `
            --threshold $Threshold `
            --summary-json $summary
        if ($LASTEXITCODE -ne 0) { throw "Preprocess failed for $bed" }
    }
}

$bins = 1..4 | ForEach-Object { Join-Path $preproc ("Bed{0}.bin" -f $_) }
foreach ($b in $bins) {
    if (-not (Test-Path -LiteralPath $b)) { throw "Missing projection bin: $b" }
}
$dfList = ($bins -join ";")

if ($SkipRecon) { exit 0 }

$results = @()
foreach ($overlap in $overlapList) {
    $ov = [int]$overlap
    $crop = if ($ov -eq 0) { 0 } else { $CropEndVoxels }
    $caseName = "overlap_{0}_crop_{1}" -f $ov, $crop
    $outDir = Join-Path $reconRoot $caseName
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $prefix = Join-Path $outDir "mouse_mlem"
    $log = Join-Path $outDir "recon.log"

    $bedOverlapArg = if ($ov -eq 0) { "0,0,0,0,0" } else { "$ov,$ov,$ov,0,0" }
    $args = @(
        "-dfList", $dfList,
        "-chunkCSRDir", $MatrixDir,
        "-opti", "mlem",
        "-it", ("{0}:5" -f $Iterations),
        "-th", "8",
        "-bedOverlap", $bedOverlapArg,
        "-cropEnd", ("{0},{0}" -f $crop),
        "-zbedflip", "0",
        "-zoutflip", "0",
        "-fout", $prefix,
        "-vb", "1"
    )

    Write-Host "[recon] $caseName (bedOverlap=$bedOverlapArg cropEnd=$crop)"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $argLine = ($args | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " "
    $proc = Start-Process -FilePath $SpectExe -ArgumentList $argLine -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError (Join-Path $outDir "recon.err.log")
    $code = $proc.ExitCode
    $sw.Stop()
    if ($code -ne 0) { throw "Reconstruction failed for $caseName (exit $code)" }

    $mhd = Get-ChildItem -LiteralPath $outDir -Filter "*_it${Iterations}.mhd" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $mhd) {
        $mhd = Get-ChildItem -LiteralPath $outDir -Filter "*.mhd" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    $exportDir = ""
    if ($mhd -and $ExportFormat -ne "none") {
        $exportDir = Join-Path $outDir "export"
        $patient = Split-Path $SpectRoot -Leaf
        $exportArgs = @(
            (Join-Path $repo "scripts\export_recon_volume.py"),
            "--image", $mhd.FullName,
            "--out-dir", $exportDir,
            "--format", $ExportFormat,
            "--png-mode", $ExportPngMode,
            "--series-description", "SPECT_mouse_mlem",
            "--patient-name", $patient,
            "--fov-diameter-mm", $FovDiameterMm
        )
        if ($FovCropXY) { $exportArgs += "--fov-crop-xy" }
        if ($ExportCropZMm) { $exportArgs += @("--crop-z-mm", $ExportCropZMm) }
        & $py @exportArgs
        if ($LASTEXITCODE -ne 0) { throw "Export failed for $caseName" }
        Write-Host "[export] $exportDir"
    }
    $results += [pscustomobject]@{
        case = $caseName
        bed_overlap = $bedOverlapArg
        crop_end = "$crop,$crop"
        output_mhd = if ($mhd) { $mhd.FullName } else { "" }
        export_dir = $exportDir
        wall_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        log = $log
    }
}

$summaryPath = Join-Path $reconRoot "run_summary.json"
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host "[done] summary -> $summaryPath"
$results | Format-Table -AutoSize
