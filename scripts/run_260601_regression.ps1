# 260601 compact canonical 数值回归（收敛重构后的正式验收入口）。
# 跑出 5 个固定 case，b1_plain 对 archived 参考做硬阈值对比；
# 指定 -GoldenDir 时逐 case 对冻结基线做近零差对比（重构回归用）。
param(
    [string]$Exe = "build\Release\bin\Release\SPECT_recon.exe",
    [string]$OutRoot = "recon_out\regression_260601",
    [string]$GoldenDir = "",
    [string]$ProjBin = "test_data\sys_260601_recon_ref\PSC_6module_24pin_10MBq_600s_test_1.bin",
    [string]$RefImg = "test_data\sys_260601_recon_ref\recon_iter050.img",
    [string]$MatrixDir = "test_data\6module\260601"
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$emptyCfg = Join-Path $OutRoot "empty.ini"
if (-not (Test-Path $emptyCfg)) { New-Item -ItemType File -Path $emptyCfg | Out-Null }

$common = @("-config", $emptyCfg, "-chunkCSRDir", $MatrixDir, "-th", "4")

function Invoke-Case([string]$Name, [string[]]$Extra) {
    $prefix = Join-Path $OutRoot "$Name\recon"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $prefix) | Out-Null
    $logPath = Join-Path $OutRoot "$Name.log"
    $argLine = ((@($common) + $Extra + @("-fout", $prefix)) | ForEach-Object { '"{0}"' -f $_ }) -join " "
    cmd /c "`"$Exe`" $argLine > `"$logPath`" 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "Case $Name failed with exit $LASTEXITCODE (see $logPath)" }
    Write-Host "case $Name ok"
}

Invoke-Case "b1_plain"     @("-df", $ProjBin, "-opti", "mlem", "-it", "50")
Invoke-Case "b1_butter"    @("-df", $ProjBin, "-opti", "mlem", "-it", "50", "-butter", "0.35,2")
Invoke-Case "bsrem20"      @("-df", $ProjBin, "-opti", "bsrem", "-it", "20")
Invoke-Case "sieve_castor20" @("-df", $ProjBin, "-opti", "mlem", "-it", "20", "-psf", "1.3,3", "-psfMode", "sieve", "-psfKernel", "castor")

# 两床 stitch： Bed1/Bed2 由同一投影复制（目录 beds\ 随 OutRoot 自动准备）
$beds = Join-Path $OutRoot "beds"
New-Item -ItemType Directory -Force -Path $beds | Out-Null
Copy-Item $ProjBin (Join-Path $beds "Bed1.bin") -Force
Copy-Item $ProjBin (Join-Path $beds "Bed2.bin") -Force
Invoke-Case "stitch2bed" @("-dfList", "$beds\Bed1.bin;$beds\Bed2.bin", "-bedOverlap", "164", "-opti", "mlem", "-it", "20")

New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot "metrics") | Out-Null
$failed = @()

# 硬锚点： b1_plain vs archived 参考（阈值 0.02 / 0.005）
$cand = Join-Path $OutRoot "b1_plain\recon_it50.mhd"
$json = Join-Path $OutRoot "metrics\b1_plain_vs_ref.json"
python scripts\compare_compact_legacy.py --candidate $cand --reference $RefImg --out-json $json
if ($LASTEXITCODE -ne 0) { $failed += "b1_plain_vs_ref" }

# 可选： 冻结基线逐 case 近零差对比（期望 float 噪声量级）
if ($GoldenDir) {
    $finals = @{ b1_plain = "recon_it50"; b1_butter = "recon_it50"; bsrem20 = "recon_it20"; sieve_castor20 = "recon_it20"; stitch2bed = "recon_it20" }
    foreach ($case in $finals.Keys) {
        $a = Join-Path $OutRoot "$case\$($finals[$case]).mhd"
        $b = Join-Path $GoldenDir "$case\$($finals[$case]).mhd"
        $j = Join-Path $OutRoot "metrics\${case}_vs_golden.json"
        python scripts\compare_compact_legacy.py --candidate $a --reference $b --out-json $j `
            --max-rmse-over-mean 1e-4 --max-sum-rel-diff 1e-6
        if ($LASTEXITCODE -ne 0) { $failed += "${case}_vs_golden" }
    }
}

if ($failed.Count -gt 0) { throw "REGRESSION FAILED: $($failed -join ', ') (metrics in $OutRoot\metrics)" }
Write-Host "REGRESSION PASSED -> $OutRoot\metrics"
