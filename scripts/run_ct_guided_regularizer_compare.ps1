param(
    [string]$OutputRoot = "F:\DATA\predata\SUPER_PSC_1_0\SPECT\ct_guided_regularizer_compare_20260714",
    [string[]]$Datasets = @("mouse", "S", "capillary", "zimu"),
    [int]$Iterations = 20,
    [double]$RdpBeta = 0.3,
    [double]$TvBeta = 0.3,
    [int]$MaxGpuUtilPct = 20,
    [switch]$AllowBusyGpu
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo "build\Release\bin\Release\SPECT_recon.exe"
$matrix = "E:\Xiang_Project\SPECT\6Module\release\SPECT_6Module_260707\bin\recon\matrix\260601"
$config = Join-Path $repo "multipinhole\config\spect_config.ini"
$mlemConf = Join-Path $repo "multipinhole\config\algo_mlem.conf"
$bsremConf = Join-Path $repo "multipinhole\config\algo_bsrem.conf"
$ctRoot = "F:\DATA\predata\SUPER_PSC_1_0\SPECT\overlap10_vs11_joint_20260713\ct_guided_v1"

$contracts = @{
    mouse = @{
        projection = "F:\DATA\rawdata\SUPER_PSC_1_0\SPECT\20260709_spect_mouse\preproc_identity_spect_20260710f"
        mu = Join-Path $ctRoot "mouse_ct_mu_ov10.mhd"
    }
    S = @{
        projection = "F:\DATA\rawdata\SUPER_PSC_1_0\SPECT\rdp_overlap8_sweep_20260713\inputs\S"
        mu = Join-Path $ctRoot "S_ct_mu_ov10.mhd"
    }
    capillary = @{
        projection = "F:\DATA\predata\SUPER_PSC_1_0\SPECT\20260713_spect_maoxiguan\20260713_103948\SPECT"
        mu = Join-Path $ctRoot "capillary_ct_mu_ov10.mhd"
    }
    zimu = @{
        projection = "F:\DATA\rawdata\SUPER_PSC_1_0\SPECT\20260710_spect_zimu\preproc_identity_spect_20260710f"
        mu = Join-Path $ctRoot "zimu_ct_mu_ov10.mhd"
    }
}

foreach ($path in @($exe, $matrix, $config, $mlemConf, $bsremConf)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}

$selected = foreach ($name in $Datasets) {
    if (-not $contracts.ContainsKey($name)) { throw "Unknown dataset: $name" }
    $contract = $contracts[$name]
    $beds = 1..4 | ForEach-Object { Join-Path $contract.projection "Bed$_.bin" }
    foreach ($path in @($contract.mu) + $beds) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Dataset $name is incomplete: $path" }
    }
    [pscustomobject]@{ name = $name; mu = $contract.mu; beds = $beds }
}

if (-not $AllowBusyGpu) {
    $samples = 1..3 | ForEach-Object {
        $value = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) { throw "nvidia-smi GPU precheck failed" }
        Start-Sleep -Seconds 2
        [int](($value | Select-Object -First 1).Trim())
    }
    if (($samples | Measure-Object -Maximum).Maximum -gt $MaxGpuUtilPct) {
        throw "GPU is busy: utilization samples=$($samples -join ',')%, limit=$MaxGpuUtilPct%."
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$cases = @(
    @{ name = "mlem_noct"; algorithm = "MLEM"; ctac = $false; reg = $null },
    @{ name = "mlem_ctac"; algorithm = "MLEM"; ctac = $true; reg = $null },
    @{ name = "bsrem0_ctac"; algorithm = "BSREM"; ctac = $true; reg = "RDP,0,0,0.01" },
    @{ name = "rdp_ctac"; algorithm = "BSREM"; ctac = $true; reg = "RDP,$RdpBeta,0,0.01" },
    @{ name = "tv_ctac"; algorithm = "BSREM"; ctac = $true; reg = "TV,$TvBeta,1e-6" }
)

$runs = @()
foreach ($dataset in $selected) {
    $datasetRoot = Join-Path $OutputRoot $dataset.name
    $noCtDenom = Join-Path $datasetRoot "cache\noct_denom.mhd"
    $ctDenom = Join-Path $datasetRoot "cache\ctac_denom.mhd"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $noCtDenom) | Out-Null
    foreach ($case in $cases) {
        $runDir = Join-Path $datasetRoot $case.name
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $args = @(
            "-config", $config,
            "-algoConf", $(if ($case.algorithm -eq "MLEM") { $mlemConf } else { $bsremConf }),
            "-dfList", ($dataset.beds -join ";"),
            "-bedOverlap", "10,10,10,0,0",
            "-chunkCSRDir", $matrix,
            "-compactDenomCache", $(if ($case.ctac) { $ctDenom } else { $noCtDenom }),
            "-fout", (Join-Path $runDir "recon"),
            "-opti", $case.algorithm,
            "-it", "$Iterations`:$Iterations",
            "-th", "20",
            "-psf", "1.5,4",
            "-psfMode", "Operator",
            "-dim", "108,108,80",
            "-vox", "0.3,0.3,0.3",
            "-ang", "360",
            "-detN", "48,48",
            "-projectionResidual", "on",
            "-vb", "2"
        )
        if ($case.ctac) { $args += @("-ctac", "-attMap", $dataset.mu) }
        if ($case.algorithm -eq "BSREM") {
            $args += @("-bsrem", "1.0,0.05", "-bsremCurv", "off", "-bsremKappa", "on")
        }
        if ($case.reg) {
            $args += @("-reg", $case.reg, "-regCT", $dataset.mu,
                       "-regCtSigma", "0.01", "-regCtFloor", "0.05")
        }
        $log = Join-Path $runDir "console.log"
        & $exe @args 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "$($dataset.name)/$($case.name) failed" }
        $runs += [pscustomobject]@{
            dataset = $dataset.name; case = $case.name; mu = $dataset.mu
            output = Join-Path $runDir "recon_it$Iterations.mhd"; log = $log
        }
    }
}

$manifest = [ordered]@{
    contract = "overlap10 joint, FOV15/T4, Operator PSF 1.5 mm, fixed CUDA CTAC"
    iterations = $Iterations; rdp_beta = $RdpBeta; tv_beta = $TvBeta
    datasets = $Datasets; runs = $runs
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputRoot "run_manifest.json") -Encoding utf8
& python (Join-Path $PSScriptRoot "eval_ct_guided_regularizer_compare.py") --root $OutputRoot --iterations $Iterations
if ($LASTEXITCODE -ne 0) { throw "QA evaluation failed" }
