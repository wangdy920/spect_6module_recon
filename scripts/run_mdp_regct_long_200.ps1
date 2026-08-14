param(
    [string]$OutputRoot = "recon_out\mdp_regct_long_20260814_localxy",
    [string]$RegCT = "recon_out\ct_spect_reg_20260814_localxy\20260729_mdp\regct_hu_recon220.mhd",
    [string]$ControlRoot = "",
    [int]$Iterations = 200,
    [int]$SaveEvery = 20
)

$ErrorActionPreference = "Stop"
# PowerShell 7 treats native stderr as NativeCommandError. SPECT_recon writes
# host-cache budget lines to stderr; those must not abort a successful run.
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$repo = Split-Path -Parent $PSScriptRoot
$output = Join-Path $repo $OutputRoot
$exe = Join-Path $repo "build\verify-vs\bin\Release\SPECT_recon.exe"
$config = Join-Path $repo "multipinhole\config\spect_config.ct_guided.ini"
$algo = Join-Path $repo "multipinhole\config\algo_bsrem.conf"
$matrix = "F:\matrix_PSC_PIN26x28x25_SCALE2_141keV_120x120x220"
$projectionRoot = Join-Path $repo "recon_out\ctac_mdp_smoke_20260813\preproc\20260729_mdp\SPECT"
$beds = 1..5 | ForEach-Object { Join-Path $projectionRoot "Bed$_.bin" }
$denom = Join-Path $repo "recon_out\mdp_regct_smoke_20260814\denom\mdp_noctac_rdp_v4.mhd"
$regct = if ([IO.Path]::IsPathRooted($RegCT)) { $RegCT } else { Join-Path $repo $RegCT }
$controlRootAbs = if ($ControlRoot) {
    if ([IO.Path]::IsPathRooted($ControlRoot)) { $ControlRoot } else { Join-Path $repo $ControlRoot }
} else { "" }

if ($Iterations -le 0 -or $SaveEvery -le 0 -or $Iterations % $SaveEvery -ne 0) {
    throw "Iterations must be positive and divisible by SaveEvery."
}
foreach ($path in @($exe, $config, $algo, $matrix, $denom, $regct) + $beds) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
if ($controlRootAbs -and -not (Test-Path -LiteralPath $controlRootAbs)) {
    throw "Missing ControlRoot: $controlRootAbs"
}
if (Get-Process SPECT_recon -ErrorAction SilentlyContinue) {
    throw "Another SPECT_recon process is active; this experiment is single-GPU sequential."
}
$regctHeader = Get-Content -LiteralPath $regct -Raw
foreach ($tag in @(
    "SPECTMapRole = regularizer_reconstruction_storage",
    "SPECTBeds = 1,2,3,4,5",
    "SPECTZLayout = reconstruction_storage_z",
    "SPECTIntensityUnits = HU"
)) {
    if (-not $regctHeader.Contains($tag)) { throw "regCT contract tag missing: $tag" }
}

$cases = @(
    [ordered]@{ name = "bare_bsrem"; reg = "None"; ct = $false },
    [ordered]@{ name = "tv_plain"; reg = "TV,0.3,1e-6"; ct = $false },
    [ordered]@{ name = "tv_ct"; reg = "TV,0.3,1e-6"; ct = $true },
    [ordered]@{ name = "rdp_plain"; reg = "RDP,0.3,0,0.01"; ct = $false },
    [ordered]@{ name = "rdp_ct"; reg = "RDP,0.3,0,0.01"; ct = $true }
)

New-Item -ItemType Directory -Force -Path $output | Out-Null
$contract = [ordered]@{
    status = "running"
    started_at = (Get-Date).ToString("o")
    question = "200-iteration paired effects of HU-guided TV/RDP after local-COM CT-SPECT registration"
    protected_baseline = "bare_bsrem"
    ctac = $false
    iterations = $Iterations
    save_every = $SaveEvery
    control_root = $controlRootAbs
    map = "ct_anchored_local_com"
    shared = [ordered]@{
        executable = $exe; config = $config; algorithm_config = $algo
        matrix = $matrix; projections = $beds; denominator_cache = $denom
        dimension_xyz = @(120, 120, 220); voxel_mm_xyz = @(0.3, 0.3, 0.3)
        output_dimension_xyz = @(100, 100, 290); z_bed_flip = 1
        psf = "sieve/castor/1.3 mm/3 sigmas"; cylinder_mm = @(36, 66)
        bsrem = "alpha0=1.0,gamma=0.05,curv=off,kappa=on"
    }
    ct_guidance = [ordered]@{ image = $regct; units = "HU"; sigma_hu = 100; floor = 0.05 }
    evidence = [ordered]@{
        sha256 = [ordered]@{}
        matrix_files = @(Get-ChildItem -LiteralPath $matrix -File | Sort-Object Name |
            ForEach-Object { [ordered]@{ name = $_.Name; bytes = $_.Length; modified_utc = $_.LastWriteTimeUtc.ToString("o") } })
    }
    cases = $cases
    runs = @()
}
$hashInputs = @($exe, $config, $algo, $regct,
    [IO.Path]::ChangeExtension($regct, ".raw"), $denom,
    [IO.Path]::ChangeExtension($denom, ".raw"), [IO.Path]::ChangeExtension($denom, ".manifest.txt"),
    (Join-Path $matrix "metadata.bin"), (Join-Path $repo "multipinhole\config\bad_dector")) + $beds
foreach ($path in $hashInputs) {
    $contract.evidence.sha256[(Resolve-Path -LiteralPath $path).Path] =
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}
$contractPath = Join-Path $output "experiment.json"
if ($controlRootAbs) {
    $controlExperiment = Join-Path $controlRootAbs "experiment.json"
    if (-not (Test-Path -LiteralPath $controlExperiment)) {
        throw "ControlRoot has no experiment.json: $controlExperiment"
    }
    $controlJson = Get-Content -LiteralPath $controlExperiment -Raw
    $exeHash = $contract.evidence.sha256[(Resolve-Path -LiteralPath $exe).Path]
    if ($controlJson -notmatch [regex]::Escape($exeHash)) {
        throw "ControlRoot exe SHA-256 does not match current SPECT_recon.exe; rebuild all five arms."
    }
}
$contract | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $contractPath -Encoding utf8

foreach ($case in $cases) {
    $runDir = Join-Path $output $case.name
    $final = Join-Path $runDir "recon_it$Iterations.mhd"
    if ($controlRootAbs -and -not $case.ct) {
        $src = Join-Path $controlRootAbs $case.name
        $srcFinal = Join-Path $src "recon_it$Iterations.mhd"
        if (-not (Test-Path -LiteralPath $srcFinal)) {
            throw "Control arm missing $($case.name): $srcFinal"
        }
        if (-not (Test-Path -LiteralPath $runDir)) {
            New-Item -ItemType Junction -Path $runDir -Target (Resolve-Path -LiteralPath $src).Path | Out-Null
        }
        if (-not (Test-Path -LiteralPath $final)) {
            throw "Control junction did not expose $($case.name) it$Iterations"
        }
        $contract.runs += [ordered]@{
            case = $case.name; status = "reused_control"; output = $final; source = $src
        }
        $contract | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $contractPath -Encoding utf8
        Write-Output "[$($case.name)] reused control from $src"
        continue
    }
    if (Test-Path -LiteralPath $final) {
        $contract.runs += [ordered]@{ case = $case.name; status = "already_complete"; output = $final }
        $contract | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $contractPath -Encoding utf8
        Write-Output "[$($case.name)] already complete; skipping"
        continue
    }
    if ((Test-Path -LiteralPath $runDir) -and
        (Get-ChildItem -LiteralPath $runDir -Force -ErrorAction SilentlyContinue)) {
        throw "Partial output exists for $($case.name): $runDir"
    }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $exeArgs = @(
        "-config", $config,
        "-algoConf", $algo,
        "-chunkCSRDir", $matrix,
        "-dfList", ($beds -join ";"),
        "-opti", "bsrem",
        "-reg", $case.reg,
        "-dim", "120,120,220",
        "-vox", "0.3,0.3,0.3",
        "-ang", "360",
        "-detN", "48,48",
        "-it", "$Iterations`:$SaveEvery",
        "-th", "20",
        "-psf", "1.3,3",
        "-psfMode", "sieve",
        "-psfKernel", "castor",
        "-zbedflip", "1",
        "-cyl", "36,66",
        "-compactDenomCache", $denom,
        "-bsrem", "1.0,0.05",
        "-bsremCurv", "off",
        "-bsremKappa", "on",
        "-projectionResidual", "on",
        "-fout", (Join-Path $runDir "recon"),
        "-vb", "1"
    )
    if ($case.ct) {
        $exeArgs += @("-regCT", $regct, "-regCtSigma", "100", "-regCtFloor", "0.05")
    }
    $started = Get-Date
    Write-Output "[$($case.name)] start $($started.ToString('o'))"
    $log = Join-Path $runDir "console.log"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $exe @exeArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($null -eq $exit) { $exit = 0 }
    if ($exit -ne 0) { throw "$($case.name) failed with exit code $exit" }
    if (-not (Test-Path -LiteralPath $final)) { throw "$($case.name) did not produce $final" }
    $finished = Get-Date
    $contract.runs += [ordered]@{
        case = $case.name; status = "complete"; output = $final
        started_at = $started.ToString("o"); finished_at = $finished.ToString("o")
        elapsed_seconds = ($finished - $started).TotalSeconds
    }
    $contract | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $contractPath -Encoding utf8
    Write-Output "[$($case.name)] complete in $([math]::Round(($finished - $started).TotalMinutes, 1)) min"
}

$contract.status = "complete"
$contract.finished_at = (Get-Date).ToString("o")
$contract | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $contractPath -Encoding utf8
Write-Output "All five runs complete: $contractPath"
