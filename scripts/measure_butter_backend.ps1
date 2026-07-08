param(
    [Parameter(Mandatory = $true)]
    [string]$ExePath,

    [Parameter(Mandatory = $true)]
    [string]$Scenario,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile,

    [string]$IterSpec = '1:1',

    [switch]$UseRegularization,
    [switch]$UsePSF
)

$ErrorActionPreference = 'Stop'

$argsList = @(
    '-dfList', './gatedata/SPECT_rectify_S_0105/data_append_bed1.mat,./gatedata/SPECT_rectify_S_0105/data_append_bed2.mat,./gatedata/SPECT_rectify_S_0105/data_append_bed3.mat,./gatedata/SPECT_rectify_S_0105/data_append_bed4.mat',
    '-dim', '100,100,100',
    '-vox', '0.3,0.3,0.3',
    '-ang', '360',
    '-detN', '40,40',
    '-it', $IterSpec,
    '-th', '8',
    '-loadCSR', './gatedata/matrix_TDA_1mm_N100_vox030_cw.csrb',
    '-mapCSR', 'on',
    '-bedOverlap', '16',
    '-cropEnd', '16,16',
    '-vb', '1',
    '-opti', 'BSREM',
    '-subsets', '1',
    '-bsrem', '1.0,0.05',
    '-butter', '0.35,2',
    '-fout', "./recon_out/cufft_pipeline/$Scenario"
)

if ($UseRegularization) {
    $argsList += @('-reg', 'RDP,1e-4,2.0,0.01')
}

if ($UsePSF) {
    $argsList += @('-psf', '1.5,4', '-psfMode', 'Operator')
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $ExePath @argsList *> $null
$exitCode = $LASTEXITCODE
$sw.Stop()

$dir = Split-Path -Parent $OutputFile
if ($dir -and -not (Test-Path $dir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$seconds = [Math]::Round($sw.Elapsed.TotalSeconds, 3)
Set-Content -Path $OutputFile -Value ("backend=cufft scenario=$Scenario iter=$IterSpec seconds=$seconds exit=$exitCode")

if ($exitCode -ne 0) {
    exit $exitCode
}
