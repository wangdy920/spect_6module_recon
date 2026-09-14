# Same-session Sampling 50-iter MLEM vs BSREM+RDP (regularization verified on)
# to measure the true RDP per-iteration pipeline cost. Uses the experiment INI
# (recon_sections_override_cli=false) so CLI -reg is not silently wiped.
# Continue: SPECT_recon prints benign NOTE lines to stderr, which "Stop"
# would turn into terminating NativeCommandError under 2>&1.
$ErrorActionPreference = "Continue"
$root = "D:\Projects\SPECT-recon"
$exe = "$root\build\verify-vs\bin\Release\SPECT_recon.exe"

# The INI resolves relative paths (bad_dector, ...) against its own directory,
# so the patched copy must live next to the original.
$srcIni = "$root\multipinhole\config\spect_config.ini"
$ini = "$root\multipinhole\config\spect_config.experiment.ini"
$text = Get-Content $srcIni -Raw
if ($text -notmatch "recon_sections_override_cli = true") {
    throw "override flag not found in $srcIni"
}
$text -replace "recon_sections_override_cli = true", `
    "recon_sections_override_cli = false" | Set-Content $ini -Encoding UTF8
$beds = (1..5 | ForEach-Object { "$root\recon_out\ctac_mdp_smoke_20260813\preproc\20260729_mdp\SPECT\Bed$_.bin" }) -join ";"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

foreach ($arm in @("mlem", "bsremrdp")) {
    $out = "$root\recon_out\perf_rdpcost_${arm}_$ts"
    New-Item -ItemType Directory -Force $out | Out-Null
    $common = @("-config", $ini, "-it", "50:1", "-save-every", "0", "-vb", "1",
                "-chunkCSRDir", "$root\test_data\6module\Sampling_Approach",
                "-dfList", $beds, "-fout", "$out\recon")
    if ($arm -eq "mlem") {
        $args_ = @("-opti", "mlem") + $common
    } else {
        $args_ = @("-opti", "bsrem", "-algoConf",
                   "$root\multipinhole\config\algo_bsrem.conf",
                   "-bsrem", "1.5,0", "-reg", "rdp,0.3,0,0.01") + $common
    }
    & $exe @args_ > "$out\console.log" 2>&1
    Write-Output "ARM=$arm exit=$LASTEXITCODE OUT=$out"
    Select-String -Path "$out\console.log" -Pattern "PERF\]" |
        ForEach-Object { $_.Line }
}
Write-Output "RDPCOST_DONE"
