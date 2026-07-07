param(
    [ValidateSet("psc", "logo", "all")]
    [string]$Case = "psc",

    [int]$Iterations = 1,
    [int]$Threads = 4,

    [string]$Exe = "staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe",
    [string]$OutputRoot = "",

    [string]$CompactChunkHostCacheBytes = "0",
    [string]$CompactDenomCacheDir = "recon_out\compact_denom_cache",

    [double]$BusyCpuThresholdPct = 25.0,
    [int]$BusyGpuUtilThresholdPct = 25,
    [switch]$IgnoreBusySystem,

    [switch]$Skip260601,
    [switch]$Skip260525,
    [switch]$SkipRender,
    [switch]$SkipMetrics
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $repo $Path)
}

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Format-ArgumentString([string[]]$ArgList) {
    $quoted = $ArgList | ForEach-Object {
        if ($_ -match '\s|;') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    return ($quoted -join ' ')
}

function Format-Command([string]$Program, [string[]]$ArgList) {
    $quoted = @($Program, (Format-ArgumentString $ArgList))
    return ($quoted -join ' ')
}

function Get-SystemMemorySnapshot {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem
        return [pscustomobject]@{
            total_visible_mib = [math]::Round(([double]$os.TotalVisibleMemorySize) / 1024.0, 1)
            free_physical_mib = [math]::Round(([double]$os.FreePhysicalMemory) / 1024.0, 1)
        }
    } catch {
        return $null
    }
}

function Get-GpuSnapshot {
    try {
        $lines = & nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null
        $gpus = @()
        foreach ($line in $lines) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }
            $parts = $line.Split(",").ForEach({ $_.Trim() })
            if ($parts.Count -lt 6) {
                continue
            }
            $gpus += [pscustomobject]@{
                index = [int]$parts[0]
                name = $parts[1]
                memory_total_mib = [int]$parts[2]
                memory_used_mib = [int]$parts[3]
                memory_free_mib = [int]$parts[4]
                utilization_gpu_pct = [int]$parts[5]
            }
        }
        if ($gpus.Count -eq 0) {
            return $null
        }
        return $gpus
    } catch {
        return $null
    }
}

function Get-BusyCpuProcesses([double]$ThresholdPct) {
    $ignorePrefixes = @("_total", "idle", "pwsh", "powershell", "cmd", "conhost", "codex", "wmiprvse")
    $isIgnored = {
        param([string]$Name)
        if ([string]::IsNullOrWhiteSpace($Name)) {
            return $true
        }
        $lower = $Name.ToLowerInvariant()
        foreach ($prefix in $ignorePrefixes) {
            if ($lower.StartsWith($prefix)) {
                return $true
            }
        }
        return $false
    }
    try {
        return @(
            Get-CimInstance Win32_PerfFormattedData_PerfProc_Process |
                Where-Object {
                    $_.Name -and
                    (-not (& $isIgnored $_.Name)) -and
                    $_.PercentProcessorTime -ge $ThresholdPct
                } |
                Sort-Object PercentProcessorTime -Descending |
                Select-Object @{Name = "InstanceName"; Expression = { $_.Name }},
                              @{Name = "CookedValue"; Expression = { [double]$_.PercentProcessorTime }}
        )
    } catch {
        $samples = Get-Counter '\Process(*)\% Processor Time' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
        return @(
            $samples.CounterSamples |
                Where-Object {
                    $_.InstanceName -and
                    (-not (& $isIgnored $_.InstanceName)) -and
                    $_.CookedValue -ge $ThresholdPct
                } |
                Sort-Object CookedValue -Descending |
                Select-Object InstanceName, CookedValue
        )
    }
}

function Assert-SystemReady([string]$OutputRootPath) {
    $cpuBusy = @(Get-BusyCpuProcesses $BusyCpuThresholdPct)
    $gpu = Get-GpuSnapshot
    $gpuBusy = $false
    if ($gpu) {
        $gpuBusy = (($gpu | Measure-Object -Property utilization_gpu_pct -Maximum).Maximum -ge $BusyGpuUtilThresholdPct)
    }

    $snapshot = [pscustomobject]@{
        checked_at = (Get-Date).ToString("o")
        busy_cpu_threshold_pct = $BusyCpuThresholdPct
        busy_gpu_util_threshold_pct = $BusyGpuUtilThresholdPct
        busy_cpu_processes = $cpuBusy
        gpu = $gpu
    }
    $snapshot | ConvertTo-Json -Depth 5 |
        Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputRootPath "system_load_precheck.json")

    if ($IgnoreBusySystem) {
        return
    }

    if (@($cpuBusy).Count -gt 0 -or $gpuBusy) {
        $details = @()
        if (@($cpuBusy).Count -gt 0) {
            $details += "busy CPU: " + (($cpuBusy | ForEach-Object {
                "{0}={1:N1}%" -f $_.InstanceName, $_.CookedValue
            }) -join ", ")
        }
        if ($gpuBusy) {
            $details += "GPU util >= $BusyGpuUtilThresholdPct%"
        }
        throw "System load precheck failed: $($details -join '; ')"
    }
}

function Parse-ProfileEntries([string]$Text) {
    $entries = [System.Collections.Generic.List[object]]::new()
    $regex = [regex]::new('(?m)^\[PROFILE\]\s+([^:]+):\s+total=([0-9.]+)s(?:,\s*calls=(\d+),\s*avg=([0-9.]+)s)?(?:,\s*bytes=(\d+))?')
    foreach ($match in $regex.Matches($Text)) {
        $calls = $null
        $avg = $null
        $bytes = $null
        if ($match.Groups[3].Success) {
            $calls = [int]$match.Groups[3].Value
        }
        if ($match.Groups[4].Success) {
            $avg = [double]$match.Groups[4].Value
        }
        if ($match.Groups[5].Success) {
            $bytes = [uint64]$match.Groups[5].Value
        }
        $entries.Add([pscustomobject]@{
            label = $match.Groups[1].Value
            total_sec = [double]$match.Groups[2].Value
            calls = $calls
            avg_sec = $avg
            bytes = $bytes
        })
    }
    return $entries
}

function Invoke-Checked([string]$Name, [string]$Program, [string[]]$ArgList, [string]$LogPath) {
    $cmdText = Format-Command $Program $ArgList
    Write-Host "[$Name] $cmdText"
    Ensure-Directory (Split-Path -Parent $LogPath)
    $runtimePath = [System.IO.Path]::ChangeExtension($LogPath, ".runtime.json")
    $samplesPath = [System.IO.Path]::ChangeExtension($LogPath, ".samples.csv")
    $monitorIntervalMs = 1000

    $startTime = Get-Date
    $systemBefore = Get-SystemMemorySnapshot
    $gpuBefore = Get-GpuSnapshot
    $peakWorkingSetMiB = 0.0
    $peakPrivateMiB = 0.0
    $peakGpuUsedMiB = $null
    $peakGpuUtilPct = $null
    $samples = [System.Collections.Generic.List[object]]::new()

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Program
    $psi.WorkingDirectory = $repo
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.Arguments = Format-ArgumentString $ArgList

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void]$proc.Start()
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()

    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds $monitorIntervalMs
        try {
            $live = Get-Process -Id $proc.Id -ErrorAction Stop
        } catch {
            continue
        }
        $elapsedSec = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 3)
        $workingSetMiB = [math]::Round($live.WorkingSet64 / 1MB, 1)
        $privateMiB = [math]::Round($live.PrivateMemorySize64 / 1MB, 1)
        $gpu = Get-GpuSnapshot
        $gpuUsedMiB = $null
        $gpuUtilPct = $null
        if ($gpu) {
            $gpuUsedMiB = [int](($gpu | Measure-Object -Property memory_used_mib -Sum).Sum)
            $gpuUtilPct = [int](($gpu | Measure-Object -Property utilization_gpu_pct -Maximum).Maximum)
            if ($null -eq $peakGpuUsedMiB -or $gpuUsedMiB -gt $peakGpuUsedMiB) {
                $peakGpuUsedMiB = $gpuUsedMiB
            }
            if ($null -eq $peakGpuUtilPct -or $gpuUtilPct -gt $peakGpuUtilPct) {
                $peakGpuUtilPct = $gpuUtilPct
            }
        }
        if ($workingSetMiB -gt $peakWorkingSetMiB) {
            $peakWorkingSetMiB = $workingSetMiB
        }
        if ($privateMiB -gt $peakPrivateMiB) {
            $peakPrivateMiB = $privateMiB
        }
        $samples.Add([pscustomobject]@{
            timestamp = (Get-Date).ToString("o")
            elapsed_sec = $elapsedSec
            working_set_mib = $workingSetMiB
            private_memory_mib = $privateMiB
            cpu_sec = [math]::Round($live.CPU, 3)
            gpu_total_used_mib = $gpuUsedMiB
            gpu_max_util_pct = $gpuUtilPct
        })
    }
    $proc.WaitForExit()

    $stdoutText = $stdoutTask.GetAwaiter().GetResult()
    $stderrText = $stderrTask.GetAwaiter().GetResult()
    $endTime = Get-Date
    $wallTimeSec = [math]::Round(($endTime - $startTime).TotalSeconds, 3)
    $systemAfter = Get-SystemMemorySnapshot
    $gpuAfter = Get-GpuSnapshot
    $profileEntries = Parse-ProfileEntries ($stdoutText + "`n" + $stderrText)

    $exitCode = $proc.ExitCode
    $peakWorkingSetMiB = [math]::Max($peakWorkingSetMiB, [math]::Round($proc.PeakWorkingSet64 / 1MB, 1))
    $cmdText | Set-Content -Encoding UTF8 -LiteralPath $LogPath
    if (-not [string]::IsNullOrWhiteSpace($stdoutText)) {
        Add-Content -Encoding UTF8 -LiteralPath $LogPath "`n=== STDOUT ==="
        Add-Content -Encoding UTF8 -LiteralPath $LogPath $stdoutText.TrimEnd()
    }
    if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
        Add-Content -Encoding UTF8 -LiteralPath $LogPath "`n=== STDERR ==="
        Add-Content -Encoding UTF8 -LiteralPath $LogPath $stderrText.TrimEnd()
    }
    if ($samples.Count -gt 0) {
        $samples | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $samplesPath
    }
    $runtime = [pscustomobject]@{
        name = $Name
        command = $cmdText
        pid = $proc.Id
        exit_code = $exitCode
        started_at = $startTime.ToString("o")
        finished_at = $endTime.ToString("o")
        wall_time_sec = $wallTimeSec
        peak_working_set_mib = $peakWorkingSetMiB
        peak_private_memory_mib = $peakPrivateMiB
        cpu_time_sec = [math]::Round($proc.TotalProcessorTime.TotalSeconds, 3)
        peak_gpu_total_used_mib = $peakGpuUsedMiB
        peak_gpu_util_pct = $peakGpuUtilPct
        sample_count = $samples.Count
        system_memory_before = $systemBefore
        system_memory_after = $systemAfter
        gpu_before = $gpuBefore
        gpu_after = $gpuAfter
        profile_entries = $profileEntries
        log_path = $LogPath
        samples_path = $(if ($samples.Count -gt 0) { $samplesPath } else { $null })
    }
    $runtime | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $runtimePath
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
    return [pscustomobject]@{
        summary_path = $runtimePath
        samples_path = $(if ($samples.Count -gt 0) { $samplesPath } else { $null })
        wall_time_sec = $wallTimeSec
        peak_working_set_mib = $peakWorkingSetMiB
        peak_private_memory_mib = $peakPrivateMiB
        peak_gpu_total_used_mib = $peakGpuUsedMiB
        peak_gpu_util_pct = $peakGpuUtilPct
        profile_entries = $profileEntries
    }
}

function Add-Result([System.Collections.Generic.List[object]]$Results,
                    [string]$Name, [string]$Prefix, [object]$RunInfo) {
    $mhd = "${Prefix}_it${Iterations}.mhd"
    $Results.Add([pscustomobject]@{
        name = $Name
        prefix = $Prefix
        mhd = $mhd
        exists = (Test-Path -LiteralPath $mhd)
        runtime_json = $RunInfo.summary_path
        samples_csv = $RunInfo.samples_path
        wall_time_sec = $RunInfo.wall_time_sec
        peak_working_set_mib = $RunInfo.peak_working_set_mib
        peak_private_memory_mib = $RunInfo.peak_private_memory_mib
        peak_gpu_total_used_mib = $RunInfo.peak_gpu_total_used_mib
        peak_gpu_util_pct = $RunInfo.peak_gpu_util_pct
        profile_entries = $RunInfo.profile_entries
    })
}

$exePath = Resolve-RepoPath $Exe
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "SPECT_recon.exe not found: $exePath"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "recon_out\260601_chunk_compare_it$Iterations"
}
$outputRootPath = Resolve-RepoPath $OutputRoot
Ensure-Directory $outputRootPath
Assert-SystemReady $outputRootPath

$compactDenomCacheDirPath = $null
if (-not [string]::IsNullOrWhiteSpace($CompactDenomCacheDir)) {
    $compactDenomCacheDirPath = Resolve-RepoPath $CompactDenomCacheDir
    Ensure-Directory $compactDenomCacheDirPath
}

$pscProjection = Resolve-RepoPath "test_data\6module\PSC_6module_24pin_10MBq_600s_1.bin"
$logoBed1 = Resolve-RepoPath "test_data\logo\Bed1.bin"
$logoBed2 = Resolve-RepoPath "test_data\logo\Bed2.bin"
$logoBed3 = Resolve-RepoPath "test_data\logo\Bed3.bin"
$matrix260601 = Resolve-RepoPath "test_data\6module\260601"
$matrix260525 = Resolve-RepoPath "test_data\6module\260525"
$load260525 = Join-Path $matrix260525 "matrix_6module.csrb"
$meta260525 = Join-Path $matrix260525 "metadata.bin"
$renderScript = Resolve-RepoPath "D:\WDY-project\Xiang_Project\SPECT\6Module\release\SPECT_6Module_Windows_20260530_125322\tools\render_recon_orthogonal_slices.py"
$metricsScript = Resolve-RepoPath "scripts\compare_recon_metrics.py"

$oldCache = $env:SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES
$env:SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES = $CompactChunkHostCacheBytes

$results = [System.Collections.Generic.List[object]]::new()

try {
    $common = @(
        "-config", "__missing_260601_compare_config__.ini",
        "-matrixLayout", "compact-symmetric",
        "-opti", "MLEM",
        "-subsets", "1",
        "-it", "$Iterations",
        "-th", "$Threads"
    )
    $standardGeometry = @(
        "-dim", "108,108,80",
        "-vox", "0.3,0.3,0.3",
        "-ang", "360",
        "-detN", "48,48"
    )

    $runPsc = $Case -eq "psc" -or $Case -eq "all"
    $runLogo = $Case -eq "logo" -or $Case -eq "all"

    if ($runPsc) {
        if (-not $Skip260601) {
            $prefix = Join-Path $outputRootPath "psc_260601\recon"
            Ensure-Directory (Split-Path -Parent $prefix)
            $cacheArgs = @()
            if ($compactDenomCacheDirPath) {
                $cacheArgs = @("-compactDenomCache",
                               (Join-Path $compactDenomCacheDirPath "psc_260601_global_denom.mhd"))
            }
            $args = @("-df", $pscProjection, "-chunkCSRDir", $matrix260601,
                      "-fout", $prefix) + $cacheArgs + $common
            $runInfo = Invoke-Checked "psc_260601" $exePath $args (Join-Path $outputRootPath "logs\psc_260601.log")
            Add-Result $results "psc_260601" $prefix $runInfo
        }
        if (-not $Skip260525) {
            $prefix = Join-Path $outputRootPath "psc_260525\recon"
            Ensure-Directory (Split-Path -Parent $prefix)
            $args = @("-df", $pscProjection, "-loadCSR", $load260525,
                      "-compactMeta", $meta260525, "-fout", $prefix) +
                    $standardGeometry + $common
            $runInfo = Invoke-Checked "psc_260525" $exePath $args (Join-Path $outputRootPath "logs\psc_260525.log")
            Add-Result $results "psc_260525" $prefix $runInfo
        }
    }

    if ($runLogo) {
        $dfList = "$logoBed1;$logoBed2;$logoBed3"
        if (-not $Skip260601) {
            $prefix = Join-Path $outputRootPath "logo_260601\recon"
            Ensure-Directory (Split-Path -Parent $prefix)
            $cacheArgs = @()
            if ($compactDenomCacheDirPath) {
                $cacheArgs = @("-compactDenomCache",
                               (Join-Path $compactDenomCacheDirPath "logo_260601_global_denom.mhd"))
            }
            $args = @("-dfList", $dfList, "-bedOverlap", "0",
                      "-chunkCSRDir", $matrix260601, "-fout", $prefix) + $cacheArgs + $common
            $runInfo = Invoke-Checked "logo_260601" $exePath $args (Join-Path $outputRootPath "logs\logo_260601.log")
            Add-Result $results "logo_260601" $prefix $runInfo
        }
        if (-not $Skip260525) {
            $prefix = Join-Path $outputRootPath "logo_260525\recon"
            Ensure-Directory (Split-Path -Parent $prefix)
            $args = @("-dfList", $dfList, "-bedOverlap", "0",
                      "-loadCSR", $load260525, "-compactMeta", $meta260525,
                      "-fout", $prefix) + $standardGeometry + $common
            $runInfo = Invoke-Checked "logo_260525" $exePath $args (Join-Path $outputRootPath "logs\logo_260525.log")
            Add-Result $results "logo_260525" $prefix $runInfo
        }
    }

    if (-not $SkipRender) {
        foreach ($result in $results) {
            if ($result.exists -and (Test-Path -LiteralPath $renderScript)) {
                $png = Join-Path $outputRootPath ("qa\" + $result.name + "_views.png")
                $args = @($renderScript, "--input", $result.mhd, "--output", $png,
                          "--title", $result.name)
                $null = Invoke-Checked ("render_" + $result.name) "python" $args (Join-Path $outputRootPath ("logs\render_" + $result.name + ".log"))
            }
        }
    }

    if (-not $SkipMetrics) {
        $pairs = @()
        if ($runPsc -and -not $Skip260601 -and -not $Skip260525) {
            $pairs += ,@("psc", (Join-Path $outputRootPath "psc_260601\recon_it${Iterations}.mhd"),
                         (Join-Path $outputRootPath "psc_260525\recon_it${Iterations}.mhd"))
        }
        if ($runLogo -and -not $Skip260601 -and -not $Skip260525) {
            $pairs += ,@("logo", (Join-Path $outputRootPath "logo_260601\recon_it${Iterations}.mhd"),
                         (Join-Path $outputRootPath "logo_260525\recon_it${Iterations}.mhd"))
        }
        foreach ($pair in $pairs) {
            $name = $pair[0]
            $candidate = $pair[1]
            $reference = $pair[2]
            if ((Test-Path -LiteralPath $candidate) -and (Test-Path -LiteralPath $reference)) {
                $outJson = Join-Path $outputRootPath ("metrics\" + $name + "_260601_vs_260525_metrics.json")
                $args = @($metricsScript, "--candidate", $candidate, "--reference", $reference,
                          "--out-json", $outJson, "--positive-union")
                $null = Invoke-Checked ("metrics_" + $name) "python" $args (Join-Path $outputRootPath ("logs\metrics_" + $name + ".log"))
            }
        }
    }

    $summary = [pscustomobject]@{
        case = $Case
        iterations = $Iterations
        threads = $Threads
        compact_chunk_host_cache_bytes = $CompactChunkHostCacheBytes
        compact_denom_cache_dir = $compactDenomCacheDirPath
        output_root = $outputRootPath
        results = $results
    }
    $summary | ConvertTo-Json -Depth 5 |
        Set-Content -Encoding UTF8 -LiteralPath (Join-Path $outputRootPath "run_summary.json")
    Write-Host "Wrote summary: $(Join-Path $outputRootPath 'run_summary.json')"
} finally {
    $env:SPECT_COMPACT_CHUNK_HOST_CACHE_BYTES = $oldCache
}
