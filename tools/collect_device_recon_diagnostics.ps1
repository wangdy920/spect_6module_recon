[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPrefix,
    [switch]$MultiBedList,
    [string]$ExePath = "",
    [string]$ConfigPath = "",
    [string]$MatrixDir = "",
    [string]$DenomCachePath = "",
    [string]$Iterations = "20:20",
    [ValidateRange(1, 9)]
    [int]$Repeats = 1,
    [int]$CudaDevice = 0,
    [ValidateRange(1, 60)]
    [int]$SampleSeconds = 5,
    [string]$ReportPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:storageDetailsByRoot = @{}

function Resolve-ExistingFile {
    param([string[]]$Candidates, [string]$Label)
    foreach ($candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and
            (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "$Label not found. Pass its explicit path."
}

function Resolve-FromBase {
    param([string]$Value, [string]$Base)
    if ([string]::IsNullOrWhiteSpace($Value)) { return "" }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Base $Value))
}

function Get-IniValue {
    param([string]$Path, [string]$Section, [string]$Key)
    $active = ""
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*\[([^]]+)\]\s*$') {
            $active = $Matches[1]
            continue
        }
        if ($active -ieq $Section -and
            $line -match ('^\s*' + [regex]::Escape($Key) + '\s*=\s*([^;#]*?)\s*(?:[;#].*)?$')) {
            return $Matches[1].Trim()
        }
    }
    return ""
}

function Quote-WindowsArgument {
    param([string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Get-SpectProcessSnapshot {
    try {
        $rows = @(Get-CimInstance Win32_Process -Filter "Name='SPECT_recon.exe'" |
            Select-Object ProcessId, ParentProcessId, CreationDate,
                          WorkingSetSize, PageFileUsage, ThreadCount,
                          KernelModeTime, UserModeTime,
                          ReadTransferCount, WriteTransferCount,
                          ExecutablePath, CommandLine)
        if ($rows.Count -eq 0) { return "(none)" }
        return ($rows | Format-List | Out-String).TrimEnd()
    } catch {
        return "process query failed: $($_.Exception.Message)"
    }
}

function Get-MemorySnapshot {
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        return "SYSTEM MEMORY: total_kib=$($os.TotalVisibleMemorySize) " +
            "free_kib=$($os.FreePhysicalMemory) " +
            "pagefile_total_kib=$($os.SizeStoredInPagingFiles) " +
            "pagefile_free_kib=$($os.FreeSpaceInPagingFiles)"
    } catch {
        return "system memory query failed: $($_.Exception.Message)"
    }
}

function Get-DiskActivitySnapshot {
    try {
        $rows = @(Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk |
            Where-Object Name -ne "_Total" |
            Select-Object Name, DiskReadBytesPersec, DiskWriteBytesPersec,
                          PercentDiskTime, CurrentDiskQueueLength)
        if ($rows.Count -eq 0) { return "SYSTEM DISK ACTIVITY: (none)" }
        return "SYSTEM DISK ACTIVITY:`r`n" +
            (($rows | Format-Table -AutoSize | Out-String).TrimEnd())
    } catch {
        return "system disk activity query failed: $($_.Exception.Message)"
    }
}

function Get-StorageSnapshot {
    param([string]$Path)
    try {
        $fullPath = [IO.Path]::GetFullPath($Path)
        $root = [IO.Path]::GetPathRoot($fullPath).TrimEnd('\')
        $prefix = @("path=$fullPath", "root=$root")
        if ($script:storageDetailsByRoot.ContainsKey($root)) {
            return (($prefix + $script:storageDetailsByRoot[$root]) -join "`r`n")
        }
        $details = @()
        if ($root.StartsWith('\\')) {
            $details += "type=network_unc"
            $script:storageDetailsByRoot[$root] = $details
            return (($prefix + $details) -join "`r`n")
        }
        $logical = Get-CimInstance Win32_LogicalDisk |
            Where-Object DeviceID -eq $root | Select-Object -First 1
        if ($null -ne $logical) {
            $driveTypes = @{ 2 = "removable"; 3 = "fixed"; 4 = "network";
                            5 = "optical"; 6 = "ramdisk" }
            $driveType = if ($driveTypes.ContainsKey([int]$logical.DriveType)) {
                $driveTypes[[int]$logical.DriveType]
            } else { "unknown" }
            $details += "volume_type=$driveType"
            $details += "filesystem=$($logical.FileSystem)"
            $details += "volume_size_bytes=$($logical.Size)"
            $details += "volume_free_bytes=$($logical.FreeSpace)"
        }
        try {
            $partition = Get-CimAssociatedInstance -InputObject $logical `
                -Association Win32_LogicalDiskToPartition `
                -ResultClassName Win32_DiskPartition | Select-Object -First 1
            $disk = Get-CimInstance -Namespace Root/Microsoft/Windows/Storage `
                -ClassName MSFT_Disk -Filter "Number=$($partition.DiskIndex)"
            $busTypes = @{ 1 = "SCSI"; 2 = "ATAPI"; 3 = "ATA"; 4 = "IEEE1394";
                           6 = "FibreChannel"; 7 = "USB"; 8 = "RAID";
                           9 = "iSCSI"; 10 = "SAS"; 11 = "SATA"; 12 = "SD";
                           13 = "MMC"; 15 = "Virtual"; 16 = "StorageSpaces";
                           17 = "NVMe"; 18 = "SCM"; 19 = "UFS" }
            $bus = if ($busTypes.ContainsKey([int]$disk.BusType)) {
                $busTypes[[int]$disk.BusType]
            } else { "unknown($($disk.BusType))" }
            $health = @{ 0 = "Healthy"; 1 = "Warning"; 2 = "Unhealthy";
                         5 = "Unknown" }
            $details += "disk_number=$($disk.Number)"
            $details += "disk_name=$($disk.FriendlyName)"
            $details += "disk_bus=$bus"
            $details += "disk_health=$(if ($health.ContainsKey([int]$disk.HealthStatus)) { $health[[int]$disk.HealthStatus] } else { $disk.HealthStatus })"
        } catch {
            $details += "physical_disk=unavailable ($($_.Exception.Message))"
        }
        $script:storageDetailsByRoot[$root] = $details
        return (($prefix + $details) -join "`r`n")
    } catch {
        return "storage query failed for '$Path': $($_.Exception.Message)"
    }
}

function Get-NvidiaSnapshot {
    param([System.Management.Automation.CommandInfo]$NvidiaSmi)
    if ($null -eq $NvidiaSmi) { return "nvidia-smi not found" }
    $gpu = & $NvidiaSmi.Source `
        --query-gpu=timestamp,index,name,driver_version,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem `
        --format=csv,noheader,nounits 2>&1 | Out-String
    $apps = & $NvidiaSmi.Source `
        --query-compute-apps=pid,process_name,used_memory `
        --format=csv,noheader,nounits 2>&1 | Out-String
    return "GPU:`r`n$($gpu.TrimEnd())`r`nCOMPUTE APPS:`r`n$($apps.TrimEnd())"
}

function Get-Median {
    param([double[]]$Values)
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $middle = [math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

function Get-RunMetrics {
    param([string]$Text, [double]$MeasuredWallSeconds)
    $perfMatches = [regex]::Matches($Text, '(?m)^.*?\[PERF\]\s+(.+)$')
    $perf = if ($perfMatches.Count -gt 0) {
        $perfMatches[$perfMatches.Count - 1].Groups[1].Value
    } else { "" }
    $cacheMatch = [regex]::Match($perf, '(?:^|\s)denom_cache=(\S+)')
    $wallMatch = [regex]::Match($perf, '(?:^|\s)full_command_s=([0-9.eE+-]+)')
    $residentMatch = [regex]::Match(
        $Text, '(?m)^.*?Resident matrix cache:\s*(\d+)\s*/\s*(\d+)')
    $overlapMatch = [regex]::Match(
        $Text, '(?m)^.*?Startup overlap:\s*resident prefix\s*(\d+)\s*/\s*(\d+)')
    [pscustomobject]@{
        DenomCache = if ($cacheMatch.Success) { $cacheMatch.Groups[1].Value } else { "unknown" }
        FullCommandSeconds = if ($wallMatch.Success) {
            [double]::Parse($wallMatch.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
        } else { $MeasuredWallSeconds }
        ResidentCached = if ($residentMatch.Success) { [int]$residentMatch.Groups[1].Value } else { -1 }
        ResidentTotal = if ($residentMatch.Success) { [int]$residentMatch.Groups[2].Value } else { -1 }
        StartupPrefix = if ($overlapMatch.Success) { [int]$overlapMatch.Groups[1].Value } else { -1 }
        StartupTotal = if ($overlapMatch.Success) { [int]$overlapMatch.Groups[2].Value } else { -1 }
        PerfLine = $perf
    }
}

$sourceRoot = Split-Path -Parent $PSScriptRoot
$installedBin = Split-Path -Parent $PSScriptRoot
$resolvedExe = Resolve-ExistingFile -Label "SPECT_recon.exe" -Candidates @(
    $ExePath,
    (Join-Path $installedBin "SPECT_recon.exe"),
    (Join-Path (Get-Location) "bin\SPECT_recon.exe"),
    (Join-Path $sourceRoot "build\Release\bin\Release\SPECT_recon.exe"),
    (Join-Path $sourceRoot "staging\SPECT_recon_rc_dev\bin\SPECT_recon.exe")
)
$exeDir = Split-Path -Parent $resolvedExe
$resolvedConfig = Resolve-ExistingFile -Label "spect_config.ini" -Candidates @(
    $ConfigPath,
    (Join-Path $exeDir "config\spect_config.ini"),
    (Join-Path $sourceRoot "multipinhole\config\spect_config.ini")
)

$matrixExplicit = $PSBoundParameters.ContainsKey("MatrixDir") -and
                  -not [string]::IsNullOrWhiteSpace($MatrixDir)
if (-not $matrixExplicit) {
    $MatrixDir = Get-IniValue $resolvedConfig "SystemMatrix" "default_chunk_csr_dir"
}
if (-not [string]::IsNullOrWhiteSpace($MatrixDir)) {
    $matrixBase = if ($matrixExplicit) { (Get-Location).Path } else { $exeDir }
    $MatrixDir = Resolve-FromBase $MatrixDir $matrixBase
} else {
    $matrixCandidates = @(Get-ChildItem -LiteralPath (Join-Path $exeDir "matrix") `
        -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "metadata.bin") })
    if ($matrixCandidates.Count -ne 1) {
        throw "Matrix auto-discovery found $($matrixCandidates.Count) candidates; pass -MatrixDir."
    }
    $MatrixDir = $matrixCandidates[0].FullName
}
if (-not (Test-Path -LiteralPath (Join-Path $MatrixDir "metadata.bin") -PathType Leaf)) {
    throw "Matrix directory has no metadata.bin: $MatrixDir"
}

$configuredDenomPath = Get-IniValue $resolvedConfig "SystemMatrix" "compact_denom_cache_path"
$configuredDenomDir = Get-IniValue $resolvedConfig "SystemMatrix" "compact_denom_cache_dir"
if (-not [string]::IsNullOrWhiteSpace($DenomCachePath)) {
    $DenomCachePath = Resolve-FromBase $DenomCachePath (Get-Location).Path
} elseif (-not [string]::IsNullOrWhiteSpace($configuredDenomPath)) {
    $DenomCachePath = Resolve-FromBase $configuredDenomPath $exeDir
}
$denomDir = if (-not [string]::IsNullOrWhiteSpace($DenomCachePath)) {
    Split-Path -Parent $DenomCachePath
} elseif (-not [string]::IsNullOrWhiteSpace($configuredDenomDir)) {
    Resolve-FromBase $configuredDenomDir $exeDir
} else {
    Resolve-FromBase "recon_out\compact_denom_cache" (Get-Location).Path
}

$resolvedOutputPrefix = Resolve-FromBase $OutputPrefix (Get-Location).Path
if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ReportPath = Join-Path (Get-Location) "spect_device_diagnostics_$stamp.txt"
}
$resolvedReport = Resolve-FromBase $ReportPath (Get-Location).Path
$reportParent = Split-Path -Parent $resolvedReport
if (-not [string]::IsNullOrWhiteSpace($reportParent)) {
    New-Item -ItemType Directory -Force -Path $reportParent | Out-Null
}

$utf8 = New-Object System.Text.UTF8Encoding($true)
$script:reportWriter = New-Object System.IO.StreamWriter($resolvedReport, $false, $utf8)
$script:reportWriter.AutoFlush = $true
function Add-Report {
    param([string]$Text = "")
    $script:reportWriter.WriteLine($Text)
}
function Add-Section {
    param([string]$Title)
    Add-Report ""
    Add-Report ("==== " + $Title + " ====")
}

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$process = $null
$runResults = @()
Write-Host "Diagnostic report: $resolvedReport"
try {
    Add-Report "SPECT_recon device diagnostic bundle"
    Add-Report "collector_schema=2"
    Add-Report "collector_path=$PSCommandPath"
    Add-Report "collector_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash)"
    Add-Report "started=$(Get-Date -Format o)"
    Add-Report "host=$env:COMPUTERNAME"
    Add-Report "user=$env:USERNAME"
    Add-Report "cwd=$((Get-Location).Path)"

    Add-Section "COMMAND"
    Add-Report "repeats=$Repeats"
    Add-Report "output_prefix_base=$resolvedOutputPrefix"
    Add-Report "report=$resolvedReport"
    Add-Report "denom_dir=$denomDir"

    Add-Section "EXECUTABLE"
    Add-Report "path=$resolvedExe"
    Add-Report "sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedExe).Hash)"
    Add-Report "version=$((& $resolvedExe --version 2>&1 | Out-String).Trim())"

    Add-Section "HOST"
    try {
        $computer = Get-CimInstance Win32_ComputerSystem
        $os = Get-CimInstance Win32_OperatingSystem
        $cpu = @(Get-CimInstance Win32_Processor)
        Add-Report "model=$($computer.Manufacturer) $($computer.Model)"
        Add-Report "ram_bytes=$($computer.TotalPhysicalMemory)"
        Add-Report "ram_free_kib=$($os.FreePhysicalMemory)"
        Add-Report "os=$($os.Caption) $($os.Version) build=$($os.BuildNumber)"
        Add-Report (($cpu | Select-Object Name, NumberOfCores,
            NumberOfLogicalProcessors, MaxClockSpeed |
            Format-List | Out-String).TrimEnd())
    } catch {
        Add-Report "host query failed: $($_.Exception.Message)"
    }
    $spectEnvironment = @([Environment]::GetEnvironmentVariables().GetEnumerator() |
        Where-Object { $_.Key -like "SPECT_*" -or $_.Key -eq "CUDA_VISIBLE_DEVICES" } |
        Sort-Object Key)
    if ($spectEnvironment.Count -eq 0) {
        Add-Report "SPECT/CUDA environment overrides: (none)"
    } else {
        Add-Report "SPECT/CUDA environment overrides:"
        foreach ($entry in $spectEnvironment) {
            Add-Report "  $($entry.Key)=$($entry.Value)"
        }
    }

    Add-Section "CONFIG"
    Add-Report "path=$resolvedConfig"
    Add-Report "sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedConfig).Hash)"
    Add-Report (Get-Content -Raw -LiteralPath $resolvedConfig)

    Add-Section "MATRIX AND DENOM CACHE"
    Add-Report "matrix=$MatrixDir"
    Add-Report "metadata_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $MatrixDir 'metadata.bin')).Hash)"
    $matrixFiles = @(Get-ChildItem -LiteralPath $MatrixDir -File |
        Where-Object { $_.Name -eq "metadata.bin" -or $_.Extension -eq ".csrb" })
    Add-Report "matrix_bytes=$(($matrixFiles | Measure-Object Length -Sum).Sum)"
    Add-Report (($matrixFiles | Select-Object Name, Length, LastWriteTime |
        Format-Table -AutoSize | Out-String).TrimEnd())
    Add-Report "denom_exact=$(if ($DenomCachePath) { $DenomCachePath } else { '(auto hash)' })"
    Add-Report "denom_dir=$denomDir"
    if (Test-Path -LiteralPath $denomDir -PathType Container) {
        $cacheFiles = @(Get-ChildItem -LiteralPath $denomDir -File |
            Where-Object { $_.Extension -in @(".mhd", ".raw", ".txt") })
        Add-Report (($cacheFiles | Select-Object Name, Length, LastWriteTime |
            Format-Table -AutoSize | Out-String).TrimEnd())
    } else {
        Add-Report "cache directory does not exist yet"
    }

    Add-Section "STORAGE"
    Add-Report "MATRIX STORAGE:"
    Add-Report (Get-StorageSnapshot $MatrixDir)
    Add-Report "DENOM STORAGE:"
    Add-Report (Get-StorageSnapshot $denomDir)
    Add-Report "OUTPUT STORAGE:"
    Add-Report (Get-StorageSnapshot $resolvedOutputPrefix)

    for ($run = 1; $run -le $Repeats; $run++) {
        $runOutputPrefix = if ($Repeats -eq 1) {
            $resolvedOutputPrefix
        } else { $resolvedOutputPrefix + "_run" + $run }
        $reconArgs = @(
            "-config", $resolvedConfig,
            $(if ($MultiBedList) { "-dfList" } else { "-df" }), $InputPath,
            "-chunkCSRDir", $MatrixDir,
            "-it", $Iterations,
            "-fout", $runOutputPrefix,
            "-cudaDevice", $CudaDevice.ToString(),
            "-vb", "1"
        )
        if (-not [string]::IsNullOrWhiteSpace($DenomCachePath)) {
            $reconArgs += @("-compactDenomCache", $DenomCachePath)
        }
        $displayCommand = (Quote-WindowsArgument $resolvedExe) + " " +
            (($reconArgs | ForEach-Object { Quote-WindowsArgument $_ }) -join " ")

        Add-Section "RUN $run/$Repeats BEFORE START"
        Add-Report "command=$displayCommand"
        Add-Report (Get-SpectProcessSnapshot)
        Add-Report (Get-MemorySnapshot)
        Add-Report (Get-DiskActivitySnapshot)
        Add-Report (Get-NvidiaSnapshot $nvidiaSmi)

        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $resolvedExe
        $argumentListProperty = $startInfo.GetType().GetProperty("ArgumentList")
        if ($null -ne $argumentListProperty) {
            foreach ($argument in $reconArgs) { $startInfo.ArgumentList.Add($argument) }
        } else {
            $startInfo.Arguments =
                ($reconArgs | ForEach-Object { Quote-WindowsArgument $_ }) -join " "
        }
        $startInfo.WorkingDirectory = (Get-Location).Path
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true

        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $startInfo
        $stopwatch = [Diagnostics.Stopwatch]::StartNew()
        if (-not $process.Start()) { throw "Failed starting SPECT_recon.exe" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        Add-Report "launched_pid=$($process.Id) at $(Get-Date -Format o)"

        do {
            Add-Section ("RUN $run SAMPLE " + (Get-Date -Format o))
            Add-Report (Get-SpectProcessSnapshot)
            Add-Report (Get-MemorySnapshot)
            Add-Report (Get-DiskActivitySnapshot)
            Add-Report (Get-NvidiaSnapshot $nvidiaSmi)
            if (-not $process.HasExited) { Start-Sleep -Seconds $SampleSeconds }
        } while (-not $process.HasExited)

        $process.WaitForExit()
        $stopwatch.Stop()
        $stdoutText = $stdoutTask.Result
        $stderrText = $stderrTask.Result
        $combinedText = $stdoutText + "`n" + $stderrText
        $metrics = Get-RunMetrics $combinedText $stopwatch.Elapsed.TotalSeconds
        $runResult = [pscustomobject]@{
            Run = $run
            ExitCode = $process.ExitCode
            MeasuredWallSeconds = $stopwatch.Elapsed.TotalSeconds
            DenomCache = $metrics.DenomCache
            FullCommandSeconds = $metrics.FullCommandSeconds
            ResidentCached = $metrics.ResidentCached
            ResidentTotal = $metrics.ResidentTotal
            StartupPrefix = $metrics.StartupPrefix
            StartupTotal = $metrics.StartupTotal
        }
        $runResults += $runResult

        Add-Section "RUN $run PROCESS RESULT"
        Add-Report ($runResult | Format-List | Out-String).TrimEnd()
        Add-Report "perf=$($metrics.PerfLine)"
        Add-Section "RUN $run STDOUT"
        Add-Report $stdoutText
        Add-Section "RUN $run STDERR"
        Add-Report $stderrText

        $logMatch = [regex]::Match(
            $combinedText, '(?m)^.*?Logging console output to:\s*(.+\.log)\s*$')
        if ($logMatch.Success -and
            (Test-Path -LiteralPath $logMatch.Groups[1].Value.Trim() -PathType Leaf)) {
            $nativeLog = $logMatch.Groups[1].Value.Trim()
            Add-Section "RUN $run NATIVE RECON LOG"
            Add-Report "path=$nativeLog"
            Add-Report (Get-Content -Raw -LiteralPath $nativeLog)
        }

        $process.Dispose()
        $process = $null
        if ($runResult.ExitCode -ne 0) {
            throw "SPECT_recon run $run exited with code $($runResult.ExitCode); report is complete."
        }
    }

    $fullTimes = @($runResults | ForEach-Object { [double]$_.FullCommandSeconds })
    Add-Section "REPEAT SUMMARY"
    Add-Report ($runResults | Format-Table -AutoSize | Out-String).TrimEnd()
    Add-Report ("median_full_command_s={0:F3}" -f (Get-Median $fullTimes))
    Add-Report "all_cache_hits=$(@($runResults | Where-Object DenomCache -ne 'hit').Count -eq 0)"
    Add-Report "all_resident_prefix_nonzero=$(@($runResults | Where-Object { $_.StartupPrefix -le 0 -and $_.ResidentCached -le 0 }).Count -eq 0)"
} catch {
    Add-Section "COLLECTOR ERROR"
    Add-Report $_.Exception.ToString()
    throw
} finally {
    if ($null -ne $script:reportWriter) {
        $script:reportWriter.Flush()
        $script:reportWriter.Dispose()
    }
    if ($null -ne $process) { $process.Dispose() }
}

Write-Host "Completed. Send this one file back for analysis: $resolvedReport"
