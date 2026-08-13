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
    [int]$CudaDevice = 0,
    [ValidateRange(1, 60)]
    [int]$SampleSeconds = 5,
    [string]$ReportPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
                          ExecutablePath, CommandLine)
        if ($rows.Count -eq 0) { return "(none)" }
        return ($rows | Format-List | Out-String).TrimEnd()
    } catch {
        return "process query failed: $($_.Exception.Message)"
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

$reconArgs = @(
    "-config", $resolvedConfig,
    $(if ($MultiBedList) { "-dfList" } else { "-df" }), $InputPath,
    "-chunkCSRDir", $MatrixDir,
    "-it", $Iterations,
    "-fout", $resolvedOutputPrefix,
    "-cudaDevice", $CudaDevice.ToString(),
    "-vb", "1"
)
if (-not [string]::IsNullOrWhiteSpace($DenomCachePath)) {
    $reconArgs += @("-compactDenomCache", $DenomCachePath)
}
$displayCommand = (Quote-WindowsArgument $resolvedExe) + " " +
    (($reconArgs | ForEach-Object { Quote-WindowsArgument $_ }) -join " ")

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
Write-Host "Diagnostic report: $resolvedReport"
try {
    Add-Report "SPECT_recon device diagnostic bundle"
    Add-Report "started=$(Get-Date -Format o)"
    Add-Report "host=$env:COMPUTERNAME"
    Add-Report "user=$env:USERNAME"
    Add-Report "cwd=$((Get-Location).Path)"

    Add-Section "COMMAND"
    Add-Report $displayCommand
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
        Add-Report "model=$($computer.Manufacturer) $($computer.Model)"
        Add-Report "ram_bytes=$($computer.TotalPhysicalMemory)"
        Add-Report "os=$($os.Caption) $($os.Version) build=$($os.BuildNumber)"
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

    Add-Section "BEFORE START"
    Add-Report (Get-SpectProcessSnapshot)
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
    if (-not $process.Start()) { throw "Failed starting SPECT_recon.exe" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    Add-Report "launched_pid=$($process.Id) at $(Get-Date -Format o)"

    do {
        Add-Section ("SAMPLE " + (Get-Date -Format o))
        Add-Report (Get-SpectProcessSnapshot)
        Add-Report (Get-NvidiaSnapshot $nvidiaSmi)
        if (-not $process.HasExited) { Start-Sleep -Seconds $SampleSeconds }
    } while (-not $process.HasExited)

    $process.WaitForExit()
    $stdoutText = $stdoutTask.Result
    $stderrText = $stderrTask.Result
    Add-Section "PROCESS RESULT"
    Add-Report "exit_code=$($process.ExitCode)"
    Add-Report "finished=$(Get-Date -Format o)"
    Add-Section "STDOUT"
    Add-Report $stdoutText
    Add-Section "STDERR"
    Add-Report $stderrText

    $logMatch = [regex]::Match(
        $stdoutText + "`n" + $stderrText,
        '(?m)^.*?Logging console output to:\s*(.+\.log)\s*$')
    if ($logMatch.Success -and
        (Test-Path -LiteralPath $logMatch.Groups[1].Value.Trim() -PathType Leaf)) {
        $nativeLog = $logMatch.Groups[1].Value.Trim()
        Add-Section "NATIVE RECON LOG"
        Add-Report "path=$nativeLog"
        Add-Report (Get-Content -Raw -LiteralPath $nativeLog)
    }

    if ($process.ExitCode -ne 0) {
        throw "SPECT_recon exited with code $($process.ExitCode); report is complete."
    }
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
