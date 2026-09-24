param(
    [string]$WorkspaceRoot = "D:\Projects\SPECT-workspace",
    [string]$PreprocessRoot = "E:\Xiang_Project\SPECT\6Module",
    [string]$ReconRoot = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
if (-not $ReconRoot) { $ReconRoot = Join-Path $PSScriptRoot ".." }
$repos = [ordered]@{
    preprocess = (Resolve-Path -LiteralPath $PreprocessRoot).Path.TrimEnd('\')
    recon = (Resolve-Path -LiteralPath $ReconRoot).Path.TrimEnd('\')
}
$root = [IO.Path]::GetFullPath($WorkspaceRoot).TrimEnd('\')
foreach ($target in $repos.Values) {
    if (-not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
        throw "Not a Git checkout: $target"
    }
    if ($root -eq $target -or $root.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Workspace must be outside both source repositories: $root"
    }
}
if (Test-Path -LiteralPath $root) {
    $item = Get-Item -LiteralPath $root -Force
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Workspace root must be a regular directory: $root"
    }
}
$workspace = Join-Path $root "SPECT.code-workspace"
$definition = [ordered]@{
    folders = @(
        @{ name = "preprocess"; path = $repos.preprocess.Replace('\', '/') },
        @{ name = "recon"; path = $repos.recon.Replace('\', '/') }
    )
    settings = @{
        "files.watcherExclude" = @{
            "**/build/**" = $true; "**/release/**" = $true
            "**/recon_out/**" = $true; "**/test_data/**" = $true
            "**/experiments/**" = $true; "**/seg_result/**" = $true
        }
    }
} | ConvertTo-Json -Depth 5
# Validate all existing destinations before creating anything; never replace user files.
if ((Test-Path -LiteralPath $workspace) -and
    [IO.File]::ReadAllText($workspace, [Text.UTF8Encoding]::new($false, $true)).Trim() -ne $definition.Trim()) {
    throw "Existing workspace differs; preserve it and choose another WorkspaceRoot: $workspace"
}
foreach ($name in $repos.Keys) {
    $path = Join-Path $root $name
    $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    if ($item -and ($item.LinkType -ne "Junction" -or
        [IO.Path]::GetFullPath([string]$item.Target).TrimEnd('\') -ne $repos[$name])) {
        throw "Existing path is not the requested junction: $path"
    }
}
New-Item -ItemType Directory -Path $root -Force | Out-Null
foreach ($name in $repos.Keys) {
    $path = Join-Path $root $name
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Junction -Path $path -Target $repos[$name] | Out-Null
    }
    Write-Host "$path -> $($repos[$name])"
}
if (-not (Test-Path -LiteralPath $workspace)) {
    [IO.File]::WriteAllText($workspace, $definition + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}
Write-Host "Open $workspace; build and package using the original repository paths."
