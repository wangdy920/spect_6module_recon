$ErrorActionPreference = "Stop"
$setup = Join-Path $PSScriptRoot "setup_spect_workspace.ps1"
$temp = Join-Path ([IO.Path]::GetTempPath()) ("spect-workspace-test-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    $root = Join-Path $temp "workspace"
    & $setup -WorkspaceRoot $root
    $file = Join-Path $root "SPECT.code-workspace"
    $before = (Get-FileHash -LiteralPath $file).Hash
    & $setup -WorkspaceRoot $root
    if ((Get-FileHash -LiteralPath $file).Hash -ne $before) { throw "Idempotence failed" }
    $definition = Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($folder in $definition.folders) {
        $link = Get-Item -LiteralPath (Join-Path $root $folder.name)
        if ($link.LinkType -ne "Junction" -or $link.Target.Replace('\', '/') -ne $folder.path) {
            throw "Wrong target: $($folder.name)"
        }
    }
    $conflict = Join-Path $temp "conflict"
    New-Item -ItemType Directory -Path (Join-Path $conflict "preprocess") -Force | Out-Null
    $rejected = $false
    try { & $setup -WorkspaceRoot $conflict } catch { $rejected = $true }
    if (-not $rejected -or (Test-Path -LiteralPath (Join-Path $conflict "recon"))) {
        throw "Conflicting directory was accepted or partially installed"
    }
    Write-Host "PASS: junction targets, idempotence, and conflict refusal"
} finally {
    # Remove only our links first; never recursively traverse their source checkouts.
    foreach ($name in @("preprocess", "recon")) {
        $path = Join-Path $temp "workspace/$name"
        $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        if ($item -and $item.LinkType -eq "Junction") { [IO.Directory]::Delete($path) }
    }
    $resolved = [IO.Path]::GetFullPath($temp)
    if (-not $resolved.StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()), [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notlike "spect-workspace-test-*") { throw "Unexpected test cleanup path" }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
