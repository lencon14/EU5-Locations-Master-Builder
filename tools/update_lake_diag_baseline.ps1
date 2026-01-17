param(
  [Parameter(Mandatory = $true)]
  [string]$Reason
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Reason)) {
  Write-Error 'Reason is required. Use: -Reason "why baseline is updated"'
  exit 2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
  $artDir = Join-Path $repoRoot "artifacts"
  if (-not (Test-Path $artDir)) {
    New-Item -ItemType Directory -Force $artDir | Out-Null
  }

  $baseline = Join-Path $artDir "diag_baseline.log"
  $metaPath = Join-Path $artDir "diag_baseline.meta.json"

  # Pick latest snapshot by filename (stable), exclude diag_baseline.log
  $snap = Get-ChildItem -Path (Join-Path $artDir "diag_*.log") -File -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -match '^diag_\d{8}_\d{6}\.log$' } |
          Sort-Object Name -Descending |
          Select-Object -First 1

  if (-not $snap) {
    throw "[ERROR] No snapshot found: artifacts\diag_*.log (run tools\run_lake_diag_regression.ps1 first)"
  }

  Copy-Item -Force $snap.FullName $baseline

  $gitCommit = (git rev-parse HEAD 2>$null | Out-String).Trim()
  $gitBranch = (git rev-parse --abbrev-ref HEAD 2>$null | Out-String).Trim()
  $pyVer     = (python --version 2>&1 | Out-String).Trim()

  $meta = [ordered]@{
    updated_utc    = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    reason         = $Reason
    git_commit     = $gitCommit
    git_branch     = $gitBranch
    python         = $pyVer
    snapshot_file  = $snap.Name
    snapshot_mtime = $snap.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ssK")
  }

  ($meta | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath $metaPath -Encoding UTF8

  Write-Host "[BASELINE UPDATED] diag_baseline.log <= $($snap.Name)"
  Write-Host "[META WRITTEN] $metaPath"
  exit 0
}
finally {
  Pop-Location
}