Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$base = "origin/accuracy-first"

$changed = (git diff --name-only "$base...HEAD" | Out-String) -split "`r?`n" | Where-Object { $_ -ne "" }

$hasLog  = $changed -contains "artifacts/diag_baseline.log"
$hasMeta = $changed -contains "artifacts/diag_baseline.meta.json"

if ($hasLog -ne $hasMeta) {
  Write-Error "Baseline contract violation: diag_baseline.log and diag_baseline.meta.json must change together."
  exit 1
}

if ($hasLog -and $hasMeta) {
  $metaRaw = Get-Content -Raw -Encoding UTF8 .\artifacts\diag_baseline.meta.json
  $meta = $metaRaw | ConvertFrom-Json
  if (-not $meta.PSObject.Properties.Name -contains "reason") {
    Write-Error "Baseline contract violation: meta must include 'reason'."
    exit 2
  }
  if ([string]::IsNullOrWhiteSpace([string]$meta.reason)) {
    Write-Error "Baseline contract violation: 'reason' must be non-empty."
    exit 3
  }
}

exit 0