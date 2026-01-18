Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$IsCI = ($env:GITHUB_ACTIONS -eq 'true') -or ($env:CI -eq 'true')
if ([string]::IsNullOrWhiteSpace($env:BASE_REF)) {
  if ($IsCI) { throw "BASE_REF is empty (CI contract broken)" }
  # local fallback
  $env:BASE_REF = "origin/master"
}
$base = $env:BASE_REF

git rev-parse --verify $base 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Error "BASE_REF '$base' not found. Run 'git branch -r' or 'git fetch --all --prune'."
  exit 4
}


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
