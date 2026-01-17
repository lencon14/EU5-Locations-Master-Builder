Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Compare against upstream tracking branch. Adjust if CI uses a different base.
$base = "origin/accuracy-first"

$changed = (git diff --name-only "$base...HEAD" | Out-String) -split "`r?`n" | Where-Object { $_ -ne "" }

$hasLog  = $changed -contains "artifacts/diag_baseline.log"
$hasMeta = $changed -contains "artifacts/diag_baseline.meta.json"

if ($hasLog -ne $hasMeta) {
  Write-Error "Baseline contract violation: diag_baseline.log and diag_baseline.meta.json must change together."
  exit 1
}

exit 0