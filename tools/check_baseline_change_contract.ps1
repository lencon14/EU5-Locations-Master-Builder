Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Always run git in repo root (avoid UNC/cwd issues)
$repoRoot = Split-Path -Parent $PSScriptRoot

# Base ref can be overridden by env var (CI-friendly)
$base = if ($env:BASE_REF) { $env:BASE_REF } else { "origin/accuracy-first" }

# --- Validate BASE_REF exists (Stop-safe for native commands)
$rc = 0
$oldEap = $ErrorActionPreference
$hadNativePref = Test-Path Variable:\PSNativeCommandUseErrorActionPreference
if ($hadNativePref) { $oldNativePref = $PSNativeCommandUseErrorActionPreference }

try {
  $ErrorActionPreference = "Continue"
  if ($hadNativePref) { $PSNativeCommandUseErrorActionPreference = $false }

  & git -C $repoRoot rev-parse --verify $base 2>$null | Out-Null
  $rc = $LASTEXITCODE
}
finally {
  if ($hadNativePref) { $PSNativeCommandUseErrorActionPreference = $oldNativePref }
  $ErrorActionPreference = $oldEap
}

if ($rc -ne 0) {
  [Console]::Error.WriteLine("BASE_REF '$base' not found. Run 'git -C `"$repoRoot`" fetch --all --prune' or set BASE_REF to an existing ref.")
  exit 4
}

# --- Compute changed files vs base...HEAD
$changed = (& git -C $repoRoot diff --name-only "$base...HEAD" | Out-String) -split "`r?`n" | Where-Object { $_ -ne "" }

$hasLog  = $changed -contains "artifacts/diag_baseline.log"
$hasMeta = $changed -contains "artifacts/diag_baseline.meta.json"

if ($hasLog -ne $hasMeta) {
  [Console]::Error.WriteLine("Baseline contract violation: diag_baseline.log and diag_baseline.meta.json must change together.")
  exit 1
}

if ($hasLog -and $hasMeta) {
  $metaPath = Join-Path $repoRoot "artifacts/diag_baseline.meta.json"
  $metaRaw  = Get-Content -Raw -Encoding UTF8 $metaPath
  $meta     = $metaRaw | ConvertFrom-Json

  if (-not ($meta.PSObject.Properties.Name -contains "reason")) {
    [Console]::Error.WriteLine("Baseline contract violation: meta must include 'reason'.")
    exit 2
  }
  if ([string]::IsNullOrWhiteSpace([string]$meta.reason)) {
    [Console]::Error.WriteLine("Baseline contract violation: 'reason' must be non-empty.")
    exit 3
  }
}

exit 0