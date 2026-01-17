<# 
tools/run_lake_diag_regression.ps1

Purpose:
  One-command diagnostic regression workflow for lake adjacency.
  - No edits to main pipeline code
  - No changes to CSV/image/logic
  - Creates timestamped diagnostic log snapshot
  - Compares against a baseline log via tools/analyze_diagnostic_lake_adjacency.py
  - STRICT BY DEFAULT: exits non-zero if deltas are detected
  - Adds SHORT TEXT SUMMARY of compare results (human-friendly)
  - Adds BASELINE FRESHNESS CHECK (age warning)

Usage:
  powershell -ExecutionPolicy Bypass -File .\tools\run_lake_diag_regression.ps1
  powershell -ExecutionPolicy Bypass -File .\tools\run_lake_diag_regression.ps1 -NonStrict
  powershell -ExecutionPolicy Bypass -File .\tools\run_lake_diag_regression.ps1 -KeepDiagLog
  powershell -ExecutionPolicy Bypass -File .\tools\run_lake_diag_regression.ps1 -BaselineMaxAgeDays 30
#>



# -----------------------------------------------------------------------------
# Exit code policy
# -----------------------------------------------------------------------------
# 0 : PASS (no deltas)
# 1 : Baseline missing / baseline precondition not satisfied
# 2 : Analyzer missing / tool precondition not satisfied
# 3 : Baseline stale (freshness threshold exceeded)
# 4 : Builder execution failed
# 5 : Compare failed (metric deltas / anomalies / parse failure)
# 9 : Unexpected failure
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
  [string]$Baseline = ".\artifacts\diag_baseline.log",
  [string]$Analyzer = ".\tools\analyze_diagnostic_lake_adjacency.py",
  [string]$Builder  = ".\src\eu5_locations_master_builder.py",
  [string]$DiagLog  = ".\artifacts\diagnostic_lake_adjacency.log",
  [string]$ArtifactsDir = ".\artifacts",
  [int]$BaselineMaxAgeDays = 14,
  [switch]$NonStrict,
  [switch]$KeepDiagLog
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Fail([string]$Message, [int]$Code = 1) {
  # Ensure exit code is preserved even if $ErrorActionPreference='Stop'.
  Write-Error -Message $Message -ErrorAction Continue
  exit $Code
}
function Write-BaselineFreshness([string]$BaselinePath, [int]$MaxAgeDays) {
  try {
    $item = Get-Item -LiteralPath $BaselinePath -ErrorAction Stop
  } catch {
    Fail "Baseline not found for freshness check: $BaselinePath" 2
  }

  $now = Get-Date
  $mtime = $item.LastWriteTime
  $age = $now - $mtime
  $ageDays = [math]::Floor($age.TotalDays)

  Write-Host ("[BASELINE] {0}  mtime={1}  age_days={2}  threshold_days={3}" -f $BaselinePath, $mtime.ToString("yyyy-MM-dd HH:mm:ss"), $ageDays, $MaxAgeDays)

  if ($age.TotalDays -gt $MaxAgeDays) {
    Write-Warning ("[BASELINE] Baseline is older than threshold ({0} days). Consider refreshing baseline." -f $MaxAgeDays)
  }
}

function Get-CompareJson([string]$AnalyzerPath, [string]$APath, [string]$BPath) {
  $json = python $AnalyzerPath compare --a $APath --b $BPath | Out-String
  if ($LASTEXITCODE -ne 0) {
    Fail "Analyzer compare failed (exit $LASTEXITCODE)" $LASTEXITCODE
  }
  try {
    return $json | ConvertFrom-Json
  } catch {
    Fail "Failed to parse compare JSON output." 4
  }
}

function Write-CompareSummary($Obj) {
  $aRec = $Obj.meta.a_records
  $bRec = $Obj.meta.b_records

  $diffItems = @()
  foreach ($p in $Obj.delta.PSObject.Properties) {
    $name = $p.Name
    $d = $p.Value

    $p50a = $null; $p50b = $null; $p50diff = $null
    $maxa = $null; $maxb = $null; $maxdiff = $null

    if ($null -ne $d.p50) { $p50a = $d.p50.a; $p50b = $d.p50.b; $p50diff = $d.p50.diff }
    if ($null -ne $d.max) { $maxa = $d.max.a; $maxb = $d.max.b; $maxdiff = $d.max.diff }

    $has = $false
    if ($null -ne $p50diff -and [int]$p50diff -ne 0) { $has = $true }
    if ($null -ne $maxdiff -and [int]$maxdiff -ne 0) { $has = $true }

    $diffItems += [pscustomobject]@{
      name = $name
      has_diff = $has
      p50_a = $p50a; p50_b = $p50b; p50_diff = $p50diff
      max_a = $maxa; max_b = $maxb; max_diff = $maxdiff
    }
  }

  $anDiff = 0
  if ($null -ne $Obj.anomalies -and $null -ne $Obj.anomalies.diff) { $anDiff = [int]$Obj.anomalies.diff }

  $changed = $diffItems | Where-Object { $_.has_diff }
  $changedCount = ($changed | Measure-Object).Count

  if ($changedCount -eq 0 -and $anDiff -eq 0) {
    Write-Host ("[SUMMARY] PASS: no metric deltas. records(a,b)=({0},{1}) anomalies_diff={2}" -f $aRec, $bRec, $anDiff)
    return
  }

  Write-Host ("[SUMMARY] FAIL: metric deltas detected. records(a,b)=({0},{1}) anomalies_diff={2}" -f $aRec, $bRec, $anDiff)
  foreach ($it in $changed) {
    $parts = @()
    if ($null -ne $it.p50_diff) { $parts += ("p50 {0}->{1} (diff {2})" -f $it.p50_a, $it.p50_b, $it.p50_diff) }
    if ($null -ne $it.max_diff) { $parts += ("max {0}->{1} (diff {2})" -f $it.max_a, $it.max_b, $it.max_diff) }
    Write-Host ("[SUMMARY] {0}: {1}" -f $it.name, ($parts -join "; "))
  }
}

if (-not (Test-Path $Analyzer)) { Fail "Analyzer not found: $Analyzer" 2 }
if (-not (Test-Path $Builder))  { Fail "Builder not found:  $Builder" 2 }
if (-not (Test-Path $ArtifactsDir)) { Fail "Artifacts dir not found: $ArtifactsDir" 2 }
if (-not (Test-Path $Baseline)) {
  Fail "Baseline log not found: $Baseline`nCreate it first: Copy-Item $DiagLog $Baseline" 1
}

# Baseline freshness check (informational, non-fatal)
Write-BaselineFreshness -BaselinePath $Baseline -MaxAgeDays $BaselineMaxAgeDays

# Strict is default unless -NonStrict is specified
$Strict = -not $NonStrict

# 1) Remove current diagnostic log to avoid append/mixing
Remove-Item $DiagLog -ErrorAction SilentlyContinue

# 2) Run builder with --diagnostic (do not suppress stdout/stderr)
python $Builder --diagnostic
if ($LASTEXITCODE -ne 0) { Fail "Builder failed with exit code $LASTEXITCODE" $LASTEXITCODE }

# 3) Confirm diagnostic log exists
if (-not (Test-Path $DiagLog)) { Fail "Diagnostic log was not created: $DiagLog" 3 }

# 4) Snapshot log with timestamp
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$Snapshot = Join-Path $ArtifactsDir ("diag_{0}.log" -f $ts)
Copy-Item $DiagLog $Snapshot -Force

# 5) Compare baseline vs snapshot (prints JSON for auditability)
python $Analyzer compare --a $Baseline --b $Snapshot
$cmpExit = $LASTEXITCODE
if ($cmpExit -ne 0) { Fail "Analyzer compare failed with exit code $cmpExit" $cmpExit }

# 6) Short text summary (always printed)
$cmpObj = Get-CompareJson -AnalyzerPath $Analyzer -APath $Baseline -BPath $Snapshot
Write-CompareSummary $cmpObj

# 7) Strict mode: fail if any diff != 0 (based on parsed object)
if ($Strict) {
  $hasDiff = $false

  if ($null -ne $cmpObj.anomalies -and $null -ne $cmpObj.anomalies.diff) {
    if ([int]$cmpObj.anomalies.diff -ne 0) { $hasDiff = $true }
  }

  if ($null -ne $cmpObj.delta) {
    foreach ($p in $cmpObj.delta.PSObject.Properties) {
      $d = $p.Value
      if ($null -ne $d.p50 -and $null -ne $d.p50.diff) {
        if ([int]$d.p50.diff -ne 0) { $hasDiff = $true }
      }
      if ($null -ne $d.max -and $null -ne $d.max.diff) {
        if ([int]$d.max.diff -ne 0) { $hasDiff = $true }
      }
    }
  }

  if ($hasDiff) {
    Write-Error ("[STRICT FAIL] Deltas detected vs baseline. Snapshot: {0}" -f $Snapshot)
    if (-not $KeepDiagLog) { Remove-Item $DiagLog -ErrorAction SilentlyContinue }
    exit 10
  } else {
    Write-Host ("[STRICT PASS] No deltas detected vs baseline. Snapshot: {0}" -f $Snapshot)
  }
} else {
  Write-Host ("[OK] Snapshot created: {0}" -f $Snapshot)
  Write-Host ("[OK] Compared against baseline: {0}" -f $Baseline)
}

# 8) Optionally remove the live diagnostic log to avoid accidental reuse
if (-not $KeepDiagLog) {
  Remove-Item $DiagLog -ErrorAction SilentlyContinue
}
