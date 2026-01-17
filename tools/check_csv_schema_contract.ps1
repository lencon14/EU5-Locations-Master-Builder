param(
  [string]$CsvPath = ".\eu5_locations_master_raw.csv",
  [string]$ExpectedHeaderPath = ".\contracts\eu5_locations_master_raw.header.txt"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExpectedHeaderPath)) {
  Write-Error "Missing expected header file: $ExpectedHeaderPath"
  exit 2
}
if (-not (Test-Path $CsvPath)) {
  Write-Error "Missing csv file: $CsvPath"
  exit 2
}

$expected = (Get-Content $ExpectedHeaderPath -TotalCount 1).Trim()
$actual   = (Get-Content $CsvPath -TotalCount 1).Trim()

if ($expected -ne $actual) {
  Write-Host "[FAIL] CSV header contract violated."
  Write-Host "Expected: $expected"
  Write-Host "Actual  : $actual"
  exit 1
}

Write-Host "[PASS] CSV header contract OK."
exit 0
