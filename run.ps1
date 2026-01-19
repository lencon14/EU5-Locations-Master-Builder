# EU5 Locations Master Builder runner (Windows / PowerShell)

[CmdletBinding()]
param(
  [string]$Eu5Root = $env:EU5_ROOT,
  [switch]$NoCache,
  [string]$CacheDir = $env:EU5_CACHE_DIR
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-Path '.tmp')) {
  New-Item -ItemType Directory -Force '.tmp' | Out-Null
}

if (-not (Test-Path '.venv')) {
  python -m venv .venv
}

$py = Join-Path '.venv' 'Scripts\python.exe'

& $py -m pip install -q --upgrade pip
& $py -m pip install -q -r requirements.txt

if ($Eu5Root) {
  $env:EU5_ROOT = $Eu5Root
}

if ($CacheDir) {
  $env:EU5_CACHE_DIR = $CacheDir
}

if ($NoCache) {
  $env:EU5_NO_CACHE = '1'
} else {
  Remove-Item Env:EU5_NO_CACHE -ErrorAction SilentlyContinue
}

& $py 'src\eu5_locations_master_builder.py'
