param(
  [Parameter(Mandatory=$false)]
  [string[]]$Paths = @(
    ".\tools\*.ps1",
    ".\contracts\*.txt",
    ".\.github\workflows\*.yml",
    ".\.gitattributes",
    ".\.gitignore"
  )
)

$ErrorActionPreference = "Stop"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

$files = @()
foreach ($pat in $Paths) {
  $files += Get-ChildItem -File $pat -ErrorAction SilentlyContinue
}
$files = $files | Sort-Object FullName -Unique

foreach ($f in $files) {
  $p = $f.FullName
  $t = [System.IO.File]::ReadAllText($p)
  $t2 = $t -replace "`r`n", "`n"
  $t2 = $t2 -replace "`r", "`n"
  if ($t2 -ne $t) {
    [System.IO.File]::WriteAllText($p, $t2, $utf8NoBom)
    Write-Host "[FIX] LF normalized: $($f.FullName)"
  }
}

Write-Host "[DONE] normalize_lf completed."
exit 0
