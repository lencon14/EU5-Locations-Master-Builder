param(
  [string]$EU5="C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V"
)

Write-Host "EU5 root: $EU5"

python .\src\eu5_locations_master_builder.py
