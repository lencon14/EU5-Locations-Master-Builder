#!/bin/bash
# Fetch EU5 game data from Windows PC via SSH.
# Usage: ./fetch_raw.sh [category ...]
# Example: ./fetch_raw.sh goods building_types religions
# No args = fetch all defined categories.

set -euo pipefail

EU5_BASE='C:\Program Files (x86)\Steam\steamapps\common\Europa Universalis V'
GAME="$EU5_BASE\\game\\in_game"
LOC="$EU5_BASE\\game\\main_menu\\localization"
RAW_DIR="$(cd "$(dirname "$0")" && pwd)/raw"

# --- helpers ---

fetch_dir() {
  local remote_path="$1"
  local local_dir="$2"
  local pattern="${3:-*}"

  mkdir -p "$local_dir"
  echo "  Fetching $remote_path → $local_dir"

  # Tar all matching files on remote, pipe to local extraction
  local count
  count=$(ssh winpc "powershell -Command \"
    chcp 65001 | Out-Null
    \\\$files = Get-ChildItem '$remote_path' -Filter '$pattern' -File
    foreach (\\\$f in \\\$files) {
      Write-Host \"FILE_START:\$(\\\$f.Name)\"
      [System.IO.File]::ReadAllText(\\\$f.FullName, [System.Text.Encoding]::UTF8)
      Write-Host 'FILE_END'
    }
  \"" 2>/dev/null | awk -v dir="$local_dir" '
    /^FILE_START:/ {
      fname = substr($0, 12)
      gsub(/\r/, "", fname)
      outfile = dir "/" fname
      writing = 1
      next
    }
    /^FILE_END/ {
      if (writing) close(outfile)
      writing = 0
      count++
      next
    }
    writing { print > outfile }
    END { print count + 0 }
  ')
  echo "    ${count} files"
}

fetch_loc() {
  local loc_name="$1"
  local local_dir="$RAW_DIR/localization"
  mkdir -p "$local_dir"

  for lang in english japanese; do
    local remote="$LOC\\$lang\\${loc_name}_l_${lang}.yml"
    local local_file="$local_dir/${loc_name}_l_${lang}.yml"
    echo "  Loc: ${loc_name}_l_${lang}.yml"
    ssh winpc "powershell -Command \"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; chcp 65001 | Out-Null; [System.IO.File]::ReadAllText('$remote', [System.Text.Encoding]::UTF8)\"" > "$local_file" 2>/dev/null || echo "    (not found, skipping)"
  done
}

# --- categories ---

fetch_goods() {
  echo "[goods]"
  fetch_dir "$GAME\\common\\goods" "$RAW_DIR/goods" "*.txt"
  fetch_loc "goods"
}

fetch_building_types() {
  echo "[building_types]"
  fetch_dir "$GAME\\common\\building_types" "$RAW_DIR/building_types" "*.txt"
  fetch_loc "buildings"
}

fetch_religions() {
  echo "[religions]"
  fetch_dir "$GAME\\common\\religions" "$RAW_DIR/religions" "*.txt"
  fetch_dir "$GAME\\common\\religion_groups" "$RAW_DIR/religion_groups" "*.txt"
  fetch_loc "religion"
}

fetch_countries() {
  echo "[countries]"
  fetch_dir "$GAME\\setup\\countries" "$RAW_DIR/countries" "*.txt"
  fetch_loc "countries"
  fetch_loc "country_names"
}

fetch_cultures() {
  echo "[cultures]"
  fetch_dir "$GAME\\common\\cultures" "$RAW_DIR/cultures" "*.txt"
  fetch_dir "$GAME\\common\\culture_groups" "$RAW_DIR/culture_groups" "*.txt"
  fetch_loc "cultures"
}

fetch_government_types() {
  echo "[government_types]"
  fetch_dir "$GAME\\common\\government_types" "$RAW_DIR/government_types" "*.txt"
  fetch_loc "government"
}

fetch_laws() {
  echo "[laws]"
  fetch_dir "$GAME\\common\\laws" "$RAW_DIR/laws" "*.txt"
  fetch_loc "laws"
}

fetch_version() {
  echo "[version]"
  local branch rev
  branch=$(ssh winpc "powershell -Command \"chcp 65001 | Out-Null; Get-Content '$EU5_BASE\\caesar_branch.txt' -Encoding UTF8\"" 2>/dev/null | tr -d '\r')
  rev=$(ssh winpc "powershell -Command \"chcp 65001 | Out-Null; Get-Content '$EU5_BASE\\caesar_rev.txt' -Encoding UTF8\"" 2>/dev/null | tr -d '\r')
  cat > "$RAW_DIR/VERSION.txt" <<VEOF
version: $(echo "$branch" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "$branch")
branch: $branch
rev: $rev
VEOF
  echo "  $(cat "$RAW_DIR/VERSION.txt" | head -1)"
}

# --- main ---

ALL_CATEGORIES=(goods building_types religions countries cultures government_types laws)

if [ $# -eq 0 ]; then
  categories=("${ALL_CATEGORIES[@]}")
else
  categories=("$@")
fi

echo "=== EU5 Raw Data Fetch ==="
fetch_version
for cat in "${categories[@]}"; do
  func="fetch_${cat}"
  if declare -f "$func" > /dev/null 2>&1; then
    "$func"
  else
    echo "[WARN] Unknown category: $cat"
  fi
done

# Normalize line endings
find "$RAW_DIR" -type f \( -name "*.txt" -o -name "*.yml" \) -exec sed -i '' $'s/\r$//' {} +

echo "=== Done ==="
