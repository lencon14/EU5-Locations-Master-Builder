#!/bin/bash
# Build-time i18n quality check
# Run after: npm run build
# Scans dist/ for untranslated raw keys and fallback indicators

set -e
DIST="dist"

if [ ! -d "$DIST" ]; then
  echo "ERROR: $DIST not found. Run 'npm run build' first."
  exit 1
fi

echo "=== i18n Quality Report ==="

# 1. Raw game term keys (pop.xxx, field.xxx, goods.cat.xxx etc.)
echo ""
echo "--- Raw keys in HTML output ---"
RAW_KEYS=$(grep -roh 'pop\.[a-z_]\+\|field\.[a-z_]\+\|goods\.cat\.[a-z_]\+\|goods\.method\.[a-z_]\+\|goods\.origin\.[a-z_]\+' "$DIST" --include="*.html" 2>/dev/null | sort -u)
if [ -n "$RAW_KEYS" ]; then
  echo "WARN: Found raw keys displayed to users:"
  echo "$RAW_KEYS" | sed 's/^/  /'
else
  echo "OK: No raw keys found."
fi

# 2. Check each language has game_terms.json with expected key count
echo ""
echo "--- game_terms.json coverage ---"
EN_COUNT=$(python3 -c "import json; print(len(json.load(open('src/data/loc/en/game_terms.json'))))")
for lang in de en es fr ja ko pl pt-br ru tr zh-hans; do
  FILE="src/data/loc/$lang/game_terms.json"
  if [ -f "$FILE" ]; then
    COUNT=$(python3 -c "import json; print(len(json.load(open('$FILE'))))")
    if [ "$COUNT" -lt "$EN_COUNT" ]; then
      echo "WARN: $lang has $COUNT terms (en has $EN_COUNT)"
    else
      echo "OK: $lang — $COUNT terms"
    fi
  else
    echo "MISSING: $FILE"
  fi
done

# 3. Per-language page count check
echo ""
echo "--- Page count per language ---"
for lang in de en es fr ja ko pl pt-br ru tr zh-hans; do
  COUNT=$(find "$DIST/$lang" -name "index.html" 2>/dev/null | wc -l | tr -d ' ')
  echo "  $lang: $COUNT pages"
done

echo ""
echo "=== Done ==="
