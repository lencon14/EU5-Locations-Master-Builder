#!/bin/bash
# Build-time i18n quality check
# Run after: npm run build
# Scans dist/ for issues and validates game_terms coverage

set -e
DIST="dist"
DATA="src/data"
ERRORS=0

if [ ! -d "$DIST" ]; then
  echo "ERROR: $DIST not found. Run 'npm run build' first."
  exit 1
fi

echo "=== i18n Quality Report ==="

# 1. Raw game term keys in HTML output
echo ""
echo "--- Raw keys in HTML output ---"
RAW_KEYS=$(grep -roh 'pop\.[a-z_]\+\|field\.[a-z_]\+\|goods\.cat\.[a-z_]\+\|goods\.method\.[a-z_]\+\|goods\.origin\.[a-z_]\+' "$DIST" --include="*.html" 2>/dev/null | sort -u)
if [ -n "$RAW_KEYS" ]; then
  echo "FAIL: Found raw keys displayed to users:"
  echo "$RAW_KEYS" | sed 's/^/  /'
  ERRORS=$((ERRORS + 1))
else
  echo "OK: No raw keys found."
fi

# 2. game_terms.json key set diff (not just count)
echo ""
echo "--- game_terms.json key coverage ---"
EN_KEYS=$(python3 -c "import json; print('\n'.join(sorted(json.load(open('$DATA/loc/en/game_terms.json')).keys())))")
for lang in de es fr ja ko pl pt-br ru tr zh-hans; do
  FILE="$DATA/loc/$lang/game_terms.json"
  if [ ! -f "$FILE" ]; then
    echo "MISSING: $FILE"
    ERRORS=$((ERRORS + 1))
    continue
  fi
  LANG_KEYS=$(python3 -c "import json; print('\n'.join(sorted(json.load(open('$FILE')).keys())))")
  MISSING=$(comm -23 <(echo "$EN_KEYS") <(echo "$LANG_KEYS"))
  EXTRA=$(comm -13 <(echo "$EN_KEYS") <(echo "$LANG_KEYS"))
  if [ -n "$MISSING" ]; then
    echo "WARN: $lang missing keys: $(echo "$MISSING" | tr '\n' ', ')"
  fi
  if [ -n "$EXTRA" ]; then
    echo "INFO: $lang extra keys: $(echo "$EXTRA" | tr '\n' ', ')"
  fi
  COUNT=$(echo "$LANG_KEYS" | wc -l | tr -d ' ')
  EN_COUNT=$(echo "$EN_KEYS" | wc -l | tr -d ' ')
  if [ -z "$MISSING" ] && [ -z "$EXTRA" ]; then
    echo "OK: $lang — $COUNT terms"
  fi
done

# 3. Validate core data keys exist in game_terms
echo ""
echo "--- Core data → game_terms coverage ---"
if [ -f "$DATA/core/goods.json" ]; then
  # Check categories
  CORE_CATS=$(python3 -c "import json; cats=set(g.get('category','') for g in json.load(open('$DATA/core/goods.json'))); print('\n'.join(sorted(c for c in cats if c)))")
  for cat in $CORE_CATS; do
    KEY="goods.cat.$cat"
    if ! python3 -c "import json,sys; d=json.load(open('$DATA/loc/en/game_terms.json')); sys.exit(0 if '$KEY' in d else 1)" 2>/dev/null; then
      echo "WARN: Core category '$cat' has no game_terms key '$KEY'"
    fi
  done

  # Check methods
  CORE_METHODS=$(python3 -c "import json; ms=set(g.get('method','') for g in json.load(open('$DATA/core/goods.json'))); print('\n'.join(sorted(m for m in ms if m)))")
  for method in $CORE_METHODS; do
    KEY="goods.method.$method"
    if ! python3 -c "import json,sys; d=json.load(open('$DATA/loc/en/game_terms.json')); sys.exit(0 if '$KEY' in d else 1)" 2>/dev/null; then
      echo "WARN: Core method '$method' has no game_terms key '$KEY'"
    fi
  done

  # Check pop keys in demand data
  POP_KEYS=$(python3 -c "
import json
pops=set()
for g in json.load(open('$DATA/core/goods.json')):
    for f in ['demand_add','demand_multiply','wealth_impact_threshold']:
        pops.update((g.get(f) or {}).keys())
for p in sorted(pops): print(p)
")
  for pop in $POP_KEYS; do
    KEY="pop.$pop"
    if ! python3 -c "import json,sys; d=json.load(open('$DATA/loc/en/game_terms.json')); sys.exit(0 if '$KEY' in d else 1)" 2>/dev/null; then
      echo "WARN: Core pop key '$pop' has no game_terms key '$KEY'"
    fi
  done

  echo "OK: Core data keys validated against game_terms."
fi

# 4. languages.py ↔ config.ts consistency
echo ""
echo "--- Language config consistency ---"
if [ -f "../pipeline/languages.py" ] && [ -f "src/i18n/config.ts" ]; then
  PY_LANGS=$(python3 -c "
import sys; sys.path.insert(0,'../pipeline')
from languages import LANGUAGES
for url,_,_ in LANGUAGES: print(url)
" | sort)
  TS_LANGS=$(grep -oP "^\s+'?[a-z][-a-z]*'?\s*:" src/i18n/config.ts | sed "s/[': ]//g" | sort)
  if [ "$PY_LANGS" = "$TS_LANGS" ]; then
    echo "OK: languages.py and config.ts have matching language codes."
  else
    echo "FAIL: Language codes differ:"
    echo "  Python: $(echo $PY_LANGS | tr '\n' ' ')"
    echo "  TS:     $(echo $TS_LANGS | tr '\n' ' ')"
    ERRORS=$((ERRORS + 1))
  fi
fi

# 5. Per-language page count
echo ""
echo "--- Page count per language ---"
for lang in de en es fr ja ko pl pt-br ru tr zh-hans; do
  COUNT=$(find "$DIST/$lang" -name "index.html" 2>/dev/null | wc -l | tr -d ' ')
  echo "  $lang: $COUNT pages"
done

echo ""
if [ $ERRORS -gt 0 ]; then
  echo "=== FAILED: $ERRORS error(s) found ==="
  exit 1
else
  echo "=== PASSED ==="
fi
