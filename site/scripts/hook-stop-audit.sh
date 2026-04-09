#!/bin/bash
# Stop hook: warn (not block) if religion-related pages have uncommitted changes.
# Real enforcement is in npm run build (check-i18n.sh + audit_manifest.py).
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

WATCHED='site/src/pages/\[lang\]/eu5/(religions|holy-sites|aspects)/|site/src/i18n/render\.ts|site/src/i18n/data\.ts|pipeline/extract_(religions|aspects|holy_sites)\.py'
CHANGED=$(git status --short 2>/dev/null | grep -E "$WATCHED" | head -1)

if [ -n "$CHANGED" ]; then
  echo "NOTE: Religion-related files have uncommitted changes. Run 'cd site && npm run build' before deploying." >&2
fi

exit 0
