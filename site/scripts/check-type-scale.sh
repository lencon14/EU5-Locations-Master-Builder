#!/usr/bin/env bash
# Reject raw font-size literals in page templates.
# All font-size values must use CSS custom properties (--type-*).
# Allowlist: Base.astro's :root definition block and body reset.
set -euo pipefail

echo "=== Type Scale Audit ==="

violations=0
while IFS= read -r file; do
  # Skip Base.astro (defines the tokens + body reset)
  [[ "$file" == *"layouts/Base.astro"* ]] && continue
  # Skip components
  [[ "$file" == *"components/"* ]] && continue

  matches=$(grep -n 'font-size:' "$file" | grep -v 'var(--type' | grep -v '^\s*//' || true)
  if [[ -n "$matches" ]]; then
    echo "FAIL: $file"
    echo "$matches" | while IFS= read -r line; do
      echo "  $line"
    done
    violations=$((violations + 1))
  fi
done < <(find src/pages -name '*.astro')

if [[ $violations -gt 0 ]]; then
  echo ""
  echo "ERROR: $violations file(s) have raw font-size values."
  echo "Use var(--type-*) tokens from Base.astro :root instead."
  exit 1
else
  echo "OK: All page templates use type scale tokens."
fi
