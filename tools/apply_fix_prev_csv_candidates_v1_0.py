#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_fix_prev_csv_candidates_v1_0.py

Purpose
-------
Fix PREV_CSV_CANDIDATES in src/eu5_locations_master_builder.py for OUT_TAG=v1_0.

Patch policy
------------
- Rewrite ONLY the PREV_CSV_CANDIDATES = [...] block.
- No other lines may change.
- Fails fast if the block is not found exactly once.

Usage
-----
  python tools/apply_fix_prev_csv_candidates_v1_0.py
"""

from __future__ import annotations

import re
from pathlib import Path


BUILDER = Path("src/eu5_locations_master_builder.py")

PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]*)PREV_CSV_CANDIDATES\s*=\s*\[(?P<body>.*?)\]\s*$"
)

REPLACEMENT = """PREV_CSV_CANDIDATES = [
]
"""


def main() -> int:
    if not BUILDER.exists():
        raise SystemExit(f"[ERROR] Builder not found: {BUILDER}")

    before = BUILDER.read_text(encoding="utf-8", errors="replace")

    matches = list(PATTERN.finditer(before))
    if len(matches) != 1:
        raise SystemExit(f"[ERROR] PREV_CSV_CANDIDATES block not found exactly once (found={len(matches)}).")

    m = matches[0]
    start, end = m.span()

    after = before[:start] + REPLACEMENT + before[end:]

    # Safety: ensure only that block changed
    if before == after:
        raise SystemExit("[ERROR] No change detected (unexpected).")

    BUILDER.write_text(after, encoding="utf-8", newline="\n")
    print("[OK] PREV_CSV_CANDIDATES fixed for v1_0 (now empty).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
