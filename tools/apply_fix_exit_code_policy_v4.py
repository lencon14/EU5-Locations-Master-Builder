# -*- coding: utf-8 -*-
"""
apply_fix_exit_code_policy_v4.py

Goal:
- Ensure baseline-missing returns exit code 1 exactly.

Why needed:
- Current behavior returns 2 for baseline missing, which breaks the policy contract.

Scope:
- tools/run_lake_diag_regression.ps1 only

Safety:
- no logic change in builder, CSV, images, or diagnostic computation.
"""

from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("tools/run_lake_diag_regression.ps1")


def detect_newline_style(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def patch_baseline_missing_exit_code(text: str) -> tuple[str, bool]:
    """
    Find the baseline-missing Fail(...) call and enforce exit code 1.

    Handles both:
      Fail "Baseline log not found: ..."          -> add ", 1"
      Fail "Baseline log not found: ..." 2        -> replace trailing 2 with 1
    """
    # Match the Fail(...) line that contains "Baseline log not found:"
    # We patch *only* that Fail call to avoid touching other error paths.
    pat = re.compile(
        r'(?m)^(?P<indent>\s*)Fail\s+"Baseline log not found:[^"]*"\s*(?P<code>\d+)?\s*$'
    )

    m = pat.search(text)
    if not m:
        # Sometimes it may be multi-line string with backticks; fallback: patch Fail line containing Baseline log not found
        pat2 = re.compile(
            r'(?m)^(?P<indent>\s*)Fail\s+".*Baseline log not found:.*"\s*(?P<code>\d+)?\s*$'
        )
        m2 = pat2.search(text)
        if not m2:
            return text, False
        m = m2
        pat = pat2

    indent = m.group("indent") or ""
    replacement = indent + 'Fail "' + m.group(0).split('Fail "',1)[1].rsplit('"',1)[0] + '" 1'
    new_text = pat.sub(replacement, text, count=1)
    return new_text, True


def main() -> int:
    if not TARGET.exists():
        print(f"[ERROR] Target not found: {TARGET}")
        return 2

    raw = TARGET.read_text(encoding="utf-8", errors="replace")
    nl = detect_newline_style(raw)
    text = normalize(raw)

    patched, ok = patch_baseline_missing_exit_code(text)
    if not ok:
        print("[ERROR] Baseline-missing Fail(...) line not found. No changes made.")
        return 3

    if patched == text:
        print(f"[OK] No change required: {TARGET}")
        return 0

    TARGET.write_text(patched.replace("\n", nl), encoding="utf-8", newline="")
    print(f"[OK] Baseline-missing exit code fixed to 1: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
