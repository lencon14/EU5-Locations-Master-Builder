#!/usr/bin/env python3
"""Fixture tests for strip_markup, _resolve_var_refs, _clean_desc.

Run with: python3 pipeline/test_strip_markup.py
   or:    python3 -m pytest pipeline/test_strip_markup.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add pipeline directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from loc_parser import strip_markup
from extract_religions import _resolve_var_refs, _clean_desc

# --- strip_markup fixtures ---

STRIP_MARKUP_CASES = [
    # (input, expected_output)
    ("[religion|e]", "religion"),
    ("[religion|el]", "religion"),
    ("[religion|l]", "religion"),
    ("[ShowReligionName('catholic')|l]", "catholic"),
    ("[ShowPopTypeName('clergy')]", "clergy"),
    ("[Concept('trade_goods', 'trade goods')|e]", "trade goods"),
    # Adjacent concept refs should remain as separate words
    ("@icon!", ""),
    ("@gold_icon!", ""),
    ("#T Important #!", "Important"),
    ("#R Red text", "Red text"),
    ("#bold Bold text", "Bold text"),
    ("#italic Italic", "Italic"),
    ("$clergy_estate$", ""),
    ("$VARIABLE$", ""),
    ("text with \\n newline", "text with newline"),
    # Underscored concept words get spaced
    ("market_center area", "market center area"),
]


def test_strip_markup():
    """Test strip_markup against known fixtures."""
    failures = []
    for input_text, expected in STRIP_MARKUP_CASES:
        result = strip_markup(input_text).strip()
        if result != expected:
            failures.append(
                f"  strip_markup({input_text!r})\n"
                f"    expected: {expected!r}\n"
                f"    got:      {result!r}"
            )
    if failures:
        print(f"FAIL: {len(failures)} strip_markup failures:")
        for f in failures:
            print(f)
        return False
    print(f"OK: {len(STRIP_MARKUP_CASES)} strip_markup tests passed")
    return True


# --- _resolve_var_refs fixtures ---

RESOLVE_VAR_CASES = [
    # (input, lookups, expected)
    ("$clergy_estate$ is good", [{"clergy_estate": "Clergy"}], "Clergy is good"),
    ("$unknown_var$ here", [{}], " here"),
    ("no variables here", [{}], "no variables here"),
    ("$A$ and $B$", [{"A": "Alpha", "B": "Beta"}], "Alpha and Beta"),
    # Chained lookups: first match wins
    ("$key$", [{"key": "First"}, {"key": "Second"}], "First"),
    ("$key$", [{}, {"key": "Fallback"}], "Fallback"),
]


def test_resolve_var_refs():
    """Test _resolve_var_refs against known fixtures."""
    failures = []
    for input_text, lookups, expected in RESOLVE_VAR_CASES:
        result = _resolve_var_refs(input_text, *lookups)
        if result != expected:
            failures.append(
                f"  _resolve_var_refs({input_text!r}, ...)\n"
                f"    expected: {expected!r}\n"
                f"    got:      {result!r}"
            )
    if failures:
        print(f"FAIL: {len(failures)} _resolve_var_refs failures:")
        for f in failures:
            print(f)
        return False
    print(f"OK: {len(RESOLVE_VAR_CASES)} _resolve_var_refs tests passed")
    return True


# --- _clean_desc fixtures ---

CLEAN_DESC_CASES = [
    # (input, expected_output)
    ("[cavalry|e][regiment|e]s", "cavalry regiments"),
    ("[GetListOfGoodsUsingMethod('farming')]", ""),
    ("#T Description #!", "Description"),
    ("text: .", "text\u3002"),
    ("simple text", "simple text"),
    ("  extra   spaces  ", "extra spaces"),
]


def test_clean_desc():
    """Test _clean_desc against known fixtures."""
    failures = []
    for input_text, expected in CLEAN_DESC_CASES:
        result = _clean_desc(input_text)
        if result != expected:
            failures.append(
                f"  _clean_desc({input_text!r})\n"
                f"    expected: {expected!r}\n"
                f"    got:      {result!r}"
            )
    if failures:
        print(f"FAIL: {len(failures)} _clean_desc failures:")
        for f in failures:
            print(f)
        return False
    print(f"OK: {len(CLEAN_DESC_CASES)} _clean_desc tests passed")
    return True


def main():
    all_pass = True
    all_pass = test_strip_markup() and all_pass
    all_pass = test_resolve_var_refs() and all_pass
    all_pass = test_clean_desc() and all_pass

    total = len(STRIP_MARKUP_CASES) + len(RESOLVE_VAR_CASES) + len(CLEAN_DESC_CASES)
    if all_pass:
        print(f"\nAll {total} fixture tests passed.")
        sys.exit(0)
    else:
        print(f"\nSome fixture tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
