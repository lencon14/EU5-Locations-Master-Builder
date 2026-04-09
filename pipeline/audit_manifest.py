#!/usr/bin/env python3
"""Audit manifest generator — verifies tooltip coverage across religion-related pages.

Scans dist/{lang}/eu5/{religions,holy-sites,aspects}/ HTML files for:
  - data-tip attribute counts (tooltip coverage)
  - [bracket] or $variable$ leaks in visible text
  - English words in non-EN pages (R3/R4 style checks)

Outputs JSON summary to pipeline/output/audit_manifest.json + prints pass/fail.

Exit code: 0 = pass, 1 = failures found.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

# --- Configuration ---

_SCRIPT_DIR = Path(__file__).resolve().parent
SITE_DIR = _SCRIPT_DIR.parent / "site"
DIST_DIR = SITE_DIR / "dist"
OUTPUT_PATH = _SCRIPT_DIR / "output" / "audit_manifest.json"

CATEGORIES = ["religions", "holy-sites", "aspects"]
LANGUAGES = ["de", "en", "es", "fr", "ja", "ko", "pl", "pt-br", "ru", "tr", "zh-hans"]

SKIP_TAGS = {"script", "style", "code", "pre"}
SKIP_CLASSES = {"en-name", "id-badge", "lang-option-name", "lang-current"}

ALLOW_EXACT = {
    "Paradoxpedia", "EU5", "EN", "DATABASE", "COMING SOON",
    "Paradoxpedia \u2014 Fan-made guide for Europa Universalis V",
    "Europa Universalis V (EU5) Paradoxpedia",
}

ALLOW_PATTERNS = [
    re.compile(r"^https?://"),
    re.compile(r"^\d"),
    re.compile(r"^[A-Z]{1,5}$"),
    re.compile(r"^[+\-]?\d"),
    re.compile(r"^\u2605"),
    re.compile(r"^Worship .+ as our Patron God$"),
    re.compile(r"^The Holy City of "),
    re.compile(r"^Europa Universalis"),
]


# --- HTML Parsers ---

class TipCounter(HTMLParser):
    """Count data-tip attributes in HTML."""

    def __init__(self):
        super().__init__()
        self.tip_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        for name, _val in attrs:
            if name == "data-tip":
                self.tip_count += 1


class VisibleTextExtractor(HTMLParser):
    """Extract visible text from HTML, respecting skip classes and tags."""

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []
        self._skip_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag in SKIP_TAGS:
            self._skip_stack.append(tag)
            return
        attr_dict = dict(attrs)
        classes = set((attr_dict.get("class") or "").split())
        if classes & SKIP_CLASSES:
            self._skip_stack.append(f"cls:{tag}")

    def handle_endtag(self, tag: str):
        if self._skip_stack:
            top = self._skip_stack[-1]
            if top == tag or top == f"cls:{tag}":
                self._skip_stack.pop()

    def handle_data(self, data: str):
        text = data.strip()
        if text and not self._skip_stack:
            self.texts.append(text)


# --- Checks ---

def has_non_latin_script(text: str) -> bool:
    """Check for CJK, Cyrillic, Arabic, etc."""
    return bool(re.search(
        r"[\u0400-\u04ff"   # Cyrillic
        r"\u3000-\u9fff"    # CJK
        r"\uff00-\uff9f"    # Fullwidth
        r"\uac00-\ud7af"    # Hangul
        r"\u0600-\u06ff]",  # Arabic
        text,
    ))


def is_allowed(text: str) -> bool:
    if text in ALLOW_EXACT:
        return True
    for pat in ALLOW_PATTERNS:
        if pat.match(text):
            return True
    return False


def check_leaks(text: str) -> str | None:
    """Check for $variable$ or [bracket] leaks."""
    if re.search(r"\$\w+\$", text):
        return f"UNRESOLVED_VAR: {text[:60]}"
    if re.search(r"\[Show\w+\(", text):
        return f"PARADOX_MARKUP: {text[:60]}"
    if re.search(r"\[Concept\(", text):
        return f"PARADOX_MARKUP: {text[:60]}"
    if re.search(r"@\w+!", text):
        return f"ICON_MARKUP: {text[:60]}"
    return None


def check_raw_key(text: str) -> str | None:
    """Check for raw underscore keys."""
    if re.match(r"^[a-z][a-z0-9_]+[a-z0-9]$", text) and "_" in text and len(text) > 10:
        return f"RAW_KEY: {text}"
    return None


def check_english_in_non_en(text: str) -> str | None:
    """Check for English phrases that should be translated (non-EN only)."""
    if has_non_latin_script(text):
        return None
    if is_allowed(text):
        return None
    if len(text) < 20:
        return None
    words = text.split()
    if len(words) <= 5 and all(w[0].isupper() or len(w) <= 3 for w in words if w):
        return None
    alpha = re.findall(r"[a-zA-Z]", text)
    if len(alpha) / max(len(text), 1) > 0.7 and len(words) > 3:
        return f"ENG_SENTENCE: {text[:80]}"
    return None


# --- Audit ---

def audit_file(path: Path, lang: str) -> dict:
    """Audit a single HTML file. Returns per-file stats."""
    content = path.read_text(encoding="utf-8", errors="replace")

    tip_counter = TipCounter()
    tip_counter.feed(content)

    text_ext = VisibleTextExtractor()
    text_ext.feed(content)

    leaks = []
    raw_keys = []
    eng_leaks = []

    for text in text_ext.texts:
        leak = check_leaks(text)
        if leak:
            leaks.append(leak)
            continue
        rk = check_raw_key(text)
        if rk:
            raw_keys.append(rk)
            continue
        # English detection only meaningful for non-Latin-script languages
        if lang in ("ja", "ko", "zh-hans", "ru"):
            eng = check_english_in_non_en(text)
            if eng:
                eng_leaks.append(eng)

    return {
        "data_tips": tip_counter.tip_count,
        "leaks": leaks,
        "raw_keys": raw_keys,
        "english_leaks": eng_leaks,
        "issue_count": len(leaks) + len(raw_keys) + len(eng_leaks),
    }


def main():
    if not DIST_DIR.exists():
        print(f"ERROR: {DIST_DIR} not found. Run 'npm run build' first.")
        sys.exit(1)

    manifest: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "languages": {},
        "summary": {
            "total_pages": 0,
            "total_data_tips": 0,
            "total_issues": 0,
            "by_type": {"leaks": 0, "raw_keys": 0, "english_leaks": 0},
        },
    }

    total_issues = 0

    for lang in LANGUAGES:
        lang_result: dict = {"categories": {}, "total_tips": 0, "total_issues": 0}

        for category in CATEGORIES:
            cat_dir = DIST_DIR / lang / "eu5" / category
            if not cat_dir.exists():
                continue

            html_files = sorted(cat_dir.rglob("*.html"))
            cat_tips = 0
            cat_issues = 0
            cat_leaks: list[str] = []
            cat_raw_keys: list[str] = []
            cat_eng: list[str] = []

            for path in html_files:
                result = audit_file(path, lang)
                cat_tips += result["data_tips"]
                cat_issues += result["issue_count"]
                cat_leaks.extend(result["leaks"])
                cat_raw_keys.extend(result["raw_keys"])
                cat_eng.extend(result["english_leaks"])

            lang_result["categories"][category] = {
                "pages": len(html_files),
                "data_tips": cat_tips,
                "issues": cat_issues,
                "leak_details": cat_leaks[:5],  # truncate for readability
                "raw_key_details": cat_raw_keys[:5],
                "english_leak_details": cat_eng[:5],
            }
            lang_result["total_tips"] += cat_tips
            lang_result["total_issues"] += cat_issues
            manifest["summary"]["total_pages"] += len(html_files)
            manifest["summary"]["total_data_tips"] += cat_tips
            manifest["summary"]["by_type"]["leaks"] += len(cat_leaks)
            manifest["summary"]["by_type"]["raw_keys"] += len(cat_raw_keys)
            manifest["summary"]["by_type"]["english_leaks"] += len(cat_eng)

        manifest["languages"][lang] = lang_result
        total_issues += lang_result["total_issues"]

    manifest["summary"]["total_issues"] = total_issues

    # Write manifest
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest written to {OUTPUT_PATH}")

    # Sanity check: abort if no pages found (incomplete build)
    if manifest["summary"]["total_pages"] == 0:
        print("ERROR: 0 pages scanned. Build may be incomplete or dist/ is stale.")
        sys.exit(1)

    # Print summary
    print(f"\n--- Audit Manifest Summary ---")
    print(f"Pages scanned: {manifest['summary']['total_pages']}")
    print(f"Total data-tip attrs: {manifest['summary']['total_data_tips']}")
    print(f"Total issues: {total_issues}")
    print(f"  Leaks ($var$, markup): {manifest['summary']['by_type']['leaks']}")
    print(f"  Raw keys: {manifest['summary']['by_type']['raw_keys']}")
    print(f"  English in non-EN: {manifest['summary']['by_type']['english_leaks']}")

    for lang in LANGUAGES:
        lr = manifest["languages"][lang]
        if lr["total_issues"] > 0:
            print(f"  {lang}: {lr['total_issues']} issues across {sum(c['pages'] for c in lr['categories'].values())} pages")

    if total_issues > 0:
        print(f"\nFAIL: {total_issues} issue(s) found.")
        sys.exit(1)
    else:
        print(f"\nPASS: All religion-related pages clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
