#!/usr/bin/env python3
"""Full-page i18n audit for all languages — catches translation leaks in built pages.

Runs after build. Parses HTML, extracts visible text (skipping script/style/code),
and checks for 5 categories of translation issues.

Exit code: 0 = pass, 1 = failures found.

Rules (all languages):
  R1: Unresolved $variable$ or Paradox markup remnants
  R2: Raw underscore keys (modifier/mechanic names)
  R5: Text starting with Japanese particle (stripped $var$ remnant) — JA only

Rules (non-EN languages):
  R3: English phrase leaks (multi-word English in non-English context)
  R4: Isolated English sentences (full English text that should be translated)
"""

from __future__ import annotations

import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# --- Configuration ---

DIST_DIR = Path(__file__).parent.parent / "dist"

ALL_LANGUAGES = ["de", "en", "es", "fr", "ja", "ko", "pl", "pt-br", "ru", "tr", "zh-hans"]

# CSS classes whose text content is intentionally English (skip checking)
SKIP_CLASSES = {"en-name", "id-badge", "lang-option-name", "lang-current"}

# Tags whose content is always skipped
SKIP_TAGS = {"script", "style", "code", "pre"}

# Allowlist: exact text values that are OK even though they look English
ALLOW_EXACT = {
    "Paradoxpedia", "EU5", "EN", "DATABASE", "COMING SOON",
    "Paradoxpedia \u2014 Fan-made guide for Europa Universalis V",
    "Europa Universalis V (EU5) Paradoxpedia",
}

# Allowlist: regex patterns for text that is OK
ALLOW_PATTERNS = [
    re.compile(r"^https?://"),            # URLs
    re.compile(r"^\d"),                    # Starts with number
    re.compile(r"^[A-Z]{1,5}$"),          # Short abbreviations (EN, DB, etc.)
    re.compile(r"^[+\-]?\d"),             # Numeric values (+10%, -0.1)
    re.compile(r"^\u2605"),               # Star ratings
    re.compile(r"^Worship .+ as our Patron God$"),  # Game-official worship aspect names
    re.compile(r"^The Holy City of "),     # Game-official holy site proper names
    re.compile(r"^Europa Universalis"),    # Site title
]

# Known existing issues that are WARN not FAIL (pre-existing, tracked separately)
KNOWN_ISSUES_PATH_PREFIX = {
    "eu5/buildings/": {
        "debug_max_profit", "tiny_production_efficiency_bonus",
        # $variable$ in building condition_lines (extract_buildings.py fix needed)
        "UNRESOLVED_VAR",
        # scaled modifier values in building pages
        "small_production_efficiency_bonus",
    },
    "eu5/religions/": {
        # scaled modifier values in religion definition_modifier (extract fix needed)
        "permanent target satisfaction", "trade efficiency",
        "tax income efficiency", "stability investment",
        "production efficiency",
    },
    "eu5/holy-sites/": {
        # scaled modifier values in holy site type modifiers (extract fix needed)
        "production efficiency", "permanent target satisfaction",
    },
}

# Regression fixtures: these specific texts MUST be caught if they appear
REGRESSION_FIXTURES = [
    "societal value monthly move",
    "medium permanent target satisfaction",
    "small permanent target satisfaction",
    "tiny permanent target satisfaction",
]


# --- HTML Parser ---

class SkipAwareExtractor(HTMLParser):
    """More precise extractor that tracks skip state via a class stack."""

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []
        self._skip_stack: list[str] = []  # stack of tags causing skip

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


# --- Rules ---

def is_allowed(text: str) -> bool:
    """Check if text matches any allowlist entry."""
    if text in ALLOW_EXACT:
        return True
    for pat in ALLOW_PATTERNS:
        if pat.match(text):
            return True
    return False


def has_japanese(text: str) -> bool:
    """Check if text contains Japanese characters."""
    return bool(re.search(r"[\u3000-\u9fff\uff00-\uff9f]", text))


def has_non_latin_script(text: str) -> bool:
    """Check for CJK, Cyrillic, Korean, etc."""
    return bool(re.search(
        r"[\u0400-\u04ff"   # Cyrillic
        r"\u3000-\u9fff"    # CJK
        r"\uff00-\uff9f"    # Fullwidth
        r"\uac00-\ud7af"    # Hangul
        r"\u0600-\u06ff"    # Arabic
        r"\u0e00-\u0e7f]",  # Thai
        text,
    ))


def check_r1_unresolved_var(text: str) -> str | None:
    """R1: Unresolved $variable$ or Paradox markup remnants."""
    if re.search(r"\$\w+\$", text):
        return f"R1:UNRESOLVED_VAR: {text[:60]}"
    if re.search(r"\[Show\w+\(", text):
        return f"R1:PARADOX_MARKUP: {text[:60]}"
    if re.search(r"\[Concept\(", text):
        return f"R1:PARADOX_MARKUP: {text[:60]}"
    if re.search(r"@\w+!", text):
        return f"R1:ICON_MARKUP: {text[:60]}"
    return None


def check_r2_raw_key(text: str) -> str | None:
    """R2: Raw underscore keys (modifier/mechanic/game_concept names)."""
    if re.match(r"^[a-z][a-z0-9_]+[a-z0-9]$", text) and "_" in text and len(text) > 10:
        return f"R2:RAW_KEY: {text}"
    return None


def check_r3_english_phrase(text: str) -> str | None:
    """R3: Multi-word English phrases that should be translated."""
    if has_non_latin_script(text):
        return None  # Mixed text with native script, skip
    if is_allowed(text):
        return None
    # Count consecutive lowercase English words (4+ chars each)
    words = text.split()
    eng_streak = 0
    max_streak = 0
    for w in words:
        clean = re.sub(r"[^a-zA-Z]", "", w)
        if len(clean) >= 4 and clean[0].islower():
            eng_streak += 1
            max_streak = max(max_streak, eng_streak)
        else:
            eng_streak = 0
    if max_streak >= 3:
        return f"R3:ENG_PHRASE: {text[:80]}"
    return None


def check_r4_english_sentence(text: str) -> str | None:
    """R4: Full English sentence/phrase (no native script chars, long enough to be suspicious).

    Skips proper nouns (Title Case, < 5 words) since game loc names are often
    English even in non-English localizations. Only flags longer phrases that
    look like untranslated UI text or descriptions.
    """
    if has_non_latin_script(text):
        return None
    if is_allowed(text):
        return None
    if len(text) < 20:
        return None
    # Skip Title Case proper nouns (game names): "Arat Sabulungan", "Tibetan Buddhism"
    words = text.split()
    if len(words) <= 5 and all(w[0].isupper() or len(w) <= 3 for w in words if w):
        return None
    # Check if it's mostly English letters
    alpha = re.findall(r"[a-zA-Z]", text)
    if len(alpha) / max(len(text), 1) > 0.7 and len(words) > 3:
        return f"R4:ENG_SENTENCE: {text[:80]}"
    return None


def check_r5_particle_start(text: str) -> str | None:
    """R5: Text starting with Japanese particle (stripped $var$ remnant).

    Low confidence — Japanese descriptions legitimately start with は/が when
    the subject is shown as a heading. Treated as WARN, not FAIL.
    JA-only rule.
    """
    if re.match(r"^[はがをにでとのへもから][\s、]", text) and len(text) > 2:
        return f"R5:PARTICLE_START: {text[:60]}"
    return None


# Rule sets by language type
RULES_ALL = [check_r1_unresolved_var, check_r2_raw_key]
RULES_NON_EN = [check_r3_english_phrase, check_r4_english_sentence]
RULES_JA_ONLY = [check_r5_particle_start]


# --- Main ---

def audit_file(path: Path, rel: str, lang: str) -> list[tuple[str, str, str]]:
    """Audit a single HTML file. Returns list of (rel_path, rule_id, detail)."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return []  # Skip files that can't be read
    ext = SkipAwareExtractor()
    ext.feed(content)

    # English detection only meaningful for non-Latin-script languages
    CJK_CYRILLIC = {"ja", "ko", "zh-hans", "ru"}
    rules = list(RULES_ALL)
    if lang in CJK_CYRILLIC:
        rules.extend(RULES_NON_EN)
    if lang == "ja":
        rules.extend(RULES_JA_ONLY)

    issues = []
    for text in ext.texts:
        if not text:
            continue
        for rule in rules:
            result = rule(text)
            if result:
                rule_id = result.split(":")[0]
                issues.append((rel, rule_id, result))
                break  # One rule per text segment
    return issues


def classify_issue(rel: str, rule_id: str, detail: str) -> str:
    """Classify issue as FAIL or WARN based on known issues and rule confidence."""
    # R5 (particle start) is low-confidence — always WARN
    if rule_id == "R5":
        return "WARN"
    # Known pre-existing issues by path prefix
    for prefix, known_texts in KNOWN_ISSUES_PATH_PREFIX.items():
        if prefix in rel:
            for known in known_texts:
                if known in detail:
                    return "WARN"
    return "FAIL"


def audit_language(lang: str) -> tuple[list[tuple[str, str, str, str]], int]:
    """Audit all pages for a single language. Returns (issues, page_count)."""
    dist_lang = DIST_DIR / lang
    if not dist_lang.exists():
        return [], 0

    html_files = sorted(dist_lang.rglob("*.html"))
    all_issues: list[tuple[str, str, str, str]] = []

    for path in html_files:
        rel = str(path.relative_to(dist_lang))
        issues = audit_file(path, rel, lang)
        for rel_path, rule_id, detail in issues:
            severity = classify_issue(rel_path, rule_id, detail)
            all_issues.append((rel_path, rule_id, detail, severity))

    return all_issues, len(html_files)


def main():
    if not DIST_DIR.exists():
        print(f"ERROR: {DIST_DIR} not found. Run 'npm run build' first.")
        sys.exit(1)

    total_fails = 0
    total_warns = 0
    total_pages = 0

    for lang in ALL_LANGUAGES:
        all_issues, page_count = audit_language(lang)
        if page_count == 0:
            continue
        total_pages += page_count

        # Separate FAIL vs WARN
        fails = [(r, rule, d) for r, rule, d, s in all_issues if s == "FAIL"]
        warns = [(r, rule, d) for r, rule, d, s in all_issues if s == "WARN"]

        # Deduplicate by detail text
        seen_details: set[str] = set()
        unique_fails: list[tuple[str, str, str]] = []
        for r, rule, d in fails:
            if d not in seen_details:
                seen_details.add(d)
                unique_fails.append((r, rule, d))

        seen_warns: set[str] = set()
        unique_warns: list[tuple[str, str, str]] = []
        for r, rule, d in warns:
            if d not in seen_warns:
                seen_warns.add(d)
                unique_warns.append((r, rule, d))

        # Print per-language summary
        if unique_fails or unique_warns:
            print(f"\n--- {lang.upper()} ({page_count} pages) ---")

        if unique_warns:
            print(f"  WARN: {len(unique_warns)} known issues")
            for r, rule, d in unique_warns[:3]:
                print(f"    [{rule}] {r}: {d}")
            if len(unique_warns) > 3:
                print(f"    ... and {len(unique_warns) - 3} more")

        if unique_fails:
            print(f"  FAIL: {len(unique_fails)} translation issues:")
            for r, rule, d in sorted(unique_fails)[:10]:
                print(f"    [{rule}] {r}: {d}")
            if len(unique_fails) > 10:
                print(f"    ... and {len(unique_fails) - 10} more")
            # Regression check (JA-specific fixtures)
            if lang == "ja":
                for fixture in REGRESSION_FIXTURES:
                    for _, _, d in unique_fails:
                        if fixture in d.lower():
                            print(f"    [REGRESSION] Caught known regression: {fixture}")
                            break

        total_fails += len(unique_fails)
        total_warns += len(unique_warns)

    # Final summary
    print(f"\n--- Full audit: {len(ALL_LANGUAGES)} languages, {total_pages} pages ---")
    if total_fails > 0:
        print(f"FAIL: {total_fails} issues across all languages. ({total_warns} known warnings)")
        sys.exit(1)
    else:
        print(f"OK: No translation issues. ({total_warns} known warnings)")
        sys.exit(0)


if __name__ == "__main__":
    main()
