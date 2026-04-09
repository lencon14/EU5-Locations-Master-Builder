"""Generate all_loc.json — flat official loc dictionary per language.

Merges all game localization files into a single flat key→string map.
Used as a central lookup for any game term, avoiding per-category extraction gaps.

Usage:
    python pipeline/generate_all_loc.py

Output:
    pipeline/output/loc/{lang}/all_loc.json
"""

from __future__ import annotations

import json
from pathlib import Path

from loc_parser import parse_loc_file, strip_markup
from languages import LANGUAGES

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"

# All loc file prefixes to merge (order matters: later overrides earlier for dupes)
LOC_SOURCES = [
    "country_names",
    "country_description_category",
    "common_used_strings",
    "cultural_and_languages",
    "cultures",
    "culture_groups",
    "religion",
    "government",
    "government_names",
    "government_reforms",
    "goods",
    "buildings",
    "laws",
    "laws_and_policies",
    "pops",
    "game_concepts",
    "estate",
    "modifiers",
    "modifier_types",
    "static_modifiers",
    "advances",
    "area",
    "location_names",
    "province_names",
    "holy_sites",
    "religious_aspects",
    "formable_countries",
    "core",
    "traits",
]


def main():
    loc_dir = RAW_DIR / "localization"
    if not loc_dir.exists():
        print(f"[WARN] {loc_dir} not found")
        return

    for url_code, game_code, display in LANGUAGES:
        merged: dict[str, str] = {}
        sources_loaded = 0

        for loc_name in LOC_SOURCES:
            path = loc_dir / f"{loc_name}_l_{game_code}.yml"
            if path.exists():
                data = parse_loc_file(path)
                for k, v in data.items():
                    if v and isinstance(v, str):
                        clean = strip_markup(v)
                        if clean:
                            merged[k] = clean
                sources_loaded += 1

        out_path = OUTPUT_DIR / "loc" / url_code / "all_loc.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(merged, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"  {url_code}: {len(merged)} keys from {sources_loaded} sources ({out_path.stat().st_size // 1024}KB)")

    print(f"\nDone: {len(LANGUAGES)} languages")


if __name__ == "__main__":
    main()
