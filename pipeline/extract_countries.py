"""Extract EU5 country data and merge with localization into site-ready JSON.

Usage:
    python pipeline/extract_countries.py

Output:
    pipeline/output/countries.json
"""

from __future__ import annotations

import json
from pathlib import Path

from paradox_parser import parse_file
from loc_parser import parse_loc_file, strip_markup

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"

SKIP_FILES = {"readme.txt", "00_readme.info"}


def build_countries() -> list[dict]:
    """Parse country setup files and merge with localization."""
    ctr_dir = RAW_DIR / "countries"
    if not ctr_dir.exists():
        print(f"[WARN] {ctr_dir} not found, skipping")
        return []

    all_countries: dict[str, dict] = {}
    for f in sorted(ctr_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                # Derive region from filename
                val["_region"] = f.stem
                all_countries[key] = val

    # Localization
    loc_en: dict[str, str] = {}
    loc_ja: dict[str, str] = {}
    loc_dir = RAW_DIR / "localization"
    for pattern in (
        "country_names_l_english.yml",
        "country_description_category_l_english.yml",
    ):
        for path in loc_dir.glob(pattern):
            loc_en.update(parse_loc_file(path))
    for pattern in (
        "country_names_l_japanese.yml",
        "country_description_category_l_japanese.yml",
    ):
        for path in loc_dir.glob(pattern):
            loc_ja.update(parse_loc_file(path))

    result = []
    for tag, props in all_countries.items():
        entry: dict = {
            "tag": tag,
            "name_en": loc_en.get(tag, tag),
            "name_ja": loc_ja.get(tag, tag),
            "desc_en": strip_markup(loc_en.get(f"{tag}_desc", "")),
            "desc_ja": strip_markup(loc_ja.get(f"{tag}_desc", "")),
            "region": props.get("_region", ""),
        }

        for field in (
            "culture_definition",
            "religion_definition",
            "description_category",
            "difficulty",
        ):
            if field in props:
                entry[field] = props[field]

        entry["source_file"] = props.get("_source_file", "")
        result.append(entry)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    countries = build_countries()
    out_path = OUTPUT_DIR / "countries.json"
    out_path.write_text(
        json.dumps(countries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(countries)} countries to {out_path}")


if __name__ == "__main__":
    main()
