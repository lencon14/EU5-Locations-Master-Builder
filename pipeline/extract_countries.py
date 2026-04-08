"""Extract EU5 country data into core + per-language loc files.

Usage:
    python pipeline/extract_countries.py

Output:
    pipeline/output/core/countries.json
    pipeline/output/loc/{lang}/countries.json
    pipeline/output/countries.json               — legacy merged format
"""

from __future__ import annotations

import json
from pathlib import Path

from paradox_parser import parse_file
from loc_parser import parse_loc_file, strip_markup
from languages import LANGUAGES

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"

SKIP_FILES = {"readme.txt", "00_readme.info"}


def load_all_loc(*loc_names: str) -> dict[str, dict[str, str]]:
    """Load and merge multiple loc files for all languages."""
    result: dict[str, dict[str, str]] = {}
    loc_dir = RAW_DIR / "localization"
    for url_code, game_code, _ in LANGUAGES:
        merged: dict[str, str] = {}
        for loc_name in loc_names:
            path = loc_dir / f"{loc_name}_l_{game_code}.yml"
            if path.exists():
                merged.update(parse_loc_file(path))
        result[url_code] = merged
    return result


def build_countries() -> tuple[list[dict], dict[str, dict]]:
    """Parse country setup files. Returns (core_list, loc_per_lang)."""
    ctr_dir = RAW_DIR / "countries"
    if not ctr_dir.exists():
        print(f"[WARN] {ctr_dir} not found, skipping")
        return [], {}

    all_countries: dict[str, dict] = {}
    for f in sorted(ctr_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                val["_region"] = f.stem
                all_countries[key] = val

    all_loc = load_all_loc("country_names", "country_description_category")

    # Core
    core_list = []
    for tag, props in all_countries.items():
        entry: dict = {
            "tag": tag,
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
        core_list.append(entry)

    # Per-language loc
    tags = [c["tag"] for c in core_list]
    loc_per_lang: dict[str, dict] = {}
    for url_code, loc_data in all_loc.items():
        lang_loc = {}
        for tag in tags:
            name = loc_data.get(tag, "")
            desc = strip_markup(loc_data.get(f"{tag}_desc", ""))
            if name or desc:
                lang_loc[tag] = {"name": name, "desc": desc}
        loc_per_lang[url_code] = lang_loc

    return core_list, loc_per_lang


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    core_list, loc_per_lang = build_countries()

    write_json(OUTPUT_DIR / "core" / "countries.json", core_list)
    print(f"Wrote {len(core_list)} countries to core/countries.json")

    for url_code, loc_data in loc_per_lang.items():
        write_json(OUTPUT_DIR / "loc" / url_code / "countries.json", loc_data)
    lang_count = sum(1 for v in loc_per_lang.values() if v)
    print(f"Wrote countries loc for {lang_count} languages")

    # Legacy merged format
    en_loc = loc_per_lang.get("en", {})
    ja_loc = loc_per_lang.get("ja", {})
    legacy = []
    for item in core_list:
        merged = dict(item)
        tag = item["tag"]
        merged["name_en"] = en_loc.get(tag, {}).get("name", tag)
        merged["name_ja"] = ja_loc.get(tag, {}).get("name", tag)
        merged["desc_en"] = en_loc.get(tag, {}).get("desc", "")
        merged["desc_ja"] = ja_loc.get(tag, {}).get("desc", "")
        legacy.append(merged)
    write_json(OUTPUT_DIR / "countries.json", legacy)
    print(f"Wrote legacy countries.json")


if __name__ == "__main__":
    main()
