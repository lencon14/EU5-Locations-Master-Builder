"""Extract EU5 government type and law data into core + per-language loc files.

Usage:
    python pipeline/extract_governments.py

Output:
    pipeline/output/core/governments.json, core/laws.json
    pipeline/output/loc/{lang}/governments.json, loc/{lang}/laws.json
    pipeline/output/governments.json, laws.json  — legacy merged format
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

SKIP_FILES = {"readme.txt"}


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


def build_governments() -> tuple[list[dict], dict[str, dict]]:
    """Parse government_types. Returns (core_list, loc_per_lang)."""
    gov_dir = RAW_DIR / "government_types"
    if not gov_dir.exists():
        print(f"[WARN] {gov_dir} not found, skipping")
        return [], {}

    all_govs: dict[str, dict] = {}
    for f in sorted(gov_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_govs[key] = val

    all_loc = load_all_loc("government", "government_names", "government_reforms")

    core_list = []
    for gov_id, props in all_govs.items():
        entry: dict = {"id": gov_id}
        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = props["modifier"]
        entry["icon"] = f"icons/government_types/{gov_id}.png"
        entry["source_file"] = props.get("_source_file", "")
        core_list.append(entry)

    gov_ids = [g["id"] for g in core_list]
    loc_per_lang: dict[str, dict] = {}
    for url_code, loc_data in all_loc.items():
        lang_loc = {}
        for gov_id in gov_ids:
            name = loc_data.get(gov_id, "")
            desc = strip_markup(loc_data.get(f"{gov_id}_desc", ""))
            if name or desc:
                lang_loc[gov_id] = {"name": name, "desc": desc}
        loc_per_lang[url_code] = lang_loc

    return core_list, loc_per_lang


def build_laws() -> tuple[list[dict], dict[str, dict]]:
    """Parse laws. Returns (core_list, loc_per_lang)."""
    law_dir = RAW_DIR / "laws"
    if not law_dir.exists():
        print(f"[WARN] {law_dir} not found, skipping")
        return [], {}

    all_laws: dict[str, dict] = {}
    for f in sorted(law_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_laws[key] = val

    all_loc = load_all_loc("laws", "laws_and_policies")

    core_list = []
    for law_id, props in all_laws.items():
        entry: dict = {"id": law_id}
        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = props["modifier"]
        entry["icon"] = f"icons/laws/{law_id}.png"
        entry["source_file"] = props.get("_source_file", "")
        core_list.append(entry)

    law_ids = [l["id"] for l in core_list]
    loc_per_lang: dict[str, dict] = {}
    for url_code, loc_data in all_loc.items():
        lang_loc = {}
        for law_id in law_ids:
            name = loc_data.get(law_id, "")
            desc = strip_markup(loc_data.get(f"{law_id}_desc", ""))
            if name or desc:
                lang_loc[law_id] = {"name": name, "desc": desc}
        loc_per_lang[url_code] = lang_loc

    return core_list, loc_per_lang


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_output(category: str, core_list: list[dict], loc_per_lang: dict[str, dict]):
    """Write core, loc, and legacy files for a category."""
    write_json(OUTPUT_DIR / "core" / f"{category}.json", core_list)
    print(f"Wrote {len(core_list)} {category} to core/{category}.json")

    for url_code, loc_data in loc_per_lang.items():
        write_json(OUTPUT_DIR / "loc" / url_code / f"{category}.json", loc_data)
    lang_count = sum(1 for v in loc_per_lang.values() if v)
    print(f"Wrote {category} loc for {lang_count} languages")

    # Legacy merged format
    en_loc = loc_per_lang.get("en", {})
    ja_loc = loc_per_lang.get("ja", {})
    legacy = []
    for item in core_list:
        merged = dict(item)
        eid = item["id"]
        merged["name_en"] = en_loc.get(eid, {}).get("name", eid)
        merged["name_ja"] = ja_loc.get(eid, {}).get("name", eid)
        merged["desc_en"] = en_loc.get(eid, {}).get("desc", "")
        merged["desc_ja"] = ja_loc.get(eid, {}).get("desc", "")
        legacy.append(merged)
    write_json(OUTPUT_DIR / f"{category}.json", legacy)
    print(f"Wrote legacy {category}.json")


def main():
    core_govs, loc_govs = build_governments()
    write_output("governments", core_govs, loc_govs)

    core_laws, loc_laws = build_laws()
    write_output("laws", core_laws, loc_laws)


if __name__ == "__main__":
    main()
