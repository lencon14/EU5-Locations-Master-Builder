"""Extract EU5 building data and merge with localization into site-ready JSON.

Usage:
    python pipeline/extract_buildings.py

Output:
    pipeline/output/buildings.json
"""

from __future__ import annotations

import json
from pathlib import Path

from paradox_parser import parse_file
from loc_parser import parse_loc_file, strip_markup

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"

SKIP_FILES = {"readme.txt"}


def build_buildings() -> list[dict]:
    """Parse all building_types files and merge with localization."""
    bld_dir = RAW_DIR / "building_types"
    if not bld_dir.exists():
        print(f"[WARN] {bld_dir} not found, skipping")
        return []

    all_buildings: dict[str, dict] = {}
    for f in sorted(bld_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_buildings[key] = val

    # Localization
    loc_en: dict[str, str] = {}
    loc_ja: dict[str, str] = {}
    loc_dir = RAW_DIR / "localization"
    for path in loc_dir.glob("buildings_l_english.yml"):
        loc_en.update(parse_loc_file(path))
    for path in loc_dir.glob("buildings_l_japanese.yml"):
        loc_ja.update(parse_loc_file(path))

    result = []
    for bld_id, props in all_buildings.items():
        entry: dict = {
            "id": bld_id,
            "name_en": loc_en.get(bld_id, bld_id),
            "name_ja": loc_ja.get(bld_id, bld_id),
            "desc_en": strip_markup(loc_en.get(f"{bld_id}_desc", "")),
            "desc_ja": strip_markup(loc_ja.get(f"{bld_id}_desc", "")),
        }

        # Key properties
        for field in (
            "category",
            "max_levels",
            "pop_type",
            "build_time",
            "expensive",
        ):
            if field in props:
                entry[field] = props[field]

        # Settlement type flags
        settlements = []
        for s in ("rural_settlement", "town", "city"):
            if props.get(s) is True:
                settlements.append(s)
        if settlements:
            entry["settlements"] = settlements

        # Modifier
        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = props["modifier"]

        # Raw modifier
        if "raw_modifier" in props and isinstance(props["raw_modifier"], dict):
            entry["raw_modifier"] = props["raw_modifier"]

        # Production methods
        if "unique_production_methods" in props and isinstance(
            props["unique_production_methods"], dict
        ):
            methods = {}
            for mk, mv in props["unique_production_methods"].items():
                if isinstance(mv, dict):
                    methods[mk] = mv
            if methods:
                entry["production_methods"] = methods

        # Construction demand
        if "construction_demand" in props:
            entry["construction_demand"] = props["construction_demand"]

        entry["icon"] = f"icons/buildings/{bld_id}.png"
        entry["source_file"] = props.get("_source_file", "")
        result.append(entry)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = build_buildings()
    out_path = OUTPUT_DIR / "buildings.json"
    out_path.write_text(
        json.dumps(buildings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(buildings)} buildings to {out_path}")


if __name__ == "__main__":
    main()
