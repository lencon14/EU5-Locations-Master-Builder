"""Extract EU5 religion data and merge with localization into site-ready JSON.

Usage:
    python pipeline/extract_religions.py

Output:
    pipeline/output/religions.json
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


def build_religions() -> list[dict]:
    """Parse religion files and merge with localization."""
    # Religion groups
    groups: dict[str, dict] = {}
    grp_dir = RAW_DIR / "religion_groups"
    if grp_dir.exists():
        for f in sorted(grp_dir.glob("*.txt")):
            if f.name.lower() in SKIP_FILES:
                continue
            data = parse_file(f)
            for key, val in data.items():
                if isinstance(val, dict):
                    groups[key] = val

    # Religions
    rel_dir = RAW_DIR / "religions"
    if not rel_dir.exists():
        print(f"[WARN] {rel_dir} not found, skipping")
        return []

    all_religions: dict[str, dict] = {}
    for f in sorted(rel_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_religions[key] = val

    # Build group lookup from religion's own "group = xxx" field
    religion_to_group: dict[str, str] = {}
    for rel_id, rel_data in all_religions.items():
        if isinstance(rel_data, dict) and "group" in rel_data:
            religion_to_group[rel_id] = rel_data["group"]

    # Localization
    loc_en: dict[str, str] = {}
    loc_ja: dict[str, str] = {}
    loc_dir = RAW_DIR / "localization"
    for path in loc_dir.glob("religion_l_english.yml"):
        loc_en.update(parse_loc_file(path))
    for path in loc_dir.glob("religion_l_japanese.yml"):
        loc_ja.update(parse_loc_file(path))

    result = []
    for rel_id, props in all_religions.items():
        entry: dict = {
            "id": rel_id,
            "name_en": loc_en.get(rel_id, rel_id),
            "name_ja": loc_ja.get(rel_id, rel_id),
            "desc_en": strip_markup(loc_en.get(f"{rel_id}_desc", "")),
            "desc_ja": strip_markup(loc_ja.get(f"{rel_id}_desc", "")),
        }

        if rel_id in religion_to_group:
            grp = religion_to_group[rel_id]
            entry["group_id"] = grp
            entry["group_en"] = loc_en.get(grp, grp)
            entry["group_ja"] = loc_ja.get(grp, grp)

        for field in ("color", "icon"):
            if field in props:
                entry[field] = props[field]

        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = props["modifier"]

        entry["icon"] = f"icons/religion/{rel_id}.png"
        entry["source_file"] = props.get("_source_file", "")
        result.append(entry)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    religions = build_religions()
    out_path = OUTPUT_DIR / "religions.json"
    out_path.write_text(
        json.dumps(religions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(religions)} religions to {out_path}")


if __name__ == "__main__":
    main()
