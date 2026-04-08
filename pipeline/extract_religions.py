"""Extract EU5 religion data into core + per-language loc files.

Usage:
    python pipeline/extract_religions.py

Output:
    pipeline/output/core/religions.json
    pipeline/output/loc/{lang}/religions.json
    pipeline/output/religions.json               — legacy merged format
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


def load_all_loc(loc_name: str) -> dict[str, dict[str, str]]:
    """Load localization for all languages. Returns {url_code: {key: value}}."""
    result = {}
    loc_dir = RAW_DIR / "localization"
    for url_code, game_code, _ in LANGUAGES:
        path = loc_dir / f"{loc_name}_l_{game_code}.yml"
        if path.exists():
            result[url_code] = parse_loc_file(path)
        else:
            result[url_code] = {}
    return result


def build_religions() -> tuple[list[dict], dict[str, dict]]:
    """Parse religion files. Returns (core_list, loc_dict_per_lang)."""
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
        return [], {}

    all_religions: dict[str, dict] = {}
    for f in sorted(rel_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_religions[key] = val

    # Build group lookup
    religion_to_group: dict[str, str] = {}
    for rel_id, rel_data in all_religions.items():
        if isinstance(rel_data, dict) and "group" in rel_data:
            religion_to_group[rel_id] = rel_data["group"]

    # Load all language loc
    all_loc = load_all_loc("religion")

    # Build core (language-independent)
    core_list = []
    for rel_id, props in all_religions.items():
        entry: dict = {"id": rel_id}

        if rel_id in religion_to_group:
            entry["group_id"] = religion_to_group[rel_id]

        for field in ("color", "icon"):
            if field in props:
                entry[field] = props[field]

        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = props["modifier"]

        entry["icon"] = f"icons/religion/{rel_id}.png"
        entry["source_file"] = props.get("_source_file", "")
        core_list.append(entry)

    # Build per-language loc (includes religion names, descs, and group names)
    rel_ids = [r["id"] for r in core_list]
    group_ids = set(religion_to_group.values())
    loc_per_lang: dict[str, dict] = {}
    for url_code, loc_data in all_loc.items():
        lang_loc = {}
        for rel_id in rel_ids:
            name = loc_data.get(rel_id, "")
            desc = strip_markup(loc_data.get(f"{rel_id}_desc", ""))
            if name or desc:
                lang_loc[rel_id] = {"name": name, "desc": desc}
        # Group names
        for grp_id in group_ids:
            name = loc_data.get(grp_id, "")
            if name:
                lang_loc[grp_id] = {"name": name}
        loc_per_lang[url_code] = lang_loc

    return core_list, loc_per_lang


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    core_list, loc_per_lang = build_religions()

    # Core
    core_path = OUTPUT_DIR / "core" / "religions.json"
    write_json(core_path, core_list)
    print(f"Wrote {len(core_list)} religions to {core_path}")

    # Per-language loc
    for url_code, loc_data in loc_per_lang.items():
        loc_path = OUTPUT_DIR / "loc" / url_code / "religions.json"
        write_json(loc_path, loc_data)
    lang_count = sum(1 for v in loc_per_lang.values() if v)
    print(f"Wrote loc for {lang_count} languages")

    # Legacy merged format
    en_loc = loc_per_lang.get("en", {})
    ja_loc = loc_per_lang.get("ja", {})
    legacy = []
    for item in core_list:
        merged = dict(item)
        rid = item["id"]
        merged["name_en"] = en_loc.get(rid, {}).get("name", rid)
        merged["name_ja"] = ja_loc.get(rid, {}).get("name", rid)
        merged["desc_en"] = en_loc.get(rid, {}).get("desc", "")
        merged["desc_ja"] = ja_loc.get(rid, {}).get("desc", "")
        if "group_id" in item:
            grp = item["group_id"]
            merged["group_en"] = en_loc.get(grp, {}).get("name", grp)
            merged["group_ja"] = ja_loc.get(grp, {}).get("name", grp)
        legacy.append(merged)
    legacy_path = OUTPUT_DIR / "religions.json"
    write_json(legacy_path, legacy)
    print(f"Wrote legacy merged format to {legacy_path}")


if __name__ == "__main__":
    main()
