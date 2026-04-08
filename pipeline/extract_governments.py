"""Extract EU5 government type data and merge with localization into site-ready JSON.

Usage:
    python pipeline/extract_governments.py

Output:
    pipeline/output/governments.json
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


def build_governments() -> list[dict]:
    """Parse government_types files and merge with localization."""
    gov_dir = RAW_DIR / "government_types"
    if not gov_dir.exists():
        print(f"[WARN] {gov_dir} not found, skipping")
        return []

    all_govs: dict[str, dict] = {}
    for f in sorted(gov_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_govs[key] = val

    # Localization
    loc_en: dict[str, str] = {}
    loc_ja: dict[str, str] = {}
    loc_dir = RAW_DIR / "localization"
    for pattern in (
        "government_l_english.yml",
        "government_names_l_english.yml",
        "government_reforms_l_english.yml",
    ):
        for path in loc_dir.glob(pattern):
            loc_en.update(parse_loc_file(path))
    for pattern in (
        "government_l_japanese.yml",
        "government_names_l_japanese.yml",
        "government_reforms_l_japanese.yml",
    ):
        for path in loc_dir.glob(pattern):
            loc_ja.update(parse_loc_file(path))

    result = []
    for gov_id, props in all_govs.items():
        entry: dict = {
            "id": gov_id,
            "name_en": loc_en.get(gov_id, gov_id),
            "name_ja": loc_ja.get(gov_id, gov_id),
            "desc_en": strip_markup(loc_en.get(f"{gov_id}_desc", "")),
            "desc_ja": strip_markup(loc_ja.get(f"{gov_id}_desc", "")),
        }

        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = props["modifier"]

        entry["source_file"] = props.get("_source_file", "")
        result.append(entry)

    return result


def build_laws() -> list[dict]:
    """Parse laws files and merge with localization."""
    law_dir = RAW_DIR / "laws"
    if not law_dir.exists():
        print(f"[WARN] {law_dir} not found, skipping")
        return []

    all_laws: dict[str, dict] = {}
    for f in sorted(law_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_laws[key] = val

    # Localization
    loc_en: dict[str, str] = {}
    loc_ja: dict[str, str] = {}
    loc_dir = RAW_DIR / "localization"
    for pattern in ("laws_l_english.yml", "laws_and_policies_l_english.yml"):
        for path in loc_dir.glob(pattern):
            loc_en.update(parse_loc_file(path))
    for pattern in ("laws_l_japanese.yml", "laws_and_policies_l_japanese.yml"):
        for path in loc_dir.glob(pattern):
            loc_ja.update(parse_loc_file(path))

    result = []
    for law_id, props in all_laws.items():
        entry: dict = {
            "id": law_id,
            "name_en": loc_en.get(law_id, law_id),
            "name_ja": loc_ja.get(law_id, law_id),
            "desc_en": strip_markup(loc_en.get(f"{law_id}_desc", "")),
            "desc_ja": strip_markup(loc_ja.get(f"{law_id}_desc", "")),
        }

        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = props["modifier"]

        entry["source_file"] = props.get("_source_file", "")
        result.append(entry)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    govs = build_governments()
    out = OUTPUT_DIR / "governments.json"
    out.write_text(json.dumps(govs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(govs)} government types to {out}")

    laws = build_laws()
    out = OUTPUT_DIR / "laws.json"
    out.write_text(json.dumps(laws, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(laws)} laws to {out}")


if __name__ == "__main__":
    main()
