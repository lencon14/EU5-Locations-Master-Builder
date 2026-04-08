"""Extract EU5 goods data into core (language-independent) + per-language loc files.

Usage:
    python pipeline/extract_goods.py

Output:
    pipeline/output/core/goods.json          — game data (no localized strings)
    pipeline/output/loc/{lang}/goods.json    — names and descriptions per language
    pipeline/output/goods.json               — legacy merged format (temporary)
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


def build_goods() -> tuple[list[dict], dict[str, dict]]:
    """Parse goods files. Returns (core_list, loc_dict_per_lang)."""
    # Parse game data
    goods_dir = RAW_DIR / "goods"
    all_goods: dict[str, dict] = {}
    source_files = sorted(goods_dir.glob("*.txt"))
    for f in source_files:
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_goods[key] = val

    # Load all language loc
    all_loc = load_all_loc("goods")

    # Build core (language-independent)
    core_list = []
    for good_id, props in all_goods.items():
        entry = {
            "id": good_id,
            "category": props.get("category", ""),
            "method": props.get("method", ""),
            "default_market_price": props.get("default_market_price", 0),
            "transport_cost": props.get("transport_cost", 0),
            "base_production": props.get("base_production", 0),
        }

        if props.get("origin_in_old_world"):
            entry["origin"] = "old_world"
        elif props.get("origin_in_new_world"):
            entry["origin"] = "new_world"

        tags = props.get("custom_tags", [])
        if isinstance(tags, list):
            entry["tags"] = tags

        if "demand_add" in props and isinstance(props["demand_add"], dict):
            entry["demand_add"] = props["demand_add"]
        if "demand_multiply" in props and isinstance(props["demand_multiply"], dict):
            entry["demand_multiply"] = props["demand_multiply"]
        if "wealth_impact_threshold" in props and isinstance(
            props["wealth_impact_threshold"], dict
        ):
            entry["wealth_impact_threshold"] = props["wealth_impact_threshold"]
        if "development_threshold" in props:
            entry["development_threshold"] = props["development_threshold"]

        entry["icon"] = f"icons/trade_goods/icon_goods_{good_id}.png"
        entry["source_file"] = props.get("_source_file", "")
        core_list.append(entry)

    # Build per-language loc
    good_ids = [g["id"] for g in core_list]
    loc_per_lang: dict[str, dict] = {}
    for url_code, loc_data in all_loc.items():
        lang_loc = {}
        for good_id in good_ids:
            name = loc_data.get(good_id, "")
            desc = strip_markup(loc_data.get(f"{good_id}_desc", ""))
            if name or desc:
                lang_loc[good_id] = {"name": name, "desc": desc}
        loc_per_lang[url_code] = lang_loc

    return core_list, loc_per_lang


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    core_list, loc_per_lang = build_goods()

    # Core data
    core_path = OUTPUT_DIR / "core" / "goods.json"
    write_json(core_path, core_list)
    print(f"Wrote {len(core_list)} goods to {core_path}")

    # Per-language loc
    for url_code, loc_data in loc_per_lang.items():
        loc_path = OUTPUT_DIR / "loc" / url_code / "goods.json"
        write_json(loc_path, loc_data)
    lang_count = sum(1 for v in loc_per_lang.values() if v)
    print(f"Wrote loc for {lang_count} languages")

    # Legacy merged format (for backward compatibility during migration)
    en_loc = loc_per_lang.get("en", {})
    ja_loc = loc_per_lang.get("ja", {})
    legacy = []
    for item in core_list:
        merged = dict(item)
        gid = item["id"]
        merged["name_en"] = en_loc.get(gid, {}).get("name", gid)
        merged["name_ja"] = ja_loc.get(gid, {}).get("name", gid)
        merged["desc_en"] = en_loc.get(gid, {}).get("desc", "")
        merged["desc_ja"] = ja_loc.get(gid, {}).get("desc", "")
        legacy.append(merged)
    legacy_path = OUTPUT_DIR / "goods.json"
    write_json(legacy_path, legacy)
    print(f"Wrote legacy merged format to {legacy_path}")


if __name__ == "__main__":
    main()
