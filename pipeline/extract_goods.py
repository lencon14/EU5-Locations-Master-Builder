"""Extract EU5 goods data and merge with localization into site-ready JSON.

Usage:
    python pipeline/extract_goods.py

Output:
    pipeline/output/goods.json
"""

from __future__ import annotations

import json
from pathlib import Path

from paradox_parser import parse_file
from loc_parser import parse_loc_file, strip_markup

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"


def build_goods() -> list[dict]:
    """Parse all goods files and merge with localization."""
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

    # Parse localization
    loc_en = parse_loc_file(RAW_DIR / "localization" / "goods_l_english.yml")
    loc_ja = parse_loc_file(RAW_DIR / "localization" / "goods_l_japanese.yml")

    # Merge
    result = []
    for good_id, props in all_goods.items():
        entry = {
            "id": good_id,
            "name_en": loc_en.get(good_id, good_id),
            "name_ja": loc_ja.get(good_id, good_id),
            "desc_en": strip_markup(loc_en.get(f"{good_id}_desc", "")),
            "desc_ja": strip_markup(loc_ja.get(f"{good_id}_desc", "")),
            "category": props.get("category", ""),
            "method": props.get("method", ""),
            "default_market_price": props.get("default_market_price", 0),
            "transport_cost": props.get("transport_cost", 0),
            "base_production": props.get("base_production", 0),
        }

        # Optional fields
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
        result.append(entry)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    goods = build_goods()
    out_path = OUTPUT_DIR / "goods.json"
    out_path.write_text(
        json.dumps(goods, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(goods)} goods to {out_path}")


if __name__ == "__main__":
    main()
