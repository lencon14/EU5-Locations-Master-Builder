"""Extract game terminology from official localization files.

Produces loc/{lang}/game_terms.json for each language.
These are NOT site UI strings — they are official in-game term translations
extracted directly from Paradox localization files.

Categories extracted:
  - pop_types: nobles, clergy, burghers, etc.
  - pop_groups: all, upper (from game_concepts)
  - goods_categories: raw_material, produced, food, special
  - goods_methods: farming, mining, gathering, hunting, forestry
  - goods_origins: old_world, new_world (from old_world_goods/new_world_goods)
  - field_labels: demand, base_production, etc. (from game_concepts)

Usage:
    cd pipeline && python3 extract_game_terms.py
"""

from __future__ import annotations

import json
from pathlib import Path

from languages import LANGUAGES
from loc_parser import parse_loc_file

RAW_DIR = Path("raw/localization")
OUTPUT_DIR = Path("output")


def _load_loc(filename: str, game_code: str) -> dict[str, str]:
    """Load a loc file, return empty dict if not found."""
    path = RAW_DIR / f"{filename}_l_{game_code}.yml"
    if path.exists():
        return parse_loc_file(path)
    return {}


def extract_terms(game_code: str) -> dict[str, str]:
    """Extract all game terms for a single language."""
    terms: dict[str, str] = {}

    goods_loc = _load_loc("goods", game_code)
    pops_loc = _load_loc("pops", game_code)
    concepts_loc = _load_loc("game_concepts", game_code)

    # --- Pop types (from pops_l) ---
    for pop_id in ["nobles", "clergy", "burghers", "laborers",
                    "soldiers", "peasants", "slaves", "tribesmen"]:
        if pop_id in pops_loc:
            terms[f"pop.{pop_id}"] = pops_loc[pop_id]

    # --- Pop groups ---
    # 'upper' from game_concepts
    if "game_concept_upper_class" in concepts_loc:
        terms["pop.upper"] = concepts_loc["game_concept_upper_class"]

    # 'all' has no official loc — use the raw key
    # (will be displayed as "all" unless manually overridden)

    # --- Goods categories (from goods_l) ---
    for cat_id in ["raw_material", "produced", "food", "special"]:
        if cat_id in goods_loc:
            terms[f"goods.cat.{cat_id}"] = goods_loc[cat_id]

    # --- Goods methods (from goods_l) ---
    for method_id in ["farming", "mining", "gathering", "hunting", "forestry"]:
        if method_id in goods_loc:
            terms[f"goods.method.{method_id}"] = goods_loc[method_id]

    # --- Goods origins (from goods_l: old_world_goods / new_world_goods) ---
    if "old_world_goods" in goods_loc:
        terms["goods.origin.old_world"] = goods_loc["old_world_goods"]
    if "new_world_goods" in goods_loc:
        terms["goods.origin.new_world"] = goods_loc["new_world_goods"]

    # --- Field labels from game_concepts ---
    concept_map = {
        "game_concept_demand": "field.demand",
        "game_concept_base_production": "field.base_production",
    }
    for concept_key, term_key in concept_map.items():
        if concept_key in concepts_loc:
            terms[term_key] = concepts_loc[concept_key]

    return terms


def main() -> None:
    print("=== Extracting game terms ===")

    for url_code, game_code, display_name in LANGUAGES:
        print(f"[{url_code}] {display_name}")
        terms = extract_terms(game_code)

        out_dir = OUTPUT_DIR / "loc" / url_code
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "game_terms.json"
        out_path.write_text(
            json.dumps(terms, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  → {out_path} ({len(terms)} terms)")

    print("Done.")


if __name__ == "__main__":
    main()
