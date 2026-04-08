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

    # 'all' has no official loc — defined per language in _FIELD_LABELS
    # as "pop.all" (meaning "all pop types")

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

    # --- Single concept terms (direct from game_concepts) ---
    single_concepts = {
        "game_concept_demand": "field.demand",
        "game_concept_base_production": "field.base_production",
        "game_concept_market_price": "field.market_price",
        "game_concept_development": "field.development",
        "game_concept_wealth": "field.wealth",
        "game_concept_cost": "field.cost",
    }
    for concept_key, term_key in single_concepts.items():
        if concept_key in concepts_loc:
            terms[term_key] = concepts_loc[concept_key]

    # --- Compound field labels ---
    # Built from official word parts. Each language's compounds were verified
    # against game_concepts_l and goods_l localization files.
    field_labels = _build_field_labels(game_code, concepts_loc)
    terms.update(field_labels)

    return terms


# Compound field labels per language.
# Sources: game_concept_target_price_desc (基準価格), game_concept_demand (需要),
# game_concept_modifier_desc (加算/乗算), game_concept_development (開発度),
# game_concept_wealth (富), game_concept_cost (コスト),
# game_concept_transport_capacity (輸送)
#
# JA labels confirmed by user. Other languages built from official word parts.
_FIELD_LABELS: dict[str, dict[str, str]] = {
    "english": {
        "pop.all": "All Pops",
        "field.method": "Method",                             # no direct game concept; RGO extraction method
        "field.origin": "Origin",                             # no direct game concept; old_world/new_world tag
        "field.default_market_price": "Base Price",           # base + price (from base_production pattern)
        "field.transport_cost": "Transport Cost",             # transport + cost
        "field.demand_add": "Demand (Additive)",              # demand + additive
        "field.demand_multiply": "Demand (Multiplicative)",   # demand + multiplicative
        "field.development_threshold": "Required Development",# required + development
        "field.wealth_impact_threshold": "Wealth Impact",     # wealth + impact
    },
    "japanese": {
        "pop.all": "全POP",
        "field.method": "生産方法",                     # game_concept_production_method
        "field.origin": "産地",                         # no direct game concept
        "field.default_market_price": "基準価格",       # from target_price_desc context
        "field.transport_cost": "輸送コスト",           # 輸送 + コスト
        "field.demand_add": "需要加算",                 # 需要 + 加算 (from modifier_desc)
        "field.demand_multiply": "需要乗算",            # 需要 + 乗算 (from modifier_desc)
        "field.development_threshold": "必要開発度",     # user confirmed
        "field.wealth_impact_threshold": "富への影響係数", # user confirmed
    },
    "german": {
        "pop.all": "Alle Schichten",
        "field.method": "Produktionsmethode",
        "field.origin": "Herkunft",
        "field.default_market_price": "Grundpreis",           # Grund + preis (from Grundproduktion pattern)
        "field.transport_cost": "Transportkosten",            # Transport + Kosten
        "field.demand_add": "Nachfrage (additiv)",            # Nachfrage + additiv
        "field.demand_multiply": "Nachfrage (multiplikativ)", # Nachfrage + multiplikativ
        "field.development_threshold": "Benötigte Entwicklung",
        "field.wealth_impact_threshold": "Reichtumseffekt",   # Reichtum + Effekt
    },
    "spanish": {
        "pop.all": "Todos",
        "field.method": "Método",
        "field.origin": "Origen",
        "field.default_market_price": "Precio base",
        "field.transport_cost": "Coste de transporte",
        "field.demand_add": "Demanda (aditiva)",
        "field.demand_multiply": "Demanda (multiplicativa)",
        "field.development_threshold": "Desarrollo requerido",
        "field.wealth_impact_threshold": "Impacto de riqueza",
    },
    "french": {
        "pop.all": "Tous",
        "field.method": "Méthode",
        "field.origin": "Origine",
        "field.default_market_price": "Prix de base",
        "field.transport_cost": "Coût de transport",
        "field.demand_add": "Demande (additive)",
        "field.demand_multiply": "Demande (multiplicative)",
        "field.development_threshold": "Développement requis",
        "field.wealth_impact_threshold": "Impact de richesse",
    },
    "korean": {
        "pop.all": "전체",
        "field.method": "생산 방법",
        "field.origin": "원산지",
        "field.default_market_price": "기준 가격",        # 기준 + 가격 (from 시장 가격)
        "field.transport_cost": "운송 비용",              # 운송 + 비용
        "field.demand_add": "수요 가산",                  # 수요 + 가산
        "field.demand_multiply": "수요 승산",             # 수요 + 승산
        "field.development_threshold": "필요 개발",       # 필요 + 개발
        "field.wealth_impact_threshold": "자산 영향",     # 자산 + 영향
    },
    "polish": {
        "pop.all": "Wszyscy",
        "field.method": "Metoda",
        "field.origin": "Pochodzenie",
        "field.default_market_price": "Cena bazowa",
        "field.transport_cost": "Koszt transportu",
        "field.demand_add": "Popyt (addytywny)",
        "field.demand_multiply": "Popyt (mnożnikowy)",
        "field.development_threshold": "Wymagany rozwój",
        "field.wealth_impact_threshold": "Wpływ bogactwa",
    },
    "braz_por": {
        "pop.all": "Todos",
        "field.method": "Método",
        "field.origin": "Origem",
        "field.default_market_price": "Preço base",
        "field.transport_cost": "Custo de transporte",
        "field.demand_add": "Demanda (aditiva)",
        "field.demand_multiply": "Demanda (multiplicativa)",
        "field.development_threshold": "Desenvolvimento necessário",
        "field.wealth_impact_threshold": "Impacto de riqueza",
    },
    "russian": {
        "pop.all": "Все",
        "field.method": "Способ добычи",
        "field.origin": "Происхождение",
        "field.default_market_price": "Базовая цена",
        "field.transport_cost": "Стоимость перевозки",
        "field.demand_add": "Спрос (аддитивный)",
        "field.demand_multiply": "Спрос (мультипликативный)",
        "field.development_threshold": "Требуемое развитие",
        "field.wealth_impact_threshold": "Влияние богатства",
    },
    "turkish": {
        "pop.all": "Tümü",
        "field.method": "Yöntem",
        "field.origin": "Köken",
        "field.default_market_price": "Taban fiyat",
        "field.transport_cost": "Nakliye maliyeti",
        "field.demand_add": "Talep (eklemeli)",
        "field.demand_multiply": "Talep (çarpımsal)",
        "field.development_threshold": "Gerekli gelişim",
        "field.wealth_impact_threshold": "Zenginlik etkisi",
    },
    "simp_chinese": {
        "pop.all": "全部",
        "field.method": "生产方式",
        "field.origin": "产地",
        "field.default_market_price": "基础价格",         # 基础 + 价格
        "field.transport_cost": "运输花费",               # 运输 + 花费
        "field.demand_add": "需求加成",                   # 需求 + 加成
        "field.demand_multiply": "需求倍率",              # 需求 + 倍率
        "field.development_threshold": "所需发展度",       # 所需 + 发展度
        "field.wealth_impact_threshold": "财富影响",       # 财富 + 影响
    },
}


def _build_field_labels(game_code: str, concepts_loc: dict[str, str]) -> dict[str, str]:
    """Return compound field labels for a language."""
    return dict(_FIELD_LABELS.get(game_code, _FIELD_LABELS["english"]))


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
