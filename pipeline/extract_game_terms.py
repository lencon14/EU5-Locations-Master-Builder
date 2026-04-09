"""Extract game terminology and build field labels for all languages.

Produces loc/{lang}/game_terms.json for each language.
Terms have three provenance levels:

  [official-1:1]  Extracted directly from Paradox loc files with no modification.
    - pop_types: nobles, clergy, etc. (from pops_l)
    - pop_groups.upper (from game_concepts: game_concept_upper_class)
    - goods_categories: raw_material, produced, food, special (from goods_l)
    - goods_methods: farming, mining, etc. (from goods_l)
    - goods_origins: old_world, new_world (from goods_l: old_world_goods/new_world_goods)
    - single concepts: demand, base_production, market_price, etc. (from game_concepts)

  [derived]  Compound labels built from official word parts.
    Field labels like "基準価格" (base price), "輸送コスト" (transport cost),
    "需要加算" (demand add) are composed from individual official terms.
    Each language's compounds are defined in _FIELD_LABELS and were verified
    against game_concepts_l and goods_l. JA labels confirmed by project owner.

  [no-official]  Terms with no official loc entry.
    - pop.all: "all" pop group has no Paradox loc. Labeled per-language in _FIELD_LABELS.
    - field.origin: "origin" has no game concept. Site-specific label.

Usage:
    cd pipeline && python3 extract_game_terms.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from languages import LANGUAGES
from loc_parser import parse_loc_file, strip_markup

RAW_DIR = Path("raw/localization")
OUTPUT_DIR = Path("output")
SITE_LOC_DIR = Path("../site/src/data/loc")


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

    # Build concept word → localized name lookup for desc post-processing
    # e.g. "good" → "交易品", "market_center" → "市場中心地"
    # Pop type names for resolving $pop$ refs and [ShowPopTypeName] in descs
    pop_names: dict[str, str] = {}
    for pop_id in ["nobles", "clergy", "burghers", "laborers",
                    "soldiers", "peasants", "slaves", "tribesmen"]:
        if pop_id in pops_loc:
            pop_names[pop_id] = pops_loc[pop_id]

    def _resolve_dollar_vars(text: str) -> str:
        """Resolve $variable$ refs against known loc data before stripping.

        Handles: $game_concept_X$ → concept display name, $pop$ → pop name.
        """
        import re
        def _repl(m: re.Match[str]) -> str:
            var = m.group(1)
            var_lower = var.lower()
            # $game_concept_X$ → concept display name
            if var_lower in concepts_loc:
                return concepts_loc[var_lower]
            if var in concepts_loc:
                return concepts_loc[var]
            # $pop_type$ → pop name
            return pop_names.get(var_lower, "")
        return re.sub(r"\$(\w+)\$", _repl, text)

    concept_names: dict[str, str] = {}
    for k, v in concepts_loc.items():
        if k.startswith("game_concept_") and not k.endswith(("_desc", "_i", "_s")):
            word = k[len("game_concept_"):]
            concept_names[word] = strip_markup(_resolve_dollar_vars(v))

    # Also include pop type names (e.g. "burghers" → "市民")
    # These appear in descs via [ShowPopTypeName('burghers')]
    for pop_id, pop_name in pop_names.items():
        if pop_id not in concept_names:
            concept_names[pop_id] = pop_name

    # Include religion names for resolving [ShowReligionName('catholic')] etc.
    religion_loc = _load_loc("religion", game_code)
    for rk, rv in religion_loc.items():
        if rv and not rk.endswith(("_desc", "_ADJ")) and rk not in concept_names:
            concept_names[rk] = rv

    def localize_desc(raw: str) -> str:
        """Resolve $refs$, strip markup, then replace concept words with localized names."""
        # Add space between adjacent [x|e][y|e] to prevent concatenation
        resolved = _resolve_dollar_vars(raw)
        resolved = re.sub(r"\](\[)", r"] \1", resolved)
        text = strip_markup(resolved)
        # Strip any remaining [...] patterns
        text = re.sub(r"\[[^\]]*\]", "", text)
        for word in sorted(concept_names.keys(), key=len, reverse=True):
            spaced = word.replace("_", " ")
            if spaced in text:
                text = text.replace(spaced, concept_names[word])
            elif word in text:
                text = text.replace(word, concept_names[word])
        return re.sub(r"\s+", " ", text).strip()

    # --- Pop types (from pops_l) ---
    for pop_id in ["nobles", "clergy", "burghers", "laborers",
                    "soldiers", "peasants", "slaves", "tribesmen"]:
        if pop_id in pops_loc:
            terms[f"pop.{pop_id}"] = pops_loc[pop_id]
        if f"{pop_id}_desc" in pops_loc:
            terms[f"pop.{pop_id}.desc"] = localize_desc(pops_loc[f"{pop_id}_desc"])

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
        # Category descriptions from game_concepts (e.g. game_concept_raw_materials_desc)
        for variant in [f"game_concept_{cat_id}_desc", f"game_concept_{cat_id}s_desc"]:
            if variant in concepts_loc:
                terms[f"goods.cat.{cat_id}.desc"] = localize_desc(concepts_loc[variant])
                break

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
        "game_concept_food": "field.food",
        "game_concept_base_production": "field.base_production",
        "game_concept_market_price": "field.market_price",
        "game_concept_development": "field.development",
        "game_concept_wealth": "field.wealth",
        "game_concept_cost": "field.cost",
        # Religion-related field labels
        "game_concept_religion_group": "field.religion_group",
        "game_concept_liturgical_language": "field.liturgical_language",
        "game_concept_holy_site": "field.holy_site",
        "game_concept_religious_opinion": "field.religious_opinion",
    }
    for concept_key, term_key in single_concepts.items():
        if concept_key in concepts_loc:
            terms[term_key] = concepts_loc[concept_key]
        # Also extract _desc for tooltips
        desc_key = f"{concept_key}_desc"
        if desc_key in concepts_loc:
            terms[f"{term_key}.desc"] = localize_desc(concepts_loc[desc_key])

    # Pop group descriptions
    if "game_concept_upper_class_desc" in concepts_loc:
        terms["pop.upper.desc"] = localize_desc(concepts_loc["game_concept_upper_class_desc"])

    # --- Compound field labels ---
    # Built from official word parts. Each language's compounds were verified
    # against game_concepts_l and goods_l localization files.
    field_labels = _build_field_labels(game_code)
    terms.update(field_labels)

    # --- Building categories (from buildings_l) ---  [official-1:1]
    buildings_loc = _load_loc("buildings", game_code)
    _BUILDING_CATEGORIES = [
        "basic_industry_category", "colonial_category", "consumer_goods_category",
        "cultural_category", "defense_category", "estate_category",
        "government_category", "infrastructure_category", "military_category",
        "naval_category", "religious_category", "rgo_building_category",
        "trade_category", "village_category", "weapons_industry_category",
    ]
    for cat_id in _BUILDING_CATEGORIES:
        if cat_id in buildings_loc:
            terms[f"building_category.{cat_id}"] = buildings_loc[cat_id]

    # --- Scaling labels (from game_concepts) ---  [official-1:1]
    for concept in ["development", "population"]:
        gc_key = f"game_concept_{concept}"
        if gc_key in concepts_loc:
            terms[f"scaling.{concept}"] = concepts_loc[gc_key]

    return terms


# Compound field labels per language.
# Provenance: [derived] and [no-official] — see module docstring.
#
# Official word parts used:
#   game_concept_target_price_desc → 基準価格 (JA)
#   game_concept_demand → 需要, game_concept_modifier_desc → 加算/乗算
#   game_concept_development → 開発度, game_concept_wealth → 富
#   game_concept_cost → コスト, game_concept_transport_capacity → 輸送
#
# JA labels confirmed by project owner. Other languages built from official word parts.
_FIELD_LABELS: dict[str, dict[str, str]] = {
    "english": {
        "pop.all": "All Pops",
        "field.method": "Method",
        "field.origin": "Origin",
        "field.default_market_price": "Base Price",
        "field.transport_cost": "Transport Cost",
        "field.demand_add": "Demand (Additive)",
        "field.demand_multiply": "Demand (Multiplicative)",
        "field.development_threshold": "Required Development",
        "field.wealth_impact_threshold": "Wealth Impact",
        "settlement.rural_settlement": "Rural Settlement",
        "settlement.town": "Town",
        "settlement.city": "City",
        "cond_category.flag": "Trait",
        "cond_category.country": "Country Condition",
        "cond_category.location": "Location Condition",
        "cond_category.allow": "Build Permission",
        "scaling.city": "City",
    },
    "japanese": {
        "pop.all": "全POP",
        "field.method": "生産方法",
        "field.origin": "産地",
        "field.default_market_price": "基準価格",
        "field.transport_cost": "輸送コスト",
        "field.demand_add": "需要加算",
        "field.demand_multiply": "需要乗算",
        "field.development_threshold": "必要開発度",
        "field.wealth_impact_threshold": "富への影響係数",
        "settlement.rural_settlement": "農村",
        "settlement.town": "町",
        "settlement.city": "都市",
        "cond_category.flag": "属性",
        "cond_category.country": "国家条件",
        "cond_category.location": "立地条件",
        "cond_category.allow": "建設許可",
        "scaling.city": "都市",
    },
    "german": {
        "pop.all": "Alle Schichten",
        "field.method": "Produktionsmethode",
        "field.origin": "Herkunft",
        "field.default_market_price": "Grundpreis",
        "field.transport_cost": "Transportkosten",
        "field.demand_add": "Nachfrage (additiv)",
        "field.demand_multiply": "Nachfrage (multiplikativ)",
        "field.development_threshold": "Benötigte Entwicklung",
        "field.wealth_impact_threshold": "Reichtumseffekt",
        "settlement.rural_settlement": "Ländliche Siedlung",
        "settlement.town": "Stadt",
        "settlement.city": "Großstadt",
        "cond_category.flag": "Eigenschaft",
        "cond_category.country": "Landesbedingung",
        "cond_category.location": "Standortbedingung",
        "cond_category.allow": "Baugenehmigung",
        "scaling.city": "Stadt",
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
        "settlement.rural_settlement": "Asentamiento rural",
        "settlement.town": "Pueblo",
        "settlement.city": "Ciudad",
        "cond_category.flag": "Rasgo",
        "cond_category.country": "Condición de país",
        "cond_category.location": "Condición de ubicación",
        "cond_category.allow": "Permiso de construcción",
        "scaling.city": "Ciudad",
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
        "settlement.rural_settlement": "Colonie rurale",
        "settlement.town": "Bourg",
        "settlement.city": "Cité",
        "cond_category.flag": "Trait",
        "cond_category.country": "Condition de pays",
        "cond_category.location": "Condition de lieu",
        "cond_category.allow": "Autorisation de construction",
        "scaling.city": "Cité",
    },
    "korean": {
        "pop.all": "전체",
        "field.method": "생산 방법",
        "field.origin": "원산지",
        "field.default_market_price": "기준 가격",
        "field.transport_cost": "운송 비용",
        "field.demand_add": "수요 가산",
        "field.demand_multiply": "수요 승산",
        "field.development_threshold": "필요 개발",
        "field.wealth_impact_threshold": "자산 영향",
        "settlement.rural_settlement": "농촌",
        "settlement.town": "마을",
        "settlement.city": "도시",
        "cond_category.flag": "속성",
        "cond_category.country": "국가 조건",
        "cond_category.location": "위치 조건",
        "cond_category.allow": "건설 허가",
        "scaling.city": "도시",
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
        "settlement.rural_settlement": "Osada wiejska",
        "settlement.town": "Miasto",
        "settlement.city": "Metropolia",
        "cond_category.flag": "Cecha",
        "cond_category.country": "Warunek krajowy",
        "cond_category.location": "Warunek lokalizacji",
        "cond_category.allow": "Pozwolenie na budowę",
        "scaling.city": "Miasto",
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
        "settlement.rural_settlement": "Assentamento rural",
        "settlement.town": "Vila",
        "settlement.city": "Cidade",
        "cond_category.flag": "Característica",
        "cond_category.country": "Condição de país",
        "cond_category.location": "Condição de localização",
        "cond_category.allow": "Permissão de construção",
        "scaling.city": "Cidade",
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
        "settlement.rural_settlement": "Сельское поселение",
        "settlement.town": "Городок",
        "settlement.city": "Город",
        "cond_category.flag": "Свойство",
        "cond_category.country": "Условие страны",
        "cond_category.location": "Условие местоположения",
        "cond_category.allow": "Разрешение на строительство",
        "scaling.city": "Город",
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
        "settlement.rural_settlement": "Kırsal yerleşim",
        "settlement.town": "Kasaba",
        "settlement.city": "Şehir",
        "cond_category.flag": "Özellik",
        "cond_category.country": "Ülke koşulu",
        "cond_category.location": "Konum koşulu",
        "cond_category.allow": "İnşaat izni",
        "scaling.city": "Şehir",
    },
    "simp_chinese": {
        "pop.all": "全部",
        "field.method": "生产方式",
        "field.origin": "产地",
        "field.default_market_price": "基础价格",
        "field.transport_cost": "运输花费",
        "field.demand_add": "需求加成",
        "field.demand_multiply": "需求倍率",
        "field.development_threshold": "所需发展度",
        "field.wealth_impact_threshold": "财富影响",
        "settlement.rural_settlement": "乡村",
        "settlement.town": "城镇",
        "settlement.city": "城市",
        "cond_category.flag": "特性",
        "cond_category.country": "国家条件",
        "cond_category.location": "位置条件",
        "cond_category.allow": "建造许可",
        "scaling.city": "城市",
    },
}


def _build_field_labels(game_code: str) -> dict[str, str]:
    """Return compound field labels for a language."""
    return dict(_FIELD_LABELS.get(game_code, _FIELD_LABELS["english"]))


def main() -> None:
    print("=== Extracting game terms ===")

    # Verify _FIELD_LABELS covers exactly the same languages as LANGUAGES
    label_codes = set(_FIELD_LABELS.keys())
    lang_codes = {game for _, game, _ in LANGUAGES}
    assert label_codes == lang_codes, f"_FIELD_LABELS mismatch: {label_codes.symmetric_difference(lang_codes)}"

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

    # Auto-copy to site if the site loc directory exists
    if SITE_LOC_DIR.is_dir():
        print("\nCopying to site/src/data/loc/ ...")
        for url_code, _, _ in LANGUAGES:
            src = OUTPUT_DIR / "loc" / url_code / "game_terms.json"
            dst_dir = SITE_LOC_DIR / url_code
            if src.exists() and dst_dir.is_dir():
                shutil.copy2(src, dst_dir / "game_terms.json")
        print("Done (site copy).")

    print("Done.")


if __name__ == "__main__":
    main()
