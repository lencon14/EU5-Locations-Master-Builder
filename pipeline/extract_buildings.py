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

# Max levels: base values from game/in_game/common/script_values/building_caps.txt
# "dynamic" means base + scaling (development/population/city bonus)
MAX_LEVEL_BASE: dict[str, int] = {
    "guild_max_level": 1,
    "workshop_max_level": 1,
    "manufactory_max_level": 5,
    "market_max_level": 1,
    "mills_max_level": 5,
    "market_warehouse_max_level": 1,
    "manpower_max_level": 1,
    "rural_building_cap": 1,
    "irrigant_cap": 2,
    "bund_cap": 2,
    "estate_building_single_level": 1,
    "estate_building_stackable_level": 1,
    "slave_market_max_level": 1,
    "trade_company_building_max_level": 2,
    "trade_company_headquarters_max_level": 1,
    "plantation_cap": 1,
    "hanseatic_kontor_max_level": 3,
    "hanseatic_shipwright_guild_max_level": 1,
    "polders_max_level": 0,
    "janissary_barracks_size": 1,
    "chancery_max_level": 1,
    "kremlin_max_level": 1,
    "venetian_palaces_max_level": 3,
    "bock_max_level": 1,
    "mercury_patio_max_level": 1,
    "construction_center_max_level": 1,
    "theodosian_walls_max_level": 1,
    "italian_merchant_quarters_max_level": 1,
    "fulani_cattle_rearing_pen_max_level": 3,
    "kurmina_headquarter_max_level": 1,
    "eghabho_nore_mansion_max_level": 1,
    "reformation_preachers_max_level": 1,
    "kurultai_max_level": 1,
    "huge_religious_building_employment": 1,
}

# Which max_level keys scale dynamically
MAX_LEVEL_DYNAMIC: dict[str, list[str]] = {
    "guild_max_level": ["開発度", "人口", "都市+5"],
    "workshop_max_level": ["開発度", "人口", "都市+10"],
    "manufactory_max_level": ["開発度", "人口", "都市+20"],
    "market_max_level": ["開発度", "人口", "都市+5"],
    "mills_max_level": ["開発度", "人口", "都市+25"],
    "market_warehouse_max_level": ["開発度", "都市+2"],
    "manpower_max_level": ["人口"],
    "rural_building_cap": ["開発度"],
    "irrigant_cap": ["開発度"],
    "bund_cap": ["開発度"],
    "estate_building_stackable_level": ["開発度"],
    "slave_market_max_level": ["開発度"],
    "polders_max_level": ["開発度"],
    "hanseatic_kontor_max_level": ["都市+2"],
    "venetian_palaces_max_level": ["開発度", "都市+2"],
    "fulani_cattle_rearing_pen_max_level": ["開発度"],
    "kurmina_headquarter_max_level": ["開発度", "人口"],
    "eghabho_nore_mansion_max_level": ["都市+1"],
    "kurultai_max_level": ["人口"],
}

# Build time in days (from game/main_menu/common/script_values/default_values.txt)
BUILD_TIME_DAYS: dict[str, int] = {
    "guild_build_time": 365,
    "workshop_build_time": 365,
    "manufactory_build_time": 450,
    "mills_build_time": 730,
    "merchant_build_time": 180,
    "plantation_build_time": 180,
    "estate_build_time": 120,
    "bank_office_build_time": 365,
    "trade_post_build_time": 365,
    "colonise_build_time": 365,
    "trade_company_build_time": 180,
    "rural_build_time": 365,
    "village_build_time": 365,
    "generic_town_build_time": 365,
    "soldier_building": 365,
    "medium_soldier_building": 365,
    "big_soldier_building": 730,
    "government_build_time": 365,
    "infrastructure_build_time": 365,
    "small_capital_build_time": 365,
    "large_capital_build_time": 730,
    "market_build_time": 365,
    "small_fort_building": 365,
    "medium_fort_building": 730,
    "large_fort_building": 1080,
    "huge_unique_build_time": 1825,
    "pirate_build_time": 180,
    "generic_event_building": 90,
    "instant_build_time": 1,
    "super_fast_temporary_house": 1,
    "trade_company_headquarters_time": 365,
    "cultural_building_time": 365,
    "large_cultural_building_time": 730,
    "religious_building_time": 365,
    "large_religious_building_time": 730,
    "large_port_building_time": 730,
    "medium_port_building_time": 365,
    "huge_port_building_time": 1825,
}


def _strip_prefix(s: str) -> str:
    """Remove a known prefix like 'culture:', 'religion:', 'government_type:', etc."""
    if isinstance(s, str) and ":" in s:
        return s.split(":", 1)[1]
    return str(s)


def _build_loc_lookup(loc_dir: Path) -> dict[str, str]:
    """Build a comprehensive Japanese localization lookup from all relevant files."""
    lookup: dict[str, str] = {}
    loc_files = [
        "government_names_l_japanese.yml",
        "government_reforms_l_japanese.yml",
        "government_l_japanese.yml",
        "cultures_l_japanese.yml",
        "culture_groups_l_japanese.yml",
        "religion_l_japanese.yml",
        "goods_l_japanese.yml",
        "countries_l_japanese.yml",
        "country_names_l_japanese.yml",
        "laws_l_japanese.yml",
        "laws_and_policies_l_japanese.yml",
        "buildings_l_japanese.yml",
        "advances_l_japanese.yml",
        "province_names_l_japanese.yml",
        "pops_l_japanese.yml",
        "estate_l_japanese.yml",
        "modifier_types_l_japanese.yml",
        "static_modifiers_l_japanese.yml",
        "location_names_l_japanese.yml",
        "units_l_japanese.yml",
        "area_l_japanese.yml",
        "holy_sites_l_japanese.yml",
        "traits_l_japanese.yml",
        "game_concepts_l_japanese.yml",
        "flavor_hab_l_japanese.yml",
    ]
    for fname in loc_files:
        path = loc_dir / fname
        if path.exists():
            data = parse_loc_file(path)
            for k, v in data.items():
                # Skip tooltip entries
                if "_tt" in k:
                    continue
                # Desc keys and modifier desc: store for var resolution & tooltips
                if "_desc" in k or k.startswith("MODIFIER_TYPE_DESC_"):
                    if len(v) < 500:
                        lookup[k] = v
                    continue
                # Non-desc: skip entries with [ markup
                if "[" in v:
                    continue
                if "$" not in v and len(v) > 30:
                    continue
                if "$" in v and len(v) > 60:
                    continue
                lookup[k] = v
                # Province names: also store without _province suffix
                if k.endswith("_province"):
                    lookup[k[: -len("_province")]] = v
                # Modifier type names: store without MODIFIER_TYPE_NAME_ prefix
                if k.startswith("MODIFIER_TYPE_NAME_"):
                    mod_key = k[len("MODIFIER_TYPE_NAME_"):]
                    if mod_key not in lookup:
                        lookup[mod_key] = v

    # Resolve $var$ within lookup values themselves
    import re as _re_loc
    for _ in range(3):
        changed = False
        for k, v in list(lookup.items()):
            if "$" in v:
                new_v = _re_loc.sub(
                    r"\$([^$]+)\$",
                    lambda m: lookup.get(m.group(1), m.group(0)),
                    v,
                )
                if new_v != v:
                    lookup[k] = new_v
                    changed = True
        if not changed:
            break

    return lookup


def _parse_conditions(
    loc_potential: dict | None,
    cty_potential: dict | None,
    allow: dict | None,
    is_foreign: bool,
    is_special: bool,
    forbidden_for_estates: bool,
    loc_ja: dict[str, str],
) -> list[dict]:
    """Convert raw condition dicts into human-readable condition entries.

    Each entry: {"category": "location"|"country"|"allow"|"flag", "text": "..."}
    """
    conds: list[dict] = []

    # --- Flags ---
    if is_special:
        conds.append({"category": "flag", "text": "特殊建物"})
    if is_foreign:
        conds.append({"category": "flag", "text": "外国建物"})
    if forbidden_for_estates:
        conds.append({"category": "flag", "text": "荘園建設不可"})

    # --- Hardcoded translations for terms not in loc files ---
    _manual: dict[str, str] = {
        # Vegetation
        "woods": "森林", "forest": "深林", "jungle": "密林",
        "sparse": "疎林", "farmland": "農地", "grasslands": "草原",
        # Topography
        "mountains": "山岳", "plateau": "高原", "hills": "丘陵",
        # Climate
        "mediterranean": "地中海性", "continental": "大陸性",
        "oceanic": "海洋性", "tropical": "熱帯", "arid": "乾燥",
        # Modifier keys
        "has a parliamentary system": "議会制",
        "can have monasteries": "修道院建設可",
        "enable pest house": "隔離所建設可",
        "enable black market buildings": "闘市建設可",
        "target of military sponsorships": "軍事支援の対象",
        "allows hanseatic federation buildings": "ハンザ同盟建物許可",
        "allow thema headquarters": "テマ司令部建設可",
        "can build kurmina headquarter": "クルミナ司令部建設可",
        "can build mamluk barracks": "マムルーク兵舎建設可",
        "has panaqas": "パナカ制",
        # Locations (not in province_names with _province suffix)
        "moscow": "モスクワ", "constantinople": "コンスタンティノープル",
        "cairo": "カイロ", "neva": "ネヴァ", "munich": "ミュンヘン",
        "amsterdam": "アムステルダム", "rome": "ローマ",
        "benin": "ベニン", "dubrovnik": "ドゥブロヴニク",
        "harlingen": "ハーリンゲン", "helsingor": "ヘルシンゲル",
        "hoorn": "ホールン", "kobenhavn": "コペンハーゲン",
        "luxembourg": "ルクセンブルク", "middelburg": "ミデルブルフ",
        "rotterdam": "ロッテルダム", "tlemcen": "トレムセン",
        "great zimbabwe": "グレート・ジンバブエ",
        "great_zimbabwe": "グレート・ジンバブエ",
        # Cultures
        "maori culture": "マオリ", "maori_culture": "マオリ",
        "khmer culture": "クメール", "khmer_culture": "クメール",
        "andalusi": "アンダルシア", "basque": "バスク",
        # Reforms with $ references
        "japanese imperial family": "日本の皇室",
        "japanese_imperial_family": "日本の皇室",
        # Advances with $ references
        "caravanserai advance": "隊商宿", "caravanserai_advance": "隊商宿",
        "german mountain toll castle advance": "ドイツ山岳関税城",
        "german_mountain_toll_castle_advance": "ドイツ山岳関税城",
        "german river toll castle advance": "ドイツ河川関税城",
        "german_river_toll_castle_advance": "ドイツ河川関税城",
        "west african caravan stop advance": "西アフリカ隊商宿",
        "west_african_caravan_stop_advance": "西アフリカ隊商宿",
        "netherlandish ship building": "ネーデルラント造船術",
        "netherlandish_ship_building": "ネーデルラント造船術",
    }

    # --- Helper to resolve a display name ---
    def _name(raw_id: str) -> str:
        clean = _strip_prefix(raw_id)
        spaced = clean.replace("_", " ")
        for candidate in (clean, spaced, clean.replace("_reform", "")):
            if candidate in _manual:
                return _manual[candidate]
            if candidate in loc_ja:
                return loc_ja[candidate]
        return spaced

    # --- Walk a condition dict and yield readable strings ---
    def _walk(obj: dict, label_cat: str) -> list[str]:
        lines: list[str] = []
        if not isinstance(obj, dict):
            return lines

        # always = False  →  event-only
        if obj.get("always") is False:
            lines.append("イベント限定（通常建設不可）")
            return lines

        for k, v in obj.items():
            if k in ("custom_tooltip", "text"):
                # Recurse into tooltip wrappers
                if isinstance(v, dict):
                    lines.extend(_walk(v, label_cat))
                continue

            # Location checks
            if k == "is_capital" and v is True:
                lines.append("首都のみ")
            elif k == "is_capital" and v is False:
                lines.append("首都以外")
            elif k == "is_port" and v is True:
                lines.append("港湾都市のみ")
            elif k == "is_coastal" and v is True:
                lines.append("沿岸のみ")
            elif k == "has_river" and v is True:
                lines.append("河川沿いのみ")
            elif k == "is_adjacent_to_lake" and v is True:
                lines.append("湖沿いのみ")
            elif k == "is_market_center" and v is True:
                lines.append("市場中心地のみ")
            elif k == "has_road_to_capital" and v is True:
                lines.append("首都への道路接続が必要")
            elif k == "is_overseas_for_owner" and v is True:
                lines.append("海外領のみ")

            # Country checks
            elif k == "has_reform":
                lines.append(f"政体改革: {_name(v)}")
            elif k == "government_type":
                if isinstance(v, list):
                    names = [_name(x) for x in v]
                    lines.append(f"政体: {' / '.join(names)}")
                else:
                    lines.append(f"政体: {_name(v)}")
            elif k == "has_advance":
                lines.append(f"進歩: {_name(v)}")
            elif k == "culture" and isinstance(v, str):
                lines.append(f"文化: {_name(v)}")
            elif k == "religion" and isinstance(v, str):
                lines.append(f"宗教: {_name(v)}")
            elif k == "tag" and isinstance(v, str):
                lines.append(f"国家: {_name(v)}")
            elif k == "has_or_had_tag" and isinstance(v, list):
                lines.append(f"国家: {' / '.join(_name(t) for t in v)}")
            elif k == "has_slavery" and v is True:
                lines.append("奴隷制が必要")
            elif k == "has_policy":
                lines.append(f"政策: {_name(v)}")

            # Culture/religion sub-checks
            elif k == "dominant_culture" and isinstance(v, str):
                lines.append(f"支配文化: {_name(v)}")
            elif k == "dominant_culture" and isinstance(v, dict):
                if "has_culture_group" in v:
                    lines.append(f"文化グループ: {_name(v['has_culture_group'])}")
                else:
                    lines.extend(_walk(v, label_cat))
            elif k == "has_culture_group":
                lines.append(f"文化グループ: {_name(v)}")
            elif k == "group" and isinstance(v, str):
                lines.append(f"宗教グループ: {_name(v)}")

            # Location reference
            elif k == "this" and isinstance(v, str) and v.startswith("location:"):
                loc_name = v.split(":", 1)[1]
                lines.append(f"特定地域: {_name(loc_name)}")
            elif k == "owns" and isinstance(v, str) and v.startswith("location:"):
                loc_name = v.split(":", 1)[1]
                lines.append(f"所有地域: {_name(loc_name)}")

            # Terrain/vegetation
            elif k == "vegetation" and isinstance(v, list):
                lines.append(f"植生: {' / '.join(_name(x) for x in v)}")
            elif k == "topography" and isinstance(v, list):
                lines.append(f"地形: {' / '.join(_name(x) for x in v)}")
            elif k == "climate" and isinstance(v, list):
                lines.append(f"気候: {' / '.join(_name(x) for x in v)}")

            # Market conditions
            elif k == "market" and isinstance(v, dict):
                if "is_produced_in_market" in v:
                    lines.append(f"市場で生産中: {_name(v['is_produced_in_market'])}")
                elif "in_trade_range_of" in v:
                    lines.append("交易圏内であること")

            # Modifier checks
            elif k.startswith("modifier:"):
                mod_name = k.split(":", 1)[1]
                if v is True:
                    lines.append(f"要補正: {_name(mod_name)}")

            # Compound: OR / AND / NOT
            elif k in ("OR", "AND", "NOT") and isinstance(v, dict):
                sub = _walk(v, label_cat)
                if k == "OR" and len(sub) > 1:
                    lines.append("いずれか: " + " / ".join(sub))
                elif k == "OR" and len(sub) == 1:
                    lines.extend(sub)
                elif k == "NOT" and sub:
                    lines.append("除外: " + "、".join(sub))
                elif k == "AND" and sub:
                    lines.extend(sub)

            # Owner sub-scope
            elif k == "owner" and isinstance(v, dict):
                sub = _walk(v, label_cat)
                lines.extend(sub)

        return lines

    # --- Process each condition block ---
    if cty_potential:
        texts = _walk(cty_potential, "country")
        for t in texts:
            conds.append({"category": "country", "text": t})

    if loc_potential:
        texts = _walk(loc_potential, "location")
        for t in texts:
            conds.append({"category": "location", "text": t})

    if allow:
        texts = _walk(allow, "allow")
        for t in texts:
            conds.append({"category": "allow", "text": t})

    return conds


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

    # Build comprehensive Japanese lookup for condition text
    loc_ja_all = _build_loc_lookup(loc_dir)

    import re as _re

    # Fallback for variables not in any loc file (game engine refs, regions, etc.)
    _var_fallback: dict[str, str] = {
        "europe": "ヨーロッパ", "india": "インド",
        "japan_region": "日本", "hre": "神聖ローマ帝国",
        "catholic_church": "カトリック教会", "reformation": "宗教改革",
        "copenhagen": "コペンハーゲン", "ilkhanate": "イルハン国",
        "turkish_culture": "トルコ文化",
        "imperial_city_of_hue": "フエの帝国都市",
        "schwaz_mine": "シュヴァーツ銀鉱山",
    }
    # Merge fallbacks into lookup (don't overwrite existing)
    for fk, fv in _var_fallback.items():
        if fk not in loc_ja_all:
            loc_ja_all[fk] = fv

    def _resolve_game_funcs(text: str, lookup: dict[str, str]) -> str:
        """Resolve [ShowXxx('key')] game engine function calls using loc data."""
        def _func_replace(m: _re.Match) -> str:
            func = m.group(1)
            key = m.group(2)
            # Map function name to loc key suffix
            if "Adjective" in func:
                loc_key = f"{key}_ADJ"
            elif "Name" in func:
                loc_key = key
            else:
                loc_key = key
            return lookup.get(loc_key, lookup.get(key, m.group(0)))

        return _re.sub(
            r"\[(?:Show|Get)\w+\('(\w+)'\)]",
            lambda m: _func_replace(_re.match(r"\[(Show|Get)(\w+?)(?:WithNoTooltip)?\('(\w+)'\)]", m.group(0)) or m),
            text,
        )

    def _resolve_game_funcs_v2(text: str, lookup: dict[str, str]) -> str:
        """Resolve [FuncName('key')] patterns."""
        def _repl(m: _re.Match) -> str:
            full = m.group(0)
            func_name = m.group(1)
            key = m.group(2)
            if "Adjective" in func_name:
                return lookup.get(f"{key}_ADJ", lookup.get(key, full))
            elif "Name" in func_name:
                return lookup.get(key, full)
            return lookup.get(key, full)

        return _re.sub(r"\[(\w+)\('(\w+)'\)\]", _repl, text)

    def _resolve_vars(text: str, lookup: dict[str, str], max_depth: int = 5) -> str:
        """Replace $var$ references and [Func('key')] calls with loc values."""
        text = _resolve_game_funcs_v2(text, lookup)
        for _ in range(max_depth):
            new = _re.sub(
                r"\$([^$]+)\$",
                lambda m: lookup.get(m.group(1), m.group(0)),
                text,
            )
            if new == text:
                break
            text = new
        return text

    result = []
    for bld_id, props in all_buildings.items():
        raw_name_en = loc_en.get(bld_id, loc_ja_all.get(bld_id, bld_id))
        raw_name_ja = loc_ja.get(bld_id, loc_ja_all.get(bld_id, bld_id))
        entry: dict = {
            "id": bld_id,
            "name_en": _resolve_vars(raw_name_en, loc_ja_all),
            "name_ja": _resolve_vars(raw_name_ja, loc_ja_all),
            "desc_en": strip_markup(_resolve_vars(loc_en.get(f"{bld_id}_desc", ""), loc_ja_all)),
            "desc_ja": strip_markup(_resolve_vars(loc_ja.get(f"{bld_id}_desc", ""), loc_ja_all)),
        }

        # Key properties
        for field in (
            "category",
            "pop_type",
            "expensive",
        ):
            if field in props:
                entry[field] = props[field]

        # Max levels: resolve named constant to base value
        ml_raw = props.get("max_levels")
        if isinstance(ml_raw, int):
            entry["max_levels"] = ml_raw
        elif isinstance(ml_raw, str) and ml_raw in MAX_LEVEL_BASE:
            entry["max_levels"] = MAX_LEVEL_BASE[ml_raw]
            if ml_raw in MAX_LEVEL_DYNAMIC:
                entry["max_levels_scaling"] = MAX_LEVEL_DYNAMIC[ml_raw]
        elif isinstance(ml_raw, str):
            entry["max_levels"] = 1  # fallback
            entry["max_levels_raw"] = ml_raw
        elif isinstance(ml_raw, dict):
            # Complex scripted value - extract base if possible
            base = ml_raw.get("value", ml_raw.get("add", {}).get("value", 1))
            entry["max_levels"] = base if isinstance(base, int) else 1

        # Build time: resolve named constant to days
        bt_key = props.get("build_time")
        if bt_key and bt_key in BUILD_TIME_DAYS:
            entry["build_days"] = BUILD_TIME_DAYS[bt_key]
        elif bt_key:
            entry["build_time"] = bt_key  # fallback: keep raw

        # Settlement type flags
        settlements = []
        for s in ("rural_settlement", "town", "city"):
            if props.get(s) is True:
                settlements.append(s)
        if settlements:
            entry["settlements"] = settlements

        # Modifier - with localized names
        def _localize_modifier(mod: dict) -> list[dict]:
            result = []
            for mk, mv in mod.items():
                item: dict = {"key": mk, "value": mv}
                loc_name = loc_ja_all.get(mk)
                if loc_name:
                    item["name_ja"] = loc_name
                # Add description for tooltip
                desc_key = f"MODIFIER_TYPE_DESC_{mk}"
                desc = loc_ja_all.get(desc_key)
                if desc:
                    # Resolve [word|e] game concept refs to Japanese
                    def _resolve_concept(m: _re.Match) -> str:
                        word = m.group(1)
                        gc_key = f"game_concept_{word}"
                        return loc_ja_all.get(gc_key, loc_ja_all.get(word, word))
                    resolved = _re.sub(r"\[(\w+)\|[eE]\]", _resolve_concept, desc)
                    cleaned = strip_markup(_resolve_vars(resolved, loc_ja_all))
                    if cleaned and len(cleaned) > 5:
                        item["desc_ja"] = cleaned
                result.append(item)
            return result

        if "modifier" in props and isinstance(props["modifier"], dict):
            entry["modifier"] = _localize_modifier(props["modifier"])

        # Raw modifier
        if "raw_modifier" in props and isinstance(props["raw_modifier"], dict):
            entry["raw_modifier"] = _localize_modifier(props["raw_modifier"])

        # Production methods
        if "unique_production_methods" in props and isinstance(
            props["unique_production_methods"], dict
        ):
            methods = {}
            for mk, mv in props["unique_production_methods"].items():
                if isinstance(mv, dict):
                    method_data = dict(mv)
                    # Add localized name
                    loc_name = loc_ja_all.get(mk)
                    if loc_name:
                        method_data["_name_ja"] = loc_name
                    # Localize category
                    cat = mv.get("category")
                    if cat:
                        cat_name = loc_ja_all.get(cat)
                        if cat_name:
                            method_data["_category_ja"] = cat_name
                    methods[mk] = method_data
            if methods:
                entry["production_methods"] = methods

        # Build conditions (human-readable)
        loc_pot = props.get("location_potential")
        cty_pot = props.get("country_potential")
        allow_cond = props.get("allow")
        is_foreign = props.get("is_foreign") is True
        is_special = props.get("is_special") is True
        forbidden = props.get("forbidden_for_estates") is True

        conditions = _parse_conditions(
            loc_pot if isinstance(loc_pot, dict) else None,
            cty_pot if isinstance(cty_pot, dict) else None,
            allow_cond if isinstance(allow_cond, dict) else None,
            is_foreign,
            is_special,
            forbidden,
            loc_ja_all,
        )
        if conditions:
            entry["conditions"] = conditions

        # Construction demand
        if "construction_demand" in props:
            entry["construction_demand"] = props["construction_demand"]

        entry["icon"] = f"icons/buildings/{bld_id}.png"
        entry["source_file"] = props.get("_source_file", "")

        # ── Normalized filter tags for UI ──
        ftags: dict[str, list[str]] = {
            "availability": [],
            "location": [],
            "country": [],
            "traits": [],
        }
        is_event_only = False
        for c in conditions:
            t = c["text"]
            cat = c["category"]
            if t == "イベント限定（通常建設不可）":
                is_event_only = True
            elif cat == "flag":
                ftags["traits"].append(t)
            elif cat == "location":
                if "首都のみ" in t:
                    ftags["location"].append("首都")
                elif "首都以外" in t:
                    ftags["location"].append("首都以外")
                elif "港湾" in t or "沿岸" in t:
                    ftags["location"].append("港湾・沿岸")
                elif "河川" in t or "湖" in t:
                    ftags["location"].append("河川・湖")
                elif "市場中心地" in t:
                    ftags["location"].append("市場中心地")
                elif "特定地域" in t:
                    ftags["location"].append("特定地域")
                elif "海外" in t:
                    ftags["location"].append("海外領")
                elif "地形" in t or "植生" in t:
                    ftags["location"].append("地形・植生")
                elif "文化" in t or "支配文化" in t:
                    ftags["location"].append("特定文化")
                elif "道路" in t:
                    ftags["location"].append("首都道路接続")
            elif cat == "country":
                if "政体:" in t and "政体改革" not in t:
                    ftags["country"].append("政体制限")
                elif "政体改革" in t:
                    ftags["country"].append("政体改革")
                elif "宗教" in t:
                    ftags["country"].append("宗教制限")
                elif "国家:" in t:
                    ftags["country"].append("特定国家")
                elif "進歩:" in t:
                    ftags["country"].append("進歩必要")
                elif "文化:" in t or "文化グループ" in t:
                    ftags["country"].append("特定文化")
                elif "要補正:" in t:
                    ftags["country"].append("特殊要件")
                elif "所有地域" in t:
                    ftags["country"].append("特定地域所有")
            elif cat == "allow":
                if "文化" in t:
                    ftags["country"].append("特定文化")
                elif "交易圏" in t:
                    ftags["country"].append("交易圏")
                elif "港湾" in t:
                    ftags["location"].append("港湾・沿岸")
                elif "市場で生産中" in t:
                    ftags["country"].append("特殊要件")

        ftags["availability"] = ["イベント限定"] if is_event_only else ["通常建設可"]
        # Deduplicate
        for k in ftags:
            ftags[k] = sorted(set(ftags[k]))
        entry["filter_tags"] = ftags

        # Decomposed conditions for filter matching
        # "いずれか: A / B" → also match "A" and "B" individually
        filter_conds: list[dict] = []
        for c in conditions:
            filter_conds.append(c)
            text = c["text"]
            if text.startswith("いずれか: "):
                parts = text[len("いずれか: "):].split(" / ")
                for part in parts:
                    part = part.strip()
                    if part:
                        filter_conds.append({"category": c["category"], "text": part})
        entry["filter_conds"] = filter_conds

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
