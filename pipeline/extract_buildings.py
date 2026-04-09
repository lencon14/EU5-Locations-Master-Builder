"""Extract EU5 building data into core + per-language loc files.

Usage:
    python pipeline/extract_buildings.py

Output:
    pipeline/output/core/buildings.json       — game data (no localized strings)
    pipeline/output/loc/{lang}/buildings.json  — names, descriptions, condition_lines, modifiers, pm
    pipeline/output/buildings.json             — legacy merged format (temporary)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from paradox_parser import parse_file
from loc_parser import parse_loc_file, strip_markup
from languages import LANGUAGES
from building_templates import TEMPLATES, TERRAIN_VALUES, ITEM_SEP, NOT_SEP

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"

SKIP_FILES = {"readme.txt"}

# Max levels: base values from game/in_game/common/script_values/building_caps.txt
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

MAX_LEVEL_DYNAMIC: dict[str, list[str]] = {
    "guild_max_level": ["development", "population", "city+5"],
    "workshop_max_level": ["development", "population", "city+10"],
    "manufactory_max_level": ["development", "population", "city+20"],
    "market_max_level": ["development", "population", "city+5"],
    "mills_max_level": ["development", "population", "city+25"],
    "market_warehouse_max_level": ["development", "city+2"],
    "manpower_max_level": ["population"],
    "rural_building_cap": ["development"],
    "irrigant_cap": ["development"],
    "bund_cap": ["development"],
    "estate_building_stackable_level": ["development"],
    "slave_market_max_level": ["development"],
    "polders_max_level": ["development"],
    "hanseatic_kontor_max_level": ["city+2"],
    "venetian_palaces_max_level": ["development", "city+2"],
    "fulani_cattle_rearing_pen_max_level": ["development"],
    "kurmina_headquarter_max_level": ["development", "population"],
    "eghabho_nore_mansion_max_level": ["city+1"],
    "kurultai_max_level": ["population"],
}

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

# ── Loc file list used by _build_loc_lookup ──

_LOC_FILES = [
    "government_names", "government_reforms", "government",
    "cultural_and_languages", "culture_groups",
    "religion", "goods",
    "countries", "country_names",
    "laws", "laws_and_policies",
    "buildings", "advances",
    "province_names", "pops",
    "estate", "modifier_types", "static_modifiers",
    "modifiers", "area",
    "units", "holy_sites", "traits",
    "game_concepts", "flavor_hab",
]


def _strip_prefix(s: str) -> str:
    """Remove a known prefix like 'culture:', 'religion:', etc."""
    if isinstance(s, str) and ":" in s:
        return s.split(":", 1)[1]
    return str(s)


def _build_loc_lookup(loc_dir: Path, game_code: str) -> dict[str, str]:
    """Build a comprehensive localization lookup for a single language."""
    lookup: dict[str, str] = {}
    for base_name in _LOC_FILES:
        path = loc_dir / f"{base_name}_l_{game_code}.yml"
        if not path.exists():
            continue
        data = parse_loc_file(path)
        for k, v in data.items():
            if "_tt" in k:
                continue
            if "_desc" in k or k.startswith("MODIFIER_TYPE_DESC_"):
                if len(v) < 500:
                    lookup[k] = v
                continue
            if "[" in v:
                continue
            if "$" not in v and len(v) > 30:
                continue
            if "$" in v and len(v) > 60:
                continue
            lookup[k] = v
            if k.endswith("_province"):
                lookup[k[: -len("_province")]] = v
            if k.startswith("MODIFIER_TYPE_NAME_"):
                mod_key = k[len("MODIFIER_TYPE_NAME_"):]
                if mod_key not in lookup:
                    lookup[mod_key] = v

    # Resolve $var$ references
    for _ in range(3):
        changed = False
        for k, v in list(lookup.items()):
            if "$" in v:
                new_v = re.sub(
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


# ── Structured requirements ──

# Collect unrecognized condition keys across all buildings (diagnostic)
_UNHANDLED_COND_KEYS: set[str] = set()


def _parse_requirements(
    loc_potential: dict | None,
    cty_potential: dict | None,
    allow: dict | None,
    is_foreign: bool,
    is_special: bool,
    forbidden_for_estates: bool,
) -> list[dict]:
    """Convert raw condition dicts into structured requirement nodes.

    Each node has: type, scope, and type-specific fields.
    """
    reqs: list[dict] = []

    # Flags
    if is_special:
        reqs.append({"type": "flag", "flag": "is_special", "scope": "flag"})
    if is_foreign:
        reqs.append({"type": "flag", "flag": "is_foreign", "scope": "flag"})
    if forbidden_for_estates:
        reqs.append({"type": "flag", "flag": "no_estates", "scope": "flag"})

    def _walk(obj: dict, scope: str) -> list[dict]:
        nodes: list[dict] = []
        if not isinstance(obj, dict):
            return nodes

        if obj.get("always") is False:
            nodes.append({"type": "event_only", "scope": scope})
            return nodes

        for k, v in obj.items():
            if k in ("custom_tooltip", "text"):
                if isinstance(v, dict):
                    nodes.extend(_walk(v, scope))
                continue

            # Location booleans
            if k in ("is_capital", "is_port", "is_coastal", "has_river",
                      "is_adjacent_to_lake", "is_market_center",
                      "has_road_to_capital", "is_overseas_for_owner"):
                if isinstance(v, bool):
                    nodes.append({"type": "loc_bool", "check": k, "value": v, "scope": scope})

            # Country booleans
            elif k == "has_slavery" and v is True:
                nodes.append({"type": "country_bool", "check": "has_slavery", "value": True, "scope": scope})

            # Reference-based conditions
            elif k == "has_reform":
                nodes.append({"type": "ref", "kind": "has_reform", "refs": [_strip_prefix(v)], "scope": scope})
            elif k == "government_type":
                refs = [_strip_prefix(x) for x in v] if isinstance(v, list) else [_strip_prefix(v)]
                nodes.append({"type": "ref", "kind": "government_type", "refs": refs, "scope": scope})
            elif k == "has_advance":
                nodes.append({"type": "ref", "kind": "has_advance", "refs": [_strip_prefix(v)], "scope": scope})
            elif k == "culture" and isinstance(v, str):
                nodes.append({"type": "ref", "kind": "culture", "refs": [_strip_prefix(v)], "scope": scope})
            elif k == "religion" and isinstance(v, str):
                nodes.append({"type": "ref", "kind": "religion", "refs": [_strip_prefix(v)], "scope": scope})
            elif k == "tag" and isinstance(v, str):
                nodes.append({"type": "ref", "kind": "tag", "refs": [v], "scope": scope})
            elif k == "has_or_had_tag":
                refs = [v] if isinstance(v, str) else list(v)
                nodes.append({"type": "ref", "kind": "has_or_had_tag", "refs": refs, "scope": scope})
            elif k == "has_policy":
                nodes.append({"type": "ref", "kind": "has_policy", "refs": [_strip_prefix(v)], "scope": scope})

            # Culture/religion sub-checks
            elif k == "dominant_culture":
                if isinstance(v, str):
                    nodes.append({"type": "ref", "kind": "dominant_culture", "refs": [_strip_prefix(v)], "scope": scope})
                elif isinstance(v, dict):
                    if "has_culture_group" in v:
                        nodes.append({"type": "ref", "kind": "has_culture_group",
                                      "refs": [_strip_prefix(v["has_culture_group"])], "scope": scope})
                    else:
                        nodes.extend(_walk(v, scope))
            elif k == "has_culture_group":
                nodes.append({"type": "ref", "kind": "has_culture_group", "refs": [_strip_prefix(v)], "scope": scope})
            elif k == "group" and isinstance(v, str):
                nodes.append({"type": "ref", "kind": "religion_group", "refs": [_strip_prefix(v)], "scope": scope})

            # Location references
            elif k == "this" and isinstance(v, str) and v.startswith("location:"):
                nodes.append({"type": "ref", "kind": "location", "refs": [v.split(":", 1)[1]], "scope": scope})
            elif k == "owns" and isinstance(v, str) and v.startswith("location:"):
                nodes.append({"type": "ref", "kind": "owns_location", "refs": [v.split(":", 1)[1]], "scope": scope})

            # Terrain/vegetation/climate (list or single string)
            elif k in ("vegetation", "topography", "climate") and (isinstance(v, list) or isinstance(v, str)):
                vals = [v] if isinstance(v, str) else list(v)
                nodes.append({"type": "terrain", "terrain_type": k, "values": vals, "scope": scope})

            # Raw material check
            elif k == "raw_material" and isinstance(v, str):
                nodes.append({"type": "ref", "kind": "market_produces", "refs": [_strip_prefix(v)], "scope": scope})

            # Region check
            elif k == "region" and isinstance(v, str) and not v.startswith("scope:"):
                nodes.append({"type": "ref", "kind": "location", "refs": [_strip_prefix(v)], "scope": scope})

            # Market conditions
            elif k == "market" and isinstance(v, dict):
                if "is_produced_in_market" in v:
                    nodes.append({"type": "ref", "kind": "market_produces",
                                  "refs": [_strip_prefix(v["is_produced_in_market"])], "scope": scope})
                elif "in_trade_range_of" in v:
                    nodes.append({"type": "ref", "kind": "in_trade_range", "refs": [], "scope": scope})

            # Modifier checks
            elif k.startswith("modifier:") and v is True:
                mod_name = k.split(":", 1)[1]
                nodes.append({"type": "modifier_check", "modifier": mod_name, "scope": scope})

            # Compound: OR / AND / NOT / NOR
            elif k in ("OR", "AND", "NOT", "NOR") and isinstance(v, dict):
                children = _walk(v, scope)
                if k == "OR" and len(children) > 1:
                    nodes.append({"type": "or", "children": children, "scope": scope})
                elif k == "OR" and len(children) == 1:
                    nodes.extend(children)
                elif k in ("NOT", "NOR") and children:
                    nodes.append({"type": "not", "children": children, "scope": scope})
                elif k == "AND" and children:
                    nodes.extend(children)

            # Owner sub-scope
            elif k == "owner" and isinstance(v, dict):
                nodes.extend(_walk(v, scope))

            # Unrecognized key — record for diagnostics
            else:
                _UNHANDLED_COND_KEYS.add(k)

        return nodes

    if cty_potential:
        reqs.extend(_walk(cty_potential, "country"))
    if loc_potential:
        reqs.extend(_walk(loc_potential, "location"))
    if allow:
        reqs.extend(_walk(allow, "allow"))

    # Deduplicate event_only (can appear in multiple scopes)
    event_only_count = sum(1 for r in reqs if r["type"] == "event_only")
    if event_only_count > 1:
        seen_event = False
        deduped = []
        for r in reqs:
            if r["type"] == "event_only":
                if not seen_event:
                    seen_event = True
                    deduped.append(r)
            else:
                deduped.append(r)
        reqs = deduped

    return reqs


# ── Facet computation ──

_FACET_MAP: dict[tuple, str] = {
    ("flag", "is_special"):           "trait:special",
    ("flag", "is_foreign"):           "trait:foreign",
    ("flag", "no_estates"):           "trait:no_estates",
    ("loc_bool", "is_capital"):       "loc:capital",
    ("loc_bool", "is_port"):          "loc:port",
    ("loc_bool", "is_coastal"):       "loc:coastal",
    ("loc_bool", "has_river"):        "loc:river",
    ("loc_bool", "is_adjacent_to_lake"): "loc:lake",
    ("loc_bool", "is_market_center"):    "loc:market_center",
    ("loc_bool", "is_overseas_for_owner"): "loc:overseas",
    ("loc_bool", "has_road_to_capital"):   "loc:road_to_capital",
    ("ref", "government_type"):       "cty:gov_type",
    ("ref", "has_reform"):            "cty:reform",
    ("ref", "has_advance"):           "cty:advance",
    ("ref", "religion"):              "cty:religion",
    ("ref", "religion_group"):        "cty:religion",
    ("ref", "tag"):                   "cty:tag",
    ("ref", "has_or_had_tag"):        "cty:tag",
    ("ref", "culture"):               "cty:culture",
    ("ref", "dominant_culture"):      "cty:culture",
    ("ref", "has_culture_group"):     "cty:culture_group",
    ("ref", "has_policy"):            "cty:policy",
    ("ref", "location"):              "loc:specific_location",
    ("ref", "owns_location"):         "cty:owns_location",
    ("ref", "market_produces"):       "cty:market_produces",
    ("ref", "in_trade_range"):        "cty:trade_range",
    ("terrain", "vegetation"):        "loc:terrain",
    ("terrain", "topography"):        "loc:terrain",
    ("terrain", "climate"):           "loc:terrain",
    ("modifier_check",):              "cty:modifier_check",
    ("country_bool",):                "cty:special",
    ("event_only",):                  "avail:event_only",
}


_FACET_NEG_MAP: dict[str, str] = {
    "is_capital": "loc:non_capital",
    "is_coastal": "loc:inland",
    "is_overseas_for_owner": "loc:domestic",
}


def _compute_facets(requirements: list[dict]) -> list[str]:
    """Derive normalized facet tokens from requirement nodes."""
    facets: set[str] = set()

    def _collect(node: dict) -> None:
        t = node["type"]
        if t in ("or", "not"):
            for child in node.get("children", []):
                _collect(child)
            return

        # loc_bool: distinguish true/false to avoid wrong-direction facets
        if t == "loc_bool" and node.get("value") is False:
            check = node.get("check", "")
            neg_facet = _FACET_NEG_MAP.get(check)
            if neg_facet:
                facets.add(neg_facet)
            return

        # Try specific key first, then generic type
        kind = node.get("kind") or node.get("check") or node.get("flag") or node.get("terrain_type")
        facet = _FACET_MAP.get((t, kind)) or _FACET_MAP.get((t,))
        if facet:
            facets.add(facet)

    for req in requirements:
        _collect(req)

    return sorted(facets)


# ── Condition line generation ──


def _resolve_name(raw_id: str, loc_lookup: dict[str, str]) -> str:
    """Resolve a game entity ID to its localized display name."""
    clean = _strip_prefix(raw_id)
    spaced = clean.replace("_", " ")
    for candidate in (clean, spaced, clean.replace("_reform", ""),
                      f"{clean}_culture", f"{clean}_group"):
        if candidate in loc_lookup:
            return loc_lookup[candidate]
    return spaced


def _generate_condition_line(req: dict, url_code: str, loc_lookup: dict[str, str]) -> str:
    """Render a single requirement node to localized display text."""
    t = req["type"]
    sep = ITEM_SEP.get(url_code, " / ")

    if t == "flag":
        key = f"flag.{req['flag']}"
        return TEMPLATES.get(key, {}).get(url_code, req["flag"])

    if t == "event_only":
        return TEMPLATES["event_only"].get(url_code, "Event only")

    if t == "loc_bool":
        key = f"loc_bool.{req['check']}.{str(req['value']).lower()}"
        return TEMPLATES.get(key, {}).get(url_code, req["check"])

    if t == "country_bool":
        key = f"country_bool.{req['check']}"
        return TEMPLATES.get(key, {}).get(url_code, req["check"])

    if t == "ref":
        kind = req["kind"]
        # Special: in_trade_range has no refs
        if kind == "in_trade_range":
            return TEMPLATES.get("ref.in_trade_range", {}).get(url_code, "Within trade range")
        template_str = TEMPLATES.get(f"ref.{kind}", {}).get(url_code, f"{kind}: {{name}}")
        names = [_resolve_name(r, loc_lookup) for r in req.get("refs", [])]
        return template_str.format(name=sep.join(names))

    if t == "terrain":
        terrain_type = req["terrain_type"]
        template_str = TEMPLATES.get(f"terrain.{terrain_type}", {}).get(url_code, f"{terrain_type}: {{name}}")
        names = []
        for val in req.get("values", []):
            tv = TERRAIN_VALUES.get(val, {})
            names.append(tv.get(url_code, val))
        return template_str.format(name=sep.join(names))

    if t == "modifier_check":
        template_str = TEMPLATES.get("modifier_check", {}).get(url_code, "Requires modifier: {name}")
        name = _resolve_name(req["modifier"], loc_lookup)
        return template_str.format(name=name)

    if t == "or":
        children_lines = [_generate_condition_line(c, url_code, loc_lookup) for c in req.get("children", [])]
        template_str = TEMPLATES.get("logic.or", {}).get(url_code, "Any of: {items}")
        return template_str.format(items=sep.join(children_lines))

    if t == "not":
        children_lines = [_generate_condition_line(c, url_code, loc_lookup) for c in req.get("children", [])]
        template_str = TEMPLATES.get("logic.not", {}).get(url_code, "Excluded: {items}")
        not_sep = NOT_SEP.get(url_code, ", ")
        return template_str.format(items=not_sep.join(children_lines))

    return str(req)


def _generate_all_condition_lines(requirements: list[dict], url_code: str,
                                   loc_lookup: dict[str, str]) -> list[str]:
    """Generate localized condition lines for all requirements."""
    return [_generate_condition_line(r, url_code, loc_lookup) for r in requirements]


# ── Modifier and PM localization ──


def _localize_modifiers(mod_dict: dict, loc_lookup: dict[str, str]) -> dict[str, dict]:
    """Generate localized modifier names and descriptions."""
    result = {}
    for mk, mv in mod_dict.items():
        entry: dict = {}
        loc_name = loc_lookup.get(mk)
        if loc_name:
            entry["name"] = loc_name
        desc_key = f"MODIFIER_TYPE_DESC_{mk}"
        desc = loc_lookup.get(desc_key)
        if desc:
            def _resolve_concept(m: re.Match) -> str:
                word = m.group(1)
                gc_key = f"game_concept_{word}"
                return loc_lookup.get(gc_key, loc_lookup.get(word, word))
            resolved = re.sub(r"\[(\w+)\|[eE]\]", _resolve_concept, desc)
            cleaned = strip_markup(_resolve_vars(resolved, loc_lookup))
            if cleaned and len(cleaned) > 5:
                entry["desc"] = cleaned
        if entry:
            result[mk] = entry
    return result


def _localize_pm(methods: dict, loc_lookup: dict[str, str]) -> dict[str, dict]:
    """Generate localized production method names and categories."""
    result = {}
    for mk, mv in methods.items():
        entry: dict = {}
        loc_name = loc_lookup.get(mk)
        if loc_name:
            entry["name"] = loc_name
        if isinstance(mv, dict):
            cat = mv.get("category")
            if cat:
                cat_name = loc_lookup.get(cat)
                if cat_name:
                    entry["category"] = cat_name
        if entry:
            result[mk] = entry
    return result


# ── Var/func resolution (shared) ──


def _resolve_game_funcs(text: str, lookup: dict[str, str]) -> str:
    """Resolve [FuncName('key')] patterns."""
    def _repl(m: re.Match) -> str:
        func_name = m.group(1)
        key = m.group(2)
        if "Adjective" in func_name:
            return lookup.get(f"{key}_ADJ", lookup.get(key, m.group(0)))
        return lookup.get(key, m.group(0))
    return re.sub(r"\[(\w+)\('(\w+)'\)\]", _repl, text)


def _resolve_vars(text: str, lookup: dict[str, str], max_depth: int = 5) -> str:
    """Replace $var$ references and [Func('key')] calls."""
    text = _resolve_game_funcs(text, lookup)
    for _ in range(max_depth):
        new = re.sub(
            r"\$([^$]+)\$",
            lambda m: lookup.get(m.group(1), m.group(0)),
            text,
        )
        if new == text:
            break
        text = new
    return text


# ── Main build function ──


def build_buildings() -> tuple[list[dict], dict[str, dict]]:
    """Parse all building_types and produce core + per-language loc data.

    Returns (core_list, {url_code: {building_id: loc_entry}}).
    """
    bld_dir = RAW_DIR / "building_types"
    if not bld_dir.exists():
        print(f"[WARN] {bld_dir} not found, skipping")
        return [], {}

    all_buildings: dict[str, dict] = {}
    for f in sorted(bld_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_buildings[key] = val

    loc_dir = RAW_DIR / "localization"

    # Build per-language loc lookups
    loc_lookups: dict[str, dict[str, str]] = {}
    for url_code, game_code, _ in LANGUAGES:
        loc_lookups[url_code] = _build_loc_lookup(loc_dir, game_code)

    # Load per-language building-specific loc (name, desc)
    bld_loc_per_lang: dict[str, dict[str, str]] = {}
    for url_code, game_code, _ in LANGUAGES:
        path = loc_dir / f"buildings_l_{game_code}.yml"
        bld_loc_per_lang[url_code] = parse_loc_file(path) if path.exists() else {}

    # Build core + loc
    core_list: list[dict] = []
    loc_per_lang: dict[str, dict] = {url_code: {} for url_code, _, _ in LANGUAGES}

    for bld_id, props in all_buildings.items():
        # ── Core entry (language-independent) ──
        core: dict = {"id": bld_id}

        for field in ("category", "pop_type", "expensive"):
            if field in props:
                core[field] = props[field]

        # Max levels
        ml_raw = props.get("max_levels")
        if isinstance(ml_raw, int):
            core["max_levels"] = ml_raw
        elif isinstance(ml_raw, str) and ml_raw in MAX_LEVEL_BASE:
            core["max_levels"] = MAX_LEVEL_BASE[ml_raw]
            if ml_raw in MAX_LEVEL_DYNAMIC:
                core["max_levels_scaling"] = MAX_LEVEL_DYNAMIC[ml_raw]
        elif isinstance(ml_raw, str):
            core["max_levels"] = 1
            core["max_levels_raw"] = ml_raw
        elif isinstance(ml_raw, dict):
            base = ml_raw.get("value", ml_raw.get("add", {}).get("value", 1))
            core["max_levels"] = base if isinstance(base, int) else 1

        # Build time
        bt_key = props.get("build_time")
        if bt_key and bt_key in BUILD_TIME_DAYS:
            core["build_days"] = BUILD_TIME_DAYS[bt_key]
        elif bt_key:
            core["build_time"] = bt_key

        # Settlements
        settlements = [s for s in ("rural_settlement", "town", "city") if props.get(s) is True]
        if settlements:
            core["settlements"] = settlements

        # Requirements (structured)
        loc_pot = props.get("location_potential")
        cty_pot = props.get("country_potential")
        allow_cond = props.get("allow")
        is_foreign = props.get("is_foreign") is True
        is_special = props.get("is_special") is True
        forbidden = props.get("forbidden_for_estates") is True

        requirements = _parse_requirements(
            loc_pot if isinstance(loc_pot, dict) else None,
            cty_pot if isinstance(cty_pot, dict) else None,
            allow_cond if isinstance(allow_cond, dict) else None,
            is_foreign, is_special, forbidden,
        )
        if requirements:
            core["requirements"] = requirements

        # Facets
        facets = _compute_facets(requirements)
        if facets:
            core["facets"] = facets

        # Modifier (core: key + value only)
        if "modifier" in props and isinstance(props["modifier"], dict):
            core["modifier"] = [{"key": mk, "value": mv} for mk, mv in props["modifier"].items()]
        if "raw_modifier" in props and isinstance(props["raw_modifier"], dict):
            core["raw_modifier"] = [{"key": mk, "value": mv} for mk, mv in props["raw_modifier"].items()]

        # Production methods (core: structural data only)
        raw_upm = props.get("unique_production_methods")
        if raw_upm is not None:
            upm_blocks = [raw_upm] if isinstance(raw_upm, dict) else [b for b in raw_upm if isinstance(b, dict)]
            methods_core = {}
            for block in upm_blocks:
                for mk, mv in block.items():
                    if isinstance(mv, dict):
                        pm_entry: dict = {}
                        if "category" in mv:
                            pm_entry["category"] = mv["category"]
                        # Collect goods inputs/outputs
                        goods_keys = {k for k in mv if k not in ("category", "produced", "output")}
                        if goods_keys:
                            pm_entry["goods"] = {k: mv[k] for k in goods_keys if isinstance(mv[k], (int, float))}
                        if "produced" in mv:
                            pm_entry["produced"] = mv["produced"]
                        if "output" in mv:
                            pm_entry["output"] = mv["output"]
                        methods_core[mk] = pm_entry
            if methods_core:
                core["production_methods"] = methods_core

        if "construction_demand" in props:
            core["construction_demand"] = props["construction_demand"]

        core["icon"] = f"icons/buildings/{bld_id}.png"
        core["source_file"] = props.get("_source_file", "")

        core_list.append(core)

        # ── Per-language loc entries ──
        for url_code, game_code, _ in LANGUAGES:
            loc_lookup = loc_lookups[url_code]
            bld_loc = bld_loc_per_lang[url_code]

            loc_entry: dict = {}

            # Name
            raw_name = bld_loc.get(bld_id, loc_lookup.get(bld_id, ""))
            if raw_name:
                loc_entry["name"] = _resolve_vars(raw_name, loc_lookup)
            else:
                # Fallback: use id
                loc_entry["name"] = bld_id

            # Description
            raw_desc = bld_loc.get(f"{bld_id}_desc", "")
            if raw_desc:
                loc_entry["desc"] = strip_markup(_resolve_vars(raw_desc, loc_lookup))

            # Condition lines
            if requirements:
                loc_entry["condition_lines"] = _generate_all_condition_lines(
                    requirements, url_code, loc_lookup
                )

            # Modifier names/descs
            if "modifier" in props and isinstance(props["modifier"], dict):
                mod_loc = _localize_modifiers(props["modifier"], loc_lookup)
                if mod_loc:
                    loc_entry["modifiers"] = mod_loc
            if "raw_modifier" in props and isinstance(props["raw_modifier"], dict):
                raw_mod_loc = _localize_modifiers(props["raw_modifier"], loc_lookup)
                if raw_mod_loc:
                    loc_entry["raw_modifiers"] = raw_mod_loc

            # Production method names
            if raw_upm is not None:
                upm_blocks_loc = [raw_upm] if isinstance(raw_upm, dict) else [b for b in raw_upm if isinstance(b, dict)]
                pm_loc: dict = {}
                for block in upm_blocks_loc:
                    for mk, mv in block.items():
                        if isinstance(mv, dict):
                            pm_loc_entry = _localize_pm({mk: mv}, loc_lookup)
                            pm_loc.update(pm_loc_entry)
                if pm_loc:
                    loc_entry["pm"] = pm_loc

            loc_per_lang[url_code][bld_id] = loc_entry

    return core_list, loc_per_lang


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    core_list, loc_per_lang = build_buildings()

    # Core data
    core_path = OUTPUT_DIR / "core" / "buildings.json"
    write_json(core_path, core_list)
    print(f"Wrote {len(core_list)} buildings to {core_path}")

    if _UNHANDLED_COND_KEYS:
        print(f"[WARN] {len(_UNHANDLED_COND_KEYS)} unhandled condition keys: {', '.join(sorted(_UNHANDLED_COND_KEYS))}")

    # Per-language loc
    for url_code, loc_data in loc_per_lang.items():
        loc_path = OUTPUT_DIR / "loc" / url_code / "buildings.json"
        write_json(loc_path, loc_data)
    lang_count = sum(1 for v in loc_per_lang.values() if v)
    print(f"Wrote loc for {lang_count} languages")

    # Legacy merged format (backward compatibility)
    en_loc = loc_per_lang.get("en", {})
    ja_loc = loc_per_lang.get("ja", {})
    legacy = []
    for item in core_list:
        merged = dict(item)
        bid = item["id"]
        merged["name_en"] = en_loc.get(bid, {}).get("name", bid)
        merged["name_ja"] = ja_loc.get(bid, {}).get("name", bid)
        merged["desc_en"] = en_loc.get(bid, {}).get("desc", "")
        merged["desc_ja"] = ja_loc.get(bid, {}).get("desc", "")

        # Legacy condition text (Japanese)
        ja_lines = ja_loc.get(bid, {}).get("condition_lines", [])
        reqs = item.get("requirements", [])
        if reqs:
            conditions = []
            for req, line in zip(reqs, ja_lines):
                conditions.append({"category": req.get("scope", "flag"), "text": line})
            merged["conditions"] = conditions

            # Legacy filter_tags
            facets = item.get("facets", [])
            ftags: dict[str, list[str]] = {"availability": [], "location": [], "country": [], "traits": []}
            is_event_only = "avail:event_only" in facets
            for f in facets:
                if f.startswith("trait:"):
                    # Map facet back to Japanese text for legacy
                    _trait_map = {"trait:special": "特殊建物", "trait:foreign": "外国建物", "trait:no_estates": "荘園建設不可"}
                    ftags["traits"].append(_trait_map.get(f, f))
                elif f.startswith("loc:"):
                    _loc_map = {
                        "loc:capital": "首都", "loc:port": "港湾・沿岸", "loc:coastal": "港湾・沿岸",
                        "loc:river": "河川・湖", "loc:lake": "河川・湖", "loc:market_center": "市場中心地",
                        "loc:specific_location": "特定地域", "loc:overseas": "海外領",
                        "loc:terrain": "地形・植生", "loc:road_to_capital": "首都道路接続",
                    }
                    tag = _loc_map.get(f)
                    if tag and tag not in ftags["location"]:
                        ftags["location"].append(tag)
                elif f.startswith("cty:"):
                    _cty_map = {
                        "cty:gov_type": "政体制限", "cty:reform": "政体改革", "cty:advance": "進歩必要",
                        "cty:religion": "宗教制限", "cty:tag": "特定国家", "cty:culture": "特定文化",
                        "cty:culture_group": "特定文化", "cty:policy": "特殊要件",
                        "cty:owns_location": "特定地域所有", "cty:market_produces": "特殊要件",
                        "cty:trade_range": "交易圏", "cty:modifier_check": "特殊要件",
                        "cty:special": "特殊要件",
                    }
                    tag = _cty_map.get(f)
                    if tag and tag not in ftags["country"]:
                        ftags["country"].append(tag)
            ftags["availability"] = ["イベント限定"] if is_event_only else ["通常建設可"]
            for k in ftags:
                ftags[k] = sorted(set(ftags[k]))
            merged["filter_tags"] = ftags

            # Legacy filter_conds
            filter_conds = []
            for cond in conditions:
                filter_conds.append(cond)
                text = cond["text"]
                if text.startswith("いずれか: "):
                    parts = text[len("いずれか: "):].split(" / ")
                    for part in parts:
                        part = part.strip()
                        if part:
                            filter_conds.append({"category": cond["category"], "text": part})
            merged["filter_conds"] = filter_conds

        # Legacy modifier with name_ja
        if "modifier" in merged and isinstance(merged["modifier"], list):
            ja_mods = ja_loc.get(bid, {}).get("modifiers", {})
            for m in merged["modifier"]:
                mod_info = ja_mods.get(m["key"], {})
                if "name" in mod_info:
                    m["name_ja"] = mod_info["name"]
                if "desc" in mod_info:
                    m["desc_ja"] = mod_info["desc"]

        legacy.append(merged)

    legacy_path = OUTPUT_DIR / "buildings.json"
    write_json(legacy_path, legacy)
    print(f"Wrote legacy merged format to {legacy_path}")


if __name__ == "__main__":
    main()
