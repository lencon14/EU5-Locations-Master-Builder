"""Extract EU5 country data into core + index + per-language loc files.

Joins data from three sources:
  - setup/countries/*.txt      → tag, color, culture, religion, difficulty, is_historic
  - country_start/10_countries.txt → capital, rank, government, tech
  - formable_countries/*.txt   → formable info (level, rule)

Also builds culture → culture_groups mapping from cultures/*.txt.

Usage:
    python pipeline/extract_countries.py

Output:
    pipeline/output/core/countries.json      — full detail data
    pipeline/output/core/countries_index.json — lightweight list data
    pipeline/output/loc/{lang}/countries.json
"""

from __future__ import annotations

import json
from pathlib import Path

import re

from paradox_parser import parse_file
from loc_parser import parse_loc_file, strip_markup
from languages import LANGUAGES

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"

SKIP_FILES = {"readme.txt", "00_readme.info"}
EXCLUDE_TAGS = {"DUMMY", "PIR", "MER"}


# ─── Data loading helpers ────────────────────────────────────────────────

def load_all_loc(*loc_names: str) -> dict[str, dict[str, str]]:
    """Load and merge multiple loc files for all languages."""
    result: dict[str, dict[str, str]] = {}
    loc_dir = RAW_DIR / "localization"
    for url_code, game_code, _ in LANGUAGES:
        merged: dict[str, str] = {}
        for loc_name in loc_names:
            path = loc_dir / f"{loc_name}_l_{game_code}.yml"
            if path.exists():
                merged.update(parse_loc_file(path))
        result[url_code] = merged
    return result


def _parse_color(props: dict) -> list[int] | None:
    """Extract RGB color from country definition.

    Handles:
      color = rgb { R G B }  → parsed as color="rgb", _values=[[R,G,B], ...]
      color = map_FRA         → named color, return None
    """
    color_val = props.get("color")
    if color_val is None:
        return None
    if isinstance(color_val, str) and color_val == "rgb":
        vals = props.get("_values", [])
        if isinstance(vals, list):
            # Find first sub-list with 3 numeric values (the color)
            for item in vals:
                if isinstance(item, list) and len(item) >= 3:
                    try:
                        return [int(float(v)) for v in item[:3]]
                    except (ValueError, TypeError):
                        continue
            # Flat list case
            if len(vals) >= 3 and all(isinstance(v, (int, float)) for v in vals[:3]):
                return [int(v) for v in vals[:3]]
    return None


# ─── Source 1: Country definitions ───────────────────────────────────────

def load_country_definitions() -> dict[str, dict]:
    """Parse setup/countries/*.txt → {tag: {culture, religion, ...}}"""
    ctr_dir = RAW_DIR / "countries"
    if not ctr_dir.exists():
        return {}

    all_countries: dict[str, dict] = {}
    for f in sorted(ctr_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict) and key not in EXCLUDE_TAGS:
                val["_source_file"] = f.name
                val["_file_region"] = f.stem
                all_countries[key] = val
    return all_countries


# ─── Source 2: Country start data ────────────────────────────────────────

def _load_templates() -> dict[str, dict]:
    """Parse setup/templates/*.txt → {template_name: {government_type, ...}}

    Resolves recursive include chains (templates including other templates).
    """
    tpl_dir = RAW_DIR / "country_start" / "templates"
    if not tpl_dir.exists():
        return {}

    # First pass: parse all templates and collect raw data + includes
    raw_templates: dict[str, tuple[dict, list[str]]] = {}
    for f in sorted(tpl_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        tpl: dict = {}
        gov = data.get("government", {})
        if isinstance(gov, dict):
            if "type" in gov:
                tpl["government_type"] = gov["type"]
            if "heir_selection" in gov:
                tpl["heir_selection"] = gov["heir_selection"]
        if "country_rank" in data:
            tpl["country_rank"] = data["country_rank"]
        # Collect includes for recursive resolution
        includes = data.get("include", [])
        if isinstance(includes, str):
            includes = [includes]
        elif not isinstance(includes, list):
            includes = []
        raw_templates[f.stem] = (tpl, [i for i in includes if isinstance(i, str)])

    # Second pass: resolve recursive includes (depth-limited)
    resolved: dict[str, dict] = {}

    visiting: set[str] = set()

    def _resolve_tpl(name: str) -> dict:
        if name in resolved:
            return resolved[name]
        if name not in raw_templates or name in visiting:
            return {}
        visiting.add(name)
        tpl, includes = raw_templates[name]
        # Start from parent templates (last include wins for override)
        merged: dict = {}
        for inc in includes:
            parent = _resolve_tpl(inc)
            merged.update(parent)
        # Own values override parents
        merged.update(tpl)
        visiting.discard(name)
        resolved[name] = merged
        return merged

    for name in raw_templates:
        _resolve_tpl(name)

    return resolved


def load_country_start() -> tuple[dict[str, dict], dict[str, dict]]:
    """Parse country_start/10_countries.txt → (countries, templates)"""
    templates = _load_templates()

    path = RAW_DIR / "country_start" / "10_countries.txt"
    if not path.exists():
        return {}, templates

    data = parse_file(path)
    countries_wrapper = data.get("countries", {})
    countries = {}
    if isinstance(countries_wrapper, dict):
        inner = countries_wrapper.get("countries", {})
        if isinstance(inner, dict):
            countries = {k: v for k, v in inner.items()
                         if isinstance(v, dict) and k not in EXCLUDE_TAGS}
    return countries, templates


# ─── Source 3: Formable countries ────────────────────────────────────────

def load_formable_countries() -> dict[str, dict]:
    """Parse formable_countries/*.txt → {target_tag: {level, rule, ...}}

    Formable entries use keys like SCA_f with tag=SCA inside.
    Returns mapping from target tag (SCA) to formable info.
    """
    formable_dir = RAW_DIR / "formable_countries"
    if not formable_dir.exists():
        return {}

    result: dict[str, dict] = {}
    for f in sorted(formable_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                target_tag = val.get("tag")
                if isinstance(target_tag, str):
                    result[target_tag] = {
                        "formable_key": key,
                        "level": val.get("level"),
                        "rule": val.get("rule"),
                        "required_locations_fraction": val.get("required_locations_fraction"),
                    }
    return result


# ─── Culture → culture_groups mapping ────────────────────────────────────

def build_culture_group_map() -> dict[str, list[str]]:
    """Parse cultures/*.txt → {culture_name: [group1, group2, ...]}"""
    culture_dir = RAW_DIR / "cultures"
    if not culture_dir.exists():
        return {}

    culture_map: dict[str, list[str]] = {}
    for f in sorted(culture_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for name, val in data.items():
            if isinstance(val, dict) and "culture_groups" in val:
                groups = val["culture_groups"]
                if isinstance(groups, str):
                    groups = [groups]
                elif isinstance(groups, list):
                    groups = [g for g in groups if isinstance(g, str)]
                elif isinstance(groups, dict):
                    groups = list(groups.keys())
                else:
                    groups = []
                if groups:
                    culture_map[name] = groups
    return culture_map


# ─── Main build ──────────────────────────────────────────────────────────

def build_countries():
    """Build country data by joining all sources."""
    # Load sources
    definitions = load_country_definitions()
    start_data, templates = load_country_start()
    formables = load_formable_countries()
    culture_groups_map = build_culture_group_map()

    print(f"Definitions: {len(definitions)} tags")
    print(f"Start data: {len(start_data)} tags")
    print(f"Templates: {len(templates)}")
    print(f"Formables: {len(formables)} entries")
    print(f"Culture→groups: {len(culture_groups_map)} cultures")

    # Join: use definitions as the base, enrich with start_data
    core_list = []
    for tag, defn in definitions.items():
        entry: dict = {"tag": tag}

        # From definitions
        entry["file_region"] = defn.get("_file_region", "")
        for field in ("culture_definition", "religion_definition",
                      "description_category", "difficulty"):
            if field in defn:
                entry[field] = defn[field]

        if defn.get("is_historic") is True:
            entry["is_historic"] = True

        color = _parse_color(defn)
        if color:
            entry["color"] = color

        # Culture groups (multi-valued)
        culture = defn.get("culture_definition", "")
        if culture and culture in culture_groups_map:
            entry["culture_groups"] = culture_groups_map[culture]

        # From start data (join by tag)
        start = start_data.get(tag, {})
        if start:
            if "capital" in start:
                entry["capital"] = start["capital"]
            if "country_rank" in start:
                entry["country_rank"] = start["country_rank"]
            if "starting_technology_level" in start:
                entry["starting_tech"] = start["starting_technology_level"]

            # Government: direct block > template fallback
            gov = start.get("government", {})
            if isinstance(gov, dict):
                if "type" in gov:
                    entry["government_type"] = gov["type"]
                if "heir_selection" in gov:
                    entry["heir_selection"] = gov["heir_selection"]

            # Template fallback: later includes override earlier (last-wins)
            includes = start.get("include", [])
            if isinstance(includes, str):
                includes = [includes]
            if isinstance(includes, list):
                # Collect template values in order; later templates override
                tpl_merged: dict = {}
                for tpl_name in includes:
                    if isinstance(tpl_name, str) and tpl_name in templates:
                        tpl_merged.update(templates[tpl_name])
                # Only fill fields not already set from direct data
                for fld in ("government_type", "heir_selection", "country_rank"):
                    if fld not in entry and fld in tpl_merged:
                        entry[fld] = tpl_merged[fld]

        # Formable info
        formable = formables.get(tag)
        if formable:
            entry["is_formable"] = True
            entry["formable_level"] = formable.get("level")
            entry["formable_rule"] = formable.get("rule")
        else:
            entry["is_formable"] = False

        # CoA icon path
        entry["icon"] = f"icons/coa/{tag}.png"

        entry["source_file"] = defn.get("_source_file", "")
        core_list.append(entry)

    return core_list


def build_index(core_list: list[dict]) -> list[dict]:
    """Build lightweight index for list page."""
    index = []
    for item in core_list:
        entry = {
            "tag": item["tag"],
            "icon": item["icon"],
            "file_region": item["file_region"],
        }
        for field in ("culture_groups", "religion_definition", "country_rank",
                      "difficulty", "description_category", "government_type",
                      "is_formable", "is_historic"):
            if field in item:
                entry[field] = item[field]
        index.append(entry)
    return index


def build_loc(tags: list[str]) -> dict[str, dict]:
    """Build per-language localization for countries."""
    # Loc sources for country names + descriptions
    all_loc = load_all_loc(
        "country_names", "country_description_category",
        "common_used_strings",
    )

    # Additional loc for resolving $variable$ in country names
    var_loc = load_all_loc(
        "location_names", "government_names", "common_used_strings",
        "cultural_and_languages", "culture_groups", "area",
    )

    # Formable country names (loaded for future use in formable descriptions)
    # formable_loc = load_all_loc("formable_countries")

    # Pre-load auxiliary loc sources (outside loop to avoid redundant parsing)
    aux_culture_loc = load_all_loc("cultural_and_languages", "cultures")
    aux_group_loc = load_all_loc("culture_groups")
    aux_capital_loc = load_all_loc("location_names")
    aux_gov_loc = load_all_loc("government", "government_names", "government_reforms")

    # Build per-language
    loc_per_lang: dict[str, dict] = {}
    for url_code, _, _ in LANGUAGES:
        loc_data = all_loc.get(url_code, {})
        var_data = var_loc.get(url_code, {})
        var_lookup = {**var_data, **loc_data}

        lang_loc: dict = {}
        for tag in tags:
            raw_name = loc_data.get(tag, "")
            name = _resolve_vars(raw_name, var_lookup) if "$" in raw_name else raw_name
            name = strip_markup(name)

            raw_desc = loc_data.get(f"{tag}_desc", "")
            desc = strip_markup(_resolve_vars(raw_desc, var_lookup) if "$" in raw_desc else raw_desc)

            if name or desc:
                lang_loc[tag] = {"name": name}
                if desc:
                    lang_loc[tag]["desc"] = desc

        # Auxiliary name maps (culture, culture_group, capital)
        culture_loc = aux_culture_loc.get(url_code, {})
        group_loc = aux_group_loc.get(url_code, {})
        capital_loc = aux_capital_loc.get(url_code, {})

        gov_loc = aux_gov_loc.get(url_code, {})

        lang_loc["_culture_names"] = {k: strip_markup(v) for k, v in culture_loc.items() if v and not k.startswith("_")}
        lang_loc["_culture_group_names"] = {k: strip_markup(v) for k, v in group_loc.items() if v and not k.startswith("_")}
        lang_loc["_capital_names"] = {k: strip_markup(v) for k, v in capital_loc.items() if v and not k.startswith("_")}
        lang_loc["_gov_names"] = {k: strip_markup(v) for k, v in gov_loc.items() if v and not k.startswith("_")}

        loc_per_lang[url_code] = lang_loc

    return loc_per_lang


def _resolve_vars(text: str, lookup: dict[str, str], depth: int = 0) -> str:
    """Resolve $variable$ references in localization text."""
    if depth > 5 or "$" not in text:
        return text
    def _replace(m):
        key = m.group(1)
        val = lookup.get(key, "")
        if val and "$" in val:
            val = _resolve_vars(val, lookup, depth + 1)
        return val if val else m.group(0)
    return re.sub(r'\$(\w+)\$', _replace, text)


# ─── Output ──────────────────────────────────────────────────────────────

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    core_list = build_countries()
    print(f"\nTotal countries: {len(core_list)}")

    # Stats
    with_capital = sum(1 for c in core_list if "capital" in c)
    with_rank = sum(1 for c in core_list if "country_rank" in c)
    with_gov = sum(1 for c in core_list if "government_type" in c)
    formable = sum(1 for c in core_list if c.get("is_formable"))
    historic = sum(1 for c in core_list if c.get("is_historic"))
    print(f"  with capital: {with_capital}")
    print(f"  with rank: {with_rank}")
    print(f"  with gov type: {with_gov}")
    print(f"  formable: {formable}")
    print(f"  is_historic: {historic}")

    # Core (full detail)
    write_json(OUTPUT_DIR / "core" / "countries.json", core_list)
    print(f"\nWrote core/countries.json ({len(core_list)} entries)")

    # Index (lightweight)
    index = build_index(core_list)
    write_json(OUTPUT_DIR / "core" / "countries_index.json", index)
    print(f"Wrote core/countries_index.json ({len(index)} entries)")

    # Per-language loc
    tags = [c["tag"] for c in core_list]
    loc_per_lang = build_loc(tags)
    for url_code, loc_data in loc_per_lang.items():
        write_json(OUTPUT_DIR / "loc" / url_code / "countries.json", loc_data)
    named = sum(1 for k, v in loc_per_lang.get("en", {}).items()
                if not k.startswith("_") and isinstance(v, dict) and v.get("name"))
    print(f"Wrote loc for {len(loc_per_lang)} languages ({named} EN names)")

    # Check $variable$ resolution (skip _ auxiliary keys)
    en_loc = loc_per_lang.get("en", {})
    unresolved_name = [tag for tag, v in en_loc.items()
                       if not tag.startswith("_") and isinstance(v, dict) and "$" in v.get("name", "")]
    unresolved_desc = [tag for tag, v in en_loc.items()
                       if not tag.startswith("_") and isinstance(v, dict) and "$" in v.get("desc", "")]
    empty_names = [tag for tag in tags
                   if tag in en_loc and isinstance(en_loc[tag], dict) and not en_loc[tag].get("name")]
    if unresolved_name:
        print(f"\n[WARN] {len(unresolved_name)} tags with unresolved $var$ in EN name:")
        for t in unresolved_name[:10]:
            print(f"  {t}: {en_loc[t]['name']}")
    if unresolved_desc:
        print(f"[WARN] {len(unresolved_desc)} tags with unresolved $var$ in EN desc")
    if empty_names:
        print(f"[WARN] {len(empty_names)} tags with empty EN name (possible stripped $var$):")
        for t in empty_names[:10]:
            print(f"  {t}")


if __name__ == "__main__":
    main()
