"""Extract EU5 holy site data into core + per-language loc files.

Usage:
    python pipeline/extract_holy_sites.py

Output:
    pipeline/output/core/holy_sites.json
    pipeline/output/core/holy_site_types.json
    pipeline/output/loc/{lang}/holy_sites.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from paradox_parser import parse_file
from loc_parser import parse_loc_file, strip_markup
from languages import LANGUAGES

PIPELINE_DIR = Path(__file__).parent
RAW_DIR = PIPELINE_DIR / "raw"
OUTPUT_DIR = PIPELINE_DIR / "output"

SKIP_FILES = {"readme.txt"}

# Names missing from loc files — resolved manually
MISSING_NAMES: dict[str, dict[str, str]] = {
    "mount_fuji": {"en": "Mount Fuji", "ja": "富士山", "ko": "후지산", "zh-hans": "富士山",
                   "de": "Fuji", "fr": "Mont Fuji", "es": "Monte Fuji", "pt-br": "Monte Fuji",
                   "ru": "Фудзи", "pl": "Fudżi", "tr": "Fuji Dağı"},
    "mount_athos": {"en": "Mount Athos", "ja": "アトス山", "ko": "아토스산", "zh-hans": "阿索斯山",
                    "de": "Athos", "fr": "Mont Athos", "es": "Monte Athos", "pt-br": "Monte Atos",
                    "ru": "Афон", "pl": "Athos", "tr": "Aynaroz"},
    "mayapan": {"en": "Mayapan", "ja": "マヤパン"},
    "shravanabelagola": {"en": "Shravanabelagola", "ja": "シュラヴァナベラゴラ"},
    "jerusalem_the_holy_city_islam": {"en": "The Holy City of al-Quds", "ja": "聖地アル＝クドゥス",
                                      "de": "Heilige Stadt al-Quds", "fr": "La ville sainte d'al-Quds",
                                      "es": "La ciudad santa de al-Quds", "ko": "성지 알쿠드스",
                                      "zh-hans": "圣城古都斯"},
}

# Holy site type names missing from loc files
MISSING_TYPE_NAMES: dict[str, dict[str, str]] = {
    "city": {"en": "City", "ja": "都市", "ko": "도시", "zh-hans": "城市",
             "de": "Stadt", "fr": "Ville", "es": "Ciudad", "pt-br": "Cidade",
             "ru": "Город", "pl": "Miasto", "tr": "Şehir"},
    "temple": {"en": "Temple", "ja": "寺院"},  # fallback if buildings loc missing
}


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


# Flat modifier keys for holy sites (same concept as religion FLAT_KEYS)
HS_FLAT_KEYS = {
    "local_max_literacy", "local_clergy_max_literacy", "local_max_control",
}

# Modifiers where negative is beneficial
HS_INVERSE_GOOD_KEYS = {
    "local_unrest",
    "invite_religious_figure_same_school_cost_modifier",
    "invite_religious_figure_different_school_cost_modifier",
}


def build_holy_site_types() -> list[dict]:
    """Parse holy_site_types. Returns list of type dicts with modifiers."""
    types_dir = RAW_DIR / "holy_site_types"
    if not types_dir.exists():
        return []

    types_list = []
    for f in sorted(types_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for type_id, props in data.items():
            if not isinstance(props, dict):
                continue
            entry: dict = {"id": type_id}
            loc_mod = props.get("location_modifier", {})
            if isinstance(loc_mod, dict) and loc_mod:
                entry["location_modifier"] = [
                    _make_mod(k, v) for k, v in loc_mod.items()
                ]
            cty_mod = props.get("country_modifier", {})
            if isinstance(cty_mod, dict) and cty_mod:
                entry["country_modifier"] = [
                    _make_mod(k, v) for k, v in cty_mod.items()
                ]
            types_list.append(entry)
    return types_list


def _make_mod(key: str, value) -> dict:
    """Create modifier entry with pct/inv flags."""
    entry: dict = {"key": key, "value": value}
    if isinstance(value, (int, float)) and key not in HS_FLAT_KEYS:
        entry["pct"] = True
    if key in HS_INVERSE_GOOD_KEYS:
        entry["inv"] = True
    return entry


def build_holy_sites() -> list[dict]:
    """Parse holy site definition files. Returns list of holy site dicts."""
    sites_dir = RAW_DIR / "holy_sites"
    if not sites_dir.exists():
        print(f"[WARN] {sites_dir} not found")
        return []

    sites = []
    for f in sorted(sites_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for site_id, props in data.items():
            if not isinstance(props, dict):
                continue
            entry: dict = {"id": site_id}
            if "location" in props:
                entry["location"] = str(props["location"])
            if "type" in props:
                entry["type"] = str(props["type"])
            if "importance" in props:
                entry["importance"] = int(props["importance"])
            if "religions" in props:
                rel = props["religions"]
                if isinstance(rel, list):
                    entry["religions"] = rel
                elif isinstance(rel, str):
                    entry["religions"] = [rel]
            entry["source_file"] = f.name
            sites.append(entry)
    # Sort by importance desc, then id
    sites.sort(key=lambda s: (-s.get("importance", 0), s["id"]))
    return sites


def _resolve_vars(text: str, loc_data: dict[str, str]) -> str:
    """Resolve $key$ references against a loc dict, then strip remaining markup."""
    resolved = re.sub(
        r"\$(\w+)\$",
        lambda m: loc_data.get(m.group(1), m.group(1).replace("_", " ").title()),
        text,
    )
    return strip_markup(resolved)


def build_loc(sites: list[dict], types: list[dict]) -> dict[str, dict]:
    """Build per-language loc for holy sites and types."""
    all_loc = load_all_loc("holy_sites")
    all_mod_loc = load_all_loc("modifier_types")
    all_bld_loc = load_all_loc("buildings")  # temple etc. defined here
    all_prov_loc = load_all_loc("location_names")  # location display names
    all_pops_loc = load_all_loc("pops")  # $clergy$ etc.
    all_estate_loc = load_all_loc("estate")  # $clergy_estate$ etc.
    all_game_terms = load_all_loc("game_concepts")

    site_ids = [s["id"] for s in sites]
    type_ids = [t["id"] for t in types]

    # Collect all locations for province name loc
    all_locations = {s.get("location") for s in sites if s.get("location")}

    # Collect all modifier keys for loc
    all_mod_keys: set[str] = set()
    for t in types:
        for m in t.get("location_modifier", []):
            all_mod_keys.add(m["key"])
        for m in t.get("country_modifier", []):
            all_mod_keys.add(m["key"])

    loc_per_lang: dict[str, dict] = {}
    for url_code, loc_data in all_loc.items():
        mod_loc = all_mod_loc.get(url_code, {})
        bld_loc = all_bld_loc.get(url_code, {})
        prov_loc = all_prov_loc.get(url_code, {})
        pops_loc = all_pops_loc.get(url_code, {})
        concepts = all_game_terms.get(url_code, {})
        lang_loc = {}

        # Build a merged lookup for $key$ resolution
        var_lookup = dict(loc_data)
        var_lookup.update(bld_loc)
        var_lookup.update(pops_loc)  # $clergy$, $nobles$ etc.
        var_lookup.update(all_estate_loc.get(url_code, {}))  # $clergy_estate$ etc.
        var_lookup.update(prov_loc)  # $jerusalem$, province names
        for ck, cv in concepts.items():
            short = ck.replace("game_concept_", "")
            if short not in var_lookup:
                var_lookup[short] = cv

        # Site names — resolve $key$ self-references before stripping markup
        for site_id in site_ids:
            name = loc_data.get(site_id, "")
            if not name:
                continue
            resolved = _resolve_vars(name, var_lookup)
            if resolved:
                lang_loc[site_id] = {"name": resolved}

        # Fill in missing or broken names (containing unresolved $...$) from MISSING_NAMES
        for site_id in site_ids:
            existing = lang_loc.get(site_id, {}).get("name", "")
            if existing and "$" not in existing:
                continue
            if site_id in MISSING_NAMES:
                mn = MISSING_NAMES[site_id]
                lang_loc[site_id] = {"name": mn.get(url_code, mn.get("en", site_id))}

        # Type names and descriptions — resolve $key$ before stripping
        for type_id in type_ids:
            name = loc_data.get(type_id, "") or bld_loc.get(type_id, "")
            desc = loc_data.get(f"{type_id}_desc", "") or bld_loc.get(f"{type_id}_desc", "")
            entry: dict = {}
            if name:
                entry["name"] = _resolve_vars(name, var_lookup)
            elif type_id in MISSING_TYPE_NAMES:
                mtn = MISSING_TYPE_NAMES[type_id]
                entry["name"] = mtn.get(url_code, mtn.get("en", type_id))
            if desc:
                entry["desc"] = _resolve_vars(desc, var_lookup)
            if entry:
                lang_loc[type_id] = entry

        # Modifier names — resolve $key$ (e.g. $clergy$) before stripping
        mod_names = {}
        for mk in sorted(all_mod_keys):
            loc_name = mod_loc.get(f"MODIFIER_TYPE_NAME_{mk}", "")
            if loc_name:
                mod_names[mk] = _resolve_vars(loc_name, var_lookup)
        if mod_names:
            lang_loc["_modifier_names"] = mod_names

        # Modifier descriptions (same pattern as extract_aspects.py)
        from extract_religions import build_concept_resolver, _resolve_var_refs, _clean_desc
        estate_loc = all_estate_loc.get(url_code, {})
        hs_extra = {ek: ev for ek, ev in estate_loc.items() if ev and not ek.endswith("_desc")}
        resolve = build_concept_resolver(concepts, hs_extra)
        mod_descs = {}
        for mk in sorted(all_mod_keys):
            raw_desc = mod_loc.get(f"MODIFIER_TYPE_DESC_{mk}", "")
            if raw_desc:
                resolved_raw = _resolve_var_refs(raw_desc, mod_loc, concepts, estate_loc)
                cleaned = resolve(_clean_desc(resolved_raw))
                if cleaned:
                    mod_descs[mk] = cleaned
        if mod_descs:
            lang_loc["_modifier_descs"] = mod_descs

        # Location (province) names
        loc_names = {}
        for loc_id in all_locations:
            pname = prov_loc.get(loc_id, "")
            if pname:
                loc_names[loc_id] = strip_markup(pname)
        if loc_names:
            lang_loc["_location_names"] = loc_names

        loc_per_lang[url_code] = lang_loc

    return loc_per_lang


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    types = build_holy_site_types()
    sites = build_holy_sites()
    loc_per_lang = build_loc(sites, types)

    # Core
    write_json(OUTPUT_DIR / "core" / "holy_sites.json", sites)
    write_json(OUTPUT_DIR / "core" / "holy_site_types.json", types)
    print(f"Wrote {len(sites)} holy sites, {len(types)} types")

    # Religions with holy sites
    rel_set = set()
    for s in sites:
        rel_set.update(s.get("religions", []))
    print(f"Religions with holy sites: {len(rel_set)}")

    # Per-language loc
    for url_code, loc_data in loc_per_lang.items():
        write_json(OUTPUT_DIR / "loc" / url_code / "holy_sites.json", loc_data)
    print(f"Wrote loc for {len(loc_per_lang)} languages")

    # Coverage
    en_loc = loc_per_lang.get("en", {})
    named = sum(1 for s in sites if s["id"] in en_loc)
    print(f"Name coverage: {named}/{len(sites)}")


if __name__ == "__main__":
    main()
