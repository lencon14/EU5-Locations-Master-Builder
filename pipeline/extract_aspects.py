"""Extract EU5 religious aspect data into core + per-language loc files.

Usage:
    python pipeline/extract_aspects.py

Output:
    pipeline/output/core/aspects.json
    pipeline/output/loc/{lang}/aspects.json
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

# --- Modifier display classification ---

# Flat keys: displayed as-is (+1, +0.1), not percentage
FLAT_KEYS = {
    "tolerance_heretic", "tolerance_heathen", "tolerance_own",
    "global_max_literacy", "global_clergy_max_literacy",
    "global_clergy_desired_pop",
    "monthly_religious_influence", "monthly_war_exhaustion",
    "monthly_yanantin",
    "global_hostile_attrition", "land_unit_attrition",
    "skill_of_new_artists", "prestige_from_land_battle",
    "clergy_estate_max_tax",
    "global_burghers_estate_power", "global_clergy_estate_power",
    "global_nobles_estate_power", "global_peasants_estate_power",
    "global_crown_estate_power",
    "global_monthly_food_modifier", "global_monthly_art_start_chance",
    "global_monthly_development", "global_monthly_prosperity",
    "global_monthly_control",
    "monthly_diplomats", "monthly_rebel_growth",
    "retreat_delay",
    "bank_interest",
    "global_life_expectancy",
    "global_population_growth",
}

# Modifiers where negative value is beneficial (cost/consumption reductions)
INVERSE_GOOD_KEYS = {
    "stability_cost", "monthly_war_exhaustion",
    "army_maintenance_cost", "navy_maintenance_cost",
    "colonial_maintenance_cost", "merchant_maintenance_cost",
    "expand_rgo_farming_cost_modifier", "expand_rgo_forestry_cost_modifier",
    "expand_rgo_mining_cost_modifier",
    "food_consumption_modifier", "global_pop_food_consumption",
    "court_spending_cost",
    "declaring_war_cost_modifier",
    "global_build_buildings_cost",
    "monthly_rebel_growth",
    "global_separatism",
    "add_religious_aspect_christian_cost_modifier",
    "remove_religious_aspect_christian_cost_modifier",
    "recruit_explorer_cost_modifier",
    "global_war_score_cost",
    "antagonism_received_modifier",
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


def _normalize_religions(val) -> list[str]:
    """Normalize religion field (single string or list) to a sorted list."""
    if isinstance(val, list):
        return sorted(set(str(v) for v in val))
    if isinstance(val, str):
        return [val]
    return []


def _enabled_to_text(enabled: dict) -> list[str]:
    """Convert enabled condition AST to human-readable exclusion list.

    Extracts aspect IDs from NOT { has_religious_aspect = religious_aspect:X }
    patterns, which represent mutual exclusivity.
    """
    excludes = []
    not_block = enabled.get("NOT")
    if not_block is None:
        return excludes

    def extract_aspect(item):
        if isinstance(item, dict):
            ra = item.get("has_religious_aspect", "")
            if isinstance(ra, str) and ra.startswith("religious_aspect:"):
                return ra.split(":", 1)[1]
        return None

    if isinstance(not_block, dict):
        asp = extract_aspect(not_block)
        if asp:
            excludes.append(asp)
    elif isinstance(not_block, list):
        for item in not_block:
            if isinstance(item, list):
                for sub in item:
                    asp = extract_aspect(sub)
                    if asp:
                        excludes.append(asp)
            else:
                asp = extract_aspect(item)
                if asp:
                    excludes.append(asp)
    return sorted(set(excludes))


def _resolve_vars(text: str, loc: dict[str, str]) -> str:
    """Resolve $variable$ references and [ShowGodName('key')] in text."""
    # Resolve [ShowGodName('key')] → god name from loc
    text = re.sub(
        r"\[ShowGodName\('(\w+)'\)\]",
        lambda m: loc.get(m.group(1), m.group(1)),
        text,
    )
    # Resolve $key$ references
    def replace_var(m):
        key = m.group(1)
        val = loc.get(key, "")
        if val:
            # Recursively resolve nested $refs$ in the resolved value
            val = re.sub(r"\$(\w+)\$", lambda m2: loc.get(m2.group(1), m2.group(1)), val)
            return val
        return m.group(0)  # Keep unresolved

    text = re.sub(r"\$(\w+)\$", replace_var, text)
    return text


def _strip_aspect_markup(text: str) -> str:
    """Strip Paradox markup but preserve readable text."""
    text = re.sub(r"#[A-Z]\b", "", text)
    text = re.sub(r"#(?:bold|high|low|positive|negative)\b\s*", "", text)
    text = re.sub(r"#T\s*", "", text)
    text = re.sub(r"#!", "", text)
    text = re.sub(r"#italic\s*", "", text)
    text = re.sub(r"#tooltip_subheading\s*", "", text)
    text = re.sub(r"@\w+!", "", text)
    text = re.sub(r"\[Concept\(\s*'[^']*'\s*,\s*'([^']*)'\s*\)\s*\|\w+\]", r"\1", text)
    text = re.sub(r"\[Show\w+\('(\w+)'\)(?:\|\w+)?\]", r"\1", text)
    text = re.sub(r"\[(\w+)\|\w+\]", r"\1", text)
    text = re.sub(r"\[(\w+)\]", r"\1", text)
    text = re.sub(r"\[\w+\.\w[\w.'()| ]*\]", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_aspects() -> tuple[list[dict], dict[str, dict]]:
    """Parse all religious aspect files. Returns (core_list, loc_dict_per_lang)."""
    asp_dir = RAW_DIR / "religious_aspects"
    if not asp_dir.exists():
        print(f"[WARN] {asp_dir} not found")
        return [], {}

    # Parse all aspect files
    all_aspects: dict[str, dict] = {}
    for f in sorted(asp_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_aspects[key] = val

    # Load loc for all languages
    all_loc = load_all_loc("religion")
    all_mod_loc = load_all_loc("modifier_types")
    all_concepts_loc = load_all_loc("game_concepts")

    # Build core (language-independent)
    core_list = []
    all_modifier_keys: set[str] = set()

    for asp_id, props in all_aspects.items():
        entry: dict = {"id": asp_id}

        # religions (list)
        religions = _normalize_religions(props.get("religion"))
        if religions:
            entry["religions"] = religions

        # icon (custom or default to asp_id)
        icon = props.get("icon")
        if icon and isinstance(icon, str):
            entry["icon"] = f"icons/religious_aspects/{icon}.png"
        else:
            entry["icon"] = f"icons/religious_aspects/{asp_id}.png"

        # modifier
        mod = props.get("modifier", {})
        if isinstance(mod, dict) and mod:
            mods = []
            for mk, mv in mod.items():
                all_modifier_keys.add(mk)
                if isinstance(mv, bool):
                    mods.append({"key": mk, "value": 1 if mv else 0, "bool": True})
                elif isinstance(mv, str):
                    mods.append({"key": mk, "value": mv, "scaled": True})
                else:
                    entry_mod: dict = {"key": mk, "value": mv}
                    if mk not in FLAT_KEYS:
                        entry_mod["pct"] = True
                    if mk in INVERSE_GOOD_KEYS:
                        entry_mod["inv"] = True
                    mods.append(entry_mod)
            entry["modifier"] = mods

        # opinions
        op = props.get("opinions", {})
        if isinstance(op, dict) and op:
            opinions = {}
            for k, v in op.items():
                if isinstance(v, (int, float)):
                    opinions[k] = v
            if opinions:
                entry["opinions"] = opinions

        # enabled (exclusion conditions)
        enabled = props.get("enabled")
        if isinstance(enabled, dict):
            excludes = _enabled_to_text(enabled)
            if excludes:
                entry["excludes"] = excludes

        entry["source_file"] = props.get("_source_file", "")
        core_list.append(entry)

    # Build per-language loc
    loc_per_lang: dict[str, dict] = {}

    for url_code, loc_data in all_loc.items():
        mod_loc = all_mod_loc.get(url_code, {})
        lang_loc: dict = {}

        for asp in core_list:
            asp_id = asp["id"]
            raw_name = loc_data.get(asp_id, "")
            raw_desc = loc_data.get(f"{asp_id}_desc", "")

            # Resolve $var$ and [ShowGodName] in both name and desc
            name = _strip_aspect_markup(_resolve_vars(raw_name, loc_data))
            desc = _strip_aspect_markup(_resolve_vars(raw_desc, loc_data))

            loc_entry: dict = {}
            if name:
                loc_entry["name"] = name
            if desc:
                loc_entry["desc"] = desc
            if loc_entry:
                lang_loc[asp_id] = loc_entry

        # Build concept resolver (reuse pattern from extract_religions)
        concepts = all_concepts_loc.get(url_code, {})
        estate_loc = load_all_loc("estate").get(url_code, {})
        from extract_religions import build_concept_resolver, _resolve_var_refs, _clean_desc
        extra_names = {ek: ev for ek, ev in estate_loc.items() if ev and not ek.endswith("_desc")}
        resolve = build_concept_resolver(concepts, extra_names)

        # Modifier name + desc localizations
        mod_names = {}
        mod_descs = {}
        for mk in sorted(all_modifier_keys):
            loc_name = mod_loc.get(f"MODIFIER_TYPE_NAME_{mk}", "")
            if loc_name:
                resolved_name = strip_markup(_resolve_var_refs(loc_name, mod_loc, concepts, estate_loc))
                if resolved_name:
                    mod_names[mk] = resolved_name
            raw_desc = mod_loc.get(f"MODIFIER_TYPE_DESC_{mk}", "")
            if raw_desc:
                resolved_raw = _resolve_var_refs(raw_desc, mod_loc, concepts, estate_loc)
                cleaned = resolve(_clean_desc(resolved_raw))
                if cleaned:
                    mod_descs[mk] = cleaned
        if mod_names:
            lang_loc["_modifier_names"] = mod_names
        if mod_descs:
            lang_loc["_modifier_descs"] = mod_descs

        loc_per_lang[url_code] = lang_loc

    return core_list, loc_per_lang


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    core_list, loc_per_lang = build_aspects()

    # Core
    core_path = OUTPUT_DIR / "core" / "aspects.json"
    write_json(core_path, core_list)
    print(f"Wrote {len(core_list)} aspects to {core_path}")

    # Stats
    with_mod = sum(1 for a in core_list if "modifier" in a)
    with_op = sum(1 for a in core_list if "opinions" in a)
    with_excl = sum(1 for a in core_list if "excludes" in a)
    rel_counts = [len(a.get("religions", [])) for a in core_list]
    print(f"  {with_mod} with modifiers, {with_op} with opinions, {with_excl} with exclusions")
    print(f"  religions per aspect: min={min(rel_counts)}, max={max(rel_counts)}, avg={sum(rel_counts)/len(rel_counts):.1f}")

    # Per-language loc
    for url_code, loc_data in loc_per_lang.items():
        loc_path = OUTPUT_DIR / "loc" / url_code / "aspects.json"
        write_json(loc_path, loc_data)
    lang_count = sum(1 for v in loc_per_lang.values() if v)
    print(f"Wrote loc for {lang_count} languages")

    # Modifier name coverage
    en_mod_names = loc_per_lang.get("en", {}).get("_modifier_names", {})
    all_mods = set()
    for a in core_list:
        for m in a.get("modifier", []):
            all_mods.add(m["key"])
    missing = all_mods - set(en_mod_names.keys())
    print(f"Modifier keys: {len(all_mods)} total, {len(en_mod_names)} localized, {len(missing)} missing")
    if missing:
        for m in sorted(missing):
            print(f"  [WARN] No loc for modifier: {m}")

    # Check for unresolved $var$ in English names/descs
    en_loc = loc_per_lang.get("en", {})
    unresolved = []
    for asp_id, entry in en_loc.items():
        if asp_id.startswith("_"):
            continue
        for field in ("name", "desc"):
            val = entry.get(field, "")
            refs = re.findall(r"\$\w+\$", val)
            if refs:
                unresolved.append((asp_id, field, refs))
    if unresolved:
        print(f"\n[WARN] Unresolved $var$ in English loc:")
        for asp_id, field, refs in unresolved:
            print(f"  {asp_id}.{field}: {refs}")
    else:
        print("No unresolved $var$ in English loc")


if __name__ == "__main__":
    main()
