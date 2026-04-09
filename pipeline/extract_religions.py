"""Extract EU5 religion data into core + per-language loc files.

Usage:
    python pipeline/extract_religions.py

Output:
    pipeline/output/core/religions.json
    pipeline/output/loc/{lang}/religions.json
    pipeline/output/religions.json               — legacy merged format
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

SKIP_FILES = {"readme.txt"}

# Modifier display format: keys in FLAT_KEYS are shown as-is (+1, +0.1),
# everything else is percentage (×100, shown as +5%).
FLAT_KEYS = {
    # Tolerance (flat integer-like values)
    "tolerance_heretic", "tolerance_heathen", "tolerance_own",
    # Caps and maximums
    "global_max_literacy", "global_clergy_max_literacy",
    "maximum_religious_influence",
    # Counts
    "number_of_allowed_religious_figures", "retreat_delay",
    "global_hostile_attrition",
    # Monthly values (small flat changes per month)
    "monthly_karma", "monthly_legitimacy", "monthly_prestige",
    "monthly_war_exhaustion", "monthly_religious_influence",
    "global_monthly_literacy", "global_monthly_art_start_chance",
    "global_monthly_development", "global_monthly_food_modifier",
    # Attrition costs (very small flat)
    "land_morale_attrition_cost", "naval_morale_attrition_cost",
    "land_unit_attrition",
    # Misc flat
    "skill_of_new_artists", "prestige_from_land_battle",
    "clergy_estate_max_tax",
    # Estate power (flat change to estate influence weight)
    "global_burghers_estate_power", "global_clergy_estate_power",
    "global_nobles_estate_power", "global_peasants_estate_power",
}

# Modifiers where negative value is beneficial (cost/consumption reductions)
INVERSE_GOOD_KEYS = {
    "army_auxiliary_maintenance_cost_modifier", "army_reinforce_cost",
    "expand_rgo_farming_cost_modifier", "expand_rgo_forestry_cost_modifier",
    "exploration_maintenance_cost", "food_consumption_modifier",
    "global_build_buildings_cost", "global_pop_food_consumption",
    "land_morale_attrition_cost", "naval_morale_attrition_cost",
    "land_unit_attrition", "merchant_maintenance_cost",
    "monthly_war_exhaustion", "navy_maintenance_cost",
    "stability_cost", "trade_sea_movement_cost_modifier",
    "global_separatism", "global_monthly_art_start_chance",
}

# Boolean mechanic flags to extract from religion definitions
MECHANIC_FLAGS = [
    "has_karma", "has_purity", "has_honor", "has_yanantin",
    "has_religious_influence", "has_canonization",
    "has_autocephalous_patriarchates", "has_patriarchs",
    "has_religious_head", "has_cardinals", "has_rite_power",
    "has_avatars", "use_icons", "needs_reform",
    "allow_mysticism_vs_jurisprudence",
    "culture_locked",
]

# Map mechanic flags → game_concept loc keys
MECHANIC_CONCEPT_KEYS = {
    "has_karma": "game_concept_karma",
    "has_purity": "game_concept_purity",
    "has_honor": "game_concept_honor",
    "has_yanantin": "game_concept_yanantin",
    "has_religious_influence": "game_concept_religious_influence",
    "has_canonization": "game_concept_canonization",
    "has_autocephalous_patriarchates": "game_concept_autocephalous_patriarchates",
    "has_patriarchs": "game_concept_patriarchs",
    "has_religious_head": "game_concept_religious_head",
    "has_cardinals": "game_concept_cardinals",
    "has_rite_power": "game_concept_rite_power",
    "has_avatars": "game_concept_avatars",
    "use_icons": "game_concept_use_icons",
    "needs_reform": "game_concept_reform_desire",
    "allow_mysticism_vs_jurisprudence": "game_concept_mysticism_vs_jurisprudence",
}

# Mechanic flags with no game_concept entry — manual loc
MECHANIC_MANUAL_NAMES: dict[str, dict[str, str]] = {
    "use_icons": {"en": "Icon Painting", "ja": "イコン（聖像画）"},
    "culture_locked": {"en": "Culture Locked", "ja": "文化固定"},
}

# Additional loc sources to resolve $key$ refs in descriptions
# These keys appear in descriptions but aren't in religion/game_terms loc.
# Keys are looked up across: culture_groups, country_names, advances, area,
# laws_and_policies, plus a manual fallback for runtime-generated names.
EXTRA_LOC_SOURCES = [
    "culture_groups", "country_names", "advances", "area", "laws_and_policies",
    "cultures", "cultural_and_languages",
]

# Runtime-generated names not in any loc file (ShowRegionName etc.)
# English values used as base; other languages override via their own entries.
RUNTIME_NAMES: dict[str, dict[str, str]] = {
    "japan_region": {"en": "Japan", "ja": "日本", "ko": "일본", "zh-hans": "日本",
                     "de": "Japan", "fr": "Japon", "es": "Japón", "pt-br": "Japão",
                     "ru": "Япония", "pl": "Japonia", "tr": "Japonya"},
    "japanese_group": {"en": "Japanese", "ja": "日本"},
    "chinese_group": {"en": "Chinese", "ja": "中国"},
    "asia": {"en": "Asia", "ja": "アジア", "ko": "아시아", "zh-hans": "亚洲",
             "de": "Asien", "fr": "Asie", "es": "Asia", "pt-br": "Ásia",
             "ru": "Азия", "pl": "Azja", "tr": "Asya"},
    "europe": {"en": "Europe", "ja": "ヨーロッパ", "ko": "유럽", "zh-hans": "欧洲",
               "de": "Europa", "fr": "Europe", "es": "Europa", "pt-br": "Europa",
               "ru": "Европа", "pl": "Europa", "tr": "Avrupa"},
    "china": {"en": "China", "ja": "中国", "ko": "중국", "zh-hans": "中国",
              "de": "China", "fr": "Chine", "es": "China", "pt-br": "China",
              "ru": "Китай", "pl": "Chiny", "tr": "Çin"},
    "india": {"en": "India", "ja": "インド", "ko": "인도", "zh-hans": "印度",
              "de": "Indien", "fr": "Inde", "es": "India", "pt-br": "Índia",
              "ru": "Индия", "pl": "Indie", "tr": "Hindistan"},
    "tibet_region": {"en": "Tibet", "ja": "チベット"},
    "andes_region": {"en": "the Andes", "ja": "アンデス"},
    "russian_region": {"en": "Russia", "ja": "ロシア"},
    "scandinavian_region": {"en": "Scandinavia", "ja": "スカンジナビア"},
    "bodhisattva": {"en": "Bodhisattva", "ja": "菩薩"},
    "sutra": {"en": "Sūtra", "ja": "経典"},
    "dharma": {"en": "Dharma", "ja": "ダルマ"},
    "SECT_NYIGMA_NAME": {"en": "Nyingma", "ja": "ニンマ派"},
    "confucianism_policy": {"en": "Confucianism", "ja": "儒教", "ko": "유교",
                            "zh-hans": "儒学", "de": "Konfuzianismus", "fr": "Confucianisme",
                            "es": "Confucianismo", "pt-br": "Confucionismo",
                            "ru": "Конфуцианство", "pl": "Konfucjanizm", "tr": "Konfüçyüsçülük"},
    "daoism_policy": {"en": "Daoism", "ja": "道教", "ko": "도교",
                      "zh-hans": "道教", "de": "Daoismus", "fr": "Taoïsme",
                      "es": "Taoísmo", "pt-br": "Taoísmo",
                      "ru": "Даосизм", "pl": "Taoizm", "tr": "Taoizm"},
    "finnish_language": {"en": "Finnish", "ja": "フィンランド語"},
    "germanic_language_family": {"en": "Germanic", "ja": "ゲルマン語"},
    "greek_culture": {"en": "Greek", "ja": "ギリシャ"},
    "samaritan_culture": {"en": "Samaritan", "ja": "サマリア人"},
    "imamate": {"en": "Imamate", "ja": "イマーム制"},
    "character_title_guru": {"en": "Guru", "ja": "グル"},
    "rank_empire_muslim_ruler_male": {"en": "Caliph", "ja": "カリフ"},
    "sapmi": {"en": "Sápmi", "ja": "サーミ"},
}

# Opinion level display order
OPINION_LEVELS = {"kindred": 2, "positive": 1, "negative": -1, "enemy": -2}


def _resolve_var_refs(raw: str, *lookups: dict[str, str]) -> str:
    """Resolve $variable$ references by chaining through lookup dicts.

    Follows one level of indirection: $MODIFIER_TYPE_DESC_X$ → look up that key.
    """
    def repl(m: re.Match) -> str:
        var = m.group(1)
        for d in lookups:
            if var in d:
                return d[var]
        return ""
    return re.sub(r"\$(\w+)\$", repl, raw)


def _clean_desc(text: str) -> str:
    """Post-process description text: fix adjacent concept refs, strip residual markup."""
    # Add space between adjacent [x|e][y|e] before they get concatenated
    text = re.sub(r"\](\[)", r"] \1", text)
    text = strip_markup(text)
    # Strip any remaining [...] patterns that strip_markup missed
    text = re.sub(r"\[[^\]]*\]", "", text)
    # Clean broken sentence endings from stripped dynamic refs (e.g. ": ." or ": ,")
    text = re.sub(r"[：:]\s*[.,。]", "。", text)
    return re.sub(r"\s+", " ", text).strip()


def build_concept_resolver(
    concepts_data: dict[str, str],
    extra_names: dict[str, str] | None = None,
):
    """Build a function that replaces English concept words with localized names.

    Compiles all concept words into a single regex for O(1) per-string resolution
    instead of O(n_concepts) per-string with individual regexes.

    extra_names: additional word → localized mappings (e.g. religion names, goods names).
    """
    cmap: dict[str, str] = {}
    for k, v in concepts_data.items():
        if not k.startswith("game_concept_"):
            continue
        if k.endswith(("_desc", "_i", "_s")):
            continue
        word = k[len("game_concept_"):]
        localized = strip_markup(v)
        if not localized:
            continue
        cmap[word] = localized
        if "_" in word:
            cmap[word.replace("_", " ")] = localized

    # Merge extra names (religion names, goods names, etc.)
    if extra_names:
        for word, localized in extra_names.items():
            if localized and word not in cmap:
                cmap[word] = localized

    if not cmap:
        return lambda text: text

    # Build single alternation regex, longest-first for proper matching
    sorted_words = sorted(cmap.keys(), key=len, reverse=True)
    alt = "|".join(re.escape(w) for w in sorted_words)
    compiled = re.compile(
        r"(?<![a-zA-Z_])(" + alt + r")s?(?![a-zA-Z_])",
        re.IGNORECASE,
    )
    lower_map = {k.lower(): v for k, v in cmap.items()}

    def resolve(text: str) -> str:
        if not text:
            return text
        result = compiled.sub(lambda m: lower_map.get(m.group(1).lower(), m.group(0)), text)
        return re.sub(r"\s+", " ", result).strip()

    return resolve


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


def build_religions() -> tuple[list[dict], dict[str, dict]]:
    """Parse religion files. Returns (core_list, loc_dict_per_lang)."""
    # Religion groups
    groups: dict[str, dict] = {}
    grp_dir = RAW_DIR / "religion_groups"
    if grp_dir.exists():
        for f in sorted(grp_dir.glob("*.txt")):
            if f.name.lower() in SKIP_FILES:
                continue
            data = parse_file(f)
            for key, val in data.items():
                if isinstance(val, dict):
                    groups[key] = val

    # Religions
    rel_dir = RAW_DIR / "religions"
    if not rel_dir.exists():
        print(f"[WARN] {rel_dir} not found, skipping")
        return [], {}

    all_religions: dict[str, dict] = {}
    for f in sorted(rel_dir.glob("*.txt")):
        if f.name.lower() in SKIP_FILES:
            continue
        data = parse_file(f)
        for key, val in data.items():
            if isinstance(val, dict):
                val["_source_file"] = f.name
                all_religions[key] = val

    # Build group lookup
    religion_to_group: dict[str, str] = {}
    for rel_id, rel_data in all_religions.items():
        if isinstance(rel_data, dict) and "group" in rel_data:
            religion_to_group[rel_id] = rel_data["group"]

    # Load all language loc
    all_loc = load_all_loc("religion")
    all_mod_loc = load_all_loc("modifier_types")
    all_concepts_loc = load_all_loc("game_concepts")
    all_estate_loc = load_all_loc("estate")
    # Extra loc sources for $key$ resolution in descriptions
    extra_locs: dict[str, dict[str, str]] = {}
    for src in EXTRA_LOC_SOURCES:
        for url_code, data in load_all_loc(src).items():
            if url_code not in extra_locs:
                extra_locs[url_code] = {}
            extra_locs[url_code].update(data)

    # Build core (language-independent)
    core_list = []
    all_modifier_keys: set[str] = set()
    for rel_id, props in all_religions.items():
        entry: dict = {"id": rel_id}

        if rel_id in religion_to_group:
            entry["group_id"] = religion_to_group[rel_id]

        # definition_modifier → modifier[]
        dm = props.get("definition_modifier", {})
        if isinstance(dm, dict) and dm:
            mods = []
            for mk, mv in dm.items():
                all_modifier_keys.add(mk)
                # Normalize: bool → 1/0, str stays str (scaling variables)
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
            entry["opinions"] = {k: v for k, v in op.items() if v in OPINION_LEVELS}

        # religious_aspects (int)
        ra = props.get("religious_aspects")
        if ra is not None:
            entry["religious_aspects"] = int(ra) if not isinstance(ra, int) else ra

        # max_sects
        ms = props.get("max_sects")
        if ms is not None:
            entry["max_sects"] = int(ms) if not isinstance(ms, int) else ms

        # enable date
        enable = props.get("enable")
        if enable is not None:
            entry["enable"] = str(enable)

        # language
        lang_val = props.get("language")
        if lang_val:
            entry["language"] = str(lang_val)

        # mechanic flags
        flags = []
        for flag in MECHANIC_FLAGS:
            if props.get(flag):
                flags.append(flag)
        if flags:
            entry["mechanics"] = flags

        entry["icon"] = f"icons/religion/{rel_id}.png"
        entry["source_file"] = props.get("_source_file", "")
        core_list.append(entry)

    # Build per-language loc (includes religion names, descs, group names, modifier names)
    rel_ids = [r["id"] for r in core_list]
    group_ids = set(religion_to_group.values())

    # Collect all liturgical language keys for loc extraction
    all_languages = {r.get("language") for r in core_list if r.get("language")}

    # Collect $key$ references from English descriptions to extract shared keys
    en_loc = all_loc.get("en", {})
    shared_refs: set[str] = set()
    for rel_id in rel_ids:
        raw_desc = en_loc.get(f"{rel_id}_desc", "")
        shared_refs.update(re.findall(r"\$(\w+)\$", raw_desc))
    all_known = set(rel_ids) | group_ids
    shared_keys = {k for k in shared_refs if k not in all_known}

    loc_per_lang: dict[str, dict] = {}
    for url_code, loc_data in all_loc.items():
        mod_loc = all_mod_loc.get(url_code, {})
        lang_loc = {}

        for rel_id in rel_ids:
            name = loc_data.get(rel_id, "")
            raw_desc = loc_data.get(f"{rel_id}_desc", "")
            desc = _strip_markup_keep_refs(raw_desc)
            loc_entry: dict = {}
            if name:
                loc_entry["name"] = name
            if desc:
                loc_entry["desc"] = desc
            if loc_entry:
                lang_loc[rel_id] = loc_entry

        # Group names
        for grp_id in group_ids:
            name = loc_data.get(grp_id, "")
            if name:
                lang_loc[grp_id] = {"name": name}

        # Shared keys referenced by $...$ in descriptions
        # Resolve from: religion loc → extra locs → runtime fallback
        extra = extra_locs.get(url_code, {})
        for key in shared_keys:
            if key in lang_loc:
                continue
            val = loc_data.get(key, "") or extra.get(key, "")
            # Strip $..$ from resolved values (e.g. "$SECT_CONFUCIANISM_NAME$")
            if val:
                val = strip_markup(re.sub(r"\$\w+\$", "", val).strip())
            if val:
                lang_loc[key] = {"name": val}
            elif key in RUNTIME_NAMES:
                rn = RUNTIME_NAMES[key]
                lang_loc[key] = {"name": rn.get(url_code, rn.get("en", key))}

        # Build concept resolver with religion names + goods names + estate names
        concepts = all_concepts_loc.get(url_code, {})
        estate_loc = all_estate_loc.get(url_code, {})
        goods_loc = load_all_loc("goods").get(url_code, {})
        extra_names: dict[str, str] = {}
        # Add religion names (shinto → 神道, catholic → カトリック, etc.)
        for rel_id in rel_ids:
            name = loc_data.get(rel_id, "")
            if name:
                extra_names[rel_id] = name
        for grp_id in group_ids:
            name = loc_data.get(grp_id, "")
            if name:
                extra_names[grp_id] = name
        # Add goods names (beer → ビール, wine → ワイン, etc.)
        for gk, gv in goods_loc.items():
            if gv and not gk.endswith("_desc"):
                extra_names[gk] = gv
        # Add estate names (clergy_estate → 聖職者, etc.)
        for ek, ev in estate_loc.items():
            if ev and not ek.endswith("_desc"):
                extra_names[ek] = ev
        resolve = build_concept_resolver(concepts, extra_names)

        # Modifier name localizations
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
                # Resolve $variable$ references first (e.g. $game_concept_X_desc$)
                resolved_raw = _resolve_var_refs(raw_desc, mod_loc, concepts, estate_loc)
                cleaned = resolve(_clean_desc(resolved_raw))
                if cleaned:
                    mod_descs[mk] = cleaned
        if mod_names:
            lang_loc["_modifier_names"] = mod_names
        if mod_descs:
            lang_loc["_modifier_descs"] = mod_descs

        # Mechanic flag names from game_concepts + manual fallback
        mech_names = {}
        mech_descs = {}
        for flag, concept_key in MECHANIC_CONCEPT_KEYS.items():
            val = concepts.get(concept_key, "")
            if val:
                mech_names[flag] = strip_markup(val)
            desc_val = concepts.get(f"{concept_key}_desc", "")
            if desc_val:
                mech_descs[flag] = resolve(_clean_desc(_resolve_var_refs(desc_val, mod_loc, concepts, estate_loc)))
        for flag, names in MECHANIC_MANUAL_NAMES.items():
            if flag not in mech_names:
                mech_names[flag] = names.get(url_code, names.get("en", flag))
        if mech_names:
            lang_loc["_mechanic_names"] = mech_names
        if mech_descs:
            lang_loc["_mechanic_descs"] = mech_descs

        # Liturgical language names from cultures/cultural_and_languages loc
        lang_names = {}
        for lang_key in all_languages:
            val = extra.get(lang_key, "")
            if val:
                lang_names[lang_key] = strip_markup(val)
        if lang_names:
            lang_loc["_language_names"] = lang_names

        loc_per_lang[url_code] = lang_loc

    return core_list, loc_per_lang


def _strip_markup_keep_refs(text: str) -> str:
    """Strip Paradox markup but preserve $concept$ references for frontend resolution."""
    text = re.sub(r"#[A-Z]\b", "", text)
    text = re.sub(r"#(?:bold|high|low|positive|negative)\b\s*", "", text)
    text = re.sub(r"#T\s*", "", text)
    text = re.sub(r"#!", "", text)
    text = re.sub(r"#italic\s*", "", text)
    text = re.sub(r"#tooltip_subheading\s*", "", text)
    text = re.sub(r"@\w+!", "", text)
    # Preserve $VARIABLE$ — do NOT strip them
    text = re.sub(r"\[Concept\(\s*'[^']*'\s*,\s*'([^']*)'\s*\)\s*\|\w+\]", r"\1", text)
    text = re.sub(r"\[Show\w+\('(\w+)'\)(?:\|\w+)?\]", r"\1", text)
    text = re.sub(r"\[(\w+)\|\w+\]", r"\1", text)
    text = re.sub(r"\[(\w+)\]", r"\1", text)
    text = re.sub(r"\[\w+\.\w[\w.'()| ]*\]", "", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    core_list, loc_per_lang = build_religions()

    # Core
    core_path = OUTPUT_DIR / "core" / "religions.json"
    write_json(core_path, core_list)
    print(f"Wrote {len(core_list)} religions to {core_path}")

    # Stats
    with_mod = sum(1 for r in core_list if "modifier" in r)
    with_op = sum(1 for r in core_list if "opinions" in r)
    with_mech = sum(1 for r in core_list if "mechanics" in r)
    print(f"  {with_mod} with modifiers, {with_op} with opinions, {with_mech} with mechanics")

    # Per-language loc
    for url_code, loc_data in loc_per_lang.items():
        loc_path = OUTPUT_DIR / "loc" / url_code / "religions.json"
        write_json(loc_path, loc_data)
    lang_count = sum(1 for v in loc_per_lang.values() if v)
    print(f"Wrote loc for {lang_count} languages")

    # Modifier name coverage
    en_mod_names = loc_per_lang.get("en", {}).get("_modifier_names", {})
    all_mods = set()
    for r in core_list:
        for m in r.get("modifier", []):
            all_mods.add(m["key"])
    missing = all_mods - set(en_mod_names.keys())
    print(f"Modifier keys: {len(all_mods)} total, {len(en_mod_names)} localized, {len(missing)} missing")
    if missing:
        for m in sorted(missing):
            print(f"  [WARN] No loc for modifier: {m}")

    # Legacy merged format
    en_loc = loc_per_lang.get("en", {})
    ja_loc = loc_per_lang.get("ja", {})
    legacy = []
    for item in core_list:
        merged = dict(item)
        rid = item["id"]
        merged["name_en"] = en_loc.get(rid, {}).get("name", rid)
        merged["name_ja"] = ja_loc.get(rid, {}).get("name", rid)
        merged["desc_en"] = en_loc.get(rid, {}).get("desc", "")
        merged["desc_ja"] = ja_loc.get(rid, {}).get("desc", "")
        if "group_id" in item:
            grp = item["group_id"]
            merged["group_en"] = en_loc.get(grp, {}).get("name", grp)
            merged["group_ja"] = ja_loc.get(grp, {}).get("name", grp)
        legacy.append(merged)
    legacy_path = OUTPUT_DIR / "religions.json"
    write_json(legacy_path, legacy)
    print(f"Wrote legacy merged format to {legacy_path}")


if __name__ == "__main__":
    main()
