/**
 * Data loader — combines core (language-independent) + loc (localized names/descriptions).
 *
 * Uses import.meta.glob so Vite resolves all JSON at build time.
 * Fallback: if a locale file is missing for a given category, English is used.
 */
import { DEFAULT_LANG, type Lang } from './config';

// -- Category type (prevents typos in loadCore) --
// Note: countries/buildings/religions/holy_sites/aspects loc have special keys
// and are excluded from locModules. Use dedicated loaders instead of loadLoc().
export type Category = 'goods' | 'buildings' | 'countries' | 'religions' | 'governments' | 'laws' | 'holy_sites' | 'holy_site_types' | 'aspects';

// -- Core data (arrays) --
const coreModules = import.meta.glob<unknown[]>(
  '../data/core/*.json',
  { eager: true, import: 'default' },
);

// -- Localization data (keyed objects: { [id]: { name, desc? } }) --
// game_terms.json is excluded here — it has a different shape (flat key-value)
// and is loaded separately via termModules. Do NOT pass 'game_terms' to loadLoc().
const locModules = import.meta.glob<Record<string, { name: string; desc?: string }>>(
  ['../data/loc/*/*.json', '!../data/loc/*/game_terms.json', '!../data/loc/*/buildings.json', '!../data/loc/*/religions.json', '!../data/loc/*/holy_sites.json', '!../data/loc/*/aspects.json', '!../data/loc/*/countries.json'],
  { eager: true, import: 'default' },
);

type LocEntry = { name: string; desc?: string };
type LocMap = Record<string, LocEntry>;

// -- Caches (populated on first access, reused across pages in same build) --
const _locCache = new Map<string, LocMap>();
const _termsCache = new Map<string, Record<string, string>>();

/** Load core (language-independent) data for a category */
export function loadCore<T = unknown>(category: Category): T[] {
  const data = (coreModules[`../data/core/${category}.json`] as T[] | undefined) ?? [];
  if (data.length === 0) {
    console.warn(`[i18n] loadCore: no data found for category "${category}"`);
  }
  return data;
}

/** Load localization map for a category + language, with English fallback.
 *  Note: 'game_terms' is NOT a valid Category. game_terms.json is excluded from
 *  locModules (different shape) and loaded via loadGameTerms() / termModules instead. */
export function loadLoc(category: Category, lang: Lang): LocMap {
  const cacheKey = `${lang}:${category}`;
  const cached = _locCache.get(cacheKey);
  if (cached) return cached;

  const primary = locModules[`../data/loc/${lang}/${category}.json`];
  if (lang === DEFAULT_LANG) {
    const result = primary ?? {};
    if (!primary) {
      console.warn(`[i18n] loadLoc: English data missing for category "${category}"`);
    }
    _locCache.set(cacheKey, result);
    return result;
  }

  const fallback = locModules[`../data/loc/${DEFAULT_LANG}/${category}.json`] ?? {};
  if (!locModules[`../data/loc/${DEFAULT_LANG}/${category}.json`]) {
    console.warn(`[i18n] loadLoc: English fallback data missing for category "${category}"`);
  }
  if (!primary) {
    _locCache.set(cacheKey, fallback);
    return fallback;
  }

  // Merge: use primary where available, fall back to English per-field
  const merged: LocMap = {};
  for (const id of Object.keys(fallback)) {
    const p = primary[id];
    const f = fallback[id];
    merged[id] = {
      name: p?.name ?? f?.name ?? id,
      desc: p?.desc ?? f?.desc,
    };
  }
  for (const id of Object.keys(primary)) {
    if (!(id in merged)) merged[id] = primary[id];
  }
  _locCache.set(cacheKey, merged);
  return merged;
}

/** Convenience: get a single item's localized name */
export function getName(category: Category, id: string, lang: Lang): string {
  const loc = loadLoc(category, lang);
  return loc[id]?.name ?? id;
}

// -- Game terms (flat key-value: { "pop.nobles": "Nobles", ... }) --
const termModules = import.meta.glob<Record<string, string>>(
  '../data/loc/*/game_terms.json',
  { eager: true, import: 'default' },
);

/** Load game terms for a language, with English fallback */
export function loadGameTerms(lang: Lang): Record<string, string> {
  const cached = _termsCache.get(lang);
  if (cached) return cached;

  const primary = termModules[`../data/loc/${lang}/game_terms.json`] ?? {};
  let result: Record<string, string>;
  if (lang === DEFAULT_LANG) {
    result = primary;
  } else {
    const fallback = termModules[`../data/loc/${DEFAULT_LANG}/game_terms.json`] ?? {};
    result = { ...fallback, ...primary };
  }
  _termsCache.set(lang, result);
  return result;
}

/** Get a single game term */
export function gameTerm(lang: Lang, key: string): string {
  const terms = loadGameTerms(lang);
  return terms[key] ?? key;
}

// -- Official loc (flat key-value: all game loc merged) --
// TODO: Phase 1 of translation leak prevention. Currently blocked by Vite SSR
// bundling issue with node:fs and OOM with eager import.meta.glob for 11×4MB.
// For now, individual _*_names auxiliary maps in category loc files are used.
// Future: resolve Vite/Astro compatibility for large JSON lazy loading.

// -- Religion loc (excluded from locModules due to _modifier_names/_mechanic_names special keys) --
type ModNameMap = Record<string, string>;
const _relLocCache = new Map<string, LocMap>();
const _relModCache = new Map<string, ModNameMap>();

const relRawModules = import.meta.glob<Record<string, unknown>>(
  '../data/loc/*/religions.json',
  { eager: true, import: 'default' },
);

/** Load religion loc, with English fallback. Handles _modifier_names exclusion. */
export function loadReligionLoc(lang: Lang): LocMap {
  const cached = _relLocCache.get(lang);
  if (cached) return cached;

  const raw = relRawModules[`../data/loc/${lang}/religions.json`] as Record<string, unknown> | undefined;
  const rawEn = relRawModules[`../data/loc/${DEFAULT_LANG}/religions.json`] as Record<string, unknown> | undefined;

  // Filter out special keys (starting with _)
  const toLocMap = (r: Record<string, unknown> | undefined): LocMap => {
    if (!r) return {};
    const map: LocMap = {};
    for (const [k, v] of Object.entries(r)) {
      if (k.startsWith('_') || typeof v !== 'object' || v === null) continue;
      const entry = v as { name?: string; desc?: string };
      if (entry.name != null) map[k] = { name: entry.name, desc: entry.desc };
    }
    return map;
  };

  const primary = toLocMap(raw);
  if (lang === DEFAULT_LANG) {
    _relLocCache.set(lang, primary);
    return primary;
  }
  const fallback = toLocMap(rawEn);
  const merged: LocMap = {};
  for (const id of Object.keys(fallback)) {
    const p = primary[id];
    const f = fallback[id];
    merged[id] = { name: p?.name ?? f?.name ?? id, desc: p?.desc ?? f?.desc };
  }
  for (const id of Object.keys(primary)) {
    if (!(id in merged)) merged[id] = primary[id];
  }
  _relLocCache.set(lang, merged);
  return merged;
}

/** Load religion modifier display names for a language, with English fallback. */
export function loadReligionModNames(lang: Lang): ModNameMap {
  const cached = _relModCache.get(lang);
  if (cached) return cached;

  const primary = (relRawModules[`../data/loc/${lang}/religions.json`] as Record<string, unknown> | undefined)?.["_modifier_names"] as ModNameMap | undefined;
  const fallback = (relRawModules[`../data/loc/${DEFAULT_LANG}/religions.json`] as Record<string, unknown> | undefined)?.["_modifier_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _relModCache.set(lang, result);
  return result;
}

/** Load religion mechanic flag names (from game_concepts), with English fallback. */
const _relMechCache = new Map<string, ModNameMap>();
export function loadReligionMechNames(lang: Lang): ModNameMap {
  const cached = _relMechCache.get(lang);
  if (cached) return cached;

  const primary = (relRawModules[`../data/loc/${lang}/religions.json`] as Record<string, unknown> | undefined)?.["_mechanic_names"] as ModNameMap | undefined;
  const fallback = (relRawModules[`../data/loc/${DEFAULT_LANG}/religions.json`] as Record<string, unknown> | undefined)?.["_mechanic_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _relMechCache.set(lang, result);
  return result;
}

/** Load religion modifier descriptions, with English fallback. */
const _relModDescCache = new Map<string, ModNameMap>();
export function loadReligionModDescs(lang: Lang): ModNameMap {
  const cached = _relModDescCache.get(lang);
  if (cached) return cached;

  const primary = (relRawModules[`../data/loc/${lang}/religions.json`] as Record<string, unknown> | undefined)?.["_modifier_descs"] as ModNameMap | undefined;
  const fallback = (relRawModules[`../data/loc/${DEFAULT_LANG}/religions.json`] as Record<string, unknown> | undefined)?.["_modifier_descs"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _relModDescCache.set(lang, result);
  return result;
}

/** Load religion mechanic descriptions (from game_concepts), with English fallback. */
const _relMechDescCache = new Map<string, ModNameMap>();
export function loadReligionMechDescs(lang: Lang): ModNameMap {
  const cached = _relMechDescCache.get(lang);
  if (cached) return cached;

  const primary = (relRawModules[`../data/loc/${lang}/religions.json`] as Record<string, unknown> | undefined)?.["_mechanic_descs"] as ModNameMap | undefined;
  const fallback = (relRawModules[`../data/loc/${DEFAULT_LANG}/religions.json`] as Record<string, unknown> | undefined)?.["_mechanic_descs"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _relMechDescCache.set(lang, result);
  return result;
}

// -- Holy sites loc (excluded from locModules due to _modifier_names) --
const hsRawModules = import.meta.glob<Record<string, unknown>>(
  '../data/loc/*/holy_sites.json',
  { eager: true, import: 'default' },
);

const _hsLocCache = new Map<string, LocMap>();

/** Load holy sites loc, with English fallback. */
export function loadHolySiteLoc(lang: Lang): LocMap {
  const cached = _hsLocCache.get(lang);
  if (cached) return cached;

  const raw = hsRawModules[`../data/loc/${lang}/holy_sites.json`] as Record<string, unknown> | undefined;
  const rawEn = hsRawModules[`../data/loc/${DEFAULT_LANG}/holy_sites.json`] as Record<string, unknown> | undefined;

  const toLocMap = (r: Record<string, unknown> | undefined): LocMap => {
    if (!r) return {};
    const map: LocMap = {};
    for (const [k, v] of Object.entries(r)) {
      if (k.startsWith('_') || typeof v !== 'object' || v === null) continue;
      const entry = v as { name?: string; desc?: string };
      if (entry.name != null) map[k] = { name: entry.name, desc: entry.desc };
    }
    return map;
  };

  const primary = toLocMap(raw);
  if (lang === DEFAULT_LANG) {
    _hsLocCache.set(lang, primary);
    return primary;
  }
  const fallback = toLocMap(rawEn);
  const merged: LocMap = {};
  for (const id of Object.keys(fallback)) {
    const p = primary[id];
    const f = fallback[id];
    merged[id] = { name: p?.name ?? f?.name ?? id, desc: p?.desc ?? f?.desc };
  }
  for (const id of Object.keys(primary)) {
    if (!(id in merged)) merged[id] = primary[id];
  }
  _hsLocCache.set(lang, merged);
  return merged;
}

/** Load holy site location (province) display names, with English fallback. */
const _hsLocNameCache = new Map<string, ModNameMap>();
export function loadHolySiteLocationNames(lang: Lang): ModNameMap {
  const cached = _hsLocNameCache.get(lang);
  if (cached) return cached;

  const primary = (hsRawModules[`../data/loc/${lang}/holy_sites.json`] as Record<string, unknown> | undefined)?.["_location_names"] as ModNameMap | undefined;
  const fallback = (hsRawModules[`../data/loc/${DEFAULT_LANG}/holy_sites.json`] as Record<string, unknown> | undefined)?.["_location_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _hsLocNameCache.set(lang, result);
  return result;
}

/** Load holy site modifier display names, with English fallback. */
const _hsModCache = new Map<string, ModNameMap>();
export function loadHolySiteModNames(lang: Lang): ModNameMap {
  const cached = _hsModCache.get(lang);
  if (cached) return cached;

  const primary = (hsRawModules[`../data/loc/${lang}/holy_sites.json`] as Record<string, unknown> | undefined)?.["_modifier_names"] as ModNameMap | undefined;
  const fallback = (hsRawModules[`../data/loc/${DEFAULT_LANG}/holy_sites.json`] as Record<string, unknown> | undefined)?.["_modifier_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _hsModCache.set(lang, result);
  return result;
}

/** Load holy site modifier descriptions, with English fallback. */
const _hsModDescCache = new Map<string, ModNameMap>();
export function loadHolySiteModDescs(lang: Lang): ModNameMap {
  const cached = _hsModDescCache.get(lang);
  if (cached) return cached;

  const primary = (hsRawModules[`../data/loc/${lang}/holy_sites.json`] as Record<string, unknown> | undefined)?.["_modifier_descs"] as ModNameMap | undefined;
  const fallback = (hsRawModules[`../data/loc/${DEFAULT_LANG}/holy_sites.json`] as Record<string, unknown> | undefined)?.["_modifier_descs"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _hsModDescCache.set(lang, result);
  return result;
}

/** Load liturgical language display names, with English fallback. */
const _relLangCache = new Map<string, ModNameMap>();
export function loadReligionLangNames(lang: Lang): ModNameMap {
  const cached = _relLangCache.get(lang);
  if (cached) return cached;

  const primary = (relRawModules[`../data/loc/${lang}/religions.json`] as Record<string, unknown> | undefined)?.["_language_names"] as ModNameMap | undefined;
  const fallback = (relRawModules[`../data/loc/${DEFAULT_LANG}/religions.json`] as Record<string, unknown> | undefined)?.["_language_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _relLangCache.set(lang, result);
  return result;
}

// -- Aspects loc (excluded from locModules due to _modifier_names) --
const aspRawModules = import.meta.glob<Record<string, unknown>>(
  '../data/loc/*/aspects.json',
  { eager: true, import: 'default' },
);

const _aspLocCache = new Map<string, LocMap>();

/** Load aspect loc, with English fallback. */
export function loadAspectLoc(lang: Lang): LocMap {
  const cached = _aspLocCache.get(lang);
  if (cached) return cached;

  const raw = aspRawModules[`../data/loc/${lang}/aspects.json`] as Record<string, unknown> | undefined;
  const rawEn = aspRawModules[`../data/loc/${DEFAULT_LANG}/aspects.json`] as Record<string, unknown> | undefined;

  const toLocMap = (r: Record<string, unknown> | undefined): LocMap => {
    if (!r) return {};
    const map: LocMap = {};
    for (const [k, v] of Object.entries(r)) {
      if (k.startsWith('_') || typeof v !== 'object' || v === null) continue;
      const entry = v as { name?: string; desc?: string };
      if (entry.name != null) map[k] = { name: entry.name, desc: entry.desc };
    }
    return map;
  };

  const primary = toLocMap(raw);
  if (lang === DEFAULT_LANG) {
    _aspLocCache.set(lang, primary);
    return primary;
  }
  const fallback = toLocMap(rawEn);
  const merged: LocMap = {};
  for (const id of Object.keys(fallback)) {
    const p = primary[id];
    const f = fallback[id];
    merged[id] = { name: p?.name ?? f?.name ?? id, desc: p?.desc ?? f?.desc };
  }
  for (const id of Object.keys(primary)) {
    if (!(id in merged)) merged[id] = primary[id];
  }
  _aspLocCache.set(lang, merged);
  return merged;
}

/** Load aspect modifier display names, with English fallback. */
const _aspModCache = new Map<string, ModNameMap>();
export function loadAspectModNames(lang: Lang): ModNameMap {
  const cached = _aspModCache.get(lang);
  if (cached) return cached;

  const primary = (aspRawModules[`../data/loc/${lang}/aspects.json`] as Record<string, unknown> | undefined)?.["_modifier_names"] as ModNameMap | undefined;
  const fallback = (aspRawModules[`../data/loc/${DEFAULT_LANG}/aspects.json`] as Record<string, unknown> | undefined)?.["_modifier_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _aspModCache.set(lang, result);
  return result;
}

/** Load aspect modifier descriptions, with English fallback. */
const _aspModDescCache = new Map<string, ModNameMap>();
export function loadAspectModDescs(lang: Lang): ModNameMap {
  const cached = _aspModDescCache.get(lang);
  if (cached) return cached;

  const primary = (aspRawModules[`../data/loc/${lang}/aspects.json`] as Record<string, unknown> | undefined)?.["_modifier_descs"] as ModNameMap | undefined;
  const fallback = (aspRawModules[`../data/loc/${DEFAULT_LANG}/aspects.json`] as Record<string, unknown> | undefined)?.["_modifier_descs"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _aspModDescCache.set(lang, result);
  return result;
}

// -- Building-specific extended loc (condition_lines, modifiers, pm) --
import type { BuildingLocEntry } from './types';

type BuildingLocMap = Record<string, BuildingLocEntry>;
const _bldLocCache = new Map<string, BuildingLocMap>();

const bldLocModules = import.meta.glob<BuildingLocMap>(
  '../data/loc/*/buildings.json',
  { eager: true, import: 'default' },
);

/** Load extended building loc for a language, with English fallback. */
export function loadBuildingLoc(lang: Lang): BuildingLocMap {
  const cached = _bldLocCache.get(lang);
  if (cached) return cached;

  const primary = bldLocModules[`../data/loc/${lang}/buildings.json`];
  if (lang === DEFAULT_LANG) {
    const result = primary ?? {};
    _bldLocCache.set(lang, result);
    return result;
  }

  const fallback = bldLocModules[`../data/loc/${DEFAULT_LANG}/buildings.json`] ?? {};
  if (!primary) {
    _bldLocCache.set(lang, fallback);
    return fallback;
  }

  const merged: BuildingLocMap = {};
  for (const id of Object.keys(fallback)) {
    const p = primary[id];
    const f = fallback[id];
    merged[id] = {
      name: p?.name ?? f?.name ?? id,
      desc: p?.desc ?? f?.desc,
      condition_lines: p?.condition_lines ?? f?.condition_lines,
      modifiers: p?.modifiers ?? f?.modifiers,
      raw_modifiers: p?.raw_modifiers ?? f?.raw_modifiers,
      pm: p?.pm ?? f?.pm,
    };
  }
  for (const id of Object.keys(primary)) {
    if (!(id in merged)) merged[id] = primary[id];
  }
  _bldLocCache.set(lang, merged);
  return merged;
}

// -- Countries loc (excluded from locModules due to _culture_names/_culture_group_names/_capital_names) --
const ctyRawModules = import.meta.glob<Record<string, unknown>>(
  '../data/loc/*/countries.json',
  { eager: true, import: 'default' },
);

const _ctyLocCache = new Map<string, LocMap>();

/** Load countries loc, with English fallback. */
export function loadCountryLoc(lang: Lang): LocMap {
  const cached = _ctyLocCache.get(lang);
  if (cached) return cached;

  const raw = ctyRawModules[`../data/loc/${lang}/countries.json`] as Record<string, unknown> | undefined;
  const rawEn = ctyRawModules[`../data/loc/${DEFAULT_LANG}/countries.json`] as Record<string, unknown> | undefined;

  const toLocMap = (r: Record<string, unknown> | undefined): LocMap => {
    if (!r) return {};
    const map: LocMap = {};
    for (const [k, v] of Object.entries(r)) {
      if (k.startsWith('_') || typeof v !== 'object' || v === null) continue;
      const entry = v as { name?: string; desc?: string };
      if (entry.name != null) map[k] = { name: entry.name, desc: entry.desc };
    }
    return map;
  };

  const primary = toLocMap(raw);
  if (lang === DEFAULT_LANG) {
    _ctyLocCache.set(lang, primary);
    return primary;
  }
  const fallback = toLocMap(rawEn);
  const merged: LocMap = {};
  for (const id of Object.keys(fallback)) {
    const p = primary[id];
    const f = fallback[id];
    merged[id] = { name: p?.name ?? f?.name ?? id, desc: p?.desc ?? f?.desc };
  }
  for (const id of Object.keys(primary)) {
    if (!(id in merged)) merged[id] = primary[id];
  }
  _ctyLocCache.set(lang, merged);
  return merged;
}

/** Load culture display names, with English fallback. */
const _ctyCultureCache = new Map<string, ModNameMap>();
export function loadCultureNames(lang: Lang): ModNameMap {
  const cached = _ctyCultureCache.get(lang);
  if (cached) return cached;
  const primary = (ctyRawModules[`../data/loc/${lang}/countries.json`] as Record<string, unknown> | undefined)?.["_culture_names"] as ModNameMap | undefined;
  const fallback = (ctyRawModules[`../data/loc/${DEFAULT_LANG}/countries.json`] as Record<string, unknown> | undefined)?.["_culture_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _ctyCultureCache.set(lang, result);
  return result;
}

/** Load culture group display names, with English fallback. */
const _ctyGroupCache = new Map<string, ModNameMap>();
export function loadCultureGroupNames(lang: Lang): ModNameMap {
  const cached = _ctyGroupCache.get(lang);
  if (cached) return cached;
  const primary = (ctyRawModules[`../data/loc/${lang}/countries.json`] as Record<string, unknown> | undefined)?.["_culture_group_names"] as ModNameMap | undefined;
  const fallback = (ctyRawModules[`../data/loc/${DEFAULT_LANG}/countries.json`] as Record<string, unknown> | undefined)?.["_culture_group_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _ctyGroupCache.set(lang, result);
  return result;
}

/** Load government/heir display names (from government + government_names + government_reforms loc). */
const _ctyGovCache = new Map<string, ModNameMap>();
export function loadCountryGovNames(lang: Lang): ModNameMap {
  const cached = _ctyGovCache.get(lang);
  if (cached) return cached;
  const primary = (ctyRawModules[`../data/loc/${lang}/countries.json`] as Record<string, unknown> | undefined)?.["_gov_names"] as ModNameMap | undefined;
  const fallback = (ctyRawModules[`../data/loc/${DEFAULT_LANG}/countries.json`] as Record<string, unknown> | undefined)?.["_gov_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _ctyGovCache.set(lang, result);
  return result;
}

/** Load capital/location display names, with English fallback. */
const _ctyCapitalCache = new Map<string, ModNameMap>();
export function loadCapitalNames(lang: Lang): ModNameMap {
  const cached = _ctyCapitalCache.get(lang);
  if (cached) return cached;
  const primary = (ctyRawModules[`../data/loc/${lang}/countries.json`] as Record<string, unknown> | undefined)?.["_capital_names"] as ModNameMap | undefined;
  const fallback = (ctyRawModules[`../data/loc/${DEFAULT_LANG}/countries.json`] as Record<string, unknown> | undefined)?.["_capital_names"] as ModNameMap | undefined;
  const result = { ...fallback, ...primary };
  _ctyCapitalCache.set(lang, result);
  return result;
}
