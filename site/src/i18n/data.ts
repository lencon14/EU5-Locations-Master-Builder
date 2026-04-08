/**
 * Data loader — combines core (language-independent) + loc (localized names/descriptions).
 *
 * Uses import.meta.glob so Vite resolves all JSON at build time.
 * Fallback: if a locale file is missing for a given category, English is used.
 */
import { DEFAULT_LANG, type Lang } from './config';

// -- Category type (prevents typos) --
export type Category = 'goods' | 'buildings' | 'countries' | 'religions' | 'governments' | 'laws';

// -- Core data (arrays) --
const coreModules = import.meta.glob<unknown[]>(
  '../data/core/*.json',
  { eager: true, import: 'default' },
);

// -- Localization data (keyed objects: { [id]: { name, desc? } }) --
const locModules = import.meta.glob<Record<string, { name: string; desc?: string }>>(
  '../data/loc/*/*.json',
  { eager: true, import: 'default' },
);

type LocEntry = { name: string; desc?: string };
type LocMap = Record<string, LocEntry>;

// -- Caches (populated on first access, reused across pages in same build) --
const _locCache = new Map<string, LocMap>();
const _termsCache = new Map<string, Record<string, string>>();

/** Load core (language-independent) data for a category */
export function loadCore<T = unknown>(category: Category): T[] {
  return (coreModules[`../data/core/${category}.json`] as T[] | undefined) ?? [];
}

/** Load localization map for a category + language, with English fallback */
export function loadLoc(category: Category, lang: Lang): LocMap {
  const cacheKey = `${lang}:${category}`;
  const cached = _locCache.get(cacheKey);
  if (cached) return cached;

  const primary = locModules[`../data/loc/${lang}/${category}.json`];
  if (lang === DEFAULT_LANG) {
    const result = primary ?? {};
    _locCache.set(cacheKey, result);
    return result;
  }

  const fallback = locModules[`../data/loc/${DEFAULT_LANG}/${category}.json`] ?? {};
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
