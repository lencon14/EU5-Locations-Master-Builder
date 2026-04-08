/**
 * Data loader — combines core (language-independent) + loc (localized names/descriptions).
 *
 * Uses import.meta.glob so Vite resolves all JSON at build time.
 * Fallback: if a locale file is missing for a given category, English is used.
 */
import { DEFAULT_LANG, type Lang } from './config';

// -- Core data (arrays) --
const coreModules = import.meta.glob<any[]>(
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

/** Load core (language-independent) data for a category */
export function loadCore(category: string): any[] {
  return coreModules[`../data/core/${category}.json`] ?? [];
}

/** Load localization map for a category + language, with English fallback */
export function loadLoc(category: string, lang: Lang): LocMap {
  const primary = locModules[`../data/loc/${lang}/${category}.json`];
  if (lang === DEFAULT_LANG) return primary ?? {};

  const fallback = locModules[`../data/loc/${DEFAULT_LANG}/${category}.json`] ?? {};
  if (!primary) return fallback;

  // Merge: use primary where available, fall back to English per-field
  const merged: LocMap = {};
  for (const id of Object.keys(fallback)) {
    const p = primary[id];
    const f = fallback[id];
    merged[id] = {
      name: p?.name || f?.name || id,
      desc: p?.desc || f?.desc,
    };
  }
  // Include items only in primary (shouldn't happen, but safe)
  for (const id of Object.keys(primary)) {
    if (!(id in merged)) merged[id] = primary[id];
  }
  return merged;
}

/** Convenience: get a single item's localized name */
export function getName(category: string, id: string, lang: Lang): string {
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
  const primary = termModules[`../data/loc/${lang}/game_terms.json`] ?? {};
  if (lang === DEFAULT_LANG) return primary;
  const fallback = termModules[`../data/loc/${DEFAULT_LANG}/game_terms.json`] ?? {};
  return { ...fallback, ...primary };
}

/** Get a single game term */
export function gameTerm(lang: Lang, key: string): string {
  return loadGameTerms(lang)[key] ?? key;
}
