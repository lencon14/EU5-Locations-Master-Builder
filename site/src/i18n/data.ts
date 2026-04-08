/**
 * Data loader — combines core (language-independent) + loc (localized names/descriptions).
 *
 * Uses import.meta.glob so Vite resolves all JSON at build time.
 * Fallback: if a locale file is missing for a given category, English is used.
 * Fallback events are tracked and reported at build time via reportFallbacks().
 */
import { DEFAULT_LANG, type Lang } from './config';

// -- Fallback tracking (build-time diagnostics) --
const _fallbacks: { type: string; lang: string; key: string }[] = [];

function trackFallback(type: string, lang: string, key: string): void {
  _fallbacks.push({ type, lang, key });
}

/** Call at end of build to report fallback/missing translation stats */
export function reportFallbacks(): void {
  if (_fallbacks.length === 0) return;
  const byLang: Record<string, number> = {};
  for (const f of _fallbacks) {
    byLang[f.lang] = (byLang[f.lang] || 0) + 1;
  }
  console.warn(`[i18n] ${_fallbacks.length} fallback(s) to English:`);
  for (const [lang, count] of Object.entries(byLang).sort((a, b) => b[1] - a[1])) {
    console.warn(`  ${lang}: ${count}`);
  }
  // Log first 10 unique missing keys for debugging
  const unique = [...new Set(_fallbacks.map(f => `${f.lang}:${f.type}:${f.key}`))];
  if (unique.length > 0) {
    console.warn(`  Sample keys: ${unique.slice(0, 10).join(', ')}`);
  }
}

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
  if (!primary) {
    trackFallback('loc', lang, `${category}/*`);
    return fallback;
  }

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
  const terms = loadGameTerms(lang);
  if (!(key in terms)) {
    trackFallback('gameTerm', lang, key);
  }
  return terms[key] ?? key;
}
