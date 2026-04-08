/** Language definitions matching pipeline/languages.py */

export const LANGUAGES = {
  de:        { name: 'Deutsch',          hreflang: 'de' },
  en:        { name: 'English',          hreflang: 'en' },
  es:        { name: 'Español',          hreflang: 'es' },
  fr:        { name: 'Français',         hreflang: 'fr' },
  ja:        { name: '日本語',            hreflang: 'ja' },
  ko:        { name: '한국어',            hreflang: 'ko' },
  pl:        { name: 'Polski',           hreflang: 'pl' },
  'pt-br':   { name: 'Português (BR)',   hreflang: 'pt-BR' },
  ru:        { name: 'Русский',          hreflang: 'ru' },
  tr:        { name: 'Türkçe',           hreflang: 'tr' },
  'zh-hans': { name: '简体中文',          hreflang: 'zh-Hans' },
} as const;

export type Lang = keyof typeof LANGUAGES;
export const LANG_CODES = Object.keys(LANGUAGES) as Lang[];
export const DEFAULT_LANG: Lang = 'en';

export function isValidLang(lang: string): lang is Lang {
  return lang in LANGUAGES;
}

export function getHreflang(lang: Lang): string {
  return LANGUAGES[lang].hreflang;
}

/** Build a locale-prefixed path: /{lang}/eu5{path} */
export function localePath(lang: Lang, path: string = '/'): string {
  const p = path.startsWith('/') ? path : `/${path}`;
  return `/${lang}/eu5${p}`;
}

/** Build alternate URLs for all languages (for hreflang tags) */
export function getAlternates(path: string): { lang: Lang; hreflang: string; href: string }[] {
  return LANG_CODES.map((lang) => ({
    lang,
    hreflang: getHreflang(lang),
    href: localePath(lang, path),
  }));
}
