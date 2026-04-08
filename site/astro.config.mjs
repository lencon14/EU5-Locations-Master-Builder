// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  site: 'https://paradoxpedia.com',
  i18n: {
    defaultLocale: 'en',
    locales: [
      'de', 'en', 'es', 'fr', 'ja', 'ko', 'pl',
      { path: 'pt-br', codes: ['pt-BR', 'pt'] },
      'ru', 'tr',
      { path: 'zh-hans', codes: ['zh-Hans', 'zh'] },
    ],
    routing: 'manual',
  },
});
