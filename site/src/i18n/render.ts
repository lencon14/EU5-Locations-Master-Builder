/**
 * Shared render functions for displaying localized game data.
 *
 * Used across religion, holy-site, and aspect page templates
 * to avoid duplicating formatting logic.
 */
import { t } from './ui';
import type { Lang } from './config';

/** Format a modifier value for display. String values (scaled) use localized label. */
export function formatModValue(lang: Lang, val: number | string, isPct: boolean): string {
  if (typeof val === "string") return t(lang, "common.scaled");
  if (isPct) {
    const pct = val * 100;
    const sign = pct > 0 ? "+" : "";
    return `${sign}${pct % 1 === 0 ? pct.toFixed(0) : pct.toFixed(1)}%`;
  }
  const sign = val > 0 ? "+" : "";
  return `${sign}${val}`;
}

/**
 * Convert a description with $key$ refs to plain tooltip text (truncated).
 *
 * Each page provides its own `resolve` function that knows how to look up
 * keys from its specific set of loc dictionaries.
 *
 * @param raw - Raw description text possibly containing $key$ references
 * @param resolve - Function that resolves a $key$ to a display string, or undefined
 * @param maxLen - Maximum length before truncation (default: 150)
 */
export function descToTooltip(
  raw: string | undefined,
  resolve: (key: string) => string | undefined,
  maxLen = 150,
): string | undefined {
  if (!raw) return undefined;
  const text = raw.replace(/\$([^$]+)\$/g, (_match, key: string) => {
    return resolve(key) ?? key.replace(/_/g, ' ');
  });
  const trimmed = text.replace(/\s+/g, ' ').trim();
  if (trimmed.length < 4) return undefined;
  return trimmed.length > maxLen ? trimmed.slice(0, maxLen - 3) + '\u2026' : trimmed;
}

/**
 * Strip $key$ refs from description text for simple tooltip display.
 *
 * Used on index/list pages where full resolution isn't needed.
 * Removes $refs$ entirely instead of resolving them.
 */
export function descToSimpleTooltip(
  raw: string | undefined,
  maxLen = 150,
): string | undefined {
  if (!raw) return undefined;
  const plain = raw.replace(/\$[^$]+\$/g, '').replace(/\s+/g, ' ').trim();
  if (plain.length < 4) return undefined;
  return plain.length > maxLen ? plain.slice(0, maxLen - 3) + '\u2026' : plain;
}
