// spec: ./SPEC.md

// project imports
import type { Seed } from '@models/trees';
import type { DashboardSeedParams } from './types';

/// The URL query key for a file seed (value: sha256).
const KEY_SAMPLE = 'sample';
/// The URL query key for an entity seed (value: entity uuid, any kind incl. devices).
const KEY_ENTITY = 'entity';
/// The URL query key for a repo seed (value: full repo url).
const KEY_REPO = 'repo';
/// The URL query key for a tag seed (value: encKey ':' encValue).
const KEY_TAG = 'tag';
/// The URL query key for the crawl depth.
const KEY_DEPTH = 'depth';
/// The default crawl depth when `depth` is absent or invalid.
const DEFAULT_DEPTH = 2;
/**
 * The maximum crawl depth; larger values are clamped. Deeper crawling flows through the omnibar `depth`
 * clause -> `growToDepth`, not the seed. Exported so the builder's depth picker and depth control clamp to
 * the exact same bound the codec enforces (single source of truth).
 */
export const MAX_DEPTH = 10;

/**
 * The URL query keys the dashboard seed codec owns, in encode order (`sample`, `entity`, `repo`, `tag`,
 * `depth`). Exported so callers that strip stale seed params (e.g. the builder's mirror-to-URL effect) use
 * the codec's own key list rather than a duplicated literal that could drift.
 */
export const SEED_PARAM_KEYS = [KEY_SAMPLE, KEY_ENTITY, KEY_REPO, KEY_TAG, KEY_DEPTH] as const;

/**
 * Deduplicate a list of strings while preserving first-seen order.
 *
 * Repeated identical seed params (e.g. `sample=X&sample=X`) must collapse so the decoded seed is
 * stable regardless of how a URL was assembled.
 *
 * @param values - The raw values for a single seed key.
 * @returns The values with exact duplicates removed, in first-seen order.
 */
function dedupe(values: string[]): string[] {
  return Array.from(new Set(values));
}

/**
 * Split an encoded `tag` param value into its `[key, value]` halves.
 *
 * The stored value is `encodeURIComponent(key) ':' encodeURIComponent(value)`. Splitting on the
 * first colon is unambiguous because each half was percent-encoded, so any literal `:` inside a
 * key or value is `%3A` and cannot be mistaken for the separator. Both halves are then
 * `decodeURIComponent`-ed. A value with no colon is malformed (a bare key cannot seed
 * `Seed.tags`, which maps key -> value[]) and yields `null`.
 *
 * @param raw - The (outer-decoded) `tag` param value, i.e. `encKey:encValue`.
 * @returns The decoded `[key, value]` pair, or `null` if there is no colon.
 */
function decodeTag(raw: string): [string, string] | null {
  const colon = raw.indexOf(':');
  if (colon === -1) {
    return null;
  }
  const encKey = raw.slice(0, colon);
  const encValue = raw.slice(colon + 1);
  return [safeDecode(encKey), safeDecode(encValue)];
}

/**
 * `decodeURIComponent` that returns the input unchanged on malformed escapes instead of throwing.
 *
 * A shared URL can carry a stray `%` that is not a valid escape; the codec must not throw on such
 * input, so it falls back to the raw string.
 *
 * @param value - The possibly percent-encoded string.
 * @returns The decoded string, or the original string if decoding fails.
 */
function safeDecode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

/**
 * Encode a tag `key`/`value` pair into a single `tag` param value.
 *
 * Each half is `encodeURIComponent`-ed so embedded `:`, `=`, `&`, URLs, timestamps, and unicode
 * survive, then joined with a literal `:` separator. The outer `URLSearchParams` layer encodes the
 * result again on `toString()`.
 *
 * @param key - The tag key.
 * @param value - The tag value (may be empty).
 * @returns The joined `encKey:encValue` string.
 */
function encodeTag(key: string, value: string): string {
  return `${encodeURIComponent(key)}:${encodeURIComponent(value)}`;
}

/**
 * Parse and validate the crawl depth from raw param values.
 *
 * Falls back to {@link DEFAULT_DEPTH} when the value is absent, empty, `NaN`, non-integer, or
 * negative; clamps values above {@link MAX_DEPTH} down to the bound. Only the first `depth` param
 * is considered.
 *
 * @param raw - The raw `depth` value, or `null`/`undefined` when absent.
 * @returns A valid depth in the range `0..=MAX_DEPTH`.
 */
function parseDepth(raw: string | null | undefined): number {
  if (raw == null || raw.trim() === '') {
    return DEFAULT_DEPTH;
  }
  const parsed = Number.parseInt(raw, 10);
  // reject NaN, non-integers (e.g. "2.5" -> 2 would silently drop the fraction, so require an
  // exact integer string match after parse), and negatives
  if (Number.isNaN(parsed) || String(parsed) !== raw.trim() || parsed < 0) {
    return DEFAULT_DEPTH;
  }
  return Math.min(parsed, MAX_DEPTH);
}

/**
 * Decode dashboard seed params from a `URLSearchParams` into a {@link Seed} plus a crawl depth.
 *
 * Reads only the dashboard's own keys (`sample`, `entity`, `repo`, `tag`, `depth`) and ignores all
 * others (omnibar-clause keys, tab hash, etc.), so it is safe to call on a merged params object.
 * Empty resource values are dropped; duplicate identical params are deduped; malformed `tag` values
 * (no colon) are dropped; `depth` is validated/clamped. Absent resource params yield an empty seed
 * and the default depth.
 *
 * Devices-vs-entities routing: every `entity=<uuid>` is routed into `Seed.entities` (never
 * `Seed.devices`), because devices are one entity kind and the only real seeding caller
 * (`EntityDetails`) puts device ids under `entities`; the server resolves an id to its kind.
 *
 * @param params - The URL search params to decode.
 * @returns The decoded `{ seed, depth }`.
 */
export function decodeSeedParams(params: URLSearchParams): DashboardSeedParams {
  const seed: Seed = {};
  // samples: sha256s, non-empty and deduped
  const samples = dedupe(params.getAll(KEY_SAMPLE).filter((v) => v !== ''));
  if (samples.length > 0) {
    seed.samples = samples;
  }
  // entities: uuids (any entity kind incl. devices), non-empty and deduped
  const entities = dedupe(params.getAll(KEY_ENTITY).filter((v) => v !== ''));
  if (entities.length > 0) {
    seed.entities = entities;
  }
  // repos: full urls, non-empty and deduped
  const repos = dedupe(params.getAll(KEY_REPO).filter((v) => v !== ''));
  if (repos.length > 0) {
    seed.repos = repos;
  }
  // tags: key -> value[]; drop malformed (no colon) entries and dedupe values within a key
  const tags: { [key: string]: string[] } = {};
  for (const raw of params.getAll(KEY_TAG)) {
    const parsed = decodeTag(raw);
    if (parsed === null) {
      continue;
    }
    const [key, value] = parsed;
    const existing = tags[key] ?? [];
    if (!existing.includes(value)) {
      existing.push(value);
    }
    tags[key] = existing;
  }
  if (Object.keys(tags).length > 0) {
    seed.tags = tags;
  }
  return { seed, depth: parseDepth(params.get(KEY_DEPTH)) };
}

/**
 * Encode a {@link Seed} plus crawl depth into a `URLSearchParams` — the inverse of
 * {@link decodeSeedParams}.
 *
 * Writes only the dashboard's own keys in a stable order (`sample`, `entity`, `repo`, `tag`, then
 * `depth`) so encoded URLs are deterministic and diffable, and never touches omnibar-clause keys.
 * Values are deduped; the `depth` is validated/clamped through the same rules as decode so the
 * output is always a valid, re-decodable dashboard URL.
 *
 * @param seed - The seed to encode.
 * @param depth - The crawl depth to encode (validated/clamped on write).
 * @returns The encoded params.
 */
export function encodeSeedParams(seed: Seed, depth: number): URLSearchParams {
  const params = new URLSearchParams();
  // samples
  for (const sha256 of dedupe((seed.samples ?? []).filter((v) => v !== ''))) {
    params.append(KEY_SAMPLE, sha256);
  }
  // entities
  for (const id of dedupe((seed.entities ?? []).filter((v) => v !== ''))) {
    params.append(KEY_ENTITY, id);
  }
  // repos
  for (const url of dedupe((seed.repos ?? []).filter((v) => v !== ''))) {
    params.append(KEY_REPO, url);
  }
  // tags: emit one param per (key, value), deduping values within a key
  if (seed.tags) {
    for (const key of Object.keys(seed.tags)) {
      for (const value of dedupe(seed.tags[key])) {
        params.append(KEY_TAG, encodeTag(key, value));
      }
    }
  }
  // depth is normalized through the same validation as decode so round-trips are stable
  params.set(KEY_DEPTH, String(parseDepth(String(depth))));
  return params;
}
