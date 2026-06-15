/**
 * Generic, domain-agnostic codecs for syncing typed state to/from the URL query string.
 *
 * A {@link ParamCodec} maps a single logical slice of state (`T`) to and from a set of query
 * params that it exclusively owns. Because each codec only touches its own keys, many codecs
 * can compose on a single URL without stepping on one another. This module knows nothing about
 * the omnibar (or any other feature) — feature-specific codecs are built on top of it.
 */

export interface ParamCodec<T> {
  /**
   * The param keys this codec owns for the given params. The writer deletes these before
   * re-encoding so stale entries don't linger. Takes the current params so codecs with dynamic
   * keys (e.g. `tags[KEY]`) can discover which keys they currently own.
   */
  keys: (params: URLSearchParams) => string[];
  /** Write `value` into `params` (mutates in place). */
  encode: (value: T, params: URLSearchParams) => void;
  /** Read this codec's value from `params`, or `undefined` if it isn't present. */
  decode: (params: URLSearchParams) => T | undefined;
}

/**
 * A single string param, e.g. `?index=All`. Empty strings are treated as absent (not written).
 */
export function stringCodec(key: string): ParamCodec<string> {
  return {
    keys: () => [key],
    encode: (value, params) => {
      if (value) {
        params.set(key, value);
      }
    },
    decode: (params) => params.get(key) ?? undefined,
  };
}

/**
 * A repeated param as an ordered list, e.g. `?open=a&open=b` ↔ `['a', 'b']`. An empty list is
 * treated as absent (decodes to `undefined`, encodes to nothing).
 */
export function listCodec(key: string): ParamCodec<string[]> {
  return {
    keys: () => [key],
    encode: (value, params) => {
      value.forEach((item) => params.append(key, item));
    },
    decode: (params) => {
      const all = params.getAll(key);
      return all.length > 0 ? all : undefined;
    },
  };
}

/**
 * A boolean flag, e.g. `?flagged=1` ↔ `true`. `false` is treated as absent (not written); a
 * `1`-valued param decodes to `true`, anything else (including a missing key) decodes to
 * `undefined`.
 *
 * @param key - The single param key this codec owns.
 * @returns A {@link ParamCodec} binding a boolean flag to `key`.
 */
export function boolCodec(key: string): ParamCodec<boolean> {
  return {
    keys: () => [key],
    encode: (value, params) => {
      if (value) {
        params.set(key, '1');
      }
    },
    decode: (params) => (params.get(key) === '1' ? true : undefined),
  };
}

/**
 * A repeated param as an unordered {@link Set}, e.g. `?hidden=a&hidden=b` ↔ `Set(['a', 'b'])`,
 * preserving insertion order. An empty set is treated as absent (decodes to `undefined`, encodes to
 * nothing). Duplicate URL occurrences collapse into the set naturally.
 *
 * @param key - The single param key this codec owns.
 * @returns A {@link ParamCodec} binding a set of strings to repeated `key` params.
 */
export function setCodec(key: string): ParamCodec<Set<string>> {
  return {
    keys: () => [key],
    encode: (value, params) => {
      value.forEach((item) => params.append(key, item));
    },
    decode: (params) => {
      const all = params.getAll(key);
      return all.length > 0 ? new Set(all) : undefined;
    },
  };
}
