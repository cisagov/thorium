import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';

// project imports
import type { ParamCodec } from './codecs';

/**
 * Bind a typed slice of state to the URL query string, using `codec` to (de)serialize it.
 *
 * The URL is the single source of truth: the returned value is decoded from the current params,
 * and the setter re-encodes into the URL. Writes use a functional updater so independent codecs
 * sharing the URL never clobber one another, and `{ replace: true }` so rapid changes (typing in
 * a filter) don't flood browser history.
 *
 * The setter accepts either a new value or an updater function (like `useState`), resolving the
 * updater against the value decoded from the live URL so it's never stale.
 *
 * This is the generic seam for URL state (pattern A1). Swapping the backing store later (e.g. a
 * batched context provider) means rewriting only this file — call sites keep the same
 * `[value, setValue]` shape.
 */
export function useUrlState<T>(codec: ParamCodec<T>, fallback: T): [T, (next: T | ((prev: T) => T)) => void] {
  const [searchParams, setSearchParams] = useSearchParams();

  // Re-decode only when the query string actually changes, so the returned value is referentially
  // stable across unrelated re-renders (callers can safely depend on it in effects).
  const paramsString = searchParams.toString();
  // codec/fallback are expected to be stable for a given call site, so we key only on the URL
  const value = useMemo(() => {
    const decoded = codec.decode(new URLSearchParams(paramsString));
    return decoded !== undefined ? decoded : fallback;
  }, [paramsString]);

  const setValue = useCallback(
    (next: T | ((prev: T) => T)) => {
      setSearchParams(
        (prev) => {
          const out = new URLSearchParams(prev);
          // resolve updater functions against the value currently in the URL (never stale)
          const resolved = typeof next === 'function' ? (next as (p: T) => T)(codec.decode(new URLSearchParams(prev)) ?? fallback) : next;
          // clear this codec's existing keys, then write the new value
          codec.keys(out).forEach((key) => out.delete(key));
          codec.encode(resolved, out);
          return out;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [value, setValue];
}
