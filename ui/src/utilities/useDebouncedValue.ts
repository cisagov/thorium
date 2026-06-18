import { useEffect, useState } from 'react';

/**
 * Returns a debounced copy of `value` that only updates after `delayMs`
 * have elapsed without `value` changing. Useful for gating expensive work
 * (e.g. network requests) on a rapidly-changing input such as editor text.
 *
 * @param value - The value to debounce.
 * @param delayMs - Delay in milliseconds before the update propagates.
 * @returns The most recent value, delayed by `delayMs`.
 */
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}
