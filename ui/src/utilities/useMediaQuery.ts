import { useEffect, useState } from 'react';

/**
 * Subscribe to a CSS media query and re-render when it starts or stops matching.
 *
 * Wraps `window.matchMedia` and keeps a boolean in sync with the query's `matches` state. The hook
 * is guarded for environments without a DOM (SSR, tests running in a `node` environment): when
 * `window`/`window.matchMedia` is unavailable it returns `false` and never subscribes, so importing
 * a component that uses it never throws outside a browser.
 *
 * @param query - A CSS media query string (e.g. `'(min-width: 2000px)'`).
 * @returns `true` while the query matches the current viewport, `false` otherwise (and always
 *   `false` when there is no DOM).
 */
export function useMediaQuery(query: string): boolean {
  // lazy initializer reads the current match once so the first paint is correct in the browser and
  // stays false when there is no DOM (matchMedia absent)
  const [matches, setMatches] = useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return false;
    }
    return window.matchMedia(query).matches;
  });
  useEffect(() => {
    // bail out of subscribing when there is no DOM to observe
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return;
    }
    const mediaQueryList = window.matchMedia(query);
    // resync on mount in case the viewport changed between the initial render and the effect
    setMatches(mediaQueryList.matches);
    const listener = (event: MediaQueryListEvent): void => setMatches(event.matches);
    mediaQueryList.addEventListener('change', listener);
    return () => mediaQueryList.removeEventListener('change', listener);
  }, [query]);
  return matches;
}

export default useMediaQuery;
