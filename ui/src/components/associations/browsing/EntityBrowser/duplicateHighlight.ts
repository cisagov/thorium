// spec: ./EntityBrowser.spec.md

/**
 * The class toggled on a row header to visually mark it as part of the currently highlighted duplicate
 * group. Applied imperatively (not via React) to every visible occurrence of a node id so a re-render of a
 * memoized row never has to run — see the styled `RowHeader` rule of the same name.
 */
export const DUPLICATE_HIGHLIGHT_CLASS = 'duplicate-highlight';

/**
 * The attribute {@link https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/dataset dataset} key each
 * row header carries (`data-node-id`), used to find every occurrence of the same graph node in the DOM.
 */
const NODE_ID_ATTR = 'data-node-id';

/**
 * The scope attribute on the browser root (`data-testid="entity-browser"`) that bounds occurrence queries to a
 * single browser instance — the side-by-side association tree reuses the same node ids in its own DOM, so an
 * unscoped query would cross-highlight it.
 */
const SCOPE_SELECTOR = '[data-testid="entity-browser"]';

/**
 * Find every currently-rendered row header for a given graph node id, scoped to the browser instance that
 * contains `origin`, in document (visual) order.
 *
 * Node ids can be repo URLs or tag-derived strings containing characters (`"`, `\`) that are unsafe in a CSS
 * attribute-value selector, so this queries by attribute *presence* and compares `dataset.nodeId` in JS rather
 * than interpolating the id into a selector. Row counts are pagination-bounded per level, so the linear scan is
 * cheap.
 *
 * @param origin - Any element inside the target browser (typically the badge or row that was interacted with).
 * @param nodeId - The graph node id whose occurrences to collect.
 * @returns The matching row header elements in document order, or an empty array if the scope can't be found.
 */
export function collectOccurrences(origin: Element, nodeId: string): HTMLElement[] {
  // bound the search to this browser instance so we never light up the sibling association tree
  const scope = origin.closest(SCOPE_SELECTOR);
  if (!scope) return [];
  // querySelectorAll preserves document order, which matches the visual top-to-bottom order of the rows
  const all = scope.querySelectorAll<HTMLElement>(`[${NODE_ID_ATTR}]`);
  const matches: HTMLElement[] = [];
  for (const el of all) {
    if (el.dataset.nodeId === nodeId) matches.push(el);
  }
  return matches;
}

/**
 * Add the {@link DUPLICATE_HIGHLIGHT_CLASS} to every element in `list`.
 *
 * @param list - The row headers to highlight.
 */
export function applyHighlight(list: HTMLElement[]): void {
  for (const el of list) el.classList.add(DUPLICATE_HIGHLIGHT_CLASS);
}

/**
 * Remove the {@link DUPLICATE_HIGHLIGHT_CLASS} from every element in `list`.
 *
 * @param list - The row headers to clear.
 */
export function clearHighlight(list: HTMLElement[]): void {
  for (const el of list) el.classList.remove(DUPLICATE_HIGHLIGHT_CLASS);
}

/**
 * The index of the next occurrence to jump to, wrapping from the last back to the first.
 *
 * @param current - The current occurrence's index (as found by element identity); a negative value (not found)
 *   starts the cycle at the first occurrence.
 * @param length - The number of occurrences currently rendered.
 * @returns The index to scroll to, or `-1` when there is nothing to jump to (0 or 1 occurrences).
 */
export function nextIndex(current: number, length: number): number {
  // with one or zero occurrences there is nowhere to jump — signal a no-op
  if (length <= 1) return -1;
  // a not-yet-located current row begins the cycle at the top
  if (current < 0) return 0;
  // advance one, wrapping the last occurrence back to the first
  return (current + 1) % length;
}

/**
 * Scroll the next occurrence of a duplicate node into view, wrapping from the last back to the first. The
 * current occurrence is located by element identity (every occurrence shares the same node id, so an id-based
 * lookup would be ambiguous).
 *
 * @param list - The occurrences in document order (from {@link collectOccurrences}).
 * @param own - The occurrence the jump was triggered from.
 */
export function scrollToNextOccurrence(list: HTMLElement[], own: HTMLElement): void {
  // locate this occurrence by identity, not by id — all occurrences carry the same node id
  const current = list.indexOf(own);
  const target = nextIndex(current, list.length);
  if (target < 0) return;
  list[target].scrollIntoView({ block: 'center', behavior: 'smooth' });
}
