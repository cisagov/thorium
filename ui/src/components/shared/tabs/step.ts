// spec: ./Tabs.spec.md

/** The minimal tab shape the keyboard-stepping helpers need: only whether a tab is selectable. */
interface SteppableTab {
  /** Whether the tab is disabled (skipped by keyboard navigation). */
  disabled?: boolean;
}

/**
 * Find the next enabled tab index when stepping from `current` in `dir`, wrapping around and
 * skipping disabled tabs.
 *
 * Walks at most one full loop so a set with no enabled tab (or only the current one) resolves without
 * spinning; when no other enabled tab exists it returns `current` unchanged (a no-op step).
 *
 * @param tabs - The tabs in display order.
 * @param current - The index the step originates from.
 * @param dir - The step direction: `1` for forward (right/down), `-1` for backward (left/up).
 * @returns The index of the next enabled tab, or `current` when none is reachable.
 */
export function nextEnabledTabIndex(tabs: SteppableTab[], current: number, dir: 1 | -1): number {
  if (tabs.length === 0) return current;
  let next = current;
  // scan up to one full wrap; break as soon as an enabled tab is found
  for (let i = 0; i < tabs.length; i++) {
    next = (next + dir + tabs.length) % tabs.length;
    if (!tabs[next]?.disabled) return next;
  }
  return current;
}

/**
 * Find the first enabled tab index (the Home-key target).
 *
 * @param tabs - The tabs in display order.
 * @returns The index of the first enabled tab, or `-1` when every tab is disabled.
 */
export function firstEnabledTabIndex(tabs: SteppableTab[]): number {
  return tabs.findIndex((tab) => !tab.disabled);
}

/**
 * Find the last enabled tab index (the End-key target).
 *
 * @param tabs - The tabs in display order.
 * @returns The index of the last enabled tab, or `-1` when every tab is disabled.
 */
export function lastEnabledTabIndex(tabs: SteppableTab[]): number {
  for (let i = tabs.length - 1; i >= 0; i--) {
    if (!tabs[i]?.disabled) return i;
  }
  return -1;
}
