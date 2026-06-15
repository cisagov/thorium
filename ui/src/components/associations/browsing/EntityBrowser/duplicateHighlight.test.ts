import { describe, it, expect } from 'vitest';

// project imports
import { nextIndex } from './duplicateHighlight';

// The DOM-touching helpers (collectOccurrences / applyHighlight / clearHighlight / scrollToNextOccurrence) are
// exercised by the entity-browser E2E specs, which run in a real browser. `nextIndex` is the pure cycle logic —
// including the "wrap back to the first after the last" behavior — so it is unit-tested here.
describe('nextIndex', () => {
  it('advances to the next occurrence', () => {
    expect(nextIndex(0, 3)).toBe(1);
    expect(nextIndex(1, 3)).toBe(2);
  });

  it('wraps from the last occurrence back to the first', () => {
    expect(nextIndex(2, 3)).toBe(0);
    expect(nextIndex(4, 5)).toBe(0);
  });

  it('starts the cycle at the first occurrence when the current row is not found', () => {
    expect(nextIndex(-1, 3)).toBe(0);
  });

  it('is a no-op (returns -1) when there is nowhere to jump', () => {
    expect(nextIndex(0, 1)).toBe(-1);
    expect(nextIndex(-1, 1)).toBe(-1);
    expect(nextIndex(0, 0)).toBe(-1);
  });
});
