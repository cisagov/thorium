import { describe, it, expect } from 'vitest';

// project imports
import { firstEnabledTabIndex, lastEnabledTabIndex, nextEnabledTabIndex } from './step';

// build a tab list from a boolean disabled mask (true = disabled)
function tabs(mask: boolean[]) {
  return mask.map((disabled) => ({ disabled }));
}

describe('nextEnabledTabIndex', () => {
  it('steps forward to the next enabled tab', () => {
    expect(nextEnabledTabIndex(tabs([false, false, false]), 0, 1)).toBe(1);
  });

  it('steps backward to the previous enabled tab', () => {
    expect(nextEnabledTabIndex(tabs([false, false, false]), 2, -1)).toBe(1);
  });

  it('wraps around when stepping forward past the end', () => {
    expect(nextEnabledTabIndex(tabs([false, false, false]), 2, 1)).toBe(0);
  });

  it('wraps around when stepping backward past the start', () => {
    expect(nextEnabledTabIndex(tabs([false, false, false]), 0, -1)).toBe(2);
  });

  it('skips disabled tabs while stepping', () => {
    // from 0, forward skips the disabled index 1 and lands on 2
    expect(nextEnabledTabIndex(tabs([false, true, false]), 0, 1)).toBe(2);
  });

  it('skips disabled tabs across the wrap boundary', () => {
    // from 1, forward wraps past disabled index 2 back to 0
    expect(nextEnabledTabIndex(tabs([false, false, true]), 1, 1)).toBe(0);
  });

  it('is a no-op when no other tab is enabled', () => {
    // only index 1 is enabled, so stepping from it returns itself
    expect(nextEnabledTabIndex(tabs([true, false, true]), 1, 1)).toBe(1);
    expect(nextEnabledTabIndex(tabs([true, false, true]), 1, -1)).toBe(1);
  });

  it('is a no-op when every tab is disabled', () => {
    expect(nextEnabledTabIndex(tabs([true, true, true]), 1, 1)).toBe(1);
  });

  it('returns current for an empty tab list', () => {
    expect(nextEnabledTabIndex([], 0, 1)).toBe(0);
  });
});

describe('firstEnabledTabIndex', () => {
  it('returns the first enabled index', () => {
    expect(firstEnabledTabIndex(tabs([true, false, false]))).toBe(1);
  });

  it('returns -1 when every tab is disabled', () => {
    expect(firstEnabledTabIndex(tabs([true, true]))).toBe(-1);
  });
});

describe('lastEnabledTabIndex', () => {
  it('returns the last enabled index', () => {
    expect(lastEnabledTabIndex(tabs([false, false, true]))).toBe(1);
  });

  it('returns -1 when every tab is disabled', () => {
    expect(lastEnabledTabIndex(tabs([true, true]))).toBe(-1);
  });
});
