import { describe, it, expect } from 'vitest';

// project imports
import { AbsoluteSelection, defaultTimeSelection, startOfLast, TimeSelectionToDateRange, TimeSelectionToStrings } from './utils';

describe('defaultTimeSelection', () => {
  it("defaults to 'all' time", () => {
    expect(defaultTimeSelection()).toEqual({ mode: 'all' });
  });
});

describe('TimeSelectionToDateRange', () => {
  it("returns [null, null] for 'all'", () => {
    expect(TimeSelectionToDateRange({ mode: 'all' })).toEqual([null, null]);
  });

  it('returns the exact start/end for an absolute selection', () => {
    const start = new Date('2026-01-01T00:00:00.000Z');
    const end = new Date('2026-01-02T00:00:00.000Z');
    const sel: AbsoluteSelection = { mode: 'absolute', start, end };
    expect(TimeSelectionToDateRange(sel)).toEqual([start, end]);
  });
});

describe('TimeSelectionToStrings', () => {
  it("returns [null, null] for 'all'", () => {
    expect(TimeSelectionToStrings({ mode: 'all' })).toEqual([null, null]);
  });

  it('serializes absolute start/end to ISO strings in [start, end] order', () => {
    const start = new Date('2026-01-01T00:00:00.000Z');
    const end = new Date('2026-01-02T12:30:00.000Z');
    expect(TimeSelectionToStrings({ mode: 'absolute', start, end })).toEqual([start.toISOString(), end.toISOString()]);
  });
});

describe('startOfLast', () => {
  const end = new Date('2026-06-15T12:00:00.000Z');

  it('subtracts fixed durations for minute/hour/day/week', () => {
    expect(startOfLast(1, 'hour', end).toISOString()).toBe('2026-06-15T11:00:00.000Z');
    expect(startOfLast(2, 'day', end).toISOString()).toBe('2026-06-13T12:00:00.000Z');
    expect(startOfLast(1, 'week', end).toISOString()).toBe('2026-06-08T12:00:00.000Z');
  });

  it('subtracts calendar months, clamping the day of month', () => {
    // March 31 minus one month clamps to the last valid day of February
    const mar31 = new Date('2026-03-31T00:00:00.000Z');
    expect(startOfLast(1, 'month', mar31).toISOString()).toBe('2026-02-28T00:00:00.000Z');
  });

  it('crosses year boundaries when subtracting months', () => {
    expect(startOfLast(2, 'month', new Date('2026-01-15T00:00:00.000Z')).toISOString()).toBe('2025-11-15T00:00:00.000Z');
  });

  it('subtracts whole years', () => {
    expect(startOfLast(1, 'year', end).toISOString()).toBe('2025-06-15T12:00:00.000Z');
  });

  it('throws on a negative amount', () => {
    expect(() => startOfLast(-1, 'day', end)).toThrow();
  });
});
