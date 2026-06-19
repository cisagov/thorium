import { describe, it, expect } from 'vitest';

// project imports
import { OmniClauseAndTimeToFilter } from './search';
import { Clause, ClauseCondition } from '@components/shared/inputs/omnibar/ClauseTypes';
import { AbsoluteSelection } from '@components/shared/inputs/omnibar/timepicker/utils';

function multi(category: string, field: string, values: string[]): Clause {
  return { category, field, condition: ClauseCondition.IsOneOf, value: { values } };
}

function single(category: string, field: string, value: string): Clause {
  return { category, field, condition: ClauseCondition.Is, value: { value } };
}

describe('OmniClauseAndTimeToFilter', () => {
  it("uses the default limit and empty ranges for 'all' time and no clauses", () => {
    expect(OmniClauseAndTimeToFilter([], { mode: 'all' })).toEqual({
      limit: 25,
      groups: [],
      tags: {},
      start: null,
      end: null,
      hideTags: [],
    });
  });

  it('extracts tag clauses into Filters.tags (and excludes hidden tags)', () => {
    const clauses: Clause[] = [
      single('tag', 'family', 'emotet'),
      single('tag', 'family', 'trickbot'),
      single('tag', 'av', 'malicious'),
      multi('hidden tags', 'hidden tags', ['Results', 'Parent']),
    ];
    const filter = OmniClauseAndTimeToFilter(clauses, { mode: 'all' });
    expect(filter.tags).toEqual({ family: ['emotet', 'trickbot'], av: ['malicious'] });
  });

  it('honors a custom default limit', () => {
    expect(OmniClauseAndTimeToFilter([], { mode: 'all' }, 50).limit).toBe(50);
  });

  it('extracts groups, limit, and hidden tags from clauses', () => {
    const clauses: Clause[] = [
      multi('group', 'group', ['team-a', 'team-b']),
      { category: 'limit', field: 'limit', condition: ClauseCondition.Is, value: { value: '100' } },
      multi('hidden tags', 'hidden tags', ['Results', 'Parent']),
    ];
    const filter = OmniClauseAndTimeToFilter(clauses, { mode: 'all' });
    expect(filter.groups).toEqual(['team-a', 'team-b']);
    expect(filter.limit).toBe(100);
    expect(filter.hideTags).toEqual(['Results', 'Parent']);
  });

  it('maps an absolute range into the inverted Filters start/end (start = latest, end = earliest)', () => {
    const earlier = new Date('2026-01-01T00:00:00.000Z');
    const later = new Date('2026-02-01T00:00:00.000Z');
    const sel: AbsoluteSelection = { mode: 'absolute', start: earlier, end: later };
    const filter = OmniClauseAndTimeToFilter([], sel);
    // Filters.start is documented as the latest date to start listing from (paging backwards)
    expect(filter.start).toBe(later.toISOString());
    expect(filter.end).toBe(earlier.toISOString());
  });
});
