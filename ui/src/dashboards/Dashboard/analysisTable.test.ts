import { describe, expect, it } from 'vitest';

// project imports
import { AnalysisSortColumn, SortDirection, reactionSortValue, sortReactions } from './analysisTable';
import { Reaction, ReactionStatus } from '@models/reactions';

/**
 * Build a minimal reaction carrying only the fields the sort reads.
 *
 * @param over - The sort-relevant fields to set (id, pipeline, group, status, first sample).
 * @returns A reaction sufficient for sort tests.
 */
function reaction(over: { id: string; pipeline: string; group: string; status: ReactionStatus; sha256: string }): Reaction {
  return { id: over.id, pipeline: over.pipeline, group: over.group, status: over.status, samples: [over.sha256] } as unknown as Reaction;
}

/// Three reactions whose columns deliberately disagree so tiebreak order is observable.
const A = reaction({ id: 'a', pipeline: 'zeta', group: 'g2', status: ReactionStatus.Completed, sha256: 'aaa' });
const B = reaction({ id: 'b', pipeline: 'alpha', group: 'g1', status: ReactionStatus.Failed, sha256: 'aaa' });
const C = reaction({ id: 'c', pipeline: 'beta', group: 'g1', status: ReactionStatus.Started, sha256: 'bbb' });

describe('reactionSortValue', () => {
  it('reads the lower-cased value per column, using the first sample for File', () => {
    const r = reaction({ id: 'x', pipeline: 'PipeA', group: 'GrpB', status: ReactionStatus.Created, sha256: 'ABC' });
    expect(reactionSortValue(r, AnalysisSortColumn.Pipeline)).toBe('pipea');
    expect(reactionSortValue(r, AnalysisSortColumn.File)).toBe('abc');
    expect(reactionSortValue(r, AnalysisSortColumn.Group)).toBe('grpb');
    expect(reactionSortValue(r, AnalysisSortColumn.Status)).toBe('created');
  });
});

describe('sortReactions — default order', () => {
  it('sorts by file, then pipeline, then status when no primary is selected', () => {
    // A and B share sha 'aaa' so pipeline breaks the tie (alpha < zeta); C's sha 'bbb' sorts last
    const sorted = sortReactions([A, B, C], null, SortDirection.Asc);
    expect(sorted.map((r) => r.id)).toEqual(['b', 'a', 'c']);
  });
  it('does not mutate its input', () => {
    const input = [A, B, C];
    sortReactions(input, null, SortDirection.Asc);
    expect(input.map((r) => r.id)).toEqual(['a', 'b', 'c']);
  });
});

describe('sortReactions — user-selected primary', () => {
  it('leads with the primary column ascending, then the default order tiebreaks', () => {
    // primary = Group: g1 before g2; within g1, default file/pipeline order → B (aaa) before C (bbb)
    const sorted = sortReactions([A, B, C], AnalysisSortColumn.Group, SortDirection.Asc);
    expect(sorted.map((r) => r.id)).toEqual(['b', 'c', 'a']);
  });
  it('applies the chosen direction to the primary column only', () => {
    // primary = Group descending: g2 (A) first, then g1 group; g1 still tiebreaks ascending → B before C
    const sorted = sortReactions([A, B, C], AnalysisSortColumn.Group, SortDirection.Desc);
    expect(sorted.map((r) => r.id)).toEqual(['a', 'b', 'c']);
  });
  it('sorts by status as the primary when selected', () => {
    // statuses: Completed(A) < Failed(B) < Started(C) alphabetically
    const sorted = sortReactions([C, B, A], AnalysisSortColumn.Status, SortDirection.Asc);
    expect(sorted.map((r) => r.id)).toEqual(['a', 'b', 'c']);
  });
});
