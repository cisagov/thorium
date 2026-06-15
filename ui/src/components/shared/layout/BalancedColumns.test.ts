import { describe, expect, it } from 'vitest';

// project imports
import {
  ASSIGNMENT_HYSTERESIS_PX,
  BalanceStrategy,
  ColumnSide,
  assignColumns,
  assignColumnsN,
  assignmentsEqual,
  computeColumnCount,
  deriveColumnCount,
  maxColumnHeight,
  roundRobinAssignment,
  shouldReplaceAssignment,
} from './BalancedColumns';

describe('assignColumns', () => {
  it('returns an empty assignment for no items', () => {
    expect(assignColumns(100, 50, [])).toEqual([]);
  });

  it('alternates starting left when bases and item heights are equal', () => {
    // 0/0 -> Left (l=10), 10/0 -> Right (r=10), 10/10 tie -> Left
    expect(assignColumns(0, 0, [10, 10, 10])).toEqual([ColumnSide.Left, ColumnSide.Right, ColumnSide.Left]);
  });

  it('fills the right column first when the left base is taller', () => {
    // l=50 vs r=0 -> Right (r=30), 50/30 -> Right (r=60), 50/60 -> Left
    expect(assignColumns(50, 0, [30, 30, 30])).toEqual([ColumnSide.Right, ColumnSide.Right, ColumnSide.Left]);
  });

  it('fills the left column first when the right base is taller', () => {
    // l=0 vs r=50 -> Left (l=30), 30/50 -> Left (l=60), 60/50 -> Right
    expect(assignColumns(0, 50, [30, 30, 30])).toEqual([ColumnSide.Left, ColumnSide.Left, ColumnSide.Right]);
  });

  it('breaks ties toward the left column', () => {
    // equal bases tie -> Left, and the running tie at 30/30 after the first two items -> Left again
    expect(assignColumns(20, 20, [10])).toEqual([ColumnSide.Left]);
    expect(assignColumns(0, 30, [30, 5])).toEqual([ColumnSide.Left, ColumnSide.Left]);
  });

  it('balances varying item heights', () => {
    // l=10 vs r=0 -> Right (r=5), 10/5 -> Right (r=25), 10/25 -> Left (l=15), 15/25 -> Left (l=25)
    expect(assignColumns(10, 0, [5, 20, 5, 10])).toEqual([ColumnSide.Right, ColumnSide.Right, ColumnSide.Left, ColumnSide.Left]);
  });
});

describe('computeColumnCount', () => {
  it('fits multiple columns when width allows', () => {
    // 3*300 + 2*20 = 940 <= 1000 but 4*300 + 3*20 = 1260 > 1000
    expect(computeColumnCount(1000, 300, 20)).toBe(3);
  });

  it('returns one column when the container is narrower than a single column', () => {
    expect(computeColumnCount(200, 300, 20)).toBe(1);
  });

  it('clamps to a minimum of one column for zero or negative widths', () => {
    expect(computeColumnCount(0, 300, 20)).toBe(1);
    expect(computeColumnCount(-100, 300, 20)).toBe(1);
  });

  it('accounts for the gap between columns', () => {
    // 2*300 + 1*20 = 620 fits exactly in 620
    expect(computeColumnCount(620, 300, 20)).toBe(2);
    // 620 needed for two columns exceeds 600, so only one fits
    expect(computeColumnCount(600, 300, 20)).toBe(1);
  });

  it('fits many columns in a large container', () => {
    // 10*300 + 9*20 = 3180 fits exactly in 3180
    expect(computeColumnCount(3180, 300, 20)).toBe(10);
  });

  it('degrades to one column when the column width and gap are non-positive', () => {
    expect(computeColumnCount(1000, 0, 0)).toBe(1);
  });
});

describe('deriveColumnCount', () => {
  it('matches computeColumnCount when no cap is given', () => {
    // 3*300 + 2*20 = 940 <= 1000 -> 3 columns, uncapped
    expect(deriveColumnCount(1000, 300, 20)).toBe(3);
  });

  it('caps the derived count at maxColumns on wide screens', () => {
    // would derive 3 (and 10) columns, but the cap holds it at 2
    expect(deriveColumnCount(1000, 300, 20, 2)).toBe(2);
    expect(deriveColumnCount(3180, 300, 20, 2)).toBe(2);
  });

  it('still collapses below the cap when narrow', () => {
    // only one column fits, which is under the cap so the cap has no effect
    expect(deriveColumnCount(200, 300, 20, 2)).toBe(1);
  });

  it('never drops below one column even with a smaller derived count', () => {
    expect(deriveColumnCount(0, 300, 20, 2)).toBe(1);
  });
});

describe('assignColumnsN', () => {
  it('stacks everything into column 0 when there is a single column', () => {
    expect(assignColumnsN(1, [10, 20], [5, 5])).toEqual({ anchorCols: [0, 0], itemCols: [0, 0] });
  });

  it('clamps a degenerate column count to a single column', () => {
    expect(assignColumnsN(0, [10], [5])).toEqual({ anchorCols: [0], itemCols: [0] });
  });

  it('seeds equal anchors left-to-right and round-robins equal items by lowest-index ties', () => {
    // anchors seed [0,1,2] at height 10 each; each item picks the lowest-index shortest column
    expect(assignColumnsN(3, [10, 10, 10], [10, 10, 10])).toEqual({ anchorCols: [0, 1, 2], itemCols: [0, 1, 2] });
  });

  it('balances anchors beyond the column count into the shortest column', () => {
    // anchors 0/1 seed columns 0 (h=10) and 1 (h=30); the overflow anchor flows to column 0
    expect(assignColumnsN(2, [10, 30, 5], [])).toEqual({ anchorCols: [0, 1, 0], itemCols: [] });
  });

  it('pushes items away from a column with a taller anchor', () => {
    // heights [100,10,10]: item ties at 10/10 -> col 1 (h=30), then col 2 (h=30), tie -> col 1
    expect(assignColumnsN(3, [100, 10, 10], [20, 20, 20])).toEqual({ anchorCols: [0, 1, 2], itemCols: [1, 2, 1] });
  });

  it('balances items by running height with ties to the lowest index', () => {
    // heights [10,0]: -> col 1 (r=5), -> col 1 (r=25), -> col 0 (l=15), -> col 0 (l=25)
    expect(assignColumnsN(2, [10, 0], [5, 20, 5, 10])).toEqual({ anchorCols: [0, 1], itemCols: [1, 1, 0, 0] });
  });

  it('defaults to InOrder when no strategy is given', () => {
    // the default arg must reproduce the explicit InOrder result exactly (no behavior change)
    const heights = [10, 30, 20];
    expect(assignColumnsN(2, [], heights)).toEqual(assignColumnsN(2, [], heights, BalanceStrategy.InOrder));
  });

  it('LongestFirst places tallest-first but returns index-aligned itemCols', () => {
    // heights [10,30,20] placed as 30(i1)->col0, 20(i2)->col1, 10(i0)->col1; itemCols stay index-aligned
    expect(assignColumnsN(2, [], [10, 30, 20], BalanceStrategy.LongestFirst)).toEqual({ anchorCols: [], itemCols: [1, 0, 1] });
  });

  it('LongestFirst balances a skewed set better than InOrder', () => {
    // one tall item among small ones: InOrder strands it atop a filled column, LongestFirst isolates it
    const items = [10, 10, 10, 10, 60];
    const inOrder = assignColumnsN(2, [], items, BalanceStrategy.InOrder);
    const longest = assignColumnsN(2, [], items, BalanceStrategy.LongestFirst);
    expect(maxColumnHeight(longest, 2, [], items)).toBeLessThan(maxColumnHeight(inOrder, 2, [], items));
  });

  it('LongestFirst breaks height ties by original index', () => {
    // all equal heights fall back to index order, matching InOrder's lowest-index-tie round-robin
    expect(assignColumnsN(3, [], [10, 10, 10], BalanceStrategy.LongestFirst)).toEqual({ anchorCols: [], itemCols: [0, 1, 2] });
  });
});

describe('roundRobinAssignment', () => {
  it('seeds anchors in their own column and cycles items across columns', () => {
    expect(roundRobinAssignment(3, 2, 5)).toEqual({ anchorCols: [0, 1], itemCols: [0, 1, 2, 0, 1] });
  });

  it('clamps a degenerate column count to a single column', () => {
    expect(roundRobinAssignment(0, 1, 3)).toEqual({ anchorCols: [0], itemCols: [0, 0, 0] });
  });

  it('clamps overflow anchors to the last column', () => {
    // anchors beyond the column count pin to the final column (min(i, count-1))
    expect(roundRobinAssignment(2, 4, 0)).toEqual({ anchorCols: [0, 1, 1, 1], itemCols: [] });
  });
});

describe('assignmentsEqual', () => {
  it('is true for identical placements and false for differing ones', () => {
    expect(assignmentsEqual({ anchorCols: [0, 1], itemCols: [0] }, { anchorCols: [0, 1], itemCols: [0] })).toBe(true);
    expect(assignmentsEqual({ anchorCols: [0, 1], itemCols: [0] }, { anchorCols: [0, 1], itemCols: [1] })).toBe(false);
  });

  it('is false for differing shapes', () => {
    expect(assignmentsEqual({ anchorCols: [0], itemCols: [0] }, { anchorCols: [0], itemCols: [0, 1] })).toBe(false);
  });

  it('compares nulls by reference', () => {
    expect(assignmentsEqual(null, null)).toBe(true);
    expect(assignmentsEqual(null, { anchorCols: [], itemCols: [] })).toBe(false);
  });
});

describe('maxColumnHeight', () => {
  it('sums each column and returns the tallest', () => {
    // col 0 gets anchor 0 (100) + item 0 (10) = 110; col 1 gets anchor 1 (20) + item 1 (10) = 30
    const assignment = { anchorCols: [0, 1], itemCols: [0, 1] };
    expect(maxColumnHeight(assignment, 2, [100, 20], [10, 10])).toBe(110);
  });

  it('ignores tiles assigned to a column beyond the count', () => {
    // the item targets column 5 which is out of range for a 2-column layout, so it is not summed
    const assignment = { anchorCols: [0], itemCols: [5] };
    expect(maxColumnHeight(assignment, 2, [40], [999])).toBe(40);
  });
});

describe('shouldReplaceAssignment', () => {
  const base = { anchorCols: [0, 1], itemCols: [0, 1] };

  it('replaces when there is no previous assignment', () => {
    expect(shouldReplaceAssignment(null, base, 2, [10, 10], [10, 10])).toBe(true);
  });

  it('replaces when the tile counts changed shape', () => {
    const next = { anchorCols: [0, 1], itemCols: [0, 1, 0] };
    expect(shouldReplaceAssignment(base, next, 2, [10, 10], [10, 10, 10])).toBe(true);
  });

  it('keeps the previous assignment when the placement is identical', () => {
    const next = { anchorCols: [0, 1], itemCols: [0, 1] };
    expect(shouldReplaceAssignment(base, next, 2, [10, 10], [10, 10])).toBe(false);
  });

  it('keeps the previous assignment when the improvement is within the hysteresis band', () => {
    // swapping the two items barely changes the tallest column, well under ASSIGNMENT_HYSTERESIS_PX
    const next = { anchorCols: [0, 1], itemCols: [1, 0] };
    const smallDelta = ASSIGNMENT_HYSTERESIS_PX - 1;
    // col heights under `base`: col0 = 100 + 0 = 100, col1 = 0 + smallDelta; under `next` they swap
    expect(shouldReplaceAssignment(base, next, 2, [100, 0], [0, smallDelta])).toBe(false);
  });

  it('replaces when a different placement improves the imbalance beyond the hysteresis band', () => {
    // moving both items off the tall column drops the max height by far more than the band
    const next = { anchorCols: [0, 1], itemCols: [1, 1] };
    expect(shouldReplaceAssignment(base, next, 2, [100, 0], [50, 50])).toBe(true);
  });
});
