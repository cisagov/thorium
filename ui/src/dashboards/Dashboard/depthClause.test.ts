import { describe, expect, it } from 'vitest';

// project imports
import { makeDepthClause, withDepthClause } from './depthClause';
import { ClauseCondition, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';

/** A non-depth text clause used to assert unrelated clauses survive. */
function textClause(value: string): Clause {
  return { category: 'text', field: 'text', condition: ClauseCondition.Is, value: { value } };
}

describe('makeDepthClause', () => {
  it('produces a depth clause with the omnibar-recognized shape', () => {
    expect(makeDepthClause(3)).toEqual({
      category: 'depth',
      field: 'depth',
      condition: ClauseCondition.Is,
      value: { value: '3' },
    });
  });

  it('stringifies the depth value', () => {
    const clause = makeDepthClause(0);
    expect(clause.value).toEqual({ value: '0' });
  });
});

describe('withDepthClause', () => {
  it('appends a depth clause when none exists', () => {
    const result = withDepthClause([textClause('foo')], 2);
    expect(result).toEqual([textClause('foo'), makeDepthClause(2)]);
  });

  it('replaces an existing depth clause without duplicating it', () => {
    const result = withDepthClause([makeDepthClause(1), textClause('foo')], 4);
    // exactly one depth clause remains, carrying the new value
    const depthClauses = result.filter((c) => c.category === 'depth');
    expect(depthClauses).toHaveLength(1);
    expect(depthClauses[0]).toEqual(makeDepthClause(4));
  });

  it('removes multiple stray depth clauses, leaving a single fresh one', () => {
    const result = withDepthClause([makeDepthClause(1), textClause('a'), makeDepthClause(2)], 5);
    expect(result.filter((c) => c.category === 'depth')).toEqual([makeDepthClause(5)]);
  });

  it('preserves non-depth clauses in order', () => {
    const result = withDepthClause([textClause('a'), makeDepthClause(1), textClause('b')], 3);
    expect(result).toEqual([textClause('a'), textClause('b'), makeDepthClause(3)]);
  });

  it('appends the fresh depth clause last', () => {
    const result = withDepthClause([textClause('a'), textClause('b')], 6);
    expect(result[result.length - 1]).toEqual(makeDepthClause(6));
  });

  it('does not mutate the input array', () => {
    const input = [textClause('a'), makeDepthClause(1)];
    const snapshot = [...input];
    withDepthClause(input, 9);
    expect(input).toEqual(snapshot);
  });
});
