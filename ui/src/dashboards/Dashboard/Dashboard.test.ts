import { describe, it, expect } from 'vitest';

// project imports
import { clausesEqual, dedupeAppend, seedHasResources } from './Dashboard';
import { ClauseCondition, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import type { Seed } from '@models/trees';

// a single-value clause helper for terse test construction
function single(category: string, field: string, value: string): Clause {
  return { category, field, condition: ClauseCondition.Is, value: { value } };
}

// a multi-value clause helper
function multi(category: string, field: string, values: string[]): Clause {
  return { category, field, condition: ClauseCondition.IsOneOf, value: { values } };
}

describe('seedHasResources', () => {
  it('is false for an empty seed', () => {
    expect(seedHasResources({})).toBe(false);
  });

  it('is false for a seed whose only tag map is empty', () => {
    expect(seedHasResources({ tags: {} })).toBe(false);
  });

  it.each<[string, Seed]>([
    ['samples', { samples: ['a'] }],
    ['entities', { entities: ['e'] }],
    ['repos', { repos: ['r'] }],
    ['tags', { tags: { FileType: ['PE32'] } }],
  ])('is true when the seed has %s', (_label, seed) => {
    expect(seedHasResources(seed)).toBe(true);
  });
});

describe('clausesEqual', () => {
  it('is true for identical single-value clauses', () => {
    expect(clausesEqual(single('Include', 'Include', 'File'), single('Include', 'Include', 'File'))).toBe(true);
  });

  it('is false when category/field/condition differ', () => {
    expect(clausesEqual(single('Include', 'Include', 'File'), single('Include', 'Include', 'Repo'))).toBe(false);
    expect(clausesEqual(single('a', 'x', 'v'), single('b', 'x', 'v'))).toBe(false);
  });

  it('compares multi-value clauses element-wise (order sensitive)', () => {
    expect(clausesEqual(multi('tag', 'k', ['a', 'b']), multi('tag', 'k', ['a', 'b']))).toBe(true);
    expect(clausesEqual(multi('tag', 'k', ['a', 'b']), multi('tag', 'k', ['b', 'a']))).toBe(false);
    expect(clausesEqual(multi('tag', 'k', ['a', 'b']), multi('tag', 'k', ['a']))).toBe(false);
  });

  it('distinguishes a single value from a one-element multi (different condition)', () => {
    expect(clausesEqual(single('tag', 'k', 'a'), multi('tag', 'k', ['a']))).toBe(false);
  });
});

describe('dedupeAppend', () => {
  it('appends a new clause', () => {
    const clauses = [single('Include', 'Include', 'File')];
    const clause = single('Include', 'Include', 'Repo');
    expect(dedupeAppend(clauses, clause)).toEqual([...clauses, clause]);
  });

  it('returns the original list (same reference) when the clause is already present', () => {
    const existing = single('Include', 'Include', 'File');
    const clauses = [existing];
    const result = dedupeAppend(clauses, single('Include', 'Include', 'File'));
    expect(result).toBe(clauses);
  });
});
