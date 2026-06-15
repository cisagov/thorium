import { describe, expect, it } from 'vitest';

// project imports
import { resetFilterClauses, toggleTagValue } from './tagFilter';
import { ClauseCondition, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';

/** Build the single `IsOneOf` tag clause for a key with the given values. */
function tagClause(key: string, values: string[]): Clause {
  return { category: 'tag', field: key, condition: ClauseCondition.IsOneOf, value: { values } };
}

/** Build a single-value `Is` tag clause for a key — the shape the URL round-trip re-decodes tag clauses into. */
function singleIsTagClause(key: string, value: string): Clause {
  return { category: 'tag', field: key, condition: ClauseCondition.Is, value: { value } };
}

/** Build a single-value `Is` tag clause (the pre-merge shape used elsewhere). */
function includeClause(value: string): Clause {
  return { category: 'Include', field: 'Include', condition: ClauseCondition.Is, value: { value } };
}

/** Build a depth clause. */
function depthClause(depth: number): Clause {
  return { category: 'depth', field: 'depth', condition: ClauseCondition.Is, value: { value: String(depth) } };
}

/** Build the default hidden-tags clause. */
function hiddenTagsClause(): Clause {
  return {
    category: 'hidden tags',
    field: 'hidden tags',
    condition: ClauseCondition.Are,
    value: { values: ['Results', 'Parent', 'submitter'] },
  };
}

describe('toggleTagValue', () => {
  it('creates a fresh IsOneOf clause when the key is absent', () => {
    const result = toggleTagValue([], 'FileType', 'PE32');
    expect(result).toEqual([tagClause('FileType', ['PE32'])]);
  });

  it('adds a value to the existing clause for the key', () => {
    const result = toggleTagValue([tagClause('FileType', ['PE32'])], 'FileType', 'ELF');
    expect(result).toEqual([tagClause('FileType', ['PE32', 'ELF'])]);
  });

  it('removes a value already present in the clause', () => {
    const result = toggleTagValue([tagClause('FileType', ['PE32', 'ELF'])], 'FileType', 'PE32');
    expect(result).toEqual([tagClause('FileType', ['ELF'])]);
  });

  it('drops the whole clause when its last value is removed', () => {
    const result = toggleTagValue([tagClause('FileType', ['PE32'])], 'FileType', 'PE32');
    expect(result).toEqual([]);
  });

  it('preserves other clauses and their order', () => {
    const clauses = [depthClause(2), tagClause('FileType', ['PE32']), includeClause('Device')];
    const result = toggleTagValue(clauses, 'FileType', 'ELF');
    expect(result).toEqual([depthClause(2), tagClause('FileType', ['PE32', 'ELF']), includeClause('Device')]);
  });

  it('appends a new key clause after existing clauses, preserving order', () => {
    const clauses = [depthClause(2), includeClause('Device')];
    const result = toggleTagValue(clauses, 'FileType', 'PE32');
    expect(result).toEqual([depthClause(2), includeClause('Device'), tagClause('FileType', ['PE32'])]);
  });

  it('does not touch a same-key clause of a different tag key', () => {
    const clauses = [tagClause('FileType', ['PE32']), tagClause('os', ['linux'])];
    const result = toggleTagValue(clauses, 'os', 'windows');
    expect(result).toEqual([tagClause('FileType', ['PE32']), tagClause('os', ['linux', 'windows'])]);
  });

  // regression: the URL round-trip re-decodes a single-value tag clause as `Is`, not `IsOneOf`; a new
  // click must merge into that clause rather than appending a second one for the same key
  it('merges into an existing single-Is tag clause for the key instead of appending', () => {
    const clauses = [singleIsTagClause('FileType', 'PE32')];
    const result = toggleTagValue(clauses, 'FileType', 'ELF');
    expect(result).toEqual([tagClause('FileType', ['PE32', 'ELF'])]);
  });

  // regression: a round-trip can leave more than one tag clause for the same key; a toggle must collapse
  // them all into a single clause at the first clause's position
  it('collapses two pre-existing duplicate tag clauses for the key into one on toggle', () => {
    const clauses = [singleIsTagClause('FileType', 'PE32'), singleIsTagClause('FileType', 'ELF')];
    const result = toggleTagValue(clauses, 'FileType', 'MachO');
    expect(result).toEqual([tagClause('FileType', ['PE32', 'ELF', 'MachO'])]);
  });

  // regression: toggling the same value twice adds then removes it, leaving no clause when it was the last
  it('toggling the same value twice removes it', () => {
    const once = toggleTagValue([], 'FileType', 'PE32');
    const twice = toggleTagValue(once, 'FileType', 'PE32');
    expect(twice).toEqual([]);
  });

  // regression: toggling off a value from a round-tripped single-Is clause drops the clause entirely
  it('removes a single-Is tag clause when its only value is toggled off', () => {
    const clauses = [singleIsTagClause('FileType', 'PE32')];
    const result = toggleTagValue(clauses, 'FileType', 'PE32');
    expect(result).toEqual([]);
  });
});

describe('resetFilterClauses', () => {
  it('returns an empty list for empty input', () => {
    expect(resetFilterClauses([])).toEqual([]);
  });

  it('keeps depth and hidden-tags clauses, dropping the rest', () => {
    const clauses = [
      depthClause(3),
      hiddenTagsClause(),
      tagClause('FileType', ['PE32']),
      includeClause('Device'),
      { category: 'text', field: 'text', condition: ClauseCondition.Is, value: { value: 'foo' } } as Clause,
    ];
    const result = resetFilterClauses(clauses);
    expect(result).toEqual([depthClause(3), hiddenTagsClause()]);
  });

  it('preserves the order of kept clauses', () => {
    const clauses = [hiddenTagsClause(), tagClause('FileType', ['PE32']), depthClause(1)];
    const result = resetFilterClauses(clauses);
    expect(result).toEqual([hiddenTagsClause(), depthClause(1)]);
  });
});
