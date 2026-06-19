import { describe, it, expect } from 'vitest';

// project imports
import {
  Clause,
  ClauseCondition,
  ClauseDraft,
  ClauseIsMulti,
  CondIsMulti,
  ConvertClauseToDraft,
  ConvertDraftToClause,
  DefaultClausesEntities,
  DraftIsComplete,
  GetMostSpecificCondition,
  GetValueString,
  NewTextClause,
  parseClauseCondition,
} from './ClauseTypes';

describe('DefaultClausesEntities', () => {
  it('hides the Results/Parent/submitter tags by default', () => {
    const clauses = DefaultClausesEntities();
    expect(clauses).toHaveLength(1);
    const [clause] = clauses;
    expect(clause.category).toBe('hidden tags');
    expect(clause.condition).toBe(ClauseCondition.Are);
    expect(ClauseIsMulti(clause)).toBe(true);
    if (ClauseIsMulti(clause)) {
      expect(clause.value.values).toEqual(['Results', 'Parent', 'submitter']);
    }
  });
});

describe('CondIsMulti / ClauseIsMulti', () => {
  it('classifies multi vs single conditions', () => {
    expect(CondIsMulti(ClauseCondition.IsOneOf)).toBe(true);
    expect(CondIsMulti(ClauseCondition.Are)).toBe(true);
    expect(CondIsMulti(ClauseCondition.Is)).toBe(false);
    expect(CondIsMulti(ClauseCondition.IsNot)).toBe(false);
  });

  it('narrows a clause by its condition', () => {
    const multi: Clause = { category: 'g', field: 'group', condition: ClauseCondition.IsOneOf, value: { values: ['a'] } };
    const single: Clause = { category: 't', field: 'text', condition: ClauseCondition.Is, value: { value: 'x' } };
    expect(ClauseIsMulti(multi)).toBe(true);
    expect(ClauseIsMulti(single)).toBe(false);
  });
});

describe('parseClauseCondition', () => {
  it('parses known conditions case-insensitively', () => {
    expect(parseClauseCondition('IS')).toBe(ClauseCondition.Is);
    expect(parseClauseCondition(' is one of ')).toBe(ClauseCondition.IsOneOf);
  });

  it('returns undefined for an unknown condition', () => {
    expect(parseClauseCondition('contains')).toBeUndefined();
  });
});

describe('GetMostSpecificCondition', () => {
  it('prefers the longest matching condition substring', () => {
    // "is one of" and "is" both appear; the longer one should win
    expect(GetMostSpecificCondition('group is one of')).toBe(ClauseCondition.IsOneOf);
    expect(GetMostSpecificCondition('name is bob')).toBe(ClauseCondition.Is);
  });

  it('returns undefined when no condition is present', () => {
    expect(GetMostSpecificCondition('just text')).toBeUndefined();
  });
});

describe('DraftIsComplete', () => {
  it('is true once category, field, condition and a value are set', () => {
    const draft: ClauseDraft = { category: 'g', field: 'group', condition: ClauseCondition.Is, value: 'a' };
    expect(DraftIsComplete(draft)).toBe(true);
  });

  it('is true for a multi condition with values', () => {
    const draft: ClauseDraft = { category: 'g', field: 'group', condition: ClauseCondition.IsOneOf, values: ['a'] };
    expect(DraftIsComplete(draft)).toBe(true);
  });

  it('is false when a value is missing', () => {
    const draft: ClauseDraft = { category: 'g', field: 'group', condition: ClauseCondition.Is };
    expect(DraftIsComplete(draft)).toBe(false);
  });
});

describe('ConvertDraftToClause / ConvertClauseToDraft', () => {
  it('round-trips a single-value clause', () => {
    const draft: ClauseDraft = { category: 'text', field: 'text', condition: ClauseCondition.Is, value: 'foo' };
    const clause = ConvertDraftToClause(draft);
    expect(clause).toEqual({ category: 'text', field: 'text', condition: ClauseCondition.Is, value: { value: 'foo' } });
    expect(ConvertClauseToDraft(clause)).toEqual(draft);
  });

  it('round-trips a multi-value clause', () => {
    const draft: ClauseDraft = { category: 'group', field: 'group', condition: ClauseCondition.IsOneOf, values: ['a', 'b'] };
    const clause = ConvertDraftToClause(draft);
    expect(clause).toEqual({ category: 'group', field: 'group', condition: ClauseCondition.IsOneOf, value: { values: ['a', 'b'] } });
    expect(ConvertClauseToDraft(clause)).toEqual(draft);
  });

  it('throws when a single condition draft has no value', () => {
    expect(() => ConvertDraftToClause({ category: 'text', field: 'text', condition: ClauseCondition.Is })).toThrow();
  });
});

describe('GetValueString', () => {
  it('joins multi values with a comma', () => {
    const clause: Clause = { category: 'g', field: 'group', condition: ClauseCondition.IsOneOf, value: { values: ['a', 'b'] } };
    expect(GetValueString(clause)).toBe('a, b');
  });

  it('returns the single value directly', () => {
    expect(GetValueString(NewTextClause('hello'))).toBe('hello');
  });
});

describe('NewTextClause', () => {
  it('builds a text/Is clause from a string', () => {
    expect(NewTextClause('q')).toEqual({ category: 'text', field: 'text', condition: ClauseCondition.Is, value: { value: 'q' } });
  });
});
