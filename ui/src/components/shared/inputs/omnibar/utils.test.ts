import { describe, it, expect } from 'vitest';

// project imports
import {
  getGroupsFromClauses,
  getHiddenTagsFromClauses,
  getIndexesFromClauses,
  getLimitFromClauses,
  getSearchTextFromClauses,
  getStringFieldFromClauses,
  getStringFieldListFromClauses,
  getTagsFromClauses,
  matchesStringClauses,
} from './utils';
import { Clause, ClauseCondition } from './ClauseTypes';
import { ElasticIndex } from '@models/search';

// helpers to build clauses without repeating the discriminated-union boilerplate
function single(category: string, field: string, value: string): Clause {
  return { category, field, condition: ClauseCondition.Is, value: { value } };
}
function multi(category: string, field: string, values: string[]): Clause {
  return { category, field, condition: ClauseCondition.IsOneOf, value: { values } };
}

describe('getSearchTextFromClauses', () => {
  it('returns empty string when there are no text clauses', () => {
    expect(getSearchTextFromClauses([single('group', 'group', 'g1')])).toBe('');
  });

  it('returns the value of a single text clause', () => {
    expect(getSearchTextFromClauses([single('text', 'text', 'malware')])).toBe('"malware"');
  });
});

describe('getGroupsFromClauses', () => {
  it('returns [] when no group clause exists', () => {
    expect(getGroupsFromClauses([single('text', 'text', 'x')])).toEqual([]);
  });

  it('reads a single group clause', () => {
    expect(getGroupsFromClauses([single('group', 'group', 'team-a')])).toEqual(['team-a']);
  });

  it('reads a multi-value group clause', () => {
    expect(getGroupsFromClauses([multi('group', 'group', ['a', 'b'])])).toEqual(['a', 'b']);
  });
});

describe('getIndexesFromClauses', () => {
  it('maps index clause values to ElasticIndex enum values', () => {
    const values = Object.values(ElasticIndex);
    const clauses = [multi('index', 'index', [values[0]])];
    expect(getIndexesFromClauses(clauses)).toEqual([values[0]]);
  });

  it('returns [] when no index clause exists', () => {
    expect(getIndexesFromClauses([single('group', 'group', 'g')])).toEqual([]);
  });
});

describe('getLimitFromClauses', () => {
  it('returns the provided default when no limit clause exists', () => {
    expect(getLimitFromClauses([], 25)).toBe(25);
    expect(getLimitFromClauses([], 50)).toBe(50);
  });

  it('parses an integer limit clause', () => {
    expect(getLimitFromClauses([single('limit', 'limit', '100')], 25)).toBe(100);
  });

  it('falls back to the default for a non-integer limit', () => {
    expect(getLimitFromClauses([single('limit', 'limit', 'abc')], 25)).toBe(25);
  });
});

describe('getStringFieldFromClauses', () => {
  it('returns the value of a matching single-value field', () => {
    expect(getStringFieldFromClauses([single('image', 'creator', 'alice')], 'creator')).toBe('alice');
  });

  it('returns empty string when the field is absent', () => {
    expect(getStringFieldFromClauses([single('image', 'name', 'tool')], 'creator')).toBe('');
  });
});

describe('getStringFieldListFromClauses', () => {
  it('collects values from both single and multi clauses for a field', () => {
    const clauses = [single('group', 'Users', 'alice'), multi('group', 'Users', ['bob', 'carol'])];
    expect(getStringFieldListFromClauses(clauses, 'Users')).toEqual(['alice', 'bob', 'carol']);
  });

  it('returns [] when no clause matches the field', () => {
    expect(getStringFieldListFromClauses([single('group', 'Owners', 'x')], 'Users')).toEqual([]);
  });
});

describe('matchesStringClauses', () => {
  const includes = (field: string, value: string): Clause => ({
    category: field,
    field,
    condition: ClauseCondition.Includes,
    value: { value },
  });

  it('returns true when no clause targets the field', () => {
    expect(matchesStringClauses([single('group', 'group', 'g')], 'username', 'alice')).toBe(true);
  });

  it('matches a substring for an "includes" clause', () => {
    expect(matchesStringClauses([includes('username', 'ali')], 'username', 'alice')).toBe(true);
    expect(matchesStringClauses([includes('username', 'ali')], 'username', 'bob')).toBe(false);
  });

  it('requires an exact match for an "is" clause', () => {
    // 'alice' is a substring of 'alice2' but exact match must fail
    expect(matchesStringClauses([single('username', 'username', 'alice')], 'username', 'alice')).toBe(true);
    expect(matchesStringClauses([single('username', 'username', 'alice')], 'username', 'alice2')).toBe(false);
  });

  it('is case-sensitive', () => {
    expect(matchesStringClauses([includes('username', 'ALI')], 'username', 'alice')).toBe(false);
  });

  it('matches one of a multi "is one of" clause exactly', () => {
    expect(matchesStringClauses([multi('username', 'username', ['alice', 'bob'])], 'username', 'bob')).toBe(true);
    expect(matchesStringClauses([multi('username', 'username', ['alice', 'bob'])], 'username', 'bo')).toBe(false);
  });
});

describe('getHiddenTagsFromClauses', () => {
  it('returns the de-duplicated union of hidden-tag multi clauses', () => {
    const clauses = [
      multi('hidden tags', 'hidden tags', ['Results', 'Parent']),
      multi('hidden tags', 'hidden tags', ['Parent', 'submitter']),
    ];
    expect(getHiddenTagsFromClauses(clauses)).toEqual(['Results', 'Parent', 'submitter']);
  });

  it('ignores single-value clauses and returns [] when none present', () => {
    expect(getHiddenTagsFromClauses([single('text', 'text', 'x')])).toEqual([]);
  });
});

describe('getTagsFromClauses', () => {
  it('returns {} when there are no tag clauses', () => {
    expect(getTagsFromClauses([single('group', 'group', 'g1')])).toEqual({});
  });

  it('collects a single tag key/value', () => {
    expect(getTagsFromClauses([single('tag', 'family', 'emotet')])).toEqual({ family: ['emotet'] });
  });

  it('merges and de-duplicates values for the same key', () => {
    const clauses = [single('tag', 'family', 'emotet'), single('tag', 'family', 'trickbot'), single('tag', 'family', 'emotet')];
    expect(getTagsFromClauses(clauses)).toEqual({ family: ['emotet', 'trickbot'] });
  });

  it('groups multiple keys and supports multi-value clauses', () => {
    const clauses = [single('tag', 'av', 'malicious'), multi('tag', 'os', ['windows', 'linux'])];
    expect(getTagsFromClauses(clauses)).toEqual({ av: ['malicious'], os: ['windows', 'linux'] });
  });

  it('excludes the hidden-tags display filter', () => {
    const clauses = [single('tag', 'family', 'emotet'), multi('hidden tags', 'hidden tags', ['Results', 'Parent'])];
    expect(getTagsFromClauses(clauses)).toEqual({ family: ['emotet'] });
  });
});
