import { describe, it, expect } from 'vitest';

// project imports
import { numLeadHashes, splitTableSections } from './Tables';

describe('numLeadHashes', () => {
  it('returns count 0 and the full string when there is no leading hash', () => {
    expect(numLeadHashes('plain heading')).toEqual({ count: 0, header: 'plain heading' });
  });

  it('counts a single leading hash and strips it from the header text', () => {
    expect(numLeadHashes('# Title')).toEqual({ count: 1, header: ' Title' });
  });

  it('counts multiple leading hashes', () => {
    expect(numLeadHashes('### Section')).toEqual({ count: 3, header: ' Section' });
  });

  it('only counts hashes at the start of the string', () => {
    expect(numLeadHashes('no # in the middle')).toEqual({ count: 0, header: 'no # in the middle' });
  });

  it('handles a string that is only hashes', () => {
    expect(numLeadHashes('##')).toEqual({ count: 2, header: '' });
  });
});

describe('splitTableSections', () => {
  it('returns a single table segment for contiguous csv rows', () => {
    const input = 'a,b\n1,2\n3,4';
    expect(splitTableSections(input)).toEqual(['a,b\n1,2\n3,4\n']);
  });

  it('emits headings and blank lines as their own segments', () => {
    const input = '# Heading\n\nplain';
    expect(splitTableSections(input)).toEqual(['# Heading', '', 'plain']);
  });

  it('closes an open table when a non-table line follows it', () => {
    const input = 'a,b\n1,2\n# Next';
    expect(splitTableSections(input)).toEqual(['a,b\n1,2\n', '# Next']);
  });

  it('emits a trailing table block when the text ends with csv rows', () => {
    const input = '# Heading\na,b\n1,2';
    expect(splitTableSections(input)).toEqual(['# Heading', 'a,b\n1,2\n']);
  });

  it('treats lines without a comma as non-table segments', () => {
    const input = 'just text\na,b';
    expect(splitTableSections(input)).toEqual(['just text', 'a,b\n']);
  });

  it('returns an empty-string segment for empty input', () => {
    expect(splitTableSections('')).toEqual(['']);
  });
});
