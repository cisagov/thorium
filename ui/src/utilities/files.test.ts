import { describe, it, expect } from 'vitest';

// project imports
import { getUniqueFileNames, isValidSha256, MAX_FILE_NAME_DISPLAY_LENGTH } from './files';
import { SubmissionChunk } from '@models/files';

/** Build a minimal SubmissionChunk carrying just the fields getUniqueFileNames reads. */
function chunk(name: string | undefined): SubmissionChunk {
  return {
    id: 'id',
    name,
    description: null,
    groups: [],
    submitter: 'tester',
    uploaded: '2026-01-01T00:00:00Z',
    // origin is not read by getUniqueFileNames; cast keeps the fixture minimal
  } as unknown as SubmissionChunk;
}

const HASH = 'a'.repeat(64);
const HASH_2 = 'b'.repeat(64);

describe('isValidSha256', () => {
  it('accepts 64 lowercase hex characters', () => {
    expect(isValidSha256('0123456789abcdef'.repeat(4))).toBe(true);
  });

  it('accepts uppercase and mixed-case hex', () => {
    expect(isValidSha256('A'.repeat(64))).toBe(true);
    expect(isValidSha256('aB'.repeat(32))).toBe(true);
  });

  it('rejects strings that are not exactly 64 chars', () => {
    expect(isValidSha256('a'.repeat(63))).toBe(false);
    expect(isValidSha256('a'.repeat(65))).toBe(false);
    expect(isValidSha256('')).toBe(false);
  });

  it('rejects 64-char strings containing non-hex characters', () => {
    expect(isValidSha256('g'.repeat(64))).toBe(false);
    expect(isValidSha256('a'.repeat(63) + ' ')).toBe(false);
  });
});

describe('getUniqueFileNames', () => {
  it('strips the agent-prepended children path', () => {
    expect(getUniqueFileNames([chunk('/tmp/thorium/children/1234-uuid/name.bin')])).toBe('name.bin');
  });

  it('strips a leading ./', () => {
    expect(getUniqueFileNames([chunk('./foo.txt')])).toBe('foo.txt');
  });

  it('de-duplicates identical names, sorts, and joins with three spaces', () => {
    expect(getUniqueFileNames([chunk('b.txt'), chunk('a.txt'), chunk('b.txt')])).toBe('a.txt,   b.txt');
  });

  it('drops sha256-hash names when a non-hash name also exists', () => {
    expect(getUniqueFileNames([chunk(HASH), chunk('real.txt')])).toBe('real.txt');
  });

  it('returns the full hash list when every name is a sha256 hash', () => {
    // hashes sort so HASH (a...) precedes HASH_2 (b...)
    expect(getUniqueFileNames([chunk(HASH_2), chunk(HASH)])).toBe(`${HASH},   ${HASH_2}`);
  });

  it('handles a missing name without throwing (treats it as empty string)', () => {
    expect(() => getUniqueFileNames([chunk(undefined)])).not.toThrow();
    expect(getUniqueFileNames([chunk(undefined)])).toBe('');
  });

  it('truncates a display string longer than the max with an ellipsis', () => {
    // one long, unique, non-hash name that exceeds the cutoff on its own
    const long = 'x'.repeat(MAX_FILE_NAME_DISPLAY_LENGTH + 50);
    const result = getUniqueFileNames([chunk(long)]);
    expect(result.length).toBe(MAX_FILE_NAME_DISPLAY_LENGTH + 3);
    expect(result.endsWith('...')).toBe(true);
  });

  it('does not truncate a display string exactly at the max length', () => {
    const exact = 'y'.repeat(MAX_FILE_NAME_DISPLAY_LENGTH);
    const result = getUniqueFileNames([chunk(exact)]);
    expect(result.length).toBe(MAX_FILE_NAME_DISPLAY_LENGTH);
    expect(result.endsWith('...')).toBe(false);
  });
});
