import { describe, expect, it } from 'vitest';

// project imports
import { generateCopyName } from './naming';

describe('generateCopyName', () => {
  it('appends a -copy suffix with no spaces or parentheses', () => {
    expect(generateCopyName('foo', [])).toBe('foo-copy');
  });

  it('numbers subsequent copies with a dash separator', () => {
    expect(generateCopyName('foo', ['foo-copy'])).toBe('foo-copy-2');
    expect(generateCopyName('foo', ['foo-copy', 'foo-copy-2'])).toBe('foo-copy-3');
  });

  it('does not stack suffixes when copying an existing copy', () => {
    expect(generateCopyName('foo-copy', ['foo-copy'])).toBe('foo-copy-2');
    expect(generateCopyName('foo-copy-2', [])).toBe('foo-copy');
  });

  it('strips a legacy " (copy)" suffix and replaces it with -copy', () => {
    expect(generateCopyName('foo (copy)', [])).toBe('foo-copy');
    expect(generateCopyName('foo (copy 3)', [])).toBe('foo-copy');
  });

  it('preserves parentheses in the base name that are not a copy suffix', () => {
    expect(generateCopyName('my image (v2)', [])).toBe('my image (v2)-copy');
  });

  it('never produces spaces or parentheses in the generated suffix', () => {
    const result = generateCopyName('foo', ['foo-copy', 'foo-copy-2', 'foo-copy-3']);
    expect(result).toBe('foo-copy-4');
    expect(result.slice('foo'.length)).not.toMatch(/[ ()]/);
  });
});
