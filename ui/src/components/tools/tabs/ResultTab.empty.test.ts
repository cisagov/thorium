import { describe, it, expect } from 'vitest';

// project imports
import { isEmptyResult } from './ResultTab';

describe('isEmptyResult', () => {
  it('treats null/undefined as empty', () => {
    expect(isEmptyResult(null)).toBe(true);
    expect(isEmptyResult(undefined as never)).toBe(true);
  });

  it('treats empty/whitespace strings as empty', () => {
    expect(isEmptyResult('')).toBe(true);
    expect(isEmptyResult('   ')).toBe(true);
  });

  it('treats literal "{}" and "[]" strings as empty', () => {
    expect(isEmptyResult('{}')).toBe(true);
    expect(isEmptyResult(' [] ')).toBe(true);
  });

  it('treats an object with no keys as empty (json with no keys)', () => {
    expect(isEmptyResult({})).toBe(true);
  });

  it('treats an empty array as empty', () => {
    expect(isEmptyResult([])).toBe(true);
  });

  it('treats populated objects/arrays/strings as non-empty', () => {
    expect(isEmptyResult({ a: 1 })).toBe(false);
    expect(isEmptyResult([1])).toBe(false);
    expect(isEmptyResult('result text')).toBe(false);
  });

  it('treats falsy scalars (0, false) as non-empty values', () => {
    expect(isEmptyResult(0)).toBe(false);
    expect(isEmptyResult(false)).toBe(false);
  });
});
