import { describe, it, expect } from 'vitest';

// project imports
import { parseRetryAfter } from './users';

describe('parseRetryAfter', () => {
  it('reads a numeric Retry-After header from a string', () => {
    expect(parseRetryAfter({ 'retry-after': '600' })).toBe(600);
  });

  it('reads a numeric Retry-After header from a number', () => {
    expect(parseRetryAfter({ 'retry-after': 42 })).toBe(42);
  });

  it('defaults to 0 when missing, non-positive, or unparseable', () => {
    expect(parseRetryAfter({})).toBe(0);
    expect(parseRetryAfter({ 'retry-after': 'soon' })).toBe(0);
    expect(parseRetryAfter({ 'retry-after': '0' })).toBe(0);
    expect(parseRetryAfter({ 'retry-after': '-5' })).toBe(0);
    expect(parseRetryAfter(undefined)).toBe(0);
    expect(parseRetryAfter(null)).toBe(0);
  });
});
