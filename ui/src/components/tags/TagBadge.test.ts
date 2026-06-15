import { describe, it, expect } from 'vitest';

// project imports
import { truncateBadgeText } from './TagBadge';

describe('truncateBadgeText', () => {
  it('returns short text unchanged', () => {
    expect(truncateBadgeText('short')).toBe('short');
  });

  it('returns text of exactly 30 chars unchanged', () => {
    const text = 'a'.repeat(30);
    expect(truncateBadgeText(text)).toBe(text);
  });

  it('truncates text longer than 30 chars to 30 chars plus an ellipsis', () => {
    const text = 'a'.repeat(40);
    const result = truncateBadgeText(text);
    expect(result).toBe('a'.repeat(30) + '...');
    expect(result).toHaveLength(33);
  });

  it('truncates a 31 char string', () => {
    const text = 'a'.repeat(31);
    expect(truncateBadgeText(text)).toBe('a'.repeat(30) + '...');
  });

  it('returns an empty string unchanged', () => {
    expect(truncateBadgeText('')).toBe('');
  });
});
