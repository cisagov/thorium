import { describe, it, expect } from 'vitest';

// project imports
import { cleanDescription } from './description';

describe('cleanDescription', () => {
  it('returns a normal description unchanged', () => {
    expect(cleanDescription('A useful pipeline')).toBe('A useful pipeline');
  });

  it('treats the literal "null" sentinel string as empty', () => {
    expect(cleanDescription('null')).toBe('');
  });

  it('returns empty string for undefined', () => {
    expect(cleanDescription(undefined)).toBe('');
  });

  it('returns empty string for null', () => {
    expect(cleanDescription(null)).toBe('');
  });

  it('returns empty string for an empty description', () => {
    expect(cleanDescription('')).toBe('');
  });
});
