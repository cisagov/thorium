import { describe, it, expect } from 'vitest';

// project imports
import { sanitizeReturnPath } from './oauthReturn';

describe('sanitizeReturnPath', () => {
  it('keeps safe same-origin relative paths', () => {
    expect(sanitizeReturnPath('/files')).toBe('/files');
    expect(sanitizeReturnPath('/graph?seed=abc#node')).toBe('/graph?seed=abc#node');
    expect(sanitizeReturnPath('/')).toBe('/');
  });

  it('falls back to / for empty/missing input', () => {
    expect(sanitizeReturnPath(undefined)).toBe('/');
    expect(sanitizeReturnPath(null)).toBe('/');
    expect(sanitizeReturnPath('')).toBe('/');
  });

  it('rejects open-redirect vectors', () => {
    expect(sanitizeReturnPath('//evil.com')).toBe('/');
    expect(sanitizeReturnPath('/\\evil.com')).toBe('/');
    expect(sanitizeReturnPath('https://evil.com')).toBe('/');
    expect(sanitizeReturnPath('javascript:alert(1)')).toBe('/');
    expect(sanitizeReturnPath('files')).toBe('/');
  });
});
