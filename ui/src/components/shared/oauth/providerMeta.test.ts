import { describe, it, expect } from 'vitest';
import { FaGoogle } from 'react-icons/fa6';
import { SiOpenid } from 'react-icons/si';

// project imports
import { getProviderMeta, sortProviders } from './providerMeta';

describe('getProviderMeta', () => {
  it('resolves known providers case-insensitively', () => {
    const meta = getProviderMeta('GOOGLE');
    expect(meta.name).toBe('GOOGLE');
    expect(meta.label).toBe('Google');
    expect(meta.Icon).toBe(FaGoogle);
  });

  it('maps aliases (azure -> Microsoft)', () => {
    expect(getProviderMeta('azure').label).toBe('Microsoft');
  });

  it('falls back to a generic icon and title-cased label for unknown providers', () => {
    const meta = getProviderMeta('corp-okta_idp');
    expect(meta.label).toBe('Corp Okta Idp');
    expect(meta.Icon).toBe(SiOpenid);
    expect(meta.name).toBe('corp-okta_idp');
  });
});

describe('sortProviders', () => {
  it('sorts alphabetically without mutating the input', () => {
    const input = ['github', 'azure', 'okta'];
    const sorted = sortProviders(input);
    expect(sorted).toEqual(['azure', 'github', 'okta']);
    expect(input).toEqual(['github', 'azure', 'okta']);
  });
});
