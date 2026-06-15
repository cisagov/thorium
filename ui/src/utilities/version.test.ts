import { describe, it, expect } from 'vitest';

// project imports
import { formatImageVersion, versionLabel } from './version';

describe('formatImageVersion', () => {
  it('returns empty string when version is undefined', () => {
    expect(formatImageVersion(undefined)).toBe('');
  });

  it('formats a basic SemVer', () => {
    expect(formatImageVersion({ SemVer: { major: 1, minor: 2, patch: 3, pre: '', build: '' } })).toBe('v1.2.3');
  });

  it('includes the pre-release suffix', () => {
    expect(formatImageVersion({ SemVer: { major: 1, minor: 0, patch: 0, pre: 'alpha.1', build: '' } })).toBe('v1.0.0-alpha.1');
  });

  it('includes the build suffix', () => {
    expect(formatImageVersion({ SemVer: { major: 2, minor: 1, patch: 0, pre: '', build: 'build.5' } })).toBe('v2.1.0+build.5');
  });

  it('includes both pre-release and build suffixes', () => {
    expect(formatImageVersion({ SemVer: { major: 1, minor: 2, patch: 3, pre: 'rc.1', build: 'build.9' } })).toBe('v1.2.3-rc.1+build.9');
  });

  it('renders a Custom version as its raw string', () => {
    expect(formatImageVersion({ Custom: 'nightly-2026-06-15' })).toBe('nightly-2026-06-15');
  });

  it('prefers Custom over SemVer when both are present', () => {
    expect(formatImageVersion({ Custom: 'edge', SemVer: { major: 9, minor: 9, patch: 9, pre: '', build: '' } })).toBe('edge');
  });

  it('returns empty string for an empty version object', () => {
    expect(formatImageVersion({})).toBe('');
  });
});

describe('versionLabel', () => {
  it('returns the raw uploaded string when the timestamp is unparseable', () => {
    expect(versionLabel('not-a-date')).toBe('not-a-date');
  });

  it('appends the formatted version when one is present', () => {
    const label = versionLabel('2026-06-15T12:00:00Z', { SemVer: { major: 1, minor: 2, patch: 3, pre: '', build: '' } });
    expect(label.endsWith(' - v1.2.3')).toBe(true);
  });

  it('returns just the date portion when no version is present', () => {
    const label = versionLabel('2026-06-15T12:00:00Z');
    expect(label.includes(' - ')).toBe(false);
    expect(label.length).toBeGreaterThan(0);
  });

  it('appends a Custom version as its raw string', () => {
    const label = versionLabel('2026-06-15T12:00:00Z', { Custom: 'nightly' });
    expect(label.endsWith(' - nightly')).toBe(true);
  });
});
