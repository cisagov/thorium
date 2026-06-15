import { describe, it, expect } from 'vitest';

// project imports
import { humanize } from './humanize';

describe('humanize', () => {
  it('splits snake_case into title-cased words', () => {
    expect(humanize('image_path')).toBe('Image Path');
    expect(humanize('source_port')).toBe('Source Port');
  });

  it('splits camelCase and PascalCase on lower→upper boundaries', () => {
    expect(humanize('windowsProcess')).toBe('Windows Process');
    expect(humanize('WindowsProcessTree')).toBe('Windows Process Tree');
    expect(humanize('NetworkConnection')).toBe('Network Connection');
  });

  it('leaves acronym runs intact (no split between consecutive capitals)', () => {
    expect(humanize('ContainsCVE')).toBe('Contains CVE');
    expect(humanize('CVE')).toBe('CVE');
  });

  it('title-cases single words and handles empty input', () => {
    expect(humanize('device')).toBe('Device');
    expect(humanize('')).toBe('');
  });
});
