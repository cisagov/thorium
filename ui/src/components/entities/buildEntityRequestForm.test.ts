import { describe, it, expect } from 'vitest';

// project imports
import { appendBracketed, buildEntityRequestForm } from './utilities';
import { EntityRequest } from '@models/entities/requests';

describe('appendBracketed', () => {
  it('encodes scalars, arrays, and nested objects with bracket keys', () => {
    const form = new FormData();
    appendBracketed(form, 'metadata[critical_system]', true);
    appendBracketed(form, 'metadata[urls]', ['a', 'b']);
    appendBracketed(form, 'metadata[nested]', { inner: ['x'] });
    expect(form.get('metadata[critical_system]')).toBe('true');
    expect(form.getAll('metadata[urls][]')).toEqual(['a', 'b']);
    expect(form.getAll('metadata[nested][inner][]')).toEqual(['x']);
  });

  it('skips null/undefined values', () => {
    const form = new FormData();
    appendBracketed(form, 'metadata[x]', null);
    appendBracketed(form, 'metadata[y]', undefined);
    expect([...form.keys()]).toHaveLength(0);
  });
});

describe('buildEntityRequestForm', () => {
  const base: EntityRequest = {
    name: 'evil.exe',
    metadata: { Device: { urls: ['http://x'], critical_system: true } },
    groups: ['system', 'team'],
    tags: { family: ['emotet'], tlp: ['amber'] },
    description: 'a flagged device',
  };

  it('serializes name, kind, groups, tags, description', () => {
    const form = buildEntityRequestForm(base);
    expect(form.get('name')).toBe('evil.exe');
    expect(form.get('kind')).toBe('Device');
    expect(form.getAll('groups[]')).toEqual(['system', 'team']);
    expect(form.getAll('tags[family][]')).toEqual(['emotet']);
    expect(form.getAll('tags[tlp][]')).toEqual(['amber']);
    expect(form.get('description')).toBe('a flagged device');
  });

  it('serializes object-variant metadata fields under metadata[...]', () => {
    const form = buildEntityRequestForm(base);
    expect(form.getAll('metadata[urls][]')).toEqual(['http://x']);
    expect(form.get('metadata[critical_system]')).toBe('true');
  });

  it('handles unit-variant (string) metadata as just a kind', () => {
    const form = buildEntityRequestForm({ ...base, metadata: 'Other', tags: {}, description: null });
    expect(form.get('kind')).toBe('Other');
    expect([...form.keys()].some((k) => k.startsWith('metadata['))).toBe(false);
  });

  it('omits an empty/missing description', () => {
    const form = buildEntityRequestForm({ ...base, description: null });
    expect(form.get('description')).toBeNull();
  });
});
