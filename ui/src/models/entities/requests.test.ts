import { describe, it, expect } from 'vitest';

// project imports
import { entityRequestKind, EntityRequest } from './requests';

const make = (metadata: EntityRequest['metadata']): EntityRequest => ({
  name: 'x',
  metadata,
  groups: [],
  tags: {},
  description: null,
});

describe('entityRequestKind', () => {
  it('reads the kind from object-variant metadata', () => {
    expect(entityRequestKind(make({ Device: { urls: [] } }))).toBe('Device');
    expect(entityRequestKind(make({ Vendor: { countries: [] } }))).toBe('Vendor');
  });

  it('reads the kind from unit-variant (string) metadata', () => {
    expect(entityRequestKind(make('Other'))).toBe('Other');
    expect(entityRequestKind(make('WindowsProcessTree'))).toBe('WindowsProcessTree');
  });

  it('returns empty string for empty metadata', () => {
    expect(entityRequestKind(make({}))).toBe('');
  });
});
