import { describe, it, expect, vi } from 'vitest';

vi.mock('@thorpi/entities', () => ({
  listEntities: vi.fn().mockResolvedValue({ entityList: [], entityCursor: null }),
}));

// project imports
import { buildUpdateEntityForm, buildCreateEntityForm, DEFAULT_LIST_LIMIT } from './utilities';
import { Entities, EntityCreateTypes, EntityTypes } from '@models/entities/entities';
import { BlankDevice } from '@models/entities/devices';
import { BlankCreateVendor, CreateVendor } from '@models/entities/vendors';

function formEntries(form: FormData): Record<string, string[]> {
  const result: Record<string, string[]> = {};
  form.forEach((value, key) => {
    if (!result[key]) result[key] = [];
    result[key].push(value as string);
  });
  return result;
}

describe('DEFAULT_LIST_LIMIT', () => {
  it('is 25', () => {
    expect(DEFAULT_LIST_LIMIT).toBe(25);
  });
});

describe('buildUpdateEntityForm', () => {
  function makeDevice(overrides: Record<string, unknown> = {}): EntityTypes {
    const base = structuredClone(BlankDevice);
    return { ...base, ...overrides };
  }

  it('includes name when changed', () => {
    const entity = makeDevice({ name: 'Old' });
    const pending = makeDevice({ name: 'New' });
    const form = buildUpdateEntityForm(entity, pending);
    expect(form.get('name')).toBe('New');
  });

  it('does not include name when unchanged', () => {
    const entity = makeDevice({ name: 'Same' });
    const pending = makeDevice({ name: 'Same' });
    const form = buildUpdateEntityForm(entity, pending);
    expect(form.has('name')).toBe(false);
  });

  it('includes added groups', () => {
    const entity = makeDevice({ groups: ['g1'] });
    const pending = makeDevice({ groups: ['g1', 'g2'] });
    const form = buildUpdateEntityForm(entity, pending);
    const entries = formEntries(form);
    expect(entries['add_groups[]']).toEqual(['g2']);
    expect(entries['remove_groups[]']).toBeUndefined();
  });

  it('includes removed groups', () => {
    const entity = makeDevice({ groups: ['g1', 'g2'] });
    const pending = makeDevice({ groups: ['g1'] });
    const form = buildUpdateEntityForm(entity, pending);
    const entries = formEntries(form);
    expect(entries['remove_groups[]']).toEqual(['g2']);
  });

  it('includes image file when provided', () => {
    const entity = makeDevice();
    const pending = makeDevice();
    const file = new File(['data'], 'icon.png');
    const form = buildUpdateEntityForm(entity, pending, file);
    expect(form.has('image')).toBe(true);
  });

  it('sets clear_image when clearImage is true and no file', () => {
    const entity = makeDevice();
    const pending = makeDevice();
    const form = buildUpdateEntityForm(entity, pending, null, true);
    expect(form.get('clear_image')).toBe('true');
  });

  it('clears description when set to empty', () => {
    const entity = makeDevice({ description: 'Old desc' });
    const pending = makeDevice({ description: '' });
    const form = buildUpdateEntityForm(entity, pending);
    expect(form.get('clear_description')).toBe('true');
  });

  it('sets description when changed', () => {
    const entity = makeDevice({ description: 'Old' });
    const pending = makeDevice({ description: 'New' });
    const form = buildUpdateEntityForm(entity, pending);
    expect(form.get('description')).toBe('New');
  });

  it('returns empty form when nothing changed', () => {
    const entity = makeDevice();
    const pending = makeDevice();
    const form = buildUpdateEntityForm(entity, pending);
    const entries = formEntries(form);
    expect(Object.keys(entries)).toHaveLength(0);
  });
});

describe('buildCreateEntityForm', () => {
  function makeCreateVendor(overrides: Partial<CreateVendor> = {}): EntityCreateTypes {
    const base: CreateVendor = structuredClone(BlankCreateVendor);
    return { ...base, ...overrides };
  }

  it('sets name and kind', () => {
    const entity = makeCreateVendor({ name: 'TestVendor' });
    const form = buildCreateEntityForm(entity);
    expect(form.get('name')).toBe('TestVendor');
    expect(form.get('kind')).toBe(Entities.Vendor);
  });

  it('includes all groups', () => {
    const entity = makeCreateVendor({ groups: ['g1', 'g2'] });
    const form = buildCreateEntityForm(entity);
    const entries = formEntries(form);
    expect(entries['groups[]']).toEqual(['g1', 'g2']);
  });

  it('includes description when present', () => {
    const entity = makeCreateVendor({ description: 'Test desc' });
    const form = buildCreateEntityForm(entity);
    expect(form.get('description')).toBe('Test desc');
  });

  it('does not include description when empty', () => {
    const entity = makeCreateVendor({ description: '' });
    const form = buildCreateEntityForm(entity);
    expect(form.has('description')).toBe(false);
  });

  it('includes tags', () => {
    const entity = makeCreateVendor({ tags: { TLP: ['RED'], MBC: ['T1059'] } });
    const form = buildCreateEntityForm(entity);
    const entries = formEntries(form);
    expect(entries['tags[TLP][]']).toEqual(['RED']);
    expect(entries['tags[MBC][]']).toEqual(['T1059']);
  });

  it('includes image file when provided', () => {
    const entity = makeCreateVendor();
    const file = new File(['data'], 'icon.png');
    const form = buildCreateEntityForm(entity, file);
    expect(form.has('image')).toBe(true);
  });

  it('includes metadata urls when present', () => {
    const entity = makeCreateVendor({
      metadata: {
        Vendor: {
          countries: [],
          critical_sectors: [],
        },
      },
    });
    const form = buildCreateEntityForm(entity);
    const entries = formEntries(form);
    expect(entries['metadata[urls][]']).toBeUndefined();
  });

  // country-list names the US "United States of America (the)"; the UI strips
  // the "(the)" suffix to match the API. The create form must still resolve the
  // stripped name to its ISO code — regression test for the broken strip logic.
  it('resolves stripped country names to their ISO codes', () => {
    const entity = makeCreateVendor({
      metadata: {
        Vendor: {
          countries: ['United States of America'],
          critical_sectors: [],
        },
      },
    });
    const form = buildCreateEntityForm(entity);
    const entries = formEntries(form);
    expect(entries['metadata[countries][]']).toEqual(['US', 'RU']);
  });
});
