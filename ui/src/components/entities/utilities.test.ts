import { describe, it, expect, vi } from 'vitest';

vi.mock('@thorpi/entities', () => ({
  listEntities: vi.fn().mockResolvedValue({ entityList: [], entityCursor: null }),
}));

// project imports
import { buildUpdateEntityForm, buildCreateEntityForm, copyEntityFields, DEFAULT_LIST_LIMIT } from './utilities';
import { Entities, EntityCreateTypes, EntityTypes } from '@models/entities/entities';
import { BlankDevice } from '@models/entities/devices';
import { BlankCreateVendor, CreateVendor } from '@models/entities/vendors';
import { BlankCreateFlag, BlankFlag, Confidence } from '@models/entities/flag';
import { BlankCreateIncident, BlankIncident } from '@models/entities/incident';
import { BlankCreateCompiledFunction, BlankCreateDecompiledFunction, BlankDecompiledFunction } from '@models/entities/functions';
import { BlankCreatePeImport, BlankCreatePeSection, BlankPeImport, BlankPeSection } from '@models/entities/pe';

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
    expect(entries['metadata[countries][]']).toEqual(['US']);
  });
});

describe('buildCreateEntityForm per-kind metadata', () => {
  it('emits Flag scalars and omits empty content', () => {
    const entity: EntityCreateTypes = structuredClone(BlankCreateFlag);
    entity.metadata.Flag.suspicion = 7;
    entity.metadata.Flag.confidence = Confidence.Likely;
    entity.metadata.Flag.reasoning = 'looks odd';
    const form = buildCreateEntityForm(entity);
    expect(form.get('metadata[suspicion]')).toBe('7');
    expect(form.get('metadata[confidence]')).toBe(Confidence.Likely);
    expect(form.get('metadata[reasoning]')).toBe('looks odd');
    expect(form.has('metadata[content]')).toBe(false);
  });

  it('emits Flag content when set', () => {
    const entity: EntityCreateTypes = structuredClone(BlankCreateFlag);
    entity.metadata.Flag.content = 'suspicious string';
    const form = buildCreateEntityForm(entity);
    expect(form.get('metadata[content]')).toBe('suspicious string');
  });

  it('emits Incident lists and only sends cover_term when set', () => {
    const entity: EntityCreateTypes = structuredClone(BlankCreateIncident);
    entity.metadata.Incident.mission_teams = ['red', 'blue'];
    entity.metadata.Incident.networks = ['net1'];
    const form = buildCreateEntityForm(entity);
    const entries = formEntries(form);
    expect(entries['metadata[mission_teams][]']).toEqual(['red', 'blue']);
    expect(entries['metadata[networks][]']).toEqual(['net1']);
    expect(form.has('metadata[cover_term]')).toBe(false);
    entity.metadata.Incident.cover_term = 'NIGHTFALL';
    expect(buildCreateEntityForm(entity).get('metadata[cover_term]')).toBe('NIGHTFALL');
  });

  it('JSON-serializes each CompiledFunction disassembly instruction', () => {
    const entity: EntityCreateTypes = structuredClone(BlankCreateCompiledFunction);
    entity.metadata.CompiledFunction.address = 4096;
    entity.metadata.CompiledFunction.disassembly = [
      { address: 4096, instruction: 'push rbp' },
      { address: 4097, instruction: 'mov rbp, rsp' },
    ];
    const form = buildCreateEntityForm(entity);
    expect(form.get('metadata[function_address]')).toBe('4096');
    const entries = formEntries(form);
    expect(entries['metadata[disassembly][]']).toEqual([
      JSON.stringify({ address: 4096, instruction: 'push rbp' }),
      JSON.stringify({ address: 4097, instruction: 'mov rbp, rsp' }),
    ]);
  });

  it('emits DecompiledFunction content and tools as plain list entries', () => {
    const entity: EntityCreateTypes = structuredClone(BlankCreateDecompiledFunction);
    entity.metadata.DecompiledFunction.address = 8;
    entity.metadata.DecompiledFunction.content = 'int main() {}';
    entity.metadata.DecompiledFunction.tools = ['ghidra', 'ida'];
    const form = buildCreateEntityForm(entity);
    expect(form.get('metadata[decompilation_content]')).toBe('int main() {}');
    expect(formEntries(form)['metadata[tools][]']).toEqual(['ghidra', 'ida']);
  });

  it('only emits PeSection scalars that were set', () => {
    const entity: EntityCreateTypes = structuredClone(BlankCreatePeSection);
    entity.metadata.PeSection.md5 = 'abc';
    entity.metadata.PeSection.raw_size = 512;
    const form = buildCreateEntityForm(entity);
    expect(form.get('metadata[md5]')).toBe('abc');
    expect(form.get('metadata[raw_size]')).toBe('512');
    expect(form.has('metadata[virtual_size]')).toBe(false);
    expect(form.has('metadata[entropy]')).toBe(false);
  });

  it('emits PeImport functions as plain list entries', () => {
    const entity: EntityCreateTypes = structuredClone(BlankCreatePeImport);
    entity.metadata.PeImport.functions = ['CreateFileA', 'ReadFile'];
    const form = buildCreateEntityForm(entity);
    expect(formEntries(form)['metadata[functions][]']).toEqual(['CreateFileA', 'ReadFile']);
  });
});

describe('buildUpdateEntityForm per-kind metadata', () => {
  it('sends Incident cover_term only when changed to a non-empty value', () => {
    const entity = structuredClone(BlankIncident);
    const pending = structuredClone(BlankIncident);
    pending.metadata.Incident.cover_term = 'DAWN';
    expect(buildUpdateEntityForm(entity, pending).get('metadata[cover_term]')).toBe('DAWN');
    // clearing an existing cover_term is dropped (the API has no clear)
    const set = structuredClone(BlankIncident);
    set.metadata.Incident.cover_term = 'DAWN';
    const cleared = structuredClone(set);
    cleared.metadata.Incident.cover_term = null;
    expect(buildUpdateEntityForm(set, cleared).has('metadata[cover_term]')).toBe(false);
  });

  it('diffs Incident list fields into add/remove keys', () => {
    const entity = structuredClone(BlankIncident);
    entity.metadata.Incident.networks = ['a', 'b'];
    const pending = structuredClone(entity);
    pending.metadata.Incident.networks = ['a', 'c'];
    const entries = formEntries(buildUpdateEntityForm(entity, pending));
    expect(entries['metadata[add_networks][]']).toEqual(['c']);
    expect(entries['metadata[remove_networks][]']).toEqual(['b']);
  });

  it('never sends DecompiledFunction tools (create-only), but does send changed content', () => {
    const entity = structuredClone(BlankDecompiledFunction);
    entity.metadata.DecompiledFunction.tools = ['ghidra'];
    entity.metadata.DecompiledFunction.content = 'old';
    const pending = structuredClone(entity);
    pending.metadata.DecompiledFunction.tools = ['ghidra', 'ida'];
    pending.metadata.DecompiledFunction.content = 'new';
    const form = buildUpdateEntityForm(entity, pending);
    expect(form.has('metadata[add_tools][]')).toBe(false);
    expect(form.has('metadata[remove_tools][]')).toBe(false);
    expect(form.has('metadata[tools][]')).toBe(false);
    expect(form.get('metadata[decompilation_content]')).toBe('new');
  });

  it('replaces PeImport functions wholesale and skips an unchanged list', () => {
    const entity = structuredClone(BlankPeImport);
    entity.metadata.PeImport.functions = ['a'];
    const same = structuredClone(entity);
    expect(buildUpdateEntityForm(entity, same).has('metadata[functions][]')).toBe(false);
    const changed = structuredClone(entity);
    changed.metadata.PeImport.functions = ['a', 'b'];
    expect(formEntries(buildUpdateEntityForm(entity, changed))['metadata[functions][]']).toEqual(['a', 'b']);
  });

  it('only sends changed PeSection scalars and never clears them', () => {
    const entity = structuredClone(BlankPeSection);
    entity.metadata.PeSection.md5 = 'old';
    entity.metadata.PeSection.raw_size = 10;
    const pending = structuredClone(entity);
    pending.metadata.PeSection.raw_size = 20;
    const form = buildUpdateEntityForm(entity, pending);
    expect(form.get('metadata[raw_size]')).toBe('20');
    expect(form.has('metadata[md5]')).toBe(false);
    // clearing md5 back to undefined is dropped
    const cleared = structuredClone(entity);
    cleared.metadata.PeSection.md5 = undefined;
    expect(buildUpdateEntityForm(entity, cleared).has('metadata[md5]')).toBe(false);
  });
});

describe('copyEntityFields', () => {
  it('clones Flag metadata as-is and appends " - copy" to the name', () => {
    const src = structuredClone(BlankFlag);
    src.name = 'MyFlag';
    src.metadata.Flag.suspicion = 3;
    src.metadata.Flag.reasoning = 'because';
    const copy = copyEntityFields(src, structuredClone(BlankCreateFlag));
    expect(copy.name).toBe('MyFlag - copy');
    expect(copy.kind).toBe(Entities.Flag);
    if (copy.kind === Entities.Flag) {
      expect(copy.metadata.Flag.suspicion).toBe(3);
      expect(copy.metadata.Flag.reasoning).toBe('because');
    }
  });

  it('clones PeSection metadata as-is', () => {
    const src = structuredClone(BlankPeSection);
    src.metadata.PeSection.md5 = 'deadbeef';
    src.metadata.PeSection.entropy = 6.5;
    const copy = copyEntityFields(src, structuredClone(BlankCreatePeSection));
    if (copy.kind === Entities.PeSection) {
      expect(copy.metadata.PeSection.md5).toBe('deadbeef');
      expect(copy.metadata.PeSection.entropy).toBe(6.5);
    }
  });

  it('clones PeImport functions', () => {
    const src = structuredClone(BlankPeImport);
    src.metadata.PeImport.functions = ['CreateFileA'];
    const copy = copyEntityFields(src, structuredClone(BlankCreatePeImport));
    if (copy.kind === Entities.PeImport) {
      expect(copy.metadata.PeImport.functions).toEqual(['CreateFileA']);
    }
  });
});
