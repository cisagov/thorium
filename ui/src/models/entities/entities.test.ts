import { describe, it, expect } from 'vitest';

// project imports
import { ENTITY_LABELS, Entities, entityLabel } from './entities';

describe('ENTITY_LABELS / entityLabel', () => {
  it('has a non-empty label for every Entities value', () => {
    for (const kind of Object.values(Entities)) {
      expect(ENTITY_LABELS[kind]).toBeTruthy();
    }
  });

  it('spaces out multi-word kinds', () => {
    expect(entityLabel(Entities.FileSystem)).toBe('File System');
    expect(entityLabel(Entities.WindowsProcessTree)).toBe('Windows Process Tree');
    expect(entityLabel(Entities.NetworkConnection)).toBe('Network Connection');
    expect(entityLabel(Entities.SigmaRule)).toBe('Sigma Rule');
    expect(entityLabel(Entities.CompiledFunction)).toBe('Compiled Function');
    expect(entityLabel(Entities.DecompiledFunction)).toBe('Decompiled Function');
  });

  it('preserves the PE acronym in labels', () => {
    expect(entityLabel(Entities.PeSection)).toBe('PE Section');
    expect(entityLabel(Entities.PeImport)).toBe('PE Import');
  });

  it('leaves single-word kinds unchanged', () => {
    expect(entityLabel(Entities.Device)).toBe('Device');
  });

  it('falls back to humanize for an unknown/string kind', () => {
    expect(entityLabel('SomeFutureKind')).toBe('Some Future Kind');
  });
});
