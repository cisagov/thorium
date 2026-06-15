import { describe, expect, it } from 'vitest';

// project imports
import { descriptorFor, labelFor, selectionsToSeedParams } from './builderDescriptors';
import { decodeSeedParams } from '../Dashboard/seedParams';
import { selectionKey } from './builderReducer';
import { BuilderSelection, SelectionKind } from './types';
import { Entity, EntityMetaMap } from '@models/entities/entities';
import { Repo } from '@models/entities/repos';
import { Sample, SubmissionChunk } from '@models/files';

/**
 * Build a minimal submission chunk with a name for file-label tests.
 *
 * @param name - The submission name.
 * @returns A submission chunk with only the fields the label derivation reads.
 */
function submission(name: string): SubmissionChunk {
  return { id: 'sub-1', name, description: null, groups: [], submitter: 'u', uploaded: '', origin: {} };
}

/**
 * Build a minimal sample for descriptor tests.
 *
 * @param sha256 - The sample sha256.
 * @param submissions - The submissions to derive the label from.
 * @returns A sample with only the fields `descriptorFor` reads.
 */
function sample(sha256: string, submissions: SubmissionChunk[]): Sample {
  return { sha256, sha1: '', md5: '', tags: {}, submissions, comments: [] };
}

describe('descriptorFor — File', () => {
  it('extracts sha256 and a derived filename label', () => {
    const s = sample('a'.repeat(64), [submission('malware.exe')]);
    const sel = descriptorFor(SelectionKind.File, s);
    expect(sel).toEqual({ kind: SelectionKind.File, sha256: 'a'.repeat(64), label: 'malware.exe' });
  });
  it('falls back to the sha256 when there are no non-hash names', () => {
    const s = sample('b'.repeat(64), []);
    const sel = descriptorFor(SelectionKind.File, s);
    expect(sel).toEqual({ kind: SelectionKind.File, sha256: 'b'.repeat(64), label: 'b'.repeat(64) });
  });
});

describe('descriptorFor — Repo', () => {
  it('extracts url and name label', () => {
    const repo = { url: 'https://github.com/x/y', name: 'y' } as Repo;
    const sel = descriptorFor(SelectionKind.Repo, repo);
    expect(sel).toEqual({ kind: SelectionKind.Repo, url: 'https://github.com/x/y', label: 'y' });
  });
  it('falls back to the url when name is empty', () => {
    const repo = { url: 'https://github.com/x/y', name: '' } as Repo;
    const sel = descriptorFor(SelectionKind.Repo, repo);
    expect(sel.label).toBe('https://github.com/x/y');
  });
});

describe('descriptorFor — Entity', () => {
  it('extracts uuid and name label', () => {
    const entity = { id: 'uuid-1', name: 'laptop-1' } as Entity<keyof EntityMetaMap>;
    const sel = descriptorFor(SelectionKind.Entity, entity);
    expect(sel).toEqual({ kind: SelectionKind.Entity, id: 'uuid-1', label: 'laptop-1 (uuid-1)' });
  });
  it('falls back to the id when name is empty', () => {
    const entity = { id: 'uuid-1', name: '' } as Entity<keyof EntityMetaMap>;
    const sel = descriptorFor(SelectionKind.Entity, entity);
    expect(sel.label).toBe('uuid-1');
  });
});

describe('descriptorFor — Tag', () => {
  it('extracts key/value and a "key: value" label', () => {
    const sel = descriptorFor(SelectionKind.Tag, { key: 'FileType', value: 'PE32' });
    expect(sel).toEqual({ kind: SelectionKind.Tag, key: 'FileType', value: 'PE32', label: 'FileType: PE32' });
  });
});

describe('labelFor', () => {
  it('formats an entity as "name (id)" and falls back to the id', () => {
    expect(labelFor(SelectionKind.Entity, 'uuid-1', 'laptop-1')).toBe('laptop-1 (uuid-1)');
    expect(labelFor(SelectionKind.Entity, 'uuid-1', '')).toBe('uuid-1');
    expect(labelFor(SelectionKind.Entity, 'uuid-1', null)).toBe('uuid-1');
  });
  it('uses the name for files/repos, falling back to the identity', () => {
    expect(labelFor(SelectionKind.File, 'a'.repeat(64), 'malware.exe')).toBe('malware.exe');
    expect(labelFor(SelectionKind.File, 'a'.repeat(64), '')).toBe('a'.repeat(64));
    expect(labelFor(SelectionKind.Repo, 'https://github.com/x/y', 'y')).toBe('y');
  });
  it('uses the identity as-is for tags', () => {
    expect(labelFor(SelectionKind.Tag, 'FileType: PE32')).toBe('FileType: PE32');
  });
});

describe('selectionsToSeedParams', () => {
  it('encodes an empty selection to a depth-only, decodable params object', () => {
    const params = selectionsToSeedParams([], 4);
    const { seed, depth } = decodeSeedParams(params);
    expect(seed).toEqual({});
    expect(depth).toBe(4);
  });

  it('round-trips mixed kinds through decodeSeedParams back to equivalent selections', () => {
    const selections: BuilderSelection[] = [
      { kind: SelectionKind.File, sha256: 'a'.repeat(64), label: 'malware.exe' },
      { kind: SelectionKind.Entity, id: 'uuid-1', label: 'laptop-1' },
      { kind: SelectionKind.Repo, url: 'https://github.com/x/y', label: 'y' },
      { kind: SelectionKind.Tag, key: 'FileType', value: 'PE32', label: 'FileType: PE32' },
    ];
    const params = selectionsToSeedParams(selections, 2);
    const { seed, depth } = decodeSeedParams(params);
    expect(depth).toBe(2);
    expect(seed.samples).toEqual(['a'.repeat(64)]);
    expect(seed.entities).toEqual(['uuid-1']);
    expect(seed.repos).toEqual(['https://github.com/x/y']);
    expect(seed.tags).toEqual({ FileType: ['PE32'] });
  });

  it('round-trips selection identity (kind + id) through encode/decode', () => {
    const selections: BuilderSelection[] = [
      { kind: SelectionKind.File, sha256: 'c'.repeat(64), label: 'x' },
      { kind: SelectionKind.Tag, key: 'a:b', value: 'c&d', label: 'a:b: c&d' },
    ];
    const params = selectionsToSeedParams(selections, 3);
    const { seed } = decodeSeedParams(params);
    // rebuild selections from the decoded seed and compare identity keys
    const rebuilt: BuilderSelection[] = [];
    for (const sha256 of seed.samples ?? []) rebuilt.push({ kind: SelectionKind.File, sha256, label: sha256 });
    if (seed.tags) {
      for (const key of Object.keys(seed.tags)) {
        for (const value of seed.tags[key]) rebuilt.push({ kind: SelectionKind.Tag, key, value, label: '' });
      }
    }
    expect(rebuilt.map(selectionKey).sort()).toEqual(selections.map(selectionKey).sort());
  });

  it('groups multiple tag values under one key', () => {
    const selections: BuilderSelection[] = [
      { kind: SelectionKind.Tag, key: 'FileType', value: 'PE32', label: 'FileType: PE32' },
      { kind: SelectionKind.Tag, key: 'FileType', value: 'ELF', label: 'FileType: ELF' },
    ];
    const params = selectionsToSeedParams(selections, 2);
    const { seed } = decodeSeedParams(params);
    expect(seed.tags?.FileType?.sort()).toEqual(['ELF', 'PE32']);
  });
});
