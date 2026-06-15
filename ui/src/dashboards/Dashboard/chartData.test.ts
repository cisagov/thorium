import { describe, expect, it } from 'vitest';

// project imports
import { collectTypeCounts, countTagKey } from './chartData';
import { Entities } from '@models/entities';
import type { Sample } from '@models/files';
import { TreeNode, TreeNodeKey } from '@models/trees';

/**
 * Build a minimal {@link Sample} node carrying the given tags.
 *
 * @param tags - The tag map, shaped `{ key: { value: groups } }`.
 * @returns A `TreeNode` wrapping the sample.
 */
function fileNode(tags: Sample['tags'] = {}): TreeNode {
  const sample = {
    sha256: `sha-${Math.random()}`,
    sha1: '',
    md5: '',
    tags,
    submissions: [],
    comments: [],
  } as unknown as Sample;
  return { [TreeNodeKey.Sample]: sample };
}

/**
 * Build a minimal entity node of the given kind, optionally with tags.
 *
 * @param kind - The entity kind string.
 * @param tags - Optional tag map.
 * @returns A `TreeNode` wrapping the entity.
 */
function entityNode(kind: string, tags: Record<string, Record<string, string[]>> = {}): TreeNode {
  return { [TreeNodeKey.Entity]: { kind, tags } as unknown as TreeNode[TreeNodeKey.Entity] };
}

/** A minimal repo node, optionally with tags. */
function repoNode(tags: Record<string, Record<string, string[]>> = {}): TreeNode {
  return { [TreeNodeKey.Repo]: { tags } as unknown as TreeNode[TreeNodeKey.Repo] };
}

describe('collectTypeCounts', () => {
  it('returns an empty list for no nodes', () => {
    expect(collectTypeCounts([])).toEqual([]);
  });

  it('blends files, repos, and one bar per entity kind', () => {
    const nodes = [
      fileNode(),
      fileNode(),
      repoNode(),
      entityNode(Entities.Folder),
      entityNode(Entities.Folder),
      entityNode(Entities.Folder),
      entityNode(Entities.Device),
    ];
    const result = collectTypeCounts(nodes);
    const byKind = Object.fromEntries(result.map((r) => [r.kind, r.value]));
    expect(byKind[Entities.File]).toBe(2);
    expect(byKind[Entities.Repo]).toBe(1);
    expect(byKind[Entities.Folder]).toBe(3);
    expect(byKind[Entities.Device]).toBe(1);
  });

  it('sorts by descending count then kind ascending', () => {
    const nodes = [entityNode(Entities.Device), entityNode(Entities.Folder), entityNode(Entities.Folder), fileNode()];
    const result = collectTypeCounts(nodes);
    // Folder(2) leads; File(1) and Device(1) tie so sort by kind ascending: Device < File
    expect(result.map((r) => r.kind)).toEqual([Entities.Folder, Entities.Device, Entities.File]);
  });

  it('falls back to Other for a kind-less entity', () => {
    const node = { [TreeNodeKey.Entity]: {} as unknown as TreeNode[TreeNodeKey.Entity] };
    const result = collectTypeCounts([node]);
    expect(result).toEqual([{ kind: Entities.Other, label: 'Other', value: 1 }]);
  });

  it('carries a human-readable label per bar', () => {
    const result = collectTypeCounts([entityNode(Entities.FileSystem)]);
    expect(result[0].label).toBe('File System');
  });
});

describe('countTagKey', () => {
  it('returns an empty list when no node carries the key', () => {
    expect(countTagKey([fileNode(), repoNode()], 'FileType')).toEqual([]);
  });

  it('counts values for the requested key across node kinds', () => {
    const nodes = [
      fileNode({ FileType: { PE32: ['g'] } }),
      fileNode({ FileType: { PE32: ['g'] } }),
      fileNode({ FileType: { ELF: ['g'] } }),
      entityNode(Entities.Folder, { FileType: { PE32: ['g'] } }),
    ];
    const result = countTagKey(nodes, 'FileType');
    expect(result).toEqual([
      { value: 'PE32', count: 3 },
      { value: 'ELF', count: 1 },
    ]);
  });

  it('ignores tags under other keys', () => {
    const nodes = [fileNode({ FileType: { PE32: ['g'] }, Other: { x: ['g'] } })];
    expect(countTagKey(nodes, 'FileTypeExtension')).toEqual([]);
    expect(countTagKey(nodes, 'FileType')).toEqual([{ value: 'PE32', count: 1 }]);
  });

  it('counts a node once per value even with multiple values under the key', () => {
    const nodes = [fileNode({ FileType: { PE32: ['g'], ELF: ['g'] } })];
    const result = countTagKey(nodes, 'FileType');
    expect(result).toEqual([
      { value: 'ELF', count: 1 },
      { value: 'PE32', count: 1 },
    ]);
  });

  it('supports a configurable key such as FileTypeExtension', () => {
    const nodes = [fileNode({ FileTypeExtension: { '.exe': ['g'] } }), fileNode({ FileTypeExtension: { '.exe': ['g'] } })];
    expect(countTagKey(nodes, 'FileTypeExtension')).toEqual([{ value: '.exe', count: 2 }]);
  });
});
