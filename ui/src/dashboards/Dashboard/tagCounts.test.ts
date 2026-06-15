import { describe, expect, it } from 'vitest';

// project imports
import { ALWAYS_HIDDEN_TAG_KEYS, collectTagCounts } from './tagCounts';
import type { Sample } from '@models/files';
import type { Tags } from '@models/tags';
import { TreeNode, TreeNodeKey } from '@models/trees';

/**
 * Build a minimal file {@link TreeNode} carrying the given tags.
 *
 * @param tags - The tag map, shaped `{ key: { value: groups } }`.
 * @returns A `TreeNode` wrapping a sample with only the fields `collectTagCounts` reads.
 */
function fileNode(tags: Tags): TreeNode {
  return { [TreeNodeKey.Sample]: { tags } as Sample };
}

describe('collectTagCounts', () => {
  it('returns an empty map for no nodes', () => {
    expect(collectTagCounts([], [])).toEqual(new Map());
  });

  it('returns an empty map when nodes have no tags', () => {
    expect(collectTagCounts([fileNode({})], [])).toEqual(new Map());
  });

  it('counts values across nodes, grouping by key', () => {
    const nodes = [
      fileNode({ FileType: { PE32: ['g'] } }),
      fileNode({ FileType: { PE32: ['g'] } }),
      fileNode({ FileType: { ELF: ['g'] }, os: { linux: ['g'] } }),
    ];
    const result = collectTagCounts(nodes, []);
    expect(result.get('FileType')).toEqual([
      { value: 'PE32', count: 2 },
      { value: 'ELF', count: 1 },
    ]);
    expect(result.get('os')).toEqual([{ value: 'linux', count: 1 }]);
  });

  it('counts a value once per node even when it lists the value with multiple group scopes', () => {
    const nodes = [fileNode({ FileType: { PE32: ['g1', 'g2'] } })];
    const result = collectTagCounts(nodes, []);
    expect(result.get('FileType')).toEqual([{ value: 'PE32', count: 1 }]);
  });

  it('excludes keys listed in hiddenKeys', () => {
    const nodes = [fileNode({ FileType: { PE32: ['g'] }, Results: { yara: ['g'] }, Parent: { p: ['g'] } })];
    const result = collectTagCounts(nodes, ['Results', 'Parent', 'submitter']);
    expect(Array.from(result.keys())).toEqual(['FileType']);
    expect(result.has('Results')).toBe(false);
    expect(result.has('Parent')).toBe(false);
  });

  it('downselects when counting over a subset of nodes', () => {
    const all = [fileNode({ FileType: { PE32: ['g'] } }), fileNode({ FileType: { ELF: ['g'] } }), fileNode({ FileType: { ELF: ['g'] } })];
    // count over only the last two nodes: PE32 drops out entirely, ELF tallies both
    const subset = collectTagCounts(all.slice(1), []);
    expect(subset.get('FileType')).toEqual([{ value: 'ELF', count: 2 }]);
  });

  it('excludes the always-hidden high-cardinality Folder*Sha256 keys when they are in the hidden list', () => {
    const nodes = [
      fileNode({
        FileType: { PE32: ['g'] },
        FolderAllSha256: { abc: ['g'] },
        FolderDataSha256: { def: ['g'] },
        FolderNamesSha256: { ghi: ['g'] },
      }),
    ];
    const result = collectTagCounts(nodes, ALWAYS_HIDDEN_TAG_KEYS);
    expect(Array.from(result.keys())).toEqual(['FileType']);
    expect(result.has('FolderAllSha256')).toBe(false);
    expect(result.has('FolderDataSha256')).toBe(false);
    expect(result.has('FolderNamesSha256')).toBe(false);
  });

  it('ALWAYS_HIDDEN_TAG_KEYS holds exactly the three Folder*Sha256 keys', () => {
    expect(ALWAYS_HIDDEN_TAG_KEYS).toEqual(['FolderAllSha256', 'FolderDataSha256', 'FolderNamesSha256']);
  });

  it("sorts a key's values by descending count then value ascending", () => {
    const nodes = [
      fileNode({ os: { windows: ['g'] } }),
      fileNode({ os: { windows: ['g'] } }),
      fileNode({ os: { linux: ['g'] } }),
      fileNode({ os: { bsd: ['g'] } }),
    ];
    const result = collectTagCounts(nodes, []);
    expect(result.get('os')).toEqual([
      { value: 'windows', count: 2 },
      { value: 'bsd', count: 1 },
      { value: 'linux', count: 1 },
    ]);
  });
});
