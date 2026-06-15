import { describe, expect, it } from 'vitest';

// project imports
import { collectSeedSummary, SeedSummaryKind } from './seedSummary';
import { TreeNode, TreeNodeKey, type Seed } from '@models/trees';

/** Build a Sample-carrying node from a sha256 and its submission names. */
function sampleNode(sha256: string, names: string[]): TreeNode {
  return {
    [TreeNodeKey.Sample]: {
      sha256,
      submissions: names.map((name) => ({ name })),
    },
  } as unknown as TreeNode;
}

/** Build an Entity-carrying node from an id and name. */
function entityNode(id: string, name: string): TreeNode {
  return { [TreeNodeKey.Entity]: { id, name, kind: 'Folder' } } as unknown as TreeNode;
}

/** Build a Repo-carrying node from a url and name. */
function repoNode(url: string, name: string): TreeNode {
  return { [TreeNodeKey.Repo]: { url, name } } as unknown as TreeNode;
}

/** Build a data_map from a list of nodes, keyed by opaque node ids. */
function dataMap(nodes: TreeNode[]): { [nodeId: string]: TreeNode } {
  return Object.fromEntries(nodes.map((node, index) => [`node-${index}`, node]));
}

describe('collectSeedSummary', () => {
  it('returns an empty list for an empty seed', () => {
    expect(collectSeedSummary({}, {})).toEqual([]);
  });

  it('resolves a file sha256 to its display name', () => {
    const seed: Seed = { samples: ['a'.repeat(64)] };
    const map = dataMap([sampleNode('a'.repeat(64), ['malware.exe'])]);
    expect(collectSeedSummary(seed, map)).toEqual([{ kind: SeedSummaryKind.File, label: 'malware.exe' }]);
  });

  it('falls back to a short sha256 when the file node is missing', () => {
    const sha256 = 'b'.repeat(64);
    const seed: Seed = { samples: [sha256] };
    expect(collectSeedSummary(seed, {})).toEqual([{ kind: SeedSummaryKind.File, label: sha256.slice(0, 12) }]);
  });

  it('falls back to a short sha256 when the file has only hash-named submissions and no name', () => {
    // a submission whose only name is the sha256 itself yields an empty derived name -> short sha256
    const sha256 = 'c'.repeat(64);
    const seed: Seed = { samples: [sha256] };
    const map = dataMap([sampleNode(sha256, [])]);
    expect(collectSeedSummary(seed, map)).toEqual([{ kind: SeedSummaryKind.File, label: sha256.slice(0, 12) }]);
  });

  it('resolves an entity id to its name', () => {
    const seed: Seed = { entities: ['abc-123'] };
    const map = dataMap([entityNode('abc-123', 'laptop-1')]);
    expect(collectSeedSummary(seed, map)).toEqual([{ kind: SeedSummaryKind.Entity, label: 'laptop-1' }]);
  });

  it('falls back to a short id when the entity node is missing', () => {
    const id = 'd'.repeat(40);
    const seed: Seed = { entities: [id] };
    expect(collectSeedSummary(seed, {})).toEqual([{ kind: SeedSummaryKind.Entity, label: id.slice(0, 12) }]);
  });

  it('resolves a repo url to its name', () => {
    const seed: Seed = { repos: ['https://github.com/org/repo'] };
    const map = dataMap([repoNode('https://github.com/org/repo', 'repo')]);
    expect(collectSeedSummary(seed, map)).toEqual([{ kind: SeedSummaryKind.Repo, label: 'repo' }]);
  });

  it('falls back to the url when the repo node is missing', () => {
    const seed: Seed = { repos: ['https://github.com/org/other'] };
    expect(collectSeedSummary(seed, {})).toEqual([{ kind: SeedSummaryKind.Repo, label: 'https://github.com/org/other' }]);
  });

  it('renders tags as key: value, one item per value', () => {
    const seed: Seed = { tags: { os: ['linux', 'windows'] } };
    expect(collectSeedSummary(seed, {})).toEqual([
      { kind: SeedSummaryKind.Tag, label: 'os: linux' },
      { kind: SeedSummaryKind.Tag, label: 'os: windows' },
    ]);
  });

  it('emits items in a stable order: files, entities, repos, then tags', () => {
    const sha256 = 'e'.repeat(64);
    const seed: Seed = {
      samples: [sha256],
      entities: ['ent-1'],
      repos: ['https://github.com/org/repo'],
      tags: { FileType: ['PE32'] },
    };
    const map = dataMap([
      entityNode('ent-1', 'device-1'),
      repoNode('https://github.com/org/repo', 'repo'),
      sampleNode(sha256, ['sample.bin']),
    ]);
    expect(collectSeedSummary(seed, map)).toEqual([
      { kind: SeedSummaryKind.File, label: 'sample.bin' },
      { kind: SeedSummaryKind.Entity, label: 'device-1' },
      { kind: SeedSummaryKind.Repo, label: 'repo' },
      { kind: SeedSummaryKind.Tag, label: 'FileType: PE32' },
    ]);
  });
});
