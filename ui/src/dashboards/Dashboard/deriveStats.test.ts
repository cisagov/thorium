import { describe, expect, it } from 'vitest';

// project imports
import { collectSampleSha256s, makeIncludeClause } from './deriveStats';
import { ClauseCondition } from '@components/shared/inputs/omnibar/ClauseTypes';
import { Entities } from '@models/entities';
import type { Sample } from '@models/files';
import { TreeNode, TreeNodeKey } from '@models/trees';

/**
 * Build a minimal entity node of the given kind.
 *
 * @param kind - The entity kind string.
 * @returns A `TreeNode` wrapping an entity with only the fields under test.
 */
function entityNode(kind: string): TreeNode {
  return { [TreeNodeKey.Entity]: { kind } as TreeNode[TreeNodeKey.Entity] };
}

/** A minimal repo node. */
function repoNode(): TreeNode {
  return { [TreeNodeKey.Repo]: {} as TreeNode[TreeNodeKey.Repo] };
}

/**
 * Assemble a `data_map` from a list of nodes with generated ids.
 *
 * @param nodes - The nodes to place in the map.
 * @returns A `{ [nodeId]: TreeNode }` map.
 */
function dataMap(nodes: TreeNode[]): Record<string, TreeNode> {
  const map: Record<string, TreeNode> = {};
  nodes.forEach((node, i) => {
    map[`node-${i}`] = node;
  });
  return map;
}

/** Build a Sample node with an explicit sha256. */
function fileNodeWithSha(sha256: string): TreeNode {
  const sample: Sample = { sha256, sha1: '', md5: '', tags: {}, submissions: [], comments: [] };
  return { [TreeNodeKey.Sample]: sample };
}

describe('collectSampleSha256s', () => {
  it('returns an empty array for an empty data_map', () => {
    expect(collectSampleSha256s({})).toEqual([]);
  });

  it('collects the sha256 of every Sample node', () => {
    const result = collectSampleSha256s(dataMap([fileNodeWithSha('aaa'), fileNodeWithSha('bbb')]));
    expect(result).toEqual(['aaa', 'bbb']);
  });

  it('dedupes repeated sha256s, preserving first-seen order', () => {
    const result = collectSampleSha256s(dataMap([fileNodeWithSha('aaa'), fileNodeWithSha('bbb'), fileNodeWithSha('aaa')]));
    expect(result).toEqual(['aaa', 'bbb']);
  });

  it('ignores non-Sample nodes', () => {
    const result = collectSampleSha256s(dataMap([fileNodeWithSha('aaa'), repoNode(), entityNode(Entities.Device)]));
    expect(result).toEqual(['aaa']);
  });
});

describe('makeIncludeClause', () => {
  it('builds an Include clause whitelisting the given kind', () => {
    expect(makeIncludeClause(Entities.Device)).toEqual({
      category: 'Include',
      field: 'Include',
      condition: ClauseCondition.Is,
      value: { value: Entities.Device },
    });
  });
});
