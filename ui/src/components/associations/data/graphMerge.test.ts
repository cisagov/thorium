import { describe, it, expect } from 'vitest';

// project imports
import { mergeGrowthInto, computeDistances } from './graphMerge';
import { Direction, BlankGraph } from '@models/trees';
import type { Graph, BranchNode } from '@models/trees';
import type { Sample } from '@models/files';

function graphWith(patch: Partial<Graph>): Graph {
  return { ...structuredClone(BlankGraph), ...patch };
}

function branch(node: string, hash: string, direction = Direction.To): BranchNode {
  return { node, direction, relationship_hash: hash, relationship: {} };
}

describe('mergeGrowthInto', () => {
  it('merges data_map entries from grown graph', () => {
    const initial = graphWith({
      data_map: { a: { Sample: { sha256: 'aaa' } as unknown as Sample } },
    });
    const grown = graphWith({
      data_map: { b: { Sample: { sha256: 'bbb' } as unknown as Sample } },
    });
    const result = mergeGrowthInto(initial, grown, []);
    expect(Object.keys(result.data_map)).toEqual(['a', 'b']);
  });

  it('overrides existing data_map entries with grown data', () => {
    const initial = graphWith({
      data_map: { a: { Sample: { sha256: 'old' } as unknown as Sample } },
    });
    const grown = graphWith({
      data_map: { a: { Sample: { sha256: 'new' } as unknown as Sample } },
    });
    const result = mergeGrowthInto(initial, grown, []);
    expect(result.data_map.a.Sample?.sha256).toBe('new');
  });

  it('merges new branches for existing source nodes without duplicates', () => {
    const b1 = branch('b', 'h1');
    const b2 = branch('c', 'h2');
    const b3 = branch('d', 'h3');
    const initial = graphWith({ branches: { a: [b1] } });
    const grown = graphWith({ branches: { a: [b1, b2], e: [b3] } });
    const result = mergeGrowthInto(initial, grown, []);
    expect(result.branches.a).toHaveLength(2);
    expect(result.branches.a[1].node).toBe('c');
    expect(result.branches.e).toHaveLength(1);
  });

  it('adds branches for new source nodes', () => {
    const initial = graphWith({ branches: {} });
    const grown = graphWith({ branches: { x: [branch('y', 'h1')] } });
    const result = mergeGrowthInto(initial, grown, []);
    expect(result.branches.x).toHaveLength(1);
  });

  it('removes grown nodes from growable and adds new growable', () => {
    const initial = graphWith({ growable: ['a', 'b', 'c'] });
    const grown = graphWith({ growable: ['d', 'e'] });
    const result = mergeGrowthInto(initial, grown, ['a', 'c']);
    expect(result.growable).toEqual(['b', 'd', 'e']);
  });

  it('preserves initial graph fields like id and groups', () => {
    const initial = graphWith({ id: 'graph-1', groups: ['g1'], initial: ['a'] });
    const grown = graphWith({ id: 'graph-2', groups: ['g2'], initial: ['b'] });
    const result = mergeGrowthInto(initial, grown, []);
    expect(result.id).toBe('graph-1');
    expect(result.groups).toEqual(['g1']);
    expect(result.initial).toEqual(['a']);
  });

  it('re-canonicalizes the merged graph so appended branches and merged keys stay id-ordered', () => {
    // grown adds a branch and a data_map node that sort BEFORE the existing ones by numeric id; the
    // merge appends/spreads in sequence, so only re-canonicalization keeps the result deterministic
    const initial = graphWith({
      data_map: { '50': {} },
      branches: { '1': [branch('50', 'h50')] },
    });
    const grown = graphWith({
      data_map: { '20': {} },
      branches: { '1': [branch('20', 'h20')] },
    });
    const result = mergeGrowthInto(initial, grown, []);
    expect(result.branches['1'].map((b) => b.node)).toEqual(['20', '50']);
    expect(Object.keys(result.data_map)).toEqual(['20', '50']);
  });

  it('does not mutate the initial graph it merges into', () => {
    const initial = graphWith({
      data_map: { '50': {} },
      branches: { '1': [branch('50', 'h50')] },
    });
    const grown = graphWith({ branches: { '1': [branch('20', 'h20')] } });
    mergeGrowthInto(initial, grown, []);
    // the source array/keys are untouched even though canonicalization reorders the merged copy
    expect(initial.branches['1'].map((b) => b.node)).toEqual(['50']);
    expect(Object.keys(initial.data_map)).toEqual(['50']);
  });
});

describe('computeDistances', () => {
  it('returns distance 0 for initial nodes', () => {
    const graph = graphWith({ initial: ['a'], branches: {} });
    const distances = computeDistances(graph);
    expect(distances.get('a')).toBe(0);
  });

  it('computes BFS distances from initial nodes', () => {
    const graph = graphWith({
      initial: ['a'],
      branches: {
        a: [branch('b', 'h1')],
        b: [branch('c', 'h2')],
      },
    });
    const distances = computeDistances(graph);
    expect(distances.get('a')).toBe(0);
    expect(distances.get('b')).toBe(1);
    expect(distances.get('c')).toBe(2);
  });

  it('handles multiple initial nodes', () => {
    const graph = graphWith({
      initial: ['a', 'c'],
      branches: {
        a: [branch('b', 'h1')],
        b: [branch('c', 'h2')],
      },
    });
    const distances = computeDistances(graph);
    expect(distances.get('a')).toBe(0);
    expect(distances.get('c')).toBe(0);
    expect(distances.get('b')).toBe(1);
  });

  it('treats branches as undirected edges', () => {
    const graph = graphWith({
      initial: ['c'],
      branches: { a: [branch('c', 'h1')] },
    });
    const distances = computeDistances(graph);
    expect(distances.get('c')).toBe(0);
    expect(distances.get('a')).toBe(1);
  });

  it('returns empty map for empty graph', () => {
    const distances = computeDistances(graphWith({}));
    expect(distances.size).toBe(0);
  });

  it('measures from an explicit seed set instead of graph.initial', () => {
    // initial is 'a', but seeding from 'b' re-bases distances on the focus root (0 at 'b')
    const graph = graphWith({
      initial: ['a'],
      branches: {
        a: [branch('b', 'h1')],
        b: [branch('c', 'h2')],
      },
    });
    const distances = computeDistances(graph, ['b']);
    expect(distances.get('b')).toBe(0);
    expect(distances.get('c')).toBe(1);
    // 'a' is one undirected hop back up from the seed 'b'
    expect(distances.get('a')).toBe(1);
  });
});
