import { describe, it, expect } from 'vitest';

// project imports
import { canonicalizeGraphOrder, compareNumericIds, normalizeGraphIds } from './trees';
import { BlankGraph, BranchNode, Direction, Graph } from '@models/trees';

/**
 * Build a graph fixture from a partial override without repeating the blank base.
 *
 * @param patch - Fields to override on top of {@link BlankGraph}.
 * @returns A graph object safe to mutate in a single test.
 */
function graphWith(patch: Partial<Graph>): Graph {
  return { ...structuredClone(BlankGraph), ...structuredClone(patch) };
}

describe('normalizeGraphIds', () => {
  it('coerces numeric ids in initial/growable/sent to strings', () => {
    const graph = graphWith({
      initial: [123 as unknown as string],
      growable: [456 as unknown as string],
      sent: [789 as unknown as string],
    });
    const result = normalizeGraphIds(graph);
    expect(result.initial).toEqual(['123']);
    expect(result.growable).toEqual(['456']);
    expect(result.sent).toEqual(['789']);
  });

  it('coerces numeric branch and hint_branch node fields to strings', () => {
    const graph = graphWith({
      branches: {
        '1': [{ relationship: {}, node: 42 as unknown as string, direction: Direction.To, relationship_hash: 'h' }],
      },
      hint_branches: {
        '1': [{ relationship: {}, node: 99 as unknown as string, direction: Direction.From, relationship_hash: 'k' }],
      },
    });
    const result = normalizeGraphIds(graph);
    expect(result.branches['1'][0].node).toBe('42');
    expect(result.hint_branches?.['1'][0].node).toBe('99');
  });

  it('leaves string ids unchanged and tolerates absent optional fields', () => {
    const graph = graphWith({ initial: ['abc'], growable: ['def'] });
    delete graph.sent;
    delete graph.hint_branches;
    const result = normalizeGraphIds(graph);
    expect(result.initial).toEqual(['abc']);
    expect(result.growable).toEqual(['def']);
    expect(result.sent).toBeUndefined();
    expect(result.hint_branches).toBeUndefined();
  });
});

describe('compareNumericIds', () => {
  it('orders large u64 ids numerically, not lexically', () => {
    // lexical order would put "10000..." before "9..." — the numeric compare must not
    const a = '2346158323794578801';
    const b = '11862460900133760000';
    expect(compareNumericIds(a, b)).toBeLessThan(0);
    expect(compareNumericIds(b, a)).toBeGreaterThan(0);
    expect(compareNumericIds(a, a)).toBe(0);
  });

  it('is a stable total order when sorting a mixed list', () => {
    const ids = ['3450496504189858957', '8855138436683789108', '2346158323794578801', '7579084672035853627'];
    expect([...ids].sort(compareNumericIds)).toEqual([
      '2346158323794578801',
      '3450496504189858957',
      '7579084672035853627',
      '8855138436683789108',
    ]);
  });

  it('falls back to a lexical order for non-numeric ids without throwing', () => {
    expect(compareNumericIds('abc', 'abd')).toBeLessThan(0);
    expect(compareNumericIds('abc', 'abc')).toBe(0);
  });
});

describe('canonicalizeGraphOrder', () => {
  /** A branch to a target node (only the fields canonicalization sorts on matter here). */
  function branch(node: string, hash: string, direction: Direction = Direction.To): BranchNode {
    return { relationship: {}, node, direction, relationship_hash: hash };
  }

  it('sorts data_map, branches, and growable by numeric id, leaving initial untouched', () => {
    const graph = graphWith({
      initial: ['300', '100'],
      growable: ['30', '10', '20'],
      data_map: { '30': {}, '10': {}, '20': {} },
      branches: {
        '30': [branch('20', 'b'), branch('10', 'a')],
        '10': [branch('30', 'c')],
      },
    });
    const result = canonicalizeGraphOrder(graph);
    expect(Object.keys(result.data_map)).toEqual(['10', '20', '30']);
    expect(Object.keys(result.branches)).toEqual(['10', '30']);
    // each branch array is ordered by target node id
    expect(result.branches['30'].map((b) => b.node)).toEqual(['10', '20']);
    expect(result.growable).toEqual(['10', '20', '30']);
    // initial keeps its caller-provided seed order
    expect(result.initial).toEqual(['300', '100']);
  });

  it('breaks branch ties by direction then relationship_hash', () => {
    const graph = graphWith({
      branches: {
        '1': [branch('5', '900', Direction.From), branch('5', '100', Direction.To), branch('5', '050', Direction.To)],
      },
    });
    const result = canonicalizeGraphOrder(graph);
    // same target node 5: To before From, and within To, hash 050 before 100
    expect(result.branches['1'].map((b) => [b.direction, b.relationship_hash])).toEqual([
      [Direction.To, '050'],
      [Direction.To, '100'],
      [Direction.From, '900'],
    ]);
  });

  it('produces identical output for two graphs that differ only in serialization order', () => {
    // same content, keys and branch arrays supplied in different orders (the reported bug)
    const graphA = graphWith({
      initial: ['1'],
      data_map: { '1': {}, '2': {}, '3': {} },
      branches: {
        '1': [branch('3', 'h3'), branch('2', 'h2')],
        '2': [branch('3', 'h4')],
      },
    });
    const graphB = graphWith({
      initial: ['1'],
      data_map: { '3': {}, '1': {}, '2': {} },
      branches: {
        '2': [branch('3', 'h4')],
        '1': [branch('2', 'h2'), branch('3', 'h3')],
      },
    });
    const canonA = canonicalizeGraphOrder(graphA);
    const canonB = canonicalizeGraphOrder(graphB);
    expect(Object.keys(canonB.data_map)).toEqual(Object.keys(canonA.data_map));
    expect(Object.keys(canonB.branches)).toEqual(Object.keys(canonA.branches));
    expect(canonB.branches).toEqual(canonA.branches);
  });

  it('does not mutate branch arrays shared with an untouched input graph', () => {
    const shared: BranchNode[] = [branch('20', 'b'), branch('10', 'a')];
    const graph = graphWith({ branches: {} });
    // simulate a caller (mergeGrowthInto) that shares an array reference
    graph.branches = { '1': shared };
    canonicalizeGraphOrder(graph);
    // the original shared array is left in its input order; canonicalization returns a sorted copy
    expect(shared.map((b) => b.node)).toEqual(['20', '10']);
  });
});
