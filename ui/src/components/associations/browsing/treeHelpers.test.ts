import { describe, it, expect } from 'vitest';

// project imports
import {
  ascendFirstParent,
  buildTreeIndex,
  contextualDisplayEdges,
  defaultBidirectional,
  displayChildren,
  DOWN_DEFAULT_CFG,
  findMultiParentNodeIds,
  hasContextualDisplayChildren,
  isStructuralEdge,
  parentIdsOf,
  REVERSE_MAX_DEPTH,
  structuralParentIdsOf,
  toDisplayCfg,
  TreeEdge,
  TreeOrientation,
} from './treeHelpers';
import { AssociationKind } from '@models/associations';
import { Entities } from '@models/entities';
import { type BranchNode, Direction, type Graph, type TreeNode, TreeNodeKey } from '@models/trees';

// --- fixture builders (mirrors browserHelpers.test.ts) ---

function entityNode(id: string, name: string, kind: Entities): TreeNode {
  return { [TreeNodeKey.Entity]: { id, name, kind, tags: {}, description: null } } as unknown as TreeNode;
}

function assocBranch(node: string, kind: AssociationKind, direction: Direction, hash = `${node}-${kind}`): BranchNode {
  return { relationship: { Association: { kind, direction } }, node, direction, relationship_hash: hash } as BranchNode;
}

function tagBranch(node: string, direction = Direction.To): BranchNode {
  return { relationship: { Tags: 'Tags' }, node, direction, relationship_hash: `${node}-tags` };
}

function mkGraph(opts: {
  dataMap: Record<string, TreeNode>;
  branches: Record<string, BranchNode[]>;
  initial?: string[];
  growable?: string[];
}): Graph {
  return {
    id: 'tree-1',
    initial: opts.initial ?? ['root'],
    growable: opts.growable ?? [],
    data_map: opts.dataMap,
    branches: opts.branches,
  };
}

/**
 * A directed relationship chain modeling the real `SigmaRule → Flag → WindowsProcess` shape: each edge is a
 * plain (non-structural) association stored `To` from the source, so from the *process* end the chain is only
 * reachable by walking parents (reverse).
 */
function sigmaFlagProcessGraph(): Graph {
  return mkGraph({
    dataMap: {
      sig: entityNode('s', 'Rule', Entities.SigmaRule),
      flag: entityNode('f', 'Suspicious', Entities.Flag),
      proc: entityNode('p', 'powershell.exe', Entities.WindowsProcess),
    },
    branches: {
      sig: [assocBranch('flag', AssociationKind.SigmaRuleHit, Direction.To)],
      flag: [assocBranch('proc', AssociationKind.AssociatedWith, Direction.To)],
    },
    initial: ['proc'],
  });
}

describe('ascendFirstParent', () => {
  // root --(To)--> mid --(To)--> leaf, so parentsOf ascends leaf → mid → root
  function chainGraph(): Graph {
    return mkGraph({
      dataMap: {
        root: entityNode('r', 'Root', Entities.Device),
        mid: entityNode('m', 'Mid', Entities.Device),
        leaf: entityNode('l', 'Leaf', Entities.Device),
      },
      branches: {
        root: [assocBranch('mid', AssociationKind.AssociatedWith, Direction.To)],
        mid: [assocBranch('leaf', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['leaf'],
    });
  }

  it('returns the start-inclusive chain nearest-first up to the top', () => {
    const idx = buildTreeIndex(chainGraph());
    expect(ascendFirstParent(idx, 'leaf', (i, id) => parentIdsOf(i, id)[0] ?? null)).toEqual(['leaf', 'mid', 'root']);
  });

  it('returns just the start when it has no selected parent', () => {
    const idx = buildTreeIndex(chainGraph());
    expect(ascendFirstParent(idx, 'root', (i, id) => parentIdsOf(i, id)[0] ?? null)).toEqual(['root']);
  });

  it('stops on a cycle, visiting each node at most once', () => {
    // a ⇄ b via bidirectional edges, so first-parent ascent would loop without the visited guard
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), b: entityNode('b', 'B', Entities.Device) },
      branches: {
        a: [assocBranch('b', AssociationKind.AssociatedWith, Direction.Bidirectional, 'H')],
        b: [assocBranch('a', AssociationKind.AssociatedWith, Direction.Bidirectional, 'H')],
      },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    expect(ascendFirstParent(idx, 'a', (i, id) => parentIdsOf(i, id)[0] ?? null)).toEqual(['a', 'b']);
  });
});

describe('parentIdsOf', () => {
  it('returns distinct parent ids in edge order', () => {
    const graph = mkGraph({
      dataMap: {
        a: entityNode('a', 'A', Entities.Device),
        b: entityNode('b', 'B', Entities.Device),
        c: entityNode('c', 'C', Entities.Device),
      },
      branches: {
        a: [assocBranch('c', AssociationKind.AssociatedWith, Direction.To)],
        b: [assocBranch('c', AssociationKind.FirmwareFor, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    expect(parentIdsOf(idx, 'c')).toEqual(['a', 'b']);
  });
  it('collapses multiple edges to the same parent to one id', () => {
    // a→c via two distinct relationship hashes: two parent edges, one distinct parent
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), c: entityNode('c', 'C', Entities.Device) },
      branches: {
        a: [
          assocBranch('c', AssociationKind.AssociatedWith, Direction.To, 'H1'),
          assocBranch('c', AssociationKind.FirmwareFor, Direction.To, 'H2'),
        ],
      },
    });
    const idx = buildTreeIndex(graph);
    expect(idx.parentsOf.get('c')).toHaveLength(2);
    expect(parentIdsOf(idx, 'c')).toEqual(['a']);
  });
  it('returns [] for a node with no parents', () => {
    const graph = mkGraph({ dataMap: { a: entityNode('a', 'A', Entities.Device) }, branches: {} });
    expect(parentIdsOf(buildTreeIndex(graph), 'a')).toEqual([]);
  });
});

describe('defaultBidirectional', () => {
  const edge = (kind: AssociationKind | undefined, relationship?: TreeEdge['relationship']): TreeEdge =>
    ({
      id: 'x',
      direction: Direction.To,
      label: 'l',
      relationship: relationship ?? { Association: { kind, direction: Direction.To } },
    }) as TreeEdge;
  it('is true for every kind in the non-structural whitelist', () => {
    expect(defaultBidirectional(edge(AssociationKind.AssociatedWith))).toBe(true);
    expect(defaultBidirectional(edge(AssociationKind.SigmaRuleHit))).toBe(true);
    expect(defaultBidirectional(edge(AssociationKind.HasNetworkConnection))).toBe(true);
    expect(defaultBidirectional(edge(AssociationKind.ContainsCVE))).toBe(true);
    expect(defaultBidirectional(edge(AssociationKind.ContainsCWE))).toBe(true);
    // the flag chain: SigmaRule -CreatedBy-> Flag -FlagFor-> WindowsProcess, both reverse-surfaced
    expect(defaultBidirectional(edge(AssociationKind.FlagFor))).toBe(true);
    expect(defaultBidirectional(edge(AssociationKind.CreatedBy))).toBe(true);
  });
  it('is false for containment and process/company parentage', () => {
    expect(defaultBidirectional(edge(AssociationKind.FileIn))).toBe(false);
    expect(defaultBidirectional(edge(AssociationKind.ChildProcess))).toBe(false);
    expect(defaultBidirectional(edge(AssociationKind.ParentCompanyOf))).toBe(false);
  });
  it('is false for semantic relationships now outside the whitelist (structural by default)', () => {
    expect(defaultBidirectional(edge(AssociationKind.FirmwareFor))).toBe(false);
    expect(defaultBidirectional(edge(AssociationKind.FileFor))).toBe(false);
    expect(defaultBidirectional(edge(AssociationKind.DevelopedBy))).toBe(false);
    expect(defaultBidirectional(edge(AssociationKind.UsedBy))).toBe(false);
  });
  it('is false for non-Association relationships (Tags)', () => {
    expect(defaultBidirectional(edge(undefined, { Tags: 'Tags' }))).toBe(false);
  });
});

describe('isStructuralEdge', () => {
  const edge = (kind: AssociationKind | undefined, relationship?: TreeEdge['relationship']): TreeEdge =>
    ({
      id: 'x',
      direction: Direction.To,
      label: 'l',
      relationship: relationship ?? { Association: { kind, direction: Direction.To } },
    }) as TreeEdge;
  it('is true for containment, process/company parentage, and semantic relationships', () => {
    expect(isStructuralEdge(edge(AssociationKind.FileIn))).toBe(true);
    expect(isStructuralEdge(edge(AssociationKind.ChildProcess))).toBe(true);
    expect(isStructuralEdge(edge(AssociationKind.FirmwareFor))).toBe(true);
    expect(isStructuralEdge(edge(AssociationKind.DevelopedBy))).toBe(true);
  });
  it('is false for the non-structural whitelist kinds', () => {
    expect(isStructuralEdge(edge(AssociationKind.AssociatedWith))).toBe(false);
    expect(isStructuralEdge(edge(AssociationKind.SigmaRuleHit))).toBe(false);
    expect(isStructuralEdge(edge(AssociationKind.HasNetworkConnection))).toBe(false);
  });
  it('is false for Tags but true for Origin (both non-Association relationships)', () => {
    expect(isStructuralEdge(edge(undefined, { Tags: 'Tags' }))).toBe(false);
    expect(isStructuralEdge(edge(undefined, { Origin: 'Unpacked' } as unknown as TreeEdge['relationship']))).toBe(true);
  });
});

describe('structuralParentIdsOf', () => {
  it('returns only structural parents, dropping relationship and Tags parents', () => {
    // shared has three parents: a structural FirmwareFor, a non-structural AssociatedWith, and a Tag
    const graph = mkGraph({
      dataMap: {
        struct: entityNode('st', 'Struct', Entities.Device),
        rel: entityNode('rl', 'Rel', Entities.Device),
        shared: entityNode('s', 'Shared', Entities.Device),
      },
      branches: {
        struct: [assocBranch('shared', AssociationKind.FirmwareFor, Direction.To)],
        rel: [assocBranch('shared', AssociationKind.AssociatedWith, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    // parentIdsOf sees both; structuralParentIdsOf drops the AssociatedWith parent
    expect(parentIdsOf(idx, 'shared').sort()).toEqual(['rel', 'struct']);
    expect(structuralParentIdsOf(idx, 'shared')).toEqual(['struct']);
  });
});

describe('toDisplayCfg', () => {
  it('defaults to down + directional (never bidirectional) when fields omitted', () => {
    const cfg = toDisplayCfg({});
    expect(cfg.orientation).toBe(TreeOrientation.Down);
    expect(cfg.bidirectional({ relationship: { Association: { kind: AssociationKind.AssociatedWith } } } as TreeEdge)).toBe(false);
  });
  it('preserves a supplied orientation and predicate', () => {
    const cfg = toDisplayCfg({ orientation: TreeOrientation.Up, bidirectional: defaultBidirectional });
    expect(cfg.orientation).toBe(TreeOrientation.Up);
    expect(cfg.bidirectional({ relationship: { Association: { kind: AssociationKind.AssociatedWith } } } as TreeEdge)).toBe(true);
  });
});

describe('displayChildren', () => {
  it('directional cfg surfaces only forward children', () => {
    const graph = sigmaFlagProcessGraph();
    const idx = buildTreeIndex(graph);
    // the process has no forward children; with a directional cfg it shows nothing
    expect(displayChildren(idx, 'proc', toDisplayCfg({})).map((e) => e.id)).toEqual([]);
    // the flag's only forward child is the process
    expect(displayChildren(idx, 'flag', toDisplayCfg({})).map((e) => e.id)).toEqual(['proc']);
  });
  it('bidirectional cfg surfaces reverse edges as reversed copies, primary first', () => {
    const graph = sigmaFlagProcessGraph();
    const idx = buildTreeIndex(graph);
    // the flag: forward child proc (primary) + reverse parent sig (reversed), primary first
    const kids = displayChildren(idx, 'flag', DOWN_DEFAULT_CFG);
    expect(kids.map((e) => e.id)).toEqual(['proc', 'sig']);
    expect(kids.find((e) => e.id === 'proc')!.reversed).toBeFalsy();
    expect(kids.find((e) => e.id === 'sig')!.reversed).toBe(true);
  });
  it('does not surface a structural parent as a reverse child', () => {
    // folder --FileIn--> file (file is child of folder, structural/containment)
    const graph = mkGraph({
      dataMap: { folder: entityNode('fold', 'dir', Entities.Folder), file: entityNode('fl', 'f.bin', Entities.File) },
      branches: { folder: [assocBranch('file', AssociationKind.FileIn, Direction.To)] },
      initial: ['file'],
    });
    const idx = buildTreeIndex(graph);
    // from the file, its parent folder is structural -> not surfaced as a bidirectional child
    expect(displayChildren(idx, 'file', DOWN_DEFAULT_CFG).map((e) => e.id)).toEqual([]);
  });
  it('dedups by id with primary winning over the reverse copy', () => {
    // a and b are mutual bidirectional peers: from a, b appears once (primary), not twice
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), b: entityNode('b', 'B', Entities.Device) },
      branches: {
        a: [assocBranch('b', AssociationKind.AssociatedWith, Direction.Bidirectional, 'H')],
        b: [assocBranch('a', AssociationKind.AssociatedWith, Direction.Bidirectional, 'H')],
      },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    const kids = displayChildren(idx, 'a', DOWN_DEFAULT_CFG);
    expect(kids.map((e) => e.id)).toEqual(['b']);
    expect(kids[0].reversed).toBeFalsy();
  });
  it('never mutates the stored index edges', () => {
    const graph = sigmaFlagProcessGraph();
    const idx = buildTreeIndex(graph);
    displayChildren(idx, 'flag', DOWN_DEFAULT_CFG);
    // the stored parent edge for the flag (sig) must not have been flagged reversed
    expect(idx.parentsOf.get('flag')!.every((e) => e.reversed === undefined)).toBe(true);
  });
});

describe('contextualDisplayEdges / reverse-depth rules', () => {
  it('a node reached via a reversed edge surfaces only further reverse edges, not forward children', () => {
    // sig has a forward child (flag) AND (hypothetically) a reverse parent; reached via reverse it must
    // suppress the forward flag so a reverse-reached SigmaRule does not re-list all its other flags
    const graph = mkGraph({
      dataMap: {
        sig: entityNode('s', 'Rule', Entities.SigmaRule),
        flagA: entityNode('fa', 'FlagA', Entities.Flag),
        flagB: entityNode('fb', 'FlagB', Entities.Flag),
      },
      branches: {
        sig: [
          assocBranch('flagA', AssociationKind.SigmaRuleHit, Direction.To),
          assocBranch('flagB', AssociationKind.SigmaRuleHit, Direction.To),
        ],
      },
      initial: ['flagA'],
    });
    const idx = buildTreeIndex(graph);
    // forward (not viaReversed): sig lists its flags
    expect(
      contextualDisplayEdges(idx, 'sig', DOWN_DEFAULT_CFG, false, 0)
        .map((e) => e.id)
        .sort(),
    ).toEqual(['flagA', 'flagB']);
    // reached via reverse: sig suppresses its forward flags entirely
    expect(contextualDisplayEdges(idx, 'sig', DOWN_DEFAULT_CFG, true, 1).map((e) => e.id)).toEqual([]);
  });
  it('stops following reverse edges at REVERSE_MAX_DEPTH', () => {
    const graph = sigmaFlagProcessGraph();
    const idx = buildTreeIndex(graph);
    // from the flag at reverseDepth just below the cap, the reverse parent (sig) still shows
    expect(contextualDisplayEdges(idx, 'flag', DOWN_DEFAULT_CFG, true, REVERSE_MAX_DEPTH - 1).map((e) => e.id)).toEqual(['sig']);
    // at the cap, reverse edges are dropped
    expect(contextualDisplayEdges(idx, 'flag', DOWN_DEFAULT_CFG, true, REVERSE_MAX_DEPTH).map((e) => e.id)).toEqual([]);
  });
  it('follows reverse edges past the default cap when maxReverseDepth is Infinity (overlay mode)', () => {
    // a 4-deep reverse chain d→c→b→a (all AssociatedWith, To from source): from a, reverse reaches b,c,d
    const graph = mkGraph({
      dataMap: {
        a: entityNode('a', 'A', Entities.Device),
        b: entityNode('b', 'B', Entities.Device),
        c: entityNode('c', 'C', Entities.Device),
        d: entityNode('d', 'D', Entities.Device),
      },
      branches: {
        d: [assocBranch('c', AssociationKind.AssociatedWith, Direction.To)],
        c: [assocBranch('b', AssociationKind.AssociatedWith, Direction.To)],
        b: [assocBranch('a', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    // at reverseDepth 3 (past the default cap of 2), Infinity still surfaces the next reverse parent
    expect(contextualDisplayEdges(idx, 'b', DOWN_DEFAULT_CFG, true, 3, Infinity).map((e) => e.id)).toEqual(['c']);
    // the default bound would have dropped it
    expect(contextualDisplayEdges(idx, 'b', DOWN_DEFAULT_CFG, true, 3).map((e) => e.id)).toEqual([]);
  });
});

describe('hasContextualDisplayChildren', () => {
  it('is true for a process that can reveal its flag via reverse, false under a directional cfg', () => {
    const graph = sigmaFlagProcessGraph();
    const idx = buildTreeIndex(graph);
    expect(hasContextualDisplayChildren(idx, 'proc', DOWN_DEFAULT_CFG, false, 0)).toBe(true);
    expect(hasContextualDisplayChildren(idx, 'proc', toDisplayCfg({}), false, 0)).toBe(false);
  });
});

describe('findMultiParentNodeIds (distinct-parent counting)', () => {
  it('flags a node with two distinct parents', () => {
    const graph = mkGraph({
      dataMap: {
        a: entityNode('a', 'A', Entities.Device),
        b: entityNode('b', 'B', Entities.Device),
        shared: entityNode('s', 'Shared', Entities.Device),
      },
      branches: {
        a: [assocBranch('shared', AssociationKind.AssociatedWith, Direction.To)],
        b: [assocBranch('shared', AssociationKind.AssociatedWith, Direction.To)],
      },
    });
    expect(findMultiParentNodeIds(graph).has('shared')).toBe(true);
  });
  it('does NOT flag a node with one parent reached via multiple relationship edges', () => {
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), shared: entityNode('s', 'Shared', Entities.Device) },
      branches: {
        a: [
          assocBranch('shared', AssociationKind.AssociatedWith, Direction.To, 'H1'),
          assocBranch('shared', AssociationKind.FirmwareFor, Direction.To, 'H2'),
        ],
      },
    });
    // two parent edges but one distinct parent -> not a duplicate
    expect(findMultiParentNodeIds(graph).has('shared')).toBe(false);
  });
  it('ignores tag branches for parentage but still counts real parents', () => {
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), t: entityNode('t', 'T', Entities.Device) },
      branches: { a: [tagBranch('t')] },
    });
    // a single tag edge is one parent, not a duplicate
    expect(findMultiParentNodeIds(graph).has('t')).toBe(false);
  });
});
