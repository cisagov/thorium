import { describe, it, expect } from 'vitest';

// project imports
import { buildTreeIndex, childIdsOf, defaultBidirectional, TreeOrientation } from '../treeHelpers';
import {
  collectGroupOptions,
  collectTagOptions,
  compareByFlagStats,
  computeFlagStats,
  effectiveChildren,
  filterTree,
  findFileNodeHash,
  focusBreadcrumb,
  getDepthFromClauses,
  getDisplayTags,
  getEntityLayerConfigFromClauses,
  groupByKind,
  nodeGroups,
  resolvePolicy,
  resolveRoots,
} from './browserHelpers';
import { FilterCriteria, LayerPolicy, SortMode, TraversalConfig } from './types';
import { Clause, ClauseCondition } from '@components/shared/inputs/omnibar/ClauseTypes';
import { AssociationKind } from '@models/associations';
import { Entities } from '@models/entities';
import { type BranchNode, Direction, type Graph, NodeType, type TreeNode, TreeNodeKey } from '@models/trees';

/** A traversal config with sensible defaults for tests (Show everything, no depth bound). */
function mkCfg(overrides?: Partial<TraversalConfig>): TraversalConfig {
  return {
    clausePolicies: {},
    includeSet: null,
    defaultPolicies: {},
    fallback: LayerPolicy.Show,
    maxDepth: null,
    distances: new Map(),
    ...overrides,
  };
}

/** A filter criteria with no active filters by default. */
function mkCriteria(overrides?: Partial<FilterCriteria>): FilterCriteria {
  return { text: '', tags: {}, groups: [], flaggedOnly: false, flaggedNodes: new Set(), ...overrides };
}

/** Build a layer clause (`Show`/`Hide`/`Exclude`/`Include` is/is-one-of <types>). */
function layerClause(category: string, values: string[]): Clause {
  if (values.length === 1) {
    return { field: category, category, condition: ClauseCondition.Is, value: { value: values[0] } };
  }
  return { field: category, category, condition: ClauseCondition.IsOneOf, value: { values } };
}

const FILE_SHA = 'a'.repeat(64);
const FILE_HASH = 'file-hash';

// --- fixture builders ---

function sampleNode(sha256: string, name: string): TreeNode {
  return { [TreeNodeKey.Sample]: { sha256, submissions: [{ name }], tags: {} } } as unknown as TreeNode;
}

function entityNode(id: string, name: string, kind: Entities, tags: Record<string, Record<string, unknown>> = {}): TreeNode {
  return { [TreeNodeKey.Entity]: { id, name, kind, tags, description: null } } as unknown as TreeNode;
}

function tagNode(tags: Record<string, string[]>): TreeNode {
  return { [TreeNodeKey.Tag]: { tags } };
}

function flagNode(id: string, suspicion: number, confidence: string): TreeNode {
  return {
    [TreeNodeKey.Entity]: { id, name: id, kind: Entities.Flag, tags: {}, metadata: { Flag: { suspicion, confidence } } },
  } as unknown as TreeNode;
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
    initial: opts.initial ?? [FILE_HASH],
    growable: opts.growable ?? [],
    data_map: opts.dataMap,
    branches: opts.branches,
  };
}

describe('findFileNodeHash', () => {
  it('finds the node id for a file sha256', () => {
    const graph = mkGraph({ dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin') }, branches: {} });
    expect(findFileNodeHash(graph, FILE_SHA)).toBe(FILE_HASH);
  });
  it('returns undefined when absent', () => {
    const graph = mkGraph({ dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin') }, branches: {} });
    expect(findFileNodeHash(graph, 'b'.repeat(64))).toBeUndefined();
  });
});

describe('buildTreeIndex (edge-carrying)', () => {
  it('carries relationship, direction and a label per edge', () => {
    const graph = mkGraph({
      dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'), e1: entityNode('id1', 'Dev', Entities.Device) },
      branches: { [FILE_HASH]: [assocBranch('e1', AssociationKind.FirmwareFor, Direction.To)] },
    });
    const idx = buildTreeIndex(graph);
    const edges = idx.childrenOf.get(FILE_HASH)!;
    expect(edges).toHaveLength(1);
    expect(edges[0].id).toBe('e1');
    expect(edges[0].direction).toBe(Direction.To);
    expect(edges[0].relationship.Association?.kind).toBe(AssociationKind.FirmwareFor);
    expect(edges[0].label).toContain('Association');
  });

  it('collapses the reverse-pair a directed edge stores on both endpoints', () => {
    // same relationship_hash stored as To on the file and From on the entity -> one edge under the file
    const graph = mkGraph({
      dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'), e1: entityNode('id1', 'Dev', Entities.Device) },
      branches: {
        [FILE_HASH]: [assocBranch('e1', AssociationKind.FirmwareFor, Direction.To, 'H')],
        e1: [assocBranch(FILE_HASH, AssociationKind.FirmwareFor, Direction.From, 'H')],
      },
    });
    const idx = buildTreeIndex(graph);
    expect(childIdsOf(idx, FILE_HASH)).toEqual(['e1']);
  });

  it('names the container for a "…In" association reached from the container side (To)', () => {
    // file --FileSystemIn--> filesystem : file is the source/container, stored To on the file
    const graph = mkGraph({
      dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'), fs: entityNode('idfs', 'dump.img', Entities.FileSystem) },
      branches: { [FILE_HASH]: [assocBranch('fs', AssociationKind.FileSystemIn, Direction.To)] },
    });
    const idx = buildTreeIndex(graph);
    const edge = idx.childrenOf.get(FILE_HASH)!.find((e) => e.id === 'fs')!;
    expect(edge.containerLabel).toBe('f.bin File');
  });

  it('names the container for a "…In" association reached from the contained side (From)', () => {
    // file --FileIn--> folder stored From on the file : the folder (source/container) becomes the parent
    const graph = mkGraph({
      dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'), folder: entityNode('idfold', 'somefolder', Entities.Folder) },
      branches: { [FILE_HASH]: [assocBranch('folder', AssociationKind.FileIn, Direction.From)] },
    });
    const idx = buildTreeIndex(graph);
    const edge = idx.childrenOf.get('folder')!.find((e) => e.id === FILE_HASH)!;
    expect(edge.containerLabel).toContain('somefolder');
  });

  it('does not add a container label for non-containment kinds (e.g. BasedIn)', () => {
    const graph = mkGraph({
      dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'), c: entityNode('idc', 'Elbonia', Entities.Vendor) },
      branches: { [FILE_HASH]: [assocBranch('c', AssociationKind.BasedIn, Direction.To)] },
    });
    const idx = buildTreeIndex(graph);
    const edge = idx.childrenOf.get(FILE_HASH)!.find((e) => e.id === 'c')!;
    expect(edge.containerLabel).toBeUndefined();
  });

  it('yields mutual parent/child entries for a bidirectional edge', () => {
    const graph = mkGraph({
      dataMap: { a: entityNode('ida', 'A', Entities.Device), b: entityNode('idb', 'B', Entities.Device) },
      branches: {
        a: [assocBranch('b', AssociationKind.AssociatedWith, Direction.Bidirectional, 'H')],
        b: [assocBranch('a', AssociationKind.AssociatedWith, Direction.Bidirectional, 'H')],
      },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    expect(childIdsOf(idx, 'a')).toEqual(['b']);
    expect(childIdsOf(idx, 'b')).toEqual(['a']);
  });
});

describe('effectiveChildren', () => {
  it('returns direct children under the default Show policy, regardless of direction', () => {
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        to: entityNode('t', 'Zeta', Entities.Device),
        from: entityNode('f', 'Alpha', Entities.Device),
        bi: entityNode('b', 'Mu', Entities.Device),
      },
      branches: {
        [FILE_HASH]: [
          assocBranch('to', AssociationKind.FirmwareFor, Direction.To),
          assocBranch('bi', AssociationKind.AssociatedWith, Direction.Bidirectional),
        ],
        from: [assocBranch(FILE_HASH, AssociationKind.DevelopedBy, Direction.From)],
      },
    });
    const idx = buildTreeIndex(graph);
    const kids = effectiveChildren(FILE_HASH, idx, graph, mkCfg(), new Set([FILE_HASH]));
    expect(kids.map((c) => c.edge.id).sort()).toEqual(['bi', 'from', 'to']);
  });

  it('prunes Skip layers entirely', () => {
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        tag: tagNode({ foo: ['bar'] }),
        e1: entityNode('id1', 'Dev', Entities.Device),
      },
      branches: { [FILE_HASH]: [tagBranch('tag'), assocBranch('e1', AssociationKind.FirmwareFor, Direction.To)] },
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ clausePolicies: { [NodeType.Tag]: LayerPolicy.Skip } });
    const kids = effectiveChildren(FILE_HASH, idx, graph, cfg, new Set([FILE_HASH]));
    expect(kids.map((c) => c.edge.id)).toEqual(['e1']);
  });

  it('grafts descendants of a PassThrough layer with a breadcrumb', () => {
    // file -> process tree (PassThrough) -> process (Show)
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        tree: entityNode('idtree', 'ProcTree', Entities.WindowsProcessTree),
        proc: entityNode('idproc', 'svchost.exe', Entities.WindowsProcess),
      },
      branches: {
        [FILE_HASH]: [assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To)],
        tree: [assocBranch('proc', AssociationKind.ChildProcess, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const kids = effectiveChildren(
      FILE_HASH,
      idx,
      graph,
      mkCfg({ clausePolicies: { [Entities.WindowsProcessTree]: LayerPolicy.PassThrough } }),
      new Set([FILE_HASH]),
    );
    expect(kids).toHaveLength(1);
    expect(kids[0].edge.id).toBe('proc');
    expect(kids[0].breadcrumb).toEqual(['ProcTree']);
  });

  it('guards cycles per path but still renders DAG re-convergence', () => {
    // root -> x -> z ; root -> y -> z (z reachable via two paths); and a cycle x -> root
    const graph = mkGraph({
      dataMap: {
        root: entityNode('idr', 'Root', Entities.Device),
        x: entityNode('idx', 'X', Entities.Device),
        y: entityNode('idy', 'Y', Entities.Device),
        z: entityNode('idz', 'Z', Entities.Device),
      },
      branches: {
        root: [
          assocBranch('x', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('y', AssociationKind.AssociatedWith, Direction.To),
        ],
        x: [
          assocBranch('z', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('root', AssociationKind.AssociatedWith, Direction.To),
        ],
        y: [assocBranch('z', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['root'],
    });
    const idx = buildTreeIndex(graph);
    // z renders under both x and y
    expect(effectiveChildren('x', idx, graph, mkCfg(), new Set(['root', 'x'])).map((c) => c.edge.id)).toContain('z');
    expect(effectiveChildren('y', idx, graph, mkCfg(), new Set(['root', 'y'])).map((c) => c.edge.id)).toEqual(['z']);
    // cycle back to root is guarded
    expect(effectiveChildren('x', idx, graph, mkCfg(), new Set(['root', 'x'])).map((c) => c.edge.id)).not.toContain('root');
  });

  it('drops a hidden leaf child', () => {
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        keep: entityNode('k', 'Keep', Entities.Device),
        gone: entityNode('g', 'Gone', Entities.Device),
      },
      branches: {
        [FILE_HASH]: [
          assocBranch('keep', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('gone', AssociationKind.AssociatedWith, Direction.To),
        ],
      },
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ hiddenNodes: new Set(['gone']) });
    expect(effectiveChildren(FILE_HASH, idx, graph, cfg, new Set([FILE_HASH])).map((c) => c.edge.id)).toEqual(['keep']);
  });

  it('hiding a mid-tree node hides its entire subtree', () => {
    // file -> mid -> deep ; hiding `mid` must also drop `deep`
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        mid: entityNode('m', 'Mid', Entities.Device),
        deep: entityNode('d', 'Deep', Entities.Device),
      },
      branches: {
        [FILE_HASH]: [assocBranch('mid', AssociationKind.AssociatedWith, Direction.To)],
        mid: [assocBranch('deep', AssociationKind.AssociatedWith, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ hiddenNodes: new Set(['mid']) });
    // mid is gone from the file's children, and nothing grafts `deep` up in its place
    expect(effectiveChildren(FILE_HASH, idx, graph, cfg, new Set([FILE_HASH])).map((c) => c.edge.id)).toEqual([]);
  });

  it('hiding a pass-through node suppresses the descendants it would otherwise graft up', () => {
    // file -> tree (PassThrough) -> proc (Show). Hiding the pass-through node drops the whole branch, so
    // `proc` is NOT grafted onto the file (hidden check runs before the policy check).
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        tree: entityNode('idtree', 'ProcTree', Entities.WindowsProcessTree),
        proc: entityNode('idproc', 'svchost.exe', Entities.WindowsProcess),
      },
      branches: {
        [FILE_HASH]: [assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To)],
        tree: [assocBranch('proc', AssociationKind.ChildProcess, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({
      clausePolicies: { [Entities.WindowsProcessTree]: LayerPolicy.PassThrough },
      hiddenNodes: new Set(['tree']),
    });
    expect(effectiveChildren(FILE_HASH, idx, graph, cfg, new Set([FILE_HASH])).map((c) => c.edge.id)).toEqual([]);
  });

  it('hides a DAG-duplicate node under every parent (hiding is by node id)', () => {
    // root -> x -> z ; root -> y -> z. Hiding `z` removes it under both x and y.
    const graph = mkGraph({
      dataMap: {
        root: entityNode('idr', 'Root', Entities.Device),
        x: entityNode('idx', 'X', Entities.Device),
        y: entityNode('idy', 'Y', Entities.Device),
        z: entityNode('idz', 'Z', Entities.Device),
      },
      branches: {
        root: [
          assocBranch('x', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('y', AssociationKind.AssociatedWith, Direction.To),
        ],
        x: [assocBranch('z', AssociationKind.AssociatedWith, Direction.To)],
        y: [assocBranch('z', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['root'],
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ hiddenNodes: new Set(['z']) });
    expect(effectiveChildren('x', idx, graph, cfg, new Set(['root', 'x'])).map((c) => c.edge.id)).toEqual([]);
    expect(effectiveChildren('y', idx, graph, cfg, new Set(['root', 'y'])).map((c) => c.edge.id)).toEqual([]);
  });

  it('prunes nodes beyond the depth bound', () => {
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        near: entityNode('n', 'Near', Entities.Device),
        far: entityNode('f', 'Far', Entities.Device),
      },
      branches: {
        [FILE_HASH]: [assocBranch('near', AssociationKind.AssociatedWith, Direction.To)],
        near: [assocBranch('far', AssociationKind.AssociatedWith, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const distances = new Map([
      [FILE_HASH, 0],
      ['near', 1],
      ['far', 2],
    ]);
    // maxDepth 1 keeps `near` but prunes `far`
    const cfg = mkCfg({ maxDepth: 1, distances });
    expect(effectiveChildren('near', idx, graph, cfg, new Set([FILE_HASH, 'near'])).map((c) => c.edge.id)).toEqual([]);
    expect(effectiveChildren(FILE_HASH, idx, graph, cfg, new Set([FILE_HASH])).map((c) => c.edge.id)).toEqual(['near']);
  });
});

describe('effectiveChildren — bidirectional relationship display', () => {
  /** A bidirectional-aware config (the entity browser's default view policy). */
  const bidiCfg = (overrides?: Partial<TraversalConfig>) =>
    mkCfg({ orientation: TreeOrientation.Down, bidirectional: defaultBidirectional, ...overrides });

  /** SigmaRule --SigmaRuleHit--> Flag --AssociatedWith--> WindowsProcess, all stored `To` from the source. */
  const sigmaChain = () =>
    mkGraph({
      dataMap: {
        sig: entityNode('s', 'Rule', Entities.SigmaRule),
        flag: entityNode('fl', 'Suspicious', Entities.Flag),
        proc: entityNode('p', 'powershell.exe', Entities.WindowsProcess),
      },
      branches: {
        sig: [assocBranch('flag', AssociationKind.SigmaRuleHit, Direction.To)],
        flag: [assocBranch('proc', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['proc'],
    });

  it('surfaces a WindowsProcess → Flag → SigmaRule via reverse edges (relationship kinds)', () => {
    const graph = sigmaChain();
    const idx = buildTreeIndex(graph);
    // from the process, its Flag is reachable as a reversed child
    const procKids = effectiveChildren('proc', idx, graph, bidiCfg(), new Set(['proc']));
    expect(procKids.map((c) => c.edge.id)).toEqual(['flag']);
    expect(procKids[0].viaReversed).toBe(true);
    expect(procKids[0].reverseDepth).toBe(1);
    // from that Flag (arrived via reverse at depth 1), the SigmaRule is the next reversed child
    const flagKids = effectiveChildren('flag', idx, graph, bidiCfg(), new Set(['proc', 'flag']), 1, true);
    expect(flagKids.map((c) => c.edge.id)).toEqual(['sig']);
    expect(flagKids[0].reverseDepth).toBe(2);
  });

  it('a reverse-reached SigmaRule does not fan back out to its other forward children', () => {
    // sig hits two flags; arriving at sig via reverse from flagA must NOT re-list flagB
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
    // sig arrived via reverse (depth 2): forward flags suppressed
    expect(effectiveChildren('sig', idx, graph, bidiCfg(), new Set(['flagA', 'sig']), 2, true).map((c) => c.edge.id)).toEqual([]);
  });

  it('keeps containment (Folder → File) directional — a File never lists its Folder as a child', () => {
    // folder --FileIn--> file (structural). Even under the bidirectional cfg the file has no reverse child.
    const graph = mkGraph({
      dataMap: { folder: entityNode('fold', 'dir', Entities.Folder), file: entityNode('fl', 'f.bin', Entities.File) },
      branches: { folder: [assocBranch('file', AssociationKind.FileIn, Direction.To)] },
      initial: ['file'],
    });
    const idx = buildTreeIndex(graph);
    expect(effectiveChildren('file', idx, graph, bidiCfg(), new Set(['file'])).map((c) => c.edge.id)).toEqual([]);
    // the folder still shows the file as a forward child
    expect(effectiveChildren('folder', idx, graph, bidiCfg(), new Set(['folder'])).map((c) => c.edge.id)).toEqual(['file']);
  });

  it('no longer reverse-surfaces a FirmwareFor edge (now structural under the whitelist)', () => {
    // device --FirmwareFor--> file: FirmwareFor left the whitelist, so from the file the device is NOT a child
    const graph = mkGraph({
      dataMap: { device: entityNode('d', 'Router', Entities.Device), file: entityNode('fl', 'fw.bin', Entities.File) },
      branches: { device: [assocBranch('file', AssociationKind.FirmwareFor, Direction.To)] },
      initial: ['file'],
    });
    const idx = buildTreeIndex(graph);
    expect(effectiveChildren('file', idx, graph, bidiCfg(), new Set(['file'])).map((c) => c.edge.id)).toEqual([]);
    // forward direction is unaffected — the device still lists the file
    expect(effectiveChildren('device', idx, graph, bidiCfg(), new Set(['device'])).map((c) => c.edge.id)).toEqual(['file']);
  });

  it('surfaces the real flag chain: WindowsProcess → Flag (FlagFor) → SigmaRule (CreatedBy)', () => {
    // the actual backend shape: SigmaRule -CreatedBy-> Flag -FlagFor-> WindowsProcess (both stored To from source)
    const graph = mkGraph({
      dataMap: {
        sig: entityNode('s', 'Rule', Entities.SigmaRule),
        flag: entityNode('fl', 'Suspicious', Entities.Flag),
        proc: entityNode('p', 'powershell.exe', Entities.WindowsProcess),
      },
      branches: {
        sig: [assocBranch('flag', AssociationKind.CreatedBy, Direction.To)],
        flag: [assocBranch('proc', AssociationKind.FlagFor, Direction.To)],
      },
      initial: ['proc'],
    });
    const idx = buildTreeIndex(graph);
    // from the process, the Flag surfaces as a reversed child (FlagFor is reverse-eligible)
    const procKids = effectiveChildren('proc', idx, graph, bidiCfg(), new Set(['proc']));
    expect(procKids.map((c) => c.edge.id)).toEqual(['flag']);
    expect(procKids[0].viaReversed).toBe(true);
    // from that Flag (reverseDepth 1), the SigmaRule surfaces via the reversed CreatedBy edge
    const flagKids = effectiveChildren('flag', idx, graph, bidiCfg(), new Set(['proc', 'flag']), 1, true);
    expect(flagKids.map((c) => c.edge.id)).toEqual(['sig']);
  });
});

describe('effectiveChildren — spanning re-root', () => {
  // walk the whole rendered tree from a root, collecting every reached node id (root inclusive)
  const reachable = (rootId: string, graph: Graph, idx: ReturnType<typeof buildTreeIndex>, cfg: TraversalConfig): Set<string> => {
    const seen = new Set<string>([rootId]);
    const visit = (id: string, path: Set<string>) => {
      for (const child of effectiveChildren(id, idx, graph, cfg, path)) {
        if (seen.has(child.edge.id)) continue;
        seen.add(child.edge.id);
        visit(child.edge.id, new Set([...path, child.edge.id]));
      }
    };
    visit(rootId, new Set([rootId]));
    return seen;
  };

  // structural chain a -> b -> c (FileFor, directional-only): from the leaf c, ancestors are only reachable by
  // walking UP, which the default (pruning) traversal never does for a structural edge
  const chain = () =>
    mkGraph({
      dataMap: {
        a: entityNode('a', 'A', Entities.Device),
        b: entityNode('b', 'B', Entities.Device),
        c: entityNode('c', 'C', Entities.Device),
      },
      branches: {
        a: [assocBranch('b', AssociationKind.FileFor, Direction.To)],
        b: [assocBranch('c', AssociationKind.FileFor, Direction.To)],
      },
      initial: ['a'],
    });

  it('re-nests every connected node under the new root, including former ancestors', () => {
    const graph = chain();
    const idx = buildTreeIndex(graph);
    expect(reachable('c', graph, idx, mkCfg({ spanning: true }))).toEqual(new Set(['a', 'b', 'c']));
  });

  it('default (pruning) traversal from the same leaf does not surface structural ancestors', () => {
    const graph = chain();
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ orientation: TreeOrientation.Down, bidirectional: defaultBidirectional });
    expect(reachable('c', graph, idx, cfg)).toEqual(new Set(['c']));
  });

  it('spans across a cycle without infinite recursion, visiting each node once', () => {
    // a sample ↔ compiled ↔ decompiled cycle with a flag on the compiled function
    const graph = mkGraph({
      dataMap: {
        sample: entityNode('s', 'Sample', Entities.Device),
        compiled: entityNode('cf', 'Sleep', Entities.Device),
        decompiled: entityNode('df', 'Sleep (decompiled)', Entities.Device),
        flag: entityNode('fl', 'Sleep call', Entities.Flag),
      },
      branches: {
        sample: [
          assocBranch('compiled', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('decompiled', AssociationKind.AssociatedWith, Direction.To),
        ],
        compiled: [assocBranch('decompiled', AssociationKind.AssociatedWith, Direction.To)],
        flag: [assocBranch('compiled', AssociationKind.FlagFor, Direction.To)],
      },
      initial: ['compiled'],
    });
    const idx = buildTreeIndex(graph);
    // re-rooted at the flag, the whole component reappears beneath it (nothing pruned, nothing duplicated away)
    expect(reachable('flag', graph, idx, mkCfg({ spanning: true }))).toEqual(new Set(['flag', 'compiled', 'decompiled', 'sample']));
  });

  it('emits neutral arrival context (viaReversed=false, reverseDepth=0), even for reverse edges', () => {
    // a --AssociatedWith(To)--> b: from b, a is only reachable via a REVERSE edge. In spanning that child must
    // still carry neutral context so downstream (nodeId, viaReversed, reverseDepth) cycle-guard keys stay finite.
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), b: entityNode('b', 'B', Entities.Device) },
      branches: { a: [assocBranch('b', AssociationKind.AssociatedWith, Direction.To)] },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ spanning: true });
    const fromB = effectiveChildren('b', idx, graph, cfg, new Set(['b']));
    expect(fromB.map((c) => c.edge.id)).toEqual(['a']);
    expect(fromB[0].viaReversed).toBe(false);
    expect(fromB[0].reverseDepth).toBe(0);
    const fromA = effectiveChildren('a', idx, graph, cfg, new Set(['a']));
    expect(fromA[0].viaReversed).toBe(false);
    expect(fromA[0].reverseDepth).toBe(0);
  });
});

describe('filterTree — spanning (re-root) cycle safety', () => {
  // A hub with three leaves. Re-rooting at a leaf makes the walk bounce leaf↔hub; before the arrival-context
  // pin this overflowed filterTree's dfs.
  const star = () =>
    mkGraph({
      dataMap: {
        hub: entityNode('h', 'Hub', Entities.Device),
        l0: entityNode('l0', 'Leaf Zero', Entities.Device),
        l1: entityNode('l1', 'Leaf One', Entities.Device),
        l2: entityNode('l2', 'Leaf Two', Entities.Device),
      },
      branches: {
        hub: [
          assocBranch('l0', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('l1', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('l2', AssociationKind.AssociatedWith, Direction.To),
        ],
      },
      initial: ['hub'],
    });

  it('star re-rooted at a leaf + text filter: terminates and returns the root→match path', () => {
    const graph = star();
    const idx = buildTreeIndex(graph);
    // re-root at l0; the match is a different leaf reached only by going up through the hub
    const visible = filterTree(['l0'], idx, graph, mkCriteria({ text: 'Leaf Two' }), mkCfg({ spanning: true }));
    expect(visible).toEqual(new Set(['l0', 'hub', 'l2']));
  });

  it('two-node A↔B bounce + filter: terminates with the correct set', () => {
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'Alpha', Entities.Device), b: entityNode('b', 'Bravo Needle', Entities.Device) },
      branches: { a: [assocBranch('b', AssociationKind.AssociatedWith, Direction.To)] },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    const visible = filterTree(['a'], idx, graph, mkCriteria({ text: 'needle' }), mkCfg({ spanning: true }));
    expect(visible).toEqual(new Set(['a', 'b']));
  });

  it('cyclic diamond: the match stays visible and reachable (discovery-chain semantics)', () => {
    // R–M–N–Z and R–P–N form an undirected cycle R-M-N-P-R (a "diamond"): N has two parents M and P.
    const graph = mkGraph({
      dataMap: {
        r: entityNode('r', 'Root', Entities.Device),
        m: entityNode('m', 'Mid', Entities.Device),
        p: entityNode('p', 'Par', Entities.Device),
        n: entityNode('n', 'Node', Entities.Device),
        z: entityNode('z', 'Zeta Needle', Entities.Device),
      },
      branches: {
        r: [assocBranch('m', AssociationKind.AssociatedWith, Direction.To), assocBranch('p', AssociationKind.AssociatedWith, Direction.To)],
        m: [assocBranch('n', AssociationKind.AssociatedWith, Direction.To)],
        p: [assocBranch('n', AssociationKind.AssociatedWith, Direction.To)],
        n: [assocBranch('z', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['r'],
    });
    const idx = buildTreeIndex(graph);
    const visible = filterTree(['r'], idx, graph, mkCriteria({ text: 'needle' }), mkCfg({ spanning: true }));
    // hard invariants: no overflow (the call returns), the match is visible, and a full root→match path is visible
    expect(visible.has('z')).toBe(true);
    expect(visible.has('n')).toBe(true);
    expect(visible.has('m')).toBe(true);
    expect(visible.has('r')).toBe(true);
    // KNOWN discovery-chain limitation of the memoized single-node-path dfs on a CYCLIC graph: the alternate
    // duplicate parent `p` is omitted (order-dependent on branch order, which canonicalization makes stable).
    // The match is never lost — it stays reachable via r→m→n→z. The phase-2 linear-flood filterTree will make
    // this complete/order-independent (include p). Pinned here intentionally; update when phase-2 lands.
    expect(visible.has('p')).toBe(false);
  });

  it('hidden hub in spanning: walk terminates and prunes the far side', () => {
    const graph = star();
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ spanning: true, hiddenNodes: new Set(['hub']) });
    // re-root at l0; hub is hidden, so nothing beyond it is reachable and the match (l2) is cut
    const visible = filterTree(['l0'], idx, graph, mkCriteria({ text: 'Leaf Two' }), cfg);
    expect(visible.has('hub')).toBe(false);
    expect(visible.has('l2')).toBe(false);
  });

  it('PassThrough hub inside a spanning cycle: terminates and grafts through', () => {
    // triangle R–PT–X–R with PT elided (PassThrough); the match X must surface (grafted), PT must not render
    const graph = mkGraph({
      dataMap: {
        r: entityNode('r', 'Root', Entities.Device),
        pt: entityNode('pt', 'Tree', Entities.WindowsProcessTree),
        x: entityNode('x', 'Xray Needle', Entities.Device),
      },
      branches: {
        r: [assocBranch('pt', AssociationKind.ProcessTreeIn, Direction.To), assocBranch('x', AssociationKind.AssociatedWith, Direction.To)],
        pt: [assocBranch('x', AssociationKind.ChildProcess, Direction.To)],
      },
      initial: ['r'],
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ spanning: true, clausePolicies: { [Entities.WindowsProcessTree]: LayerPolicy.PassThrough } });
    const visible = filterTree(['r'], idx, graph, mkCriteria({ text: 'needle' }), cfg);
    expect(visible.has('x')).toBe(true);
    expect(visible.has('r')).toBe(true);
    expect(visible.has('pt')).toBe(false); // elided (PassThrough)
  });

  it('flagged-only + spanning: surfaces a flagged node’s direct associations and terminates', () => {
    // flag -FlagFor-> proc ; proc -AssociatedWith-> assoc. Re-root at proc, Flagged-Only.
    const graph = mkGraph({
      dataMap: {
        proc: entityNode('p', 'proc', Entities.WindowsProcess),
        flag: flagNode('fl', 5, 'Likely'),
        assoc: entityNode('a', 'assoc', Entities.Device),
      },
      branches: {
        flag: [assocBranch('proc', AssociationKind.FlagFor, Direction.To)],
        proc: [assocBranch('assoc', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['proc'],
    });
    const idx = buildTreeIndex(graph);
    const { flagged: flaggedNodes } = computeFlagStats(graph, idx);
    const cfg = mkCfg({ spanning: true });
    const visible = filterTree(['proc'], idx, graph, mkCriteria({ flaggedOnly: true, flaggedNodes }), cfg);
    expect(visible.has('proc')).toBe(true);
    expect(visible.has('flag')).toBe(true);
    // decision-3: a flagged node's direct (even unflagged) association stays visible
    expect(visible.has('assoc')).toBe(true);
  });
});

describe('filterTree — reverse-reachable matches', () => {
  it('keeps a reverse-reachable SigmaRule (and the process ancestor) when it matches text', () => {
    const graph = mkGraph({
      dataMap: {
        proc: entityNode('p', 'powershell.exe', Entities.WindowsProcess),
        flag: entityNode('fl', 'Suspicious', Entities.Flag),
        sig: entityNode('s', 'NeedleRule', Entities.SigmaRule),
      },
      branches: {
        sig: [assocBranch('flag', AssociationKind.SigmaRuleHit, Direction.To)],
        flag: [assocBranch('proc', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['proc'],
    });
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ orientation: TreeOrientation.Down, bidirectional: defaultBidirectional });
    const visible = filterTree(['proc'], idx, graph, mkCriteria({ text: 'needle' }), cfg);
    // the sigma rule (reverse-reached from the process) matches and its reverse-ancestors stay visible
    expect(visible.has('sig')).toBe(true);
    expect(visible.has('flag')).toBe(true);
    expect(visible.has('proc')).toBe(true);
  });
});

describe('filterTree — Flagged Only direct associations (decision 3)', () => {
  it('keeps a flagged node’s direct associations visible but collapses deeper unflagged branches', () => {
    // MemoryDump -ProcessTreeIn-> tree -ChildProcess-> process ; flag -FlagFor-> process ; rule -CreatedBy-> flag
    // process -AssociatedWith-> assoc -AssociatedWith-> deep ; MemoryDump -AssociatedWith-> clean -AssociatedWith-> cleanDeep
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'mem.dmp'),
        tree: entityNode('t', 'ProcTree', Entities.WindowsProcessTree),
        process: entityNode('p', 'p.exe', Entities.WindowsProcess),
        flag: flagNode('fl', 5, 'Likely'),
        rule: entityNode('s', 'Rule', Entities.SigmaRule),
        assoc: entityNode('a', 'assoc', Entities.Device),
        deep: entityNode('d', 'deep', Entities.Device),
        clean: entityNode('c', 'clean', Entities.Device),
        cleanDeep: entityNode('cd', 'cleanDeep', Entities.Device),
      },
      branches: {
        [FILE_HASH]: [
          assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To),
          assocBranch('clean', AssociationKind.AssociatedWith, Direction.To),
        ],
        tree: [assocBranch('process', AssociationKind.ChildProcess, Direction.To)],
        flag: [assocBranch('process', AssociationKind.FlagFor, Direction.To)],
        rule: [assocBranch('flag', AssociationKind.CreatedBy, Direction.To)],
        process: [assocBranch('assoc', AssociationKind.AssociatedWith, Direction.To)],
        assoc: [assocBranch('deep', AssociationKind.AssociatedWith, Direction.To)],
        clean: [assocBranch('cleanDeep', AssociationKind.AssociatedWith, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const { flagged: flaggedNodes } = computeFlagStats(graph, idx);
    const cfg = mkCfg({ orientation: TreeOrientation.Down, bidirectional: defaultBidirectional });
    const visible = filterTree([FILE_HASH], idx, graph, mkCriteria({ flaggedOnly: true, flaggedNodes }), cfg);
    // the flagged spine stays visible
    expect(visible.has(FILE_HASH)).toBe(true);
    expect(visible.has('tree')).toBe(true);
    expect(visible.has('process')).toBe(true);
    expect(visible.has('flag')).toBe(true);
    // direct associations of flagged nodes stay visible even though unflagged
    expect(visible.has('rule')).toBe(true); // directly attached to the flag
    expect(visible.has('assoc')).toBe(true); // directly attached to the process
    expect(visible.has('clean')).toBe(true); // directly attached to the memory dump
    // deeper unflagged branches collapse just below the first unflagged relation
    expect(visible.has('deep')).toBe(false);
    expect(visible.has('cleanDeep')).toBe(false);
  });
});

describe('resolvePolicy', () => {
  it('applies precedence: explicit clause > include(Show) > default > include-others(PassThrough) > fallback', () => {
    const cfg = mkCfg({
      clausePolicies: { [Entities.Device]: LayerPolicy.Skip },
      includeSet: new Set<NodeType>([Entities.SigmaRule]),
      defaultPolicies: { [NodeType.Tag]: LayerPolicy.Skip },
    });
    expect(resolvePolicy(Entities.Device, cfg)).toBe(LayerPolicy.Skip); // explicit wins
    expect(resolvePolicy(Entities.SigmaRule, cfg)).toBe(LayerPolicy.Show); // in whitelist
    expect(resolvePolicy(NodeType.Tag, cfg)).toBe(LayerPolicy.Skip); // default respected over include-others
    expect(resolvePolicy(Entities.Vendor, cfg)).toBe(LayerPolicy.PassThrough); // whitelist present, not listed
  });

  it('falls back when no include set is present', () => {
    expect(resolvePolicy(Entities.Vendor, mkCfg({ fallback: LayerPolicy.Show }))).toBe(LayerPolicy.Show);
  });
});

describe('clause extractors', () => {
  it('maps Show/Hide/Exclude/Include clauses to policies + includeSet', () => {
    const clauses = [
      layerClause('Show', [Entities.Device]),
      layerClause('Hide', [Entities.WindowsProcessTree]),
      layerClause('Exclude', [NodeType.Tag]),
      layerClause('Include', [Entities.SigmaRule, Entities.NetworkConnection]),
    ];
    const { policies, includeSet } = getEntityLayerConfigFromClauses(clauses);
    expect(policies[Entities.Device]).toBe(LayerPolicy.Show);
    expect(policies[Entities.WindowsProcessTree]).toBe(LayerPolicy.PassThrough);
    expect(includeSet?.has(Entities.SigmaRule)).toBe(true);
    expect(includeSet?.has(Entities.NetworkConnection)).toBe(true);
  });

  it('reads the traversal depth (last valid positive integer)', () => {
    expect(getDepthFromClauses([layerClause('depth', ['3'])], 1)).toBe(3);
    expect(getDepthFromClauses([], 1)).toBe(1);
  });
});

describe('computeFlagStats', () => {
  it('flags each flag, the entity it flags and that entity’s wholes, plus danger-tagged nodes — never the rule', () => {
    // MemoryDump -ProcessTreeIn-> tree -ChildProcess-> proc ; flag -FlagFor-> proc ; rule -CreatedBy-> flag
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'mem.dmp'),
        tree: entityNode('t', 'ProcTree', Entities.WindowsProcessTree),
        proc: entityNode('p', 'p.exe', Entities.WindowsProcess),
        flag: flagNode('fl', 5, 'Likely'),
        rule: entityNode('s', 'Rule', Entities.SigmaRule),
        packed: entityNode('pk', 'Packed', Entities.Device, { PACKED: { true: [] } }),
        clean: entityNode('c', 'Clean', Entities.Device),
      },
      branches: {
        [FILE_HASH]: [
          assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To),
          assocBranch('packed', AssociationKind.AssociatedWith, Direction.To),
          assocBranch('clean', AssociationKind.AssociatedWith, Direction.To),
        ],
        tree: [assocBranch('proc', AssociationKind.ChildProcess, Direction.To)],
        flag: [assocBranch('proc', AssociationKind.FlagFor, Direction.To)],
        rule: [assocBranch('flag', AssociationKind.CreatedBy, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const { flagged } = computeFlagStats(graph, idx);
    expect(flagged.has('flag')).toBe(true); // the flag itself
    expect(flagged.has('proc')).toBe(true); // the entity it flags
    expect(flagged.has('tree')).toBe(true); // a whole above the process
    expect(flagged.has(FILE_HASH)).toBe(true); // the memory dump (whole; also danger-tagged via packed)
    expect(flagged.has('packed')).toBe(true); // danger-tagged
    expect(flagged.has('rule')).toBe(false); // a rule never receives a flag count
    expect(flagged.has('clean')).toBe(false); // neither
  });

  it('counts distinct flags on the containing whole and folds in max suspicion/confidence', () => {
    // MemoryDump -ProcessTreeIn-> tree ; flagA (susp 7, Likely) -FlagFor-> tree ; flagB (susp 3, Fact) -FlagFor-> tree
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'mem.dmp'),
        tree: entityNode('t', 'ProcTree', Entities.WindowsProcessTree),
        flagA: flagNode('a', 7, 'Likely'),
        flagB: flagNode('b', 3, 'Fact'),
      },
      branches: {
        [FILE_HASH]: [assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To)],
        flagA: [assocBranch('tree', AssociationKind.FlagFor, Direction.To)],
        flagB: [assocBranch('tree', AssociationKind.FlagFor, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    // both flags flag the tree → count 2, max suspicion 7, max confidence Fact(3); the memory dump aggregates them too
    expect(stats.get('tree')).toEqual({ flags: 2, suspicion: 7, confidence: 3, dangerTags: 0 });
    expect(stats.get(FILE_HASH)).toEqual({ flags: 2, suspicion: 7, confidence: 3, dangerTags: 0 });
    // a single flag counts itself
    expect(stats.get('flagA')).toEqual({ flags: 1, suspicion: 7, confidence: 2, dangerTags: 0 });
  });

  it('never propagates a flag count to the SigmaRule that created the flags', () => {
    // real backend shape: SigmaRule -CreatedBy-> {flagA, flagB, flagC}; each Flag -FlagFor-> its own process
    const graph = mkGraph({
      dataMap: {
        sig: entityNode('s', 'Rule', Entities.SigmaRule),
        flagA: flagNode('a', 5, 'Likely'),
        flagB: flagNode('b', 8, 'Fact'),
        flagC: flagNode('c', 2, 'Unsure'),
        procA: entityNode('pa', 'a.exe', Entities.WindowsProcess),
        procB: entityNode('pb', 'b.exe', Entities.WindowsProcess),
        procC: entityNode('pc', 'c.exe', Entities.WindowsProcess),
      },
      branches: {
        sig: [
          assocBranch('flagA', AssociationKind.CreatedBy, Direction.To),
          assocBranch('flagB', AssociationKind.CreatedBy, Direction.To),
          assocBranch('flagC', AssociationKind.CreatedBy, Direction.To),
        ],
        flagA: [assocBranch('procA', AssociationKind.FlagFor, Direction.To)],
        flagB: [assocBranch('procB', AssociationKind.FlagFor, Direction.To)],
        flagC: [assocBranch('procC', AssociationKind.FlagFor, Direction.To)],
      },
      initial: ['procA'],
    });
    const idx = buildTreeIndex(graph);
    const { stats, flagged } = computeFlagStats(graph, idx);
    // the rule created 3 flags but receives no flag count — a rule is implicitly interesting, not a subtree total
    expect(stats.get('sig')?.flags ?? 0).toBe(0);
    expect(flagged.has('sig')).toBe(false);
    // each flag counts itself and the process it flags — the rule never bridges back to sibling flags
    expect(stats.get('flagA')?.flags).toBe(1);
    expect(stats.get('flagB')?.flags).toBe(1);
    expect(stats.get('flagC')?.flags).toBe(1);
    expect(stats.get('procA')?.flags).toBe(1);
  });

  it('aggregates flags up the containment spine (process → tree → memory dump); the rule shows none', () => {
    // File(memory dump) -ProcessTreeIn-> tree -ChildProcess-> {procA, procB}; each process carries a Flag
    // raised by the same SigmaRule (Flag -FlagFor-> process, SigmaRule -CreatedBy-> Flag)
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'mem.dmp'),
        tree: entityNode('t', 'ProcTree', Entities.WindowsProcessTree),
        procA: entityNode('pa', 'a.exe', Entities.WindowsProcess),
        procB: entityNode('pb', 'b.exe', Entities.WindowsProcess),
        sig: entityNode('s', 'Rule', Entities.SigmaRule),
        flagA: flagNode('a', 5, 'Likely'),
        flagB: flagNode('b', 9, 'Fact'),
      },
      branches: {
        [FILE_HASH]: [assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To)],
        tree: [
          assocBranch('procA', AssociationKind.ChildProcess, Direction.To),
          assocBranch('procB', AssociationKind.ChildProcess, Direction.To),
        ],
        flagA: [assocBranch('procA', AssociationKind.FlagFor, Direction.To)],
        flagB: [assocBranch('procB', AssociationKind.FlagFor, Direction.To)],
        sig: [assocBranch('flagA', AssociationKind.CreatedBy, Direction.To), assocBranch('flagB', AssociationKind.CreatedBy, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    // the memory dump and process tree see both flags, folding in the highest suspicion/confidence
    expect(stats.get(FILE_HASH)).toEqual({ flags: 2, suspicion: 9, confidence: 3, dangerTags: 0 });
    expect(stats.get('tree')?.flags).toBe(2);
    // each process sees only its own flag; the flags and the shared rule read as a single flag
    expect(stats.get('procA')?.flags).toBe(1);
    expect(stats.get('procB')?.flags).toBe(1);
    expect(stats.get('flagA')?.flags).toBe(1);
    expect(stats.get('sig')?.flags ?? 0).toBe(0);
  });

  it('counts two flags on the same process without either flag leaking into the other', () => {
    // both flags flag the same process — the shared process must not bridge one flag's walk into the other
    const graph = mkGraph({
      dataMap: {
        proc: entityNode('p', 'p.exe', Entities.WindowsProcess),
        flagA: flagNode('a', 1, 'Unsure'),
        flagB: flagNode('b', 1, 'Unsure'),
      },
      branches: {
        flagA: [assocBranch('proc', AssociationKind.FlagFor, Direction.To)],
        flagB: [assocBranch('proc', AssociationKind.FlagFor, Direction.To)],
      },
      initial: ['proc'],
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    expect(stats.get('proc')?.flags).toBe(2);
    expect(stats.get('flagA')?.flags).toBe(1);
    expect(stats.get('flagB')?.flags).toBe(1);
  });

  it('aggregates danger-tag counts up the structural spine', () => {
    // File(memory dump) -ProcessTreeIn-> tree -ChildProcess-> {procA, procB}; procA carries two danger tags
    // (two YARAHIT values) and procB one, so the tree and memory dump show the combined total
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'mem.dmp'),
        tree: entityNode('t', 'ProcTree', Entities.WindowsProcessTree),
        procA: entityNode('pa', 'a.exe', Entities.WindowsProcess, { YARAHIT: { rule1: [], rule2: [] } }),
        procB: entityNode('pb', 'b.exe', Entities.WindowsProcess, { YARAHIT: { rule3: [] } }),
        clean: entityNode('c', 'c.exe', Entities.WindowsProcess),
      },
      branches: {
        [FILE_HASH]: [assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To)],
        tree: [
          assocBranch('procA', AssociationKind.ChildProcess, Direction.To),
          assocBranch('procB', AssociationKind.ChildProcess, Direction.To),
          assocBranch('clean', AssociationKind.ChildProcess, Direction.To),
        ],
      },
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    // the memory dump and process tree see all three danger-tag pairs; each process shows only its own
    expect(stats.get(FILE_HASH)?.dangerTags).toBe(3);
    expect(stats.get('tree')?.dangerTags).toBe(3);
    expect(stats.get('procA')?.dangerTags).toBe(2);
    expect(stats.get('procB')?.dangerTags).toBe(1);
    // an untagged sibling has no danger stat
    expect(stats.get('clean')?.dangerTags ?? 0).toBe(0);
  });

  it('does not conflate danger-tag counts with flag counts on the same node', () => {
    // a process carries a danger tag AND a flag — each count is tracked independently
    const graph = mkGraph({
      dataMap: {
        proc: entityNode('p', 'p.exe', Entities.WindowsProcess, { YARAHIT: { rule1: [] } }),
        flag: flagNode('f', 4, 'Likely'),
      },
      branches: {
        flag: [assocBranch('proc', AssociationKind.FlagFor, Direction.To)],
      },
      initial: ['proc'],
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    expect(stats.get('proc')).toEqual({ flags: 1, suspicion: 4, confidence: 2, dangerTags: 1 });
  });

  it("propagates a rule's/flag's own danger tags forward (rule → flag → entity → up), but an entity's tags never reach the flag", () => {
    // rule(YARAHIT) -CreatedBy-> flag -FlagFor-> proc(YARAHIT) ; MemoryDump -ProcessTreeIn-> tree -ChildProcess-> proc
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'mem.dmp'),
        tree: entityNode('t', 'ProcTree', Entities.WindowsProcessTree),
        proc: entityNode('p', 'p.exe', Entities.WindowsProcess, { YARAHIT: { procRule: [] } }),
        flag: flagNode('fl', 5, 'Likely'),
        rule: entityNode('s', 'Rule', Entities.SigmaRule, { YARAHIT: { ruleTag: [] } }),
      },
      branches: {
        [FILE_HASH]: [assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To)],
        tree: [assocBranch('proc', AssociationKind.ChildProcess, Direction.To)],
        flag: [assocBranch('proc', AssociationKind.FlagFor, Direction.To)],
        rule: [assocBranch('flag', AssociationKind.CreatedBy, Direction.To)],
      },
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    // the rule's own danger tag flows forward: rule → flag → proc → tree → memory dump
    expect(stats.get('rule')?.dangerTags).toBe(1);
    // the flag receives the rule's tag (rule → flag) but NOT the process's own tag (entity → flag is never followed)
    expect(stats.get('flag')?.dangerTags).toBe(1);
    // the process shows its own tag (1) plus the rule's tag that flowed through the flag (1)
    expect(stats.get('proc')?.dangerTags).toBe(2);
    expect(stats.get('tree')?.dangerTags).toBe(2);
    expect(stats.get(FILE_HASH)?.dangerTags).toBe(2);
  });

  it('trusts the stored association direction — a reversed edge aggregates the wrong way (a data-source bug)', () => {
    // a tool authored the PART as the source: CompiledFunction -AssociatedWith(To)-> DLL(danger). The DLL is
    // therefore a child of the function in the index, so its tag climbs into the function. Documents that the
    // UI trusts direction and does not compensate for reversed associations (the fix belongs in the tool).
    const dll = {
      [TreeNodeKey.Sample]: { sha256: 'd'.repeat(64), submissions: [{ name: 'lib.dll' }], tags: { ClamAV: { 'Win.Trojan': [] } } },
    } as unknown as TreeNode;
    const graph = mkGraph({
      dataMap: { fn: entityNode('fn', 'Sleep', Entities.CompiledFunction), dll },
      branches: { fn: [assocBranch('dll', AssociationKind.AssociatedWith, Direction.To)] },
      initial: ['fn'],
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    expect(stats.get('dll')?.dangerTags).toBe(1);
    expect(stats.get('fn')?.dangerTags).toBe(1); // the (expected) leak from the reversed direction
  });

  it('with the correct direction, a whole’s tags stay off its parts', () => {
    // DLL(danger, the whole) -AssociatedWith(To)-> CompiledFunction (the part): the whole is the source/parent,
    // so its tag does NOT reach the part
    const dll = {
      [TreeNodeKey.Sample]: { sha256: 'e'.repeat(64), submissions: [{ name: 'lib.dll' }], tags: { ClamAV: { 'Win.Trojan': [] } } },
    } as unknown as TreeNode;
    const graph = mkGraph({
      dataMap: { fn: entityNode('fn', 'Sleep', Entities.CompiledFunction), dll },
      branches: { dll: [assocBranch('fn', AssociationKind.AssociatedWith, Direction.To)] },
      initial: ['dll'],
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    expect(stats.get('dll')?.dangerTags).toBe(1);
    expect(stats.get('fn')?.dangerTags ?? 0).toBe(0);
  });

  it('handles duplicate re-uploaded process edges without distorting counts', () => {
    // memory dump (danger-tagged) -ProcessTreeIn-> tree -ChildProcess-> {procA, procB}; each process carries a
    // flag. The tree↔dump edge and the tree→procA edge are DUPLICATED (a process re-uploaded into the same tree).
    // Duplicates share a relationship_hash, so they must dedupe and leave counts unchanged.
    const dump = {
      [TreeNodeKey.Sample]: {
        sha256: 'f'.repeat(64),
        submissions: [{ name: 'mem.dmp' }],
        tags: { YaraRuleHits: { r1: [], r2: [], r3: [] } },
      },
    } as unknown as TreeNode;
    const graph = mkGraph({
      dataMap: {
        dump,
        tree: entityNode('t', 'ProcTree', Entities.WindowsProcessTree),
        procA: entityNode('pa', 'a.exe', Entities.WindowsProcess),
        procB: entityNode('pb', 'b.exe', Entities.WindowsProcess),
        flagA: flagNode('fa', 5, 'Likely'),
        flagB: flagNode('fb', 9, 'Fact'),
      },
      branches: {
        dump: [assocBranch('tree', AssociationKind.ProcessTreeIn, Direction.To, 'H1')],
        tree: [
          // the dump↔tree reverse-pair, stored repeatedly as if the tree were re-uploaded several times
          assocBranch('dump', AssociationKind.ProcessTreeIn, Direction.From, 'H1'),
          assocBranch('dump', AssociationKind.ProcessTreeIn, Direction.From, 'H1'),
          assocBranch('procA', AssociationKind.ChildProcess, Direction.To, 'H2'),
          assocBranch('procA', AssociationKind.ChildProcess, Direction.To, 'H2'), // duplicate child edge
          assocBranch('procB', AssociationKind.ChildProcess, Direction.To, 'H3'),
        ],
        flagA: [assocBranch('procA', AssociationKind.FlagFor, Direction.To)],
        flagB: [assocBranch('procB', AssociationKind.FlagFor, Direction.To)],
      },
      initial: ['dump'],
    });
    const idx = buildTreeIndex(graph);
    const { stats } = computeFlagStats(graph, idx);
    // flags flow part → whole; duplicates dedupe, so the tree and dump each count both flags exactly once
    expect(stats.get('tree')?.flags).toBe(2);
    expect(stats.get('dump')?.flags).toBe(2);
    expect(stats.get('procA')?.flags).toBe(1);
    expect(stats.get('procB')?.flags).toBe(1);
    // the memory dump's own danger tags stay on the dump — a whole's tags never propagate down to its parts
    expect(stats.get('dump')?.dangerTags).toBe(3);
    expect(stats.get('tree')?.dangerTags ?? 0).toBe(0);
    expect(stats.get('procA')?.dangerTags ?? 0).toBe(0);
  });
});

describe('compareByFlagStats', () => {
  const stats = new Map([
    ['x', { flags: 2, suspicion: 1, confidence: 3, dangerTags: 0 }],
    ['y', { flags: 2, suspicion: 5, confidence: 0, dangerTags: 0 }],
    ['z', { flags: 0, suspicion: 9, confidence: 3, dangerTags: 0 }],
  ]);
  it('sorts by the primary mode descending, then tiebreaks in priority order', () => {
    // primary Flags: x and y tie at 2 → tiebreak Suspicion (y 5 > x 1) → y first; z (0 flags) last
    expect([...['x', 'y', 'z']].sort((a, b) => compareByFlagStats(a, b, stats, SortMode.Flags))).toEqual(['y', 'x', 'z']);
    // primary Suspicion: z(9) > y(5) > x(1)
    expect([...['x', 'y', 'z']].sort((a, b) => compareByFlagStats(a, b, stats, SortMode.Suspicion))).toEqual(['z', 'y', 'x']);
  });
  it('treats a node with no stats as all-zero', () => {
    expect(compareByFlagStats('x', 'missing', stats, SortMode.Flags)).toBeLessThan(0);
  });
});

describe('nodeGroups / collectGroupOptions', () => {
  it('reads entity groups and collects the graph-wide set', () => {
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        d: {
          [TreeNodeKey.Entity]: { id: 'd', name: 'Dev', kind: Entities.Device, tags: {}, groups: ['alpha', 'beta'] },
        } as unknown as TreeNode,
      },
      branches: { [FILE_HASH]: [assocBranch('d', AssociationKind.FirmwareFor, Direction.To)] },
    });
    expect(nodeGroups(graph.data_map.d)).toEqual(['alpha', 'beta']);
    expect(collectGroupOptions(graph)).toEqual(['alpha', 'beta']);
  });
});

describe('collectTagOptions', () => {
  it('dedups values per key across every node in the graph', () => {
    const graph = mkGraph({
      dataMap: {
        a: entityNode('a', 'A', Entities.Device, { family: { emotet: {}, trickbot: {} }, tlp: { amber: {} } }),
        b: entityNode('b', 'B', Entities.Device, { family: { emotet: {}, qakbot: {} }, arch: { x86: {} } }),
      },
      branches: {},
      initial: ['a'],
    });
    const options = collectTagOptions(graph);
    // emotet is shared by a and b but appears once; both keys' full value sets are unioned
    expect(new Set(options.family)).toEqual(new Set(['emotet', 'trickbot', 'qakbot']));
    expect(options.tlp).toEqual(['amber']);
    expect(options.arch).toEqual(['x86']);
  });

  it('normalizes flat Tag-node tags into the same key→values shape', () => {
    const graph = mkGraph({
      dataMap: { t: tagNode({ family: ['emotet', 'x'] }) },
      branches: {},
      initial: ['t'],
    });
    expect(new Set(collectTagOptions(graph).family)).toEqual(new Set(['emotet', 'x']));
  });

  it('returns an empty object for a graph with no tagged nodes', () => {
    const graph = mkGraph({ dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin') }, branches: {} });
    expect(collectTagOptions(graph)).toEqual({});
  });
});

describe('groupByKind', () => {
  it('groups children by node type in first-appearance order', () => {
    const graph = mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin'),
        d1: entityNode('d1', 'Dev1', Entities.Device),
        s1: entityNode('s1', 'Sig1', Entities.SigmaRule),
        d2: entityNode('d2', 'Dev2', Entities.Device),
      },
      branches: {
        [FILE_HASH]: [
          assocBranch('d1', AssociationKind.FirmwareFor, Direction.To),
          assocBranch('s1', AssociationKind.SigmaRuleHit, Direction.To),
          assocBranch('d2', AssociationKind.FirmwareFor, Direction.To),
        ],
      },
    });
    const idx = buildTreeIndex(graph);
    const kids = effectiveChildren(FILE_HASH, idx, graph, mkCfg(), new Set([FILE_HASH]));
    const groups = groupByKind(kids, graph);
    expect(groups.map((g) => g.nodeType)).toEqual([Entities.Device, Entities.SigmaRule]);
    expect(groups[0].children.map((c) => c.edge.id)).toEqual(['d1', 'd2']);
  });
});

describe('resolveRoots', () => {
  it('resolves a sha256 to the file node', () => {
    const graph = mkGraph({ dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin') }, branches: {} });
    expect(resolveRoots(graph, { kind: 'sha256', sha256: FILE_SHA }).map((r) => r.id)).toEqual([FILE_HASH]);
  });
  it('passes explicit node roots through', () => {
    const graph = mkGraph({ dataMap: {}, branches: {} });
    expect(resolveRoots(graph, { kind: 'nodes', roots: [{ id: 'x', label: 'X' }] })).toEqual([{ id: 'x', label: 'X' }]);
  });
  it('ascends initial seeds to their tree root', () => {
    // root -> mid -> leaf; seed is the leaf
    const graph = mkGraph({
      dataMap: {
        root: entityNode('idr', 'Root', Entities.Device),
        mid: entityNode('idm', 'Mid', Entities.Device),
        leaf: entityNode('idl', 'Leaf', Entities.Device),
      },
      branches: {
        root: [assocBranch('mid', AssociationKind.AssociatedWith, Direction.To)],
        mid: [assocBranch('leaf', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['leaf'],
    });
    expect(resolveRoots(graph, { kind: 'initial' }).map((r) => r.id)).toEqual(['root']);
  });
});

describe('filterTree', () => {
  const buildFilterGraph = () =>
    mkGraph({
      dataMap: {
        [FILE_HASH]: sampleNode(FILE_SHA, 'root.bin'),
        keep: entityNode('k', 'NeedleEntity', Entities.Device),
        drop: entityNode('d', 'Unrelated', Entities.Device),
        danger: entityNode('g', 'Packed', Entities.Device, { PACKED: { true: [] } }),
      },
      branches: {
        [FILE_HASH]: [
          assocBranch('keep', AssociationKind.FirmwareFor, Direction.To),
          assocBranch('drop', AssociationKind.FirmwareFor, Direction.To),
          assocBranch('danger', AssociationKind.FirmwareFor, Direction.To),
        ],
      },
    });

  it('keeps text matches and their ancestors, drops the rest', () => {
    const graph = buildFilterGraph();
    const idx = buildTreeIndex(graph);
    const visible = filterTree([FILE_HASH], idx, graph, mkCriteria({ text: 'needle' }), mkCfg());
    expect(visible.has('keep')).toBe(true);
    expect(visible.has(FILE_HASH)).toBe(true); // ancestor kept
    expect(visible.has('drop')).toBe(false);
  });

  it('filters to flagged nodes and keeps their direct associations visible (Flagged Only)', () => {
    const graph = buildFilterGraph();
    const idx = buildTreeIndex(graph);
    const { flagged: flaggedNodes } = computeFlagStats(graph, idx);
    const visible = filterTree([FILE_HASH], idx, graph, mkCriteria({ flaggedOnly: true, flaggedNodes }), mkCfg());
    expect(visible.has('danger')).toBe(true); // danger-tagged
    expect(visible.has(FILE_HASH)).toBe(true); // flagged (aggregates the danger tag from `danger`)
    // keep/drop are unflagged, but they are direct associations of the flagged file, so decision 3 keeps them
    expect(visible.has('keep')).toBe(true);
    expect(visible.has('drop')).toBe(true);
  });

  it('excludes a hidden matching node (and its subtree) from the visible set', () => {
    // `keep` matches "needle" but is hidden, so filterTree — which walks via effectiveChildren — never
    // reaches it and it is not kept.
    const graph = buildFilterGraph();
    const idx = buildTreeIndex(graph);
    const cfg = mkCfg({ hiddenNodes: new Set(['keep']) });
    const visible = filterTree([FILE_HASH], idx, graph, mkCriteria({ text: 'needle' }), cfg);
    expect(visible.has('keep')).toBe(false);
  });
});

describe('resolveRoots — hidden roots (dropped by the body consumer)', () => {
  it('still resolves a root that is hidden (the EntityBrowserBody filters hidden roots out at render)', () => {
    // resolveRoots itself is hidden-agnostic — hidden roots are removed where roots are consumed, not here.
    const graph = mkGraph({ dataMap: { [FILE_HASH]: sampleNode(FILE_SHA, 'f.bin') }, branches: {} });
    const roots = resolveRoots(graph, { kind: 'sha256', sha256: FILE_SHA });
    expect(roots.map((r) => r.id)).toEqual([FILE_HASH]);
    // the consumer drops hidden roots with a plain filter
    const hiddenNodes = new Set([FILE_HASH]);
    expect(roots.filter((r) => !hiddenNodes.has(r.id))).toEqual([]);
  });
});

describe('getDisplayTags', () => {
  it('flattens key/value pairs and drops suppressed keys (incl. any *Sha256)', () => {
    const node = entityNode('e1', 'thing', Entities.Flag, {
      FileType: { PE32: [] },
      submitter: { alice: [] },
      Parent: { 'abc…': [] },
      FolderAllSha256: { deadbeef: [] },
      FsSha256: { cafef00d: [] },
    });
    const { shown, overflow } = getDisplayTags(node);
    // only FileType survives; submitter/Parent and any *Sha256 hash keys are suppressed
    expect(shown).toEqual([{ key: 'FileType', label: 'FileType', value: 'PE32' }]);
    expect(overflow).toBe(0);
  });

  it('orders non-priority pairs by significance (danger first) then key/value', () => {
    const node = entityNode('e1', 'thing', Entities.Flag, {
      Zeta: { b: [], a: [] },
      YARAHIT: { sig: [] },
      Alpha: { two: [] },
    });
    const { shown } = getDisplayTags(node);
    // YARAHIT is a danger key (rank 0) → first; the rest are general and fall back to key/value order
    expect(shown).toEqual([
      { key: 'YARAHIT', label: 'YARAHIT', value: 'sig' },
      { key: 'Alpha', label: 'Alpha', value: 'two' },
      { key: 'Zeta', label: 'Zeta', value: 'a' },
      { key: 'Zeta', label: 'Zeta', value: 'b' },
    ]);
  });

  it('curates WindowsProcess header tags: hides internals, orders, and strips the Process prefix', () => {
    const node = entityNode('p1', 'svchost.exe', Entities.WindowsProcess, {
      ProcessImagePath: { 'C:/Windows/System32/svchost.exe': [] },
      ProcessIsWow64: { true: [] },
      ProcessCommand: { 'svchost.exe -k netsvcs': [] },
      PID: { '1234': [] },
      ProcessName: { 'svchost.exe': [] },
      ProcessThreads: { '12': [] },
    });
    const { shown } = getDisplayTags(node);
    // IsWow64/Threads hidden; ordered PID → Name → Command → ImagePath; keys de-prefixed
    expect(shown).toEqual([
      { key: 'PID', label: 'PID', value: '1234' },
      { key: 'ProcessName', label: 'Name', value: 'svchost.exe' },
      { key: 'ProcessCommand', label: 'Command', value: 'svchost.exe -k netsvcs' },
      { key: 'ProcessImagePath', label: 'ImagePath', value: 'C:/Windows/System32/svchost.exe' },
    ]);
  });

  it('curates flag-scan header tags: strips the Flag prefix and hides free-text content/reasoning', () => {
    const node = entityNode('e1', 'thing', Entities.Flag, {
      FlagConfidence: { Likely: [] },
      FlagSuspicion: { '5': [] },
      FlagContent: { 'a long free-text blob that would wrap and blow out the header row': [] },
      FlagReasoning: { 'because reasons that are also very long and free-form': [] },
    });
    const { shown } = getDisplayTags(node);
    // Content/Reasoning are hidden (they belong in the details); Confidence/Suspicion keep their raw key but
    // render with the Flag prefix stripped
    expect(shown).toEqual([
      { key: 'FlagConfidence', label: 'Confidence', value: 'Likely' },
      { key: 'FlagSuspicion', label: 'Suspicion', value: '5' },
    ]);
  });

  it('caps at the limit and reports the overflow count + labels', () => {
    const node = entityNode('e1', 'thing', Entities.Flag, {
      K: { v1: [], v2: [], v3: [], v4: [] },
    });
    const { shown, overflow, overflowLabels } = getDisplayTags(node, 2);
    expect(shown).toEqual([
      { key: 'K', label: 'K', value: 'v1' },
      { key: 'K', label: 'K', value: 'v2' },
    ]);
    expect(overflow).toBe(2);
    expect(overflowLabels).toEqual(['K: v3', 'K: v4']);
  });

  it('returns an empty set for a node with no (non-suppressed) tags', () => {
    const node = entityNode('e1', 'thing', Entities.Flag, { Results: { r: [] } });
    expect(getDisplayTags(node)).toEqual({ shown: [], overflow: 0, overflowLabels: [] });
  });
});

describe('focusBreadcrumb', () => {
  // root --(To)--> mid --(To)--> leaf : parentsOf(leaf)=[mid], parentsOf(mid)=[root]
  function chainGraph(): Graph {
    return mkGraph({
      dataMap: {
        root: entityNode('idr', 'Root', Entities.Device),
        mid: entityNode('idm', 'Mid', Entities.Folder),
        leaf: entityNode('idl', 'Leaf', Entities.Flag),
      },
      branches: {
        root: [assocBranch('mid', AssociationKind.ChildProcess, Direction.To)],
        mid: [assocBranch('leaf', AssociationKind.ChildProcess, Direction.To)],
      },
      initial: ['root'],
    });
  }

  it('returns the ancestor chain top→down including the focus root, with labels', () => {
    const graph = chainGraph();
    const idx = buildTreeIndex(graph);
    const crumbs = focusBreadcrumb(graph, idx, 'leaf');
    expect(crumbs.map((c) => c.id)).toEqual(['root', 'mid', 'leaf']);
    expect(crumbs.map((c) => c.label)).toEqual(['Root', 'Mid', 'Leaf']);
  });

  it('returns just the focus root when it has no parent', () => {
    const graph = chainGraph();
    const idx = buildTreeIndex(graph);
    expect(focusBreadcrumb(graph, idx, 'root').map((c) => c.id)).toEqual(['root']);
  });
});
