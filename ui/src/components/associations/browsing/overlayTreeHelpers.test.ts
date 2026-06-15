import { describe, it, expect } from 'vitest';

// project imports
import { buildTreeIndex } from './treeHelpers';
import {
  ancestorChain,
  buildTreeRoots,
  isCompositeItemId,
  itemNodeId,
  makeChildItemId,
  overlayChildItemIds,
  overlayIsFolder,
  overlayItemPathForNode,
  parseItemId,
} from './overlayTreeHelpers';
import { AssociationKind } from '@models/associations';
import { Entities } from '@models/entities';
import { type BranchNode, Direction, type Graph, type TreeNode, TreeNodeKey } from '@models/trees';

// --- fixture builders (mirrors treeHelpers.test.ts) ---

function entityNode(id: string, name: string, kind: Entities): TreeNode {
  return { [TreeNodeKey.Entity]: { id, name, kind, tags: {}, description: null } } as unknown as TreeNode;
}

function assocBranch(node: string, kind: AssociationKind, direction: Direction, hash = `${node}-${kind}`): BranchNode {
  return { relationship: { Association: { kind, direction } }, node, direction, relationship_hash: hash } as BranchNode;
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

/** SigmaRule --SigmaRuleHit--> Flag --AssociatedWith--> WindowsProcess (all `To` from source), seeded at proc. */
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

describe('item-id codec', () => {
  it('round-trips a raw node id as its own anchor', () => {
    const ref = parseItemId('node1');
    expect(ref.nodeId).toBe('node1');
    expect(ref.path).toEqual(['node1']);
    expect(ref.viaReversed).toBe(false);
    expect(ref.reverseDepth).toBe(0);
    expect(isCompositeItemId('node1')).toBe(false);
  });
  it('embeds the anchor and every hop node in the path, with direction/depth', () => {
    // proc -(reverse)-> flag -(reverse)-> sig
    const a = makeChildItemId('proc', 'flag', true);
    const b = makeChildItemId(a, 'sig', true);
    expect(isCompositeItemId(b)).toBe(true);
    const ref = parseItemId(b);
    expect(ref.nodeId).toBe('sig');
    expect(ref.path).toEqual(['proc', 'flag', 'sig']);
    expect(ref.viaReversed).toBe(true);
    expect(ref.reverseDepth).toBe(2);
    expect(itemNodeId(b)).toBe('sig');
  });
  it('tracks a forward last-hop as not-reversed and counts only reverse hops', () => {
    const fwd = makeChildItemId('file', 'cve', false);
    const ref = parseItemId(fwd);
    expect(ref.nodeId).toBe('cve');
    expect(ref.viaReversed).toBe(false);
    expect(ref.reverseDepth).toBe(0);
    expect(itemNodeId(fwd)).toBe('cve');
  });
});

describe('overlayChildItemIds', () => {
  it('keeps a forward structural child on the backbone as a raw id', () => {
    // folder --FileIn--> file (structural containment)
    const graph = mkGraph({
      dataMap: { folder: entityNode('fold', 'dir', Entities.Folder), file: entityNode('fl', 'f.bin', Entities.File) },
      branches: { folder: [assocBranch('file', AssociationKind.FileIn, Direction.To)] },
      initial: ['folder'],
    });
    const idx = buildTreeIndex(graph);
    expect(overlayChildItemIds(idx, 'folder')).toEqual(['file']);
  });

  it('nests an incoming Flag under a process as a composite reverse child', () => {
    const idx = buildTreeIndex(sigmaFlagProcessGraph());
    const kids = overlayChildItemIds(idx, 'proc');
    expect(kids).toHaveLength(1);
    expect(isCompositeItemId(kids[0])).toBe(true);
    const ref = parseItemId(kids[0]);
    expect(ref.nodeId).toBe('flag');
    expect(ref.viaReversed).toBe(true);
  });

  it('follows the full reverse chain proc -> flag -> sig unbounded, forward-suppressed', () => {
    const idx = buildTreeIndex(sigmaFlagProcessGraph());
    const flagItem = overlayChildItemIds(idx, 'proc')[0];
    const sigChildren = overlayChildItemIds(idx, flagItem);
    // the flag (reached via reverse) surfaces ONLY the sig (reverse); its forward child proc is suppressed
    expect(sigChildren).toHaveLength(1);
    const sigRef = parseItemId(sigChildren[0]);
    expect(sigRef.nodeId).toBe('sig');
    expect(sigRef.path).toEqual(['proc', 'flag', 'sig']);
    // sig is a leaf here (no further reverse parents)
    expect(overlayChildItemIds(idx, sigChildren[0])).toEqual([]);
  });

  it('does NOT echo the parent under a non-structural forward child (file -> CVE)', () => {
    // file --ContainsCVE--> cve (non-structural forward)
    const graph = mkGraph({
      dataMap: { file: entityNode('fl', 'f.bin', Entities.File), cve: entityNode('c', 'CVE-1', Entities.Other) },
      branches: { file: [assocBranch('cve', AssociationKind.ContainsCVE, Direction.To)] },
      initial: ['file'],
    });
    const idx = buildTreeIndex(graph);
    const cveItem = overlayChildItemIds(idx, 'file')[0];
    expect(itemNodeId(cveItem)).toBe('cve');
    // the CVE's reverse edge back to the file is on its path -> guarded -> no echo
    expect(overlayChildItemIds(idx, cveItem)).toEqual([]);
    expect(overlayIsFolder(idx, cveItem)).toBe(false);
  });

  it('gives distinct composite ids to the same node under different anchors', () => {
    // both file1 and file2 contain the same CVE
    const graph = mkGraph({
      dataMap: {
        file1: entityNode('f1', 'a.bin', Entities.File),
        file2: entityNode('f2', 'b.bin', Entities.File),
        cve: entityNode('c', 'CVE-1', Entities.Other),
      },
      branches: {
        file1: [assocBranch('cve', AssociationKind.ContainsCVE, Direction.To)],
        file2: [assocBranch('cve', AssociationKind.ContainsCVE, Direction.To)],
      },
      initial: ['file1', 'file2'],
    });
    const idx = buildTreeIndex(graph);
    const a = overlayChildItemIds(idx, 'file1')[0];
    const b = overlayChildItemIds(idx, 'file2')[0];
    expect(a).not.toBe(b);
    expect(itemNodeId(a)).toBe('cve');
    expect(itemNodeId(b)).toBe('cve');
  });

  it('terminates a mutual non-structural pair via the per-path guard', () => {
    // a <-> b via AssociatedWith both directions
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), b: entityNode('b', 'B', Entities.Device) },
      branches: {
        a: [assocBranch('b', AssociationKind.AssociatedWith, Direction.To)],
        b: [assocBranch('a', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    const bItem = overlayChildItemIds(idx, 'a')[0];
    expect(itemNodeId(bItem)).toBe('b');
    // b's only edge is back to a, which is on the path -> guarded -> terminates
    expect(overlayChildItemIds(idx, bItem)).toEqual([]);
  });

  it('leaves a raw backbone mutual-structural pair to the tree flatten guard (each returns the other)', () => {
    // a <-> b via a Bidirectional FirmwareFor (structural): raw ids only self-guard, so the loop exists at
    // the helper level and is bounded by headless-tree's flatten guard (see the integration test)
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), b: entityNode('b', 'B', Entities.Device) },
      branches: {
        a: [assocBranch('b', AssociationKind.FirmwareFor, Direction.Bidirectional, 'H')],
        b: [assocBranch('a', AssociationKind.FirmwareFor, Direction.Bidirectional, 'H')],
      },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    expect(overlayChildItemIds(idx, 'a')).toEqual(['b']);
    expect(overlayChildItemIds(idx, 'b')).toEqual(['a']);
  });
});

describe('ancestorChain / buildTreeRoots (structural-only)', () => {
  it('hoists only the structural parent, ignoring a relationship parent', () => {
    // folder --FileIn--> file (structural); flag --AssociatedWith--> file (non-structural)
    const graph = mkGraph({
      dataMap: {
        folder: entityNode('fold', 'dir', Entities.Folder),
        flag: entityNode('fl', 'Flag', Entities.Flag),
        file: entityNode('f', 'f.bin', Entities.File),
      },
      branches: {
        folder: [assocBranch('file', AssociationKind.FileIn, Direction.To)],
        flag: [assocBranch('file', AssociationKind.AssociatedWith, Direction.To)],
      },
      initial: ['file'],
    });
    const idx = buildTreeIndex(graph);
    expect(ancestorChain(idx, 'file')).toEqual(['folder']);
    expect(buildTreeRoots(graph, idx)).toEqual(['folder']);
  });

  it('roots a node at itself when it has only a relationship parent', () => {
    const idx = buildTreeIndex(sigmaFlagProcessGraph());
    // proc's only parent (flag) is non-structural -> proc is its own root
    expect(ancestorChain(idx, 'proc')).toEqual([]);
    expect(buildTreeRoots(sigmaFlagProcessGraph(), idx)).toEqual(['proc']);
  });
});

describe('overlayItemPathForNode', () => {
  it('locates a structural node topping at a root as its raw id + structural chain', () => {
    const graph = mkGraph({
      dataMap: { folder: entityNode('fold', 'dir', Entities.Folder), file: entityNode('fl', 'f.bin', Entities.File) },
      branches: { folder: [assocBranch('file', AssociationKind.FileIn, Direction.To)] },
      initial: ['folder'],
    });
    const idx = buildTreeIndex(graph);
    const roots = buildTreeRoots(graph, idx);
    const located = overlayItemPathForNode(idx, roots, 'file');
    expect(located).toEqual({ itemId: 'file', expandIds: ['folder'] });
  });

  it('locates a reverse-only SigmaRule as a composite anchored at the process', () => {
    const graph = sigmaFlagProcessGraph();
    const idx = buildTreeIndex(graph);
    const roots = buildTreeRoots(graph, idx);
    const located = overlayItemPathForNode(idx, roots, 'sig');
    expect(located).not.toBeNull();
    expect(itemNodeId(located!.itemId)).toBe('sig');
    expect(parseItemId(located!.itemId).path).toEqual(['proc', 'flag', 'sig']);
    // expand chain is topmost-first: the process root, then the flag occurrence
    expect(located!.expandIds[0]).toBe('proc');
    expect(itemNodeId(located!.expandIds[1])).toBe('flag');
  });

  it('returns null for an unreachable node', () => {
    const graph = sigmaFlagProcessGraph();
    const idx = buildTreeIndex(graph);
    const roots = buildTreeRoots(graph, idx);
    expect(overlayItemPathForNode(idx, roots, 'nope')).toBeNull();
  });
});
