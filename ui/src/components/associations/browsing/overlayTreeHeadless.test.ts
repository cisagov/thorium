import { describe, it, expect, vi } from 'vitest';
import { createTree, syncDataLoaderFeature } from '@headless-tree/core';

// project imports
import { buildTreeIndex, TreeIndex } from './treeHelpers';
import { buildTreeRoots, itemNodeId, makeChildItemId, overlayChildItemIds, overlayIsFolder } from './overlayTreeHelpers';
import { AssociationKind } from '@models/associations';
import { Entities } from '@models/entities';
import { type BranchNode, Direction, type Graph, type TreeNode, TreeNodeKey } from '@models/trees';

// Integration coverage: drive @headless-tree's real flattener with the overlay's getChildren/isItemFolder to
// confirm composite ids flatten and structural cycles are bounded (not the pure-helper unit tests).

function entityNode(id: string, name: string, kind: Entities): TreeNode {
  return { [TreeNodeKey.Entity]: { id, name, kind, tags: {}, description: null } } as unknown as TreeNode;
}

function assocBranch(node: string, kind: AssociationKind, direction: Direction, hash = `${node}-${kind}`): BranchNode {
  return { relationship: { Association: { kind, direction } }, node, direction, relationship_hash: hash } as BranchNode;
}

function mkGraph(opts: { dataMap: Record<string, TreeNode>; branches: Record<string, BranchNode[]>; initial: string[] }): Graph {
  return { id: 'tree-1', initial: opts.initial, growable: [], data_map: opts.dataMap, branches: opts.branches };
}

/** Build a mounted headless tree over the overlay resolver for `graph`, expanding `expandedItems`. */
function mountOverlayTree(graph: Graph, idx: TreeIndex, expandedItems: string[]) {
  const tree = createTree<string>({
    rootItemId: 'root',
    initialState: { expandedItems },
    dataLoader: {
      getItem: (id) => id,
      getChildren: (id) => (id === 'root' ? buildTreeRoots(graph, idx) : overlayChildItemIds(idx, id)),
    },
    getItemName: (item) => itemNodeId(item.getId()),
    isItemFolder: (item) => overlayIsFolder(idx, item.getId()),
    indent: 20,
    features: [syncDataLoaderFeature],
  });
  tree.setMounted(true);
  tree.rebuildTree();
  return tree;
}

describe('overlay tree with @headless-tree flattener', () => {
  it('flattens a reverse chain of composite ids (proc -> flag -> sig)', () => {
    const graph = mkGraph({
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
    const idx = buildTreeIndex(graph);
    const flagItem = makeChildItemId('proc', 'flag', true);
    const sigItem = makeChildItemId(flagItem, 'sig', true);
    const tree = mountOverlayTree(graph, idx, ['proc', flagItem, sigItem]);
    const rendered = tree.getItems().map((i) => itemNodeId(i.getId()));
    expect(rendered).toEqual(['proc', 'flag', 'sig']);
  });

  it('bounds a structural cycle via the flattener guard instead of looping forever', () => {
    // a <-> b via a Bidirectional structural edge; expanding both would recurse without the flatten guard
    const graph = mkGraph({
      dataMap: { a: entityNode('a', 'A', Entities.Device), b: entityNode('b', 'B', Entities.Device) },
      branches: {
        a: [assocBranch('b', AssociationKind.FirmwareFor, Direction.Bidirectional, 'H')],
        b: [assocBranch('a', AssociationKind.FirmwareFor, Direction.Bidirectional, 'H')],
      },
      initial: ['a'],
    });
    const idx = buildTreeIndex(graph);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // root ascends a->b (structural), so 'b' is the root; expand both sides of the mutual pair
    const tree = mountOverlayTree(graph, idx, ['a', 'b']);
    const rendered = tree.getItems().map((i) => i.getId());
    // terminates (finite) and the flatten guard dropped the repeat rather than hanging
    expect(rendered.length).toBeLessThan(6);
    expect(rendered).toContain('b');
    warn.mockRestore();
  });
});
