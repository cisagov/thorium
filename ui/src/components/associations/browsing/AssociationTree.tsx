import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { asyncDataLoaderFeature, hotkeysCoreFeature, selectionFeature } from '@headless-tree/core';
import { useTree } from '@headless-tree/react';
import { ErrorBoundary } from 'react-error-boundary';
// project imports
import RenderErrorAlert from '../../shared/alerts/RenderErrorAlert';
import { getNodeName } from '../utilities';
import { classifyNode } from '../graph/data';
import { getNodeSvg } from '../graph/styles';
import { useGraphData, FocusSource } from '../data/GraphDataContext';
import { useSharedTreeIndex } from '../data/SharedTreeIndex';
import { TreeContainer } from './TreeContainer';
import { findMultiParentNodeIds } from './treeHelpers';
import {
  ancestorChain,
  buildTreeRoots,
  itemNodeId,
  overlayChildItemIds,
  overlayIsFolder,
  overlayItemPathForNode,
} from './overlayTreeHelpers';
import StyledSpinner from '@components/shared/fallback/Spinner.styled';
import EntitySummaryHover from '@components/shared/info/EntitySummaryHover';
import { treeNodeToInfo } from '@components/shared/info/info';
import { entityLabel } from '@models/entities';
import { TreeNode } from '@models/trees';

// spec: ./AssociationTree.spec.md

interface TreeItemOverlayProps {
  nodeData: TreeNode | undefined;
  isDuplicate: boolean;
  children: React.ReactElement;
}

/**
 * Show the shared {@link EntitySummaryHover} preview for a tree row. Falls back to the bare row when the
 * node has no describable info (matching the graph's behavior of suppressing the hover for such nodes).
 */
const TreeItemOverlay: React.FC<TreeItemOverlayProps> = ({ nodeData, isDuplicate, children }) => {
  const model = nodeData ? treeNodeToInfo(nodeData) : null;
  if (!model) return children;
  return (
    <EntitySummaryHover model={model} duplicate={isDuplicate} placement="right">
      {children}
    </EntitySummaryHover>
  );
};

const AssociationTreeComponent: React.FC = () => {
  const { graph, graphVersion, grow, growable, getGraph, focusedNodeId, focusSource, setFocusedNode } = useGraphData();
  // the tree index is derived ONCE in the shared layer (shared with the entity browser); `getIndex()` is the
  // always-fresh index for async callbacks (grow mutates the graph before the version bump commits a render)
  const { index: treeIndex, getIndex } = useSharedTreeIndex();
  const [loadingItemData, setLoadingItemData] = useState<string[]>([]);
  const [loadingItemChildrens, setLoadingItemChildrens] = useState<string[]>([]);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);

  const [expandedItems, setExpandedItems] = useState<string[]>(() => {
    const items: string[] = [];
    for (const initialId of graph.initial) {
      const s = initialId.toString();
      items.push(...ancestorChain(treeIndex, s), s);
    }
    return [...new Set(items)];
  });

  const grownNodesRef = useRef(new Set<string>());
  const [manuallyGrowing, setManuallyGrowing] = useState<Set<string>>(new Set());
  // last-known top-level roots; when growth hoists new parents above the seeds these change and we re-root
  const rootsRef = useRef<string[]>([]);

  const multiParentNodes = useMemo(() => {
    if (!graph.id) return new Set<string>();
    return findMultiParentNodeIds(graph, treeIndex);
  }, [graphVersion, treeIndex]);

  const tree = useTree<string>({
    state: { loadingItemData, loadingItemChildrens, expandedItems },
    setLoadingItemData,
    setLoadingItemChildrens,
    setExpandedItems,
    rootItemId: 'root',
    getItemName: (node) => {
      // item ids are raw node ids on the backbone and composite off it; decode before any node lookup
      const nodeId = itemNodeId(node.getId());
      const g = getGraph();
      if (g.data_map && nodeId in g.data_map) {
        return getNodeName(g.data_map[nodeId], 100);
      }
      // never leak a composite id string (carries control chars) into the UI
      return nodeId;
    },
    isItemFolder: (node) => {
      const itemId = node.getId();
      const nodeId = itemNodeId(itemId);
      // Expandable if it already has display children OR is growable in ANY arrival context. A reverse-reached
      // node must stay expandable/growable: growing it fetches ITS associations (both directions), which is how
      // the next reverse hop's edges load (e.g. growing a Flag pulls its SigmaRule). Forward children are still
      // suppressed at display time by contextualDisplayEdges, so this only surfaces the reverse chain.
      if (overlayIsFolder(getIndex(), itemId)) return true;
      return growable.has(nodeId);
    },
    createLoadingItemData: () => 'loading...',
    dataLoader: {
      getItem: (itemId) => itemId,
      getChildren: async (itemId) => {
        if (itemId === 'root') {
          return buildTreeRoots(getGraph(), getIndex());
        }
        const nodeId = itemNodeId(itemId);
        // Grow once per node id in ANY arrival context. Growing fetches the node's associations in both
        // directions, so a reverse-reached node (e.g. a Flag) must grow to load its next reverse hop (its
        // SigmaRule). Forward children stay suppressed at display time via contextualDisplayEdges; `getIndex()`
        // rebuilds on graph identity change so a duplicate occurrence still sees children grown via another.
        if (growable.has(nodeId) && !grownNodesRef.current.has(nodeId)) {
          grownNodesRef.current.add(nodeId);
          await grow(nodeId);
        }
        return overlayChildItemIds(getIndex(), itemId);
      },
    },
    indent: 20,
    features: [asyncDataLoaderFeature, selectionFeature, hotkeysCoreFeature],
  });

  const getNodeTypeInfo = useCallback(
    (nodeId: string) => {
      if (!(nodeId in graph.data_map)) return null;
      return classifyNode(nodeId, graph);
    },
    [graphVersion],
  );

  // On every growth (graphVersion bump): refresh loaded occurrences of grown nodes, and re-root when growth
  // hoists new structural parents above the current top ("amend the top"). headless-tree caches each item's
  // children, so a node grown via one occurrence — or a newly discovered ancestor of a seed — won't surface
  // until we explicitly invalidate the affected items' children.
  useEffect(() => {
    const index = getIndex();
    const roots = buildTreeRoots(getGraph(), index);
    const prev = rootsRef.current;
    const rootsChanged = roots.length !== prev.length || roots.some((r, i) => r !== prev[i]);
    // on the first pass rootsRef is empty and the tree is still building its initial roots — nothing loaded
    const firstRun = prev.length === 0;
    rootsRef.current = roots;
    if (firstRun) return;
    // duplicate-grow freshness: a node grown via one occurrence must refresh its OTHER loaded occurrences
    // (each occurrence has its own children cache), so invalidate every loaded item of a grown node id
    for (const item of tree.getItems()) {
      const id = item.getId();
      if (id === 'root') continue;
      if (grownNodesRef.current.has(itemNodeId(id))) {
        void item.invalidateChildrenIds();
      }
    }
    if (rootsChanged) {
      // open the full structural ancestor chain (topmost..seed) for every seed so hoisted parents render open
      setExpandedItems((prevExpanded) => {
        const next = new Set(prevExpanded);
        for (const initialId of getGraph().initial) {
          const s = initialId.toString();
          for (const anc of ancestorChain(index, s)) next.add(anc);
          next.add(s);
        }
        return [...next];
      });
      void tree.getItemInstance('root').invalidateChildrenIds();
    }
  }, [graphVersion]);

  const pendingFocusRef = useRef<string | null>(null);
  const initialFocusDone = useRef(false);

  useEffect(() => {
    if (initialFocusDone.current) return;
    if (!graph.initial?.length) return;
    initialFocusDone.current = true;

    let aborted = false;
    const selectInitial = async () => {
      await new Promise((r) => setTimeout(r, 150));
      if (aborted) return;
      try {
        const initialId = graph.initial[0].toString();
        const item = tree.getItemInstance(initialId);
        item.select();
      } catch {
        // Node not yet available
      }
    };

    void selectInitial();
    return () => {
      aborted = true;
    };
  }, [graphVersion]);

  // Locate a rendered occurrence of `targetNodeId` (raw on the backbone, composite off it), expand its
  // ancestor items, and select it. Defers to `pendingFocusRef` (retried on the next graphVersion) when the
  // occurrence isn't loadable yet or doesn't exist in the tree.
  const focusNodeInTree = async (targetNodeId: string) => {
    const index = getIndex();
    const roots = buildTreeRoots(getGraph(), index);
    const located = overlayItemPathForNode(index, roots, targetNodeId);
    if (!located) {
      pendingFocusRef.current = targetNodeId;
      return;
    }
    // expand each ancestor item (topmost-first), loading children as needed
    for (const ancestorId of located.expandIds) {
      try {
        const item = tree.getItemInstance(ancestorId);
        if (item.isFolder() && !item.isExpanded()) {
          item.expand();
          await tree.loadChildrenIds(ancestorId);
        }
      } catch {
        pendingFocusRef.current = targetNodeId;
        return;
      }
    }
    // small delay for the tree to rebuild after expansions, then select the located occurrence
    await new Promise((r) => setTimeout(r, 50));
    try {
      tree.getItemInstance(located.itemId).select();
    } catch {
      // occurrence not yet visible — will sync when the tree rebuilds
    }
  };

  // When the graph focuses a node, expand to and select its occurrence in the tree.
  useEffect(() => {
    if (!focusedNodeId || focusSource !== FocusSource.Graph) return;
    void focusNodeInTree(focusedNodeId);
  }, [focusedNodeId, focusSource]);

  // Retry a deferred focus after the graph version changes (tree data updated), re-running the locator.
  useEffect(() => {
    if (!pendingFocusRef.current) return;
    const pending = pendingFocusRef.current;
    pendingFocusRef.current = null;

    const retryFocus = async () => {
      await new Promise((r) => setTimeout(r, 100));
      await focusNodeInTree(pending);
    };

    void retryFocus();
  }, [graphVersion]);

  return (
    <TreeContainer>
      <div {...tree.getContainerProps()} className="tree">
        {tree.getItems().map((item) => {
          const itemId = item.getId();
          // decode the real node id — item ids are raw on the backbone and composite off it
          const nodeId = itemNodeId(itemId);
          const typeInfo = getNodeTypeInfo(nodeId);
          const isDuplicate = multiParentNodes.has(nodeId);
          const isHighlighted = highlightedNodeId !== null && highlightedNodeId === nodeId;

          return (
            <button
              key={itemId}
              {...item.getProps()}
              style={{ paddingLeft: `${item.getItemMeta().level * 20}px` }}
              onMouseEnter={() => {
                if (isDuplicate) setHighlightedNodeId(nodeId);
              }}
              onMouseLeave={() => {
                if (highlightedNodeId === nodeId) setHighlightedNodeId(null);
              }}
              onClick={(e) => {
                const isGrowable = growable.has(nodeId);
                const isExpanded = item.isExpanded();

                // grow in any arrival context — a reverse-reached node grows to load its next reverse hop
                if (isGrowable && isExpanded && !grownNodesRef.current.has(nodeId)) {
                  e.stopPropagation();
                  grownNodesRef.current.add(nodeId);
                  setManuallyGrowing((s) => new Set(s).add(nodeId));
                  void grow(nodeId).finally(() =>
                    setManuallyGrowing((s) => {
                      const next = new Set(s);
                      next.delete(nodeId);
                      return next;
                    }),
                  );
                  setFocusedNode(nodeId, FocusSource.Tree);
                  return;
                }

                (item.getProps() as { onClick?: (e: React.MouseEvent) => void }).onClick?.(e);
                setFocusedNode(nodeId, FocusSource.Tree);
                if (isDuplicate) {
                  setHighlightedNodeId((prev) => (prev === nodeId ? null : nodeId));
                }
              }}
            >
              <TreeItemOverlay nodeData={nodeId in graph.data_map ? graph.data_map[nodeId] : undefined} isDuplicate={isDuplicate}>
                <span
                  className={`treeitem${item.isFocused() ? ' focused' : ''}${item.isExpanded() ? ' expanded' : ''}${item.isSelected() ? ' selected' : ''}${item.isFolder() ? ' folder' : ''}${isDuplicate && isHighlighted ? ' duplicate-highlight' : ''}`}
                >
                  {typeInfo && (
                    <img
                      className="node-type-icon"
                      title={entityLabel(typeInfo.nodeType)}
                      alt={entityLabel(typeInfo.nodeType)}
                      src={`data:image/svg+xml;base64,${btoa(getNodeSvg(typeInfo.nodeType, typeInfo.visualState))}`}
                    />
                  )}
                  {item.getItemName()}
                  {isDuplicate && (
                    <span className="duplicate-indicator" title="Duplicate: has multiple parents in graph">
                      Duplicate
                    </span>
                  )}
                  {(item.isLoading() || manuallyGrowing.has(nodeId)) && (
                    <StyledSpinner aria-label="loading" $size={14} style={{ marginLeft: 6 }} />
                  )}
                </span>
              </TreeItemOverlay>
            </button>
          );
        })}
      </div>
    </TreeContainer>
  );
};

export const AssociationTree: React.FC = () => {
  const { graph } = useGraphData();
  return (
    <ErrorBoundary fallback={<RenderErrorAlert page={false} />}>
      {graph.id ? <AssociationTreeComponent key={graph.id} /> : <StyledSpinner aria-label="loading" $size={28} />}
    </ErrorBoundary>
  );
};

export default AssociationTree;
