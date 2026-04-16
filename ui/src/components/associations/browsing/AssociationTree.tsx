import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { Overlay, Popover, Spinner } from 'react-bootstrap';
import { asyncDataLoaderFeature, hotkeysCoreFeature, selectionFeature } from '@headless-tree/core';
import { useTree } from '@headless-tree/react';
import { ErrorBoundary } from 'react-error-boundary';
import cn from 'classnames';

// project imports
import RenderErrorAlert from '../../shared/alerts/RenderErrorAlert';
import { getNodeName } from '../utilities';
import { classifyNode } from '../graph/data';
import { getNodeSvg } from '../graph/styles';
import { useGraphData } from '../data/GraphDataContext';
import { PreviewPopover } from './PreviewPopover';
import { TreeContainer } from './TreeContainer';
import { findMultiParentNodeIds, buildTreeIndex, TreeIndex, nodeTypeKeyToLabel } from './treeHelpers';
import NodePreviewContent from './NodePreviewContent';
import { Graph, TreeNode } from '@models/trees';

function findParentFromIndex(index: TreeIndex, nodeId: string): string | null {
  const parents = index.parentsOf.get(nodeId);
  return parents?.[0] ?? null;
}

function buildTreeRoots(graph: Graph, index: TreeIndex): string[] {
  const roots: string[] = [];
  for (const initialId of graph.initial) {
    let current = initialId;
    const visited = new Set<string>();
    visited.add(current);
    let parent = findParentFromIndex(index, current);
    while (parent && !visited.has(parent)) {
      visited.add(parent);
      current = parent;
      parent = findParentFromIndex(index, current);
    }
    if (!roots.includes(current)) roots.push(current);
  }
  return roots;
}

function getDirectChildren(index: TreeIndex, nodeId: string): string[] {
  return index.childrenOf.get(nodeId) ?? [];
}

function hasDirectChildren(index: TreeIndex, nodeId: string): boolean {
  const children = index.childrenOf.get(nodeId);
  return children !== undefined && children.length > 0;
}

interface TreeItemOverlayProps {
  nodeId: string;
  nodeData: TreeNode | undefined;
  isDuplicate: boolean;
  children: React.ReactNode;
}

const SHOW_DELAY = 400;
const HIDE_DELAY = 200;

const TreeItemOverlay: React.FC<TreeItemOverlayProps> = ({ nodeId, nodeData, isDuplicate, children }) => {
  const [show, setShow] = useState(false);
  const triggerRef = useRef<HTMLSpanElement>(null);
  const showTimer = useRef<number | undefined>(undefined);
  const hideTimer = useRef<number | undefined>(undefined);

  const scheduleShow = () => {
    window.clearTimeout(hideTimer.current);
    showTimer.current = window.setTimeout(() => setShow(true), SHOW_DELAY);
  };
  const scheduleHide = () => {
    window.clearTimeout(showTimer.current);
    hideTimer.current = window.setTimeout(() => setShow(false), HIDE_DELAY);
  };
  const cancelHide = () => window.clearTimeout(hideTimer.current);

  useEffect(
    () => () => {
      window.clearTimeout(showTimer.current);
      window.clearTimeout(hideTimer.current);
    },
    [],
  );

  return (
    <>
      <span ref={triggerRef} onMouseEnter={scheduleShow} onMouseLeave={scheduleHide} className="overlay-trigger">
        {children}
      </span>
      <Overlay
        target={triggerRef.current}
        show={show && !!nodeData}
        placement="right"
        container={document.body}
        popperConfig={{ modifiers: [{ name: 'offset', options: { offset: [0, 8] } }] }}
      >
        {(props) => (
          <PreviewPopover {...props} id={`preview-${nodeId}`} onMouseEnter={cancelHide} onMouseLeave={scheduleHide}>
            <Popover.Body>{nodeData && <NodePreviewContent nodeData={nodeData} isDuplicate={isDuplicate} />}</Popover.Body>
          </PreviewPopover>
        )}
      </Overlay>
    </>
  );
};

const AssociationTreeComponent: React.FC = () => {
  const { graph, graphVersion, grow, growable, getGraph, focusedNodeId, focusSource, setFocusedNode } = useGraphData();
  const [loadingItemData, setLoadingItemData] = useState<string[]>([]);
  const [loadingItemChildrens, setLoadingItemChildrens] = useState<string[]>([]);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);

  const treeIndex = useMemo(() => {
    if (!graph.id) return buildTreeIndex({ ...graph, branches: {} });
    return buildTreeIndex(graph);
  }, [graphVersion]);

  const [expandedItems, setExpandedItems] = useState<string[]>(() => {
    const initIdx = buildTreeIndex(graph);
    const items: string[] = [];
    for (const initialId of graph.initial) {
      let current = initialId.toString();
      const chain: string[] = [];
      const visited = new Set<string>();
      visited.add(current);
      let parent = findParentFromIndex(initIdx, current);
      while (parent && !visited.has(parent)) {
        chain.push(parent);
        visited.add(parent);
        current = parent;
        parent = findParentFromIndex(initIdx, current);
      }
      items.push(...chain, initialId.toString());
    }
    return [...new Set(items)];
  });

  const grownNodesRef = useRef(new Set<string>());
  const [manuallyGrowing, setManuallyGrowing] = useState<Set<string>>(new Set());

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
      const nodeId = node.getId();
      const g = getGraph();
      if (g.data_map && nodeId in g.data_map) {
        return getNodeName(g.data_map[nodeId], 100);
      }
      return node.getItemData();
    },
    isItemFolder: (node) => {
      const nodeId = node.getId();
      const g = getGraph();
      return !!(g.growable?.includes(nodeId) || hasDirectChildren(treeIndex, nodeId));
    },
    createLoadingItemData: () => 'loading...',
    dataLoader: {
      getItem: (nodeId) => nodeId,
      getChildren: async (nodeId) => {
        if (nodeId === 'root') {
          return buildTreeRoots(getGraph(), treeIndex);
        }

        const existingChildren = getDirectChildren(treeIndex, nodeId);
        if (growable.has(nodeId) && !grownNodesRef.current.has(nodeId)) {
          grownNodesRef.current.add(nodeId);
          await grow(nodeId);
          const freshIndex = buildTreeIndex(getGraph());
          return getDirectChildren(freshIndex, nodeId);
        }
        return existingChildren;
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

  // When graph clicks a node, expand ancestors in tree, select it, and scroll to it
  useEffect(() => {
    if (!focusedNodeId || focusSource !== 'graph') return;

    const expandAndSelect = async () => {
      const g = getGraph();
      if (!g.branches) return;

      const currentIndex = buildTreeIndex(g);

      const ancestors: string[] = [];
      let current = focusedNodeId;
      let parent = findParentFromIndex(currentIndex, current);
      while (parent) {
        ancestors.unshift(parent);
        current = parent;
        parent = findParentFromIndex(currentIndex, current);
      }

      // Expand each ancestor sequentially, loading children as needed
      for (const ancestorId of ancestors) {
        try {
          const item = tree.getItemInstance(ancestorId);
          if (item.isFolder() && !item.isExpanded()) {
            item.expand();
            await tree.loadChildrenIds(ancestorId);
          }
        } catch {
          pendingFocusRef.current = focusedNodeId;
          return;
        }
      }

      // Small delay for tree to rebuild after expansions
      await new Promise((r) => setTimeout(r, 50));

      try {
        const targetItem = tree.getItemInstance(focusedNodeId);
        targetItem.select();
      } catch {
        // Node not yet visible in tree — will sync when tree rebuilds
      }
    };

    void expandAndSelect();
  }, [focusedNodeId, focusSource]);

  // Retry pending focus after graph version changes (tree data updated)
  useEffect(() => {
    if (!pendingFocusRef.current) return;
    const pending = pendingFocusRef.current;
    pendingFocusRef.current = null;

    const retryFocus = async () => {
      await new Promise((r) => setTimeout(r, 100));
      try {
        const item = tree.getItemInstance(pending);
        item.select();
      } catch {
        // Still not available
      }
    };

    void retryFocus();
  }, [graphVersion]);

  return (
    <TreeContainer>
      <div {...tree.getContainerProps()} className="tree">
        {tree.getItems().map((item) => {
          const nodeId = item.getId();
          const typeInfo = getNodeTypeInfo(nodeId);
          const isDuplicate = multiParentNodes.has(nodeId);
          const isHighlighted = highlightedNodeId !== null && highlightedNodeId === nodeId;

          return (
            <button
              key={nodeId}
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
                  setFocusedNode(nodeId, 'tree');
                  return;
                }

                item.getProps().onClick?.(e);
                setFocusedNode(nodeId, 'tree');
                if (isDuplicate) {
                  setHighlightedNodeId((prev) => (prev === nodeId ? null : nodeId));
                }
              }}
            >
              <TreeItemOverlay
                nodeId={nodeId}
                nodeData={nodeId in graph.data_map ? graph.data_map[nodeId] : undefined}
                isDuplicate={isDuplicate}
              >
                <span
                  className={cn('treeitem', {
                    focused: item.isFocused(),
                    expanded: item.isExpanded(),
                    selected: item.isSelected(),
                    folder: item.isFolder(),
                    'duplicate-highlight': isDuplicate && isHighlighted,
                  })}
                >
                  {typeInfo && (
                    <img
                      className="node-type-icon"
                      title={nodeTypeKeyToLabel(typeInfo.nodeType)}
                      alt={nodeTypeKeyToLabel(typeInfo.nodeType)}
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
                    <Spinner
                      animation="border"
                      size="sm"
                      className="loading"
                      style={{ width: 14, height: 14, marginLeft: 6, borderWidth: 2 }}
                    />
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
      {graph.id ? <AssociationTreeComponent key={graph.id} /> : <Spinner animation="border" />}
    </ErrorBoundary>
  );
};

export default AssociationTree;
