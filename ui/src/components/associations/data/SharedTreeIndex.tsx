import React, { createContext, useCallback, useContext, useMemo, useRef } from 'react';

// project imports
import { buildTreeIndex, TreeIndex } from '../browsing/treeHelpers';
import { useGraphData } from './GraphDataContext';
import { Graph } from '@models/trees';

// spec: ./SharedTreeIndex.spec.md

/**
 * Shared, version-memoized {@link TreeIndex} derived once per graph and consumed by every tree view under the
 * same {@link GraphDataProvider} (the entity browser AND the association-tree overlay). This is the single
 * source of truth for the derived index, so the two views never rebuild it independently and always agree.
 * `GraphDataContext` stays backend-data-only (SRP); the 3D graph uses `processInitialGraphData` and ignores
 * this layer.
 */
interface SharedTreeIndexValue {
  /** The index memoized on `graphVersion` — for render-time reads. */
  index: TreeIndex;
  /**
   * The latest index, rebuilt lazily when the underlying `Graph` object identity changes. Safe inside async
   * callbacks that outlive a render (e.g. the tree overlay's `getChildren`): `grow()` mutates `graphRef`
   * synchronously *before* the version bump commits a render, so a version-driven ref would be stale mid-grow.
   */
  getIndex: () => TreeIndex;
}

const SharedTreeIndexContext = createContext<SharedTreeIndexValue | undefined>(undefined);

/**
 * Read the shared tree index. Must be used within a {@link SharedTreeIndexProvider} (mounted by
 * {@link GraphDataProvider}).
 *
 * @returns The shared `{ index, getIndex }`.
 */
export const useSharedTreeIndex = (): SharedTreeIndexValue => {
  const ctx = useContext(SharedTreeIndexContext);
  if (ctx === undefined) {
    throw new Error('useSharedTreeIndex must be used within a SharedTreeIndexProvider');
  }
  return ctx;
};

/** Build a tree index for a graph, using an empty-branches graph until the graph has loaded (stable identity). */
const buildFor = (graph: Graph): TreeIndex => (graph.id ? buildTreeIndex(graph) : buildTreeIndex({ ...graph, branches: {} }));

/**
 * Provides the shared tree index. Mounted once by {@link GraphDataProvider} so both tree views under it share
 * one derived index.
 *
 * @param props.children - The subtree that consumes the shared index.
 */
export const SharedTreeIndexProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { graphVersion, getGraph } = useGraphData();
  // identity-keyed cache: rebuild only when the Graph object changes (mutated by grow before the version bump)
  const cacheRef = useRef<{ graph: Graph; index: TreeIndex } | null>(null);
  const getIndex = useCallback((): TreeIndex => {
    const current = getGraph();
    if (!cacheRef.current || cacheRef.current.graph !== current) {
      cacheRef.current = { graph: current, index: buildFor(current) };
    }
    return cacheRef.current.index;
  }, [getGraph]);
  // render-time index: recompute on version bump, reusing the identity cache so render and async agree
  const index = useMemo(() => getIndex(), [graphVersion, getIndex]);
  const value = useMemo<SharedTreeIndexValue>(() => ({ index, getIndex }), [index, getIndex]);
  return <SharedTreeIndexContext.Provider value={value}>{children}</SharedTreeIndexContext.Provider>;
};
