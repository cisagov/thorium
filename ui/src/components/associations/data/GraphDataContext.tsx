import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

// project imports
import { mergeGrowthInto, computeDistances } from './graphMerge';
import { SharedTreeIndexProvider } from './SharedTreeIndex';
import { Seed, Graph, BlankGraph } from '@models/trees';
import { getInitialTree, growTree } from '@thorpi/trees';

// spec: ./GraphDataContext.spec.md

export enum FocusSource {
  Tree = 'tree',
  Graph = 'graph',
}

interface GraphDataContextType {
  /** Snapshot of the graph ref at the last version bump. Use for render-time reads.
   *  For async callbacks that may fire after the next render, prefer `getGraph()`. */
  graph: Graph;
  graphId: string;
  graphVersion: number;
  loading: boolean;
  /** True while any grow / growToDepth is queued or running — depth controls disable on this. */
  growing: boolean;
  /** The depth the graph was initially fetched to — depth controls seed their display from this. */
  initialDepth: number;
  error: string | null;
  growable: Set<string>;
  focusedNodeId: string | null;
  focusSource: FocusSource | null;
  /** Read the latest Graph directly from the ref — safe inside async callbacks that outlive a render. */
  getGraph: () => Graph;
  grow: (nodeId: string) => Promise<void>;
  growMultiple: (nodeIds: string[], limit?: number) => Promise<void>;
  growToDepth: (depth: number) => Promise<void>;
  reload: (opts?: { filterChildless?: boolean; depth?: number }) => Promise<void>;
  setFocusedNode: (nodeId: string | null, source: FocusSource) => void;
}

const GraphDataContext = createContext<GraphDataContextType | undefined>(undefined);

export const useGraphData = (): GraphDataContextType => {
  const context = useContext(GraphDataContext);
  if (context === undefined) {
    throw new Error('useGraphData must be used within a GraphDataProvider');
  }
  return context;
};

interface GraphDataProviderProps {
  initial: Seed;
  filterChildless?: boolean;
  depth?: number;
  children: React.ReactNode;
}

export const GraphDataProvider: React.FC<GraphDataProviderProps> = ({ initial, filterChildless = false, depth = 1, children }) => {
  const graphRef = useRef<Graph>(BlankGraph);
  const [graphId, setGraphId] = useState('');
  const [graphVersion, setGraphVersion] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const [focusSource, setFocusSource] = useState<FocusSource | null>(null);
  // set the focused node and record whether the focus came from the graph or the tree, so each view can
  // decide whether to animate to it (a view ignores focus it originated itself)
  const setFocusedNode = useCallback((nodeId: string | null, source: FocusSource) => {
    setFocusedNodeId(nodeId);
    setFocusSource(source);
  }, []);
  const growChainRef = useRef<Promise<void>>(Promise.resolve());
  // sequence tag for initial fetches so only the most recently started fetch may apply its result (see the
  // cross-tree race guard in fetchInitial)
  const fetchSeqRef = useRef(0);
  const bumpVersion = useCallback(() => setGraphVersion((v) => v + 1), []);
  const handleError = useCallback((err: string) => setError(err), []);
  // grow in-flight signal: grows are serialized on `growChainRef`, but callers await concurrently, so a
  // counter (not a bare boolean) tracks how many grow/growToDepth calls are queued or running. `growing` is
  // true while any is in flight — the depth controls disable on it so rapid changes can't queue silent work.
  const growingCountRef = useRef(0);
  const [growing, setGrowing] = useState(false);
  const beginGrow = useCallback(() => {
    growingCountRef.current += 1;
    if (growingCountRef.current === 1) setGrowing(true);
  }, []);
  const endGrow = useCallback(() => {
    growingCountRef.current = Math.max(0, growingCountRef.current - 1);
    if (growingCountRef.current === 0) setGrowing(false);
  }, []);
  // provider-level, tree-scoped, success-gated depth guard: the highest depth SUCCESSFULLY grown per tree id.
  // Both the 3D graph and the entity browser call growToDepth on this one shared graph, so the guard lives
  // here (not per-consumer): redundant or lower requests become no-ops, and it advances ONLY after a clean
  // same-tree completion so a failed grow stays retryable rather than permanently blocking that depth.
  const maxGrownDepthRef = useRef<{ id: string; depth: number }>({ id: '', depth: 1 });

  const fetchInitial = useCallback(
    async (seed: Seed, fc: boolean, d: number) => {
      const seq = ++fetchSeqRef.current;
      setLoading(true);
      setError(null);
      const data = await getInitialTree(seed, fc, d, handleError);
      // only the most recently STARTED fetch may apply its result: StrictMode double-invoke and remounts each
      // POST a new server tree, and applying a stale result would leave graphId/graphRef on a tree that later
      // grow responses merge into, producing "not a valid growable node" 400s
      if (seq !== fetchSeqRef.current) {
        return;
      }
      if (data) {
        graphRef.current = data;
        setGraphId(data.id);
        bumpVersion();
      }
      setLoading(false);
    },
    [handleError, bumpVersion],
  );

  useEffect(() => {
    void fetchInitial(initial, filterChildless, depth);
  }, [initial, filterChildless, depth, fetchInitial]);

  const growMultiple = useCallback(
    async (nodeIds: string[], limit = 1) => {
      const id = graphId;
      if (!id) return;

      const doGrow = async () => {
        const data = await growTree(id, nodeIds, handleError, limit);
        if (data) {
          // drop a grow response whose tree id no longer matches the live graph (the initial fetch was swapped
          // underneath us): merging it would pollute the live tree's growable with ids that don't exist
          // server-side, producing "not a valid growable node" 400s on later grows
          if (data.id && graphRef.current.id && data.id !== graphRef.current.id) {
            return;
          }
          graphRef.current = mergeGrowthInto(graphRef.current, data, nodeIds);
          bumpVersion();
        }
      };

      beginGrow();
      growChainRef.current = growChainRef.current.then(doGrow, doGrow);
      try {
        await growChainRef.current;
      } finally {
        endGrow();
      }
    },
    [graphId, handleError, bumpVersion, beginGrow, endGrow],
  );

  const grow = useCallback(
    async (nodeId: string) => {
      await growMultiple([nodeId]);
    },
    [growMultiple],
  );

  const growToDepth = useCallback(
    async (targetDepth: number) => {
      if (!graphId || targetDepth <= 1) return;

      const doGrowToDepth = async () => {
        // pin this run to the tree that was live when it started; if the shared graph swaps to a different tree
        // (StrictMode/remount), abort rather than sending this tree's growable ids to the new tree (which lacks
        // them => 400)
        const treeId = graphId;
        // shared depth guard: skip if this tree is already grown to at least this depth (a lower/redundant
        // request from either the graph or the entity browser is a no-op)
        if (maxGrownDepthRef.current.id === treeId && targetDepth <= maxGrownDepthRef.current.depth) {
          return;
        }
        let iterations = 0;
        // limit the max loops of "BFS -> grow by depth group" we will execute in case
        // we are returned continuous stream of nodes that can also be grown
        const maxIterations = 20;
        // remember ids that failed / were dropped so we don't re-send them every iteration (which would storm
        // the API with repeated 400s)
        const failed = new Set<string>();
        // whether the loop drained naturally (nothing left to grow within targetDepth) — the success signal
        let completed = false;
        while (iterations < maxIterations) {
          // stop if the live tree changed since this loop began
          if (graphRef.current.id !== treeId) {
            break;
          }
          const distances = computeDistances(graphRef.current);
          const growableSet = graphRef.current.growable.map((n) => n.toString()).filter((n) => !failed.has(n));
          // group nodes that need to be grown by depth
          const groups = new Map<number, string[]>();
          for (const nodeId of growableSet) {
            const dist = distances.get(nodeId);
            if (dist !== undefined && dist < targetDepth) {
              const limit = targetDepth - dist;
              // add empty depth group to depth map
              if (!groups.has(limit)) groups.set(limit, []);
              groups.get(limit)!.push(nodeId);
            }
          }
          // no groups to grow this iteration
          if (groups.size === 0) {
            completed = true;
            break;
          }
          // grow the group nodes for each depth together
          let mergedThisIteration = false;
          for (const [limit, nodes] of groups) {
            const data = await growTree(treeId, nodes, handleError, limit);
            // only merge a same-tree response; otherwise record the ids as failed so the loop converges instead
            // of re-sending the same (cross-tree or rejected) ids, which would otherwise storm the API with 400s
            if (data && (!data.id || !graphRef.current.id || data.id === graphRef.current.id)) {
              // merge any returned graph data into the existing graph
              graphRef.current = mergeGrowthInto(graphRef.current, data, nodes);
              mergedThisIteration = true;
            } else {
              for (const n of nodes) failed.add(n);
            }
          }
          // bump once per iteration rather than per depth-group so every deriver (distances, index, filters,
          // 3D rebuild) runs once for the whole iteration's growth instead of once per group
          if (mergedThisIteration) bumpVersion();
          iterations++;
        }
        // advance the guard ONLY on a clean, same-tree completion with no failures, so a failed/partial grow
        // stays retryable rather than being permanently marked "grown to this depth"
        if (completed && failed.size === 0 && graphRef.current.id === treeId) {
          maxGrownDepthRef.current = { id: treeId, depth: targetDepth };
        }
      };
      beginGrow();
      growChainRef.current = growChainRef.current.then(doGrowToDepth, doGrowToDepth);
      try {
        await growChainRef.current;
      } finally {
        endGrow();
      }
    },
    [graphId, handleError, bumpVersion, beginGrow, endGrow],
  );

  const reload = useCallback(
    async (opts?: { filterChildless?: boolean; depth?: number }) => {
      await fetchInitial(initial, opts?.filterChildless ?? filterChildless, opts?.depth ?? depth);
    },
    [initial, filterChildless, depth, fetchInitial],
  );

  const growable = useMemo(() => new Set(graphRef.current.growable.map((n) => n.toString())), [graphVersion]);
  const getGraph = useCallback(() => graphRef.current, []);
  const contextValues = useMemo<GraphDataContextType>(
    () => ({
      graph: graphRef.current,
      graphId,
      graphVersion,
      loading,
      growing,
      initialDepth: depth,
      error,
      growable,
      focusedNodeId,
      focusSource,
      getGraph,
      grow,
      growMultiple,
      growToDepth,
      reload,
      setFocusedNode,
    }),
    [
      graphId,
      graphVersion,
      loading,
      growing,
      depth,
      error,
      growable,
      focusedNodeId,
      focusSource,
      getGraph,
      grow,
      growMultiple,
      growToDepth,
      reload,
      setFocusedNode,
    ],
  );
  // SharedTreeIndexProvider derives ONE tree index per graph, shared by every tree view under this provider
  // (entity browser + association-tree overlay), so they never rebuild it independently. It reads this
  // context, so it must be mounted inside it.
  return (
    <GraphDataContext.Provider value={contextValues}>
      <SharedTreeIndexProvider>{children}</SharedTreeIndexProvider>
    </GraphDataContext.Provider>
  );
};
