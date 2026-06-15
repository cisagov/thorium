// spec: ./EntityBrowser.spec.md
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

// project imports
import {
  defaultBidirectional,
  DOWN_DEFAULT_CFG,
  findMultiParentNodeIds,
  hasContextualDisplayChildren,
  TreeIndex,
  TreeOrientation,
} from '../treeHelpers';
import {
  collectGroupOptions,
  collectTagOptions,
  computeFlagStats,
  filterTree,
  focusBreadcrumb,
  getDepthFromClauses,
  getEntityLayerConfigFromClauses,
  nodeTypeOf,
  resolveRoots,
} from './browserHelpers';
import { FilterCriteria, FlagStat, LayerPolicy, LayerPolicyMap, RootDescriptor, RootSpec, SortMode, TraversalConfig } from './types';
import { useGraphData } from '../../data/GraphDataContext';
import { computeDistances } from '../../data/graphMerge';
import { getNodeName } from '../../utilities';
import { useSharedTreeIndex } from '../../data/SharedTreeIndex';
import { Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { getSearchTextFromClauses, getStringFieldListFromClauses, getTagsFromClauses } from '@components/shared/inputs/omnibar/utils';
import { TagOptions } from '@models/tags';
import { NodeType } from '@models/trees';

interface EntityBrowserContextValue {
  // graph-derived (recomputed on graphVersion)
  index: TreeIndex;
  roots: RootDescriptor[];
  /**
   * The node the tree is currently re-rooted (focused) at, or `null` for the natural roots. When set, {@link
   * roots} is that single node and the view is measured relative to it (indent resets, auto-expand is
   * focus-relative, and the depth bound is lifted so the whole loaded subtree is browsable).
   */
  focusRoot: string | null;
  /** Focus (bullseye) the tree at `id` — prune to that subtree — (or `null` to restore the natural roots). */
  setFocusRoot: (id: string | null) => void;
  /** The focus breadcrumb top→down (incl. the focus root as the last entry); empty when not focused. */
  focusAncestors: RootDescriptor[];
  /**
   * The node the tree is currently **re-rooted** (gear) at, or `null` for the natural roots. Unlike {@link
   * focusRoot}, re-rooting keeps *every* connected node visible, re-nesting them beneath the chosen node via a
   * spanning traversal (former ancestors become descendants); it reorders rather than prunes. Mutually
   * exclusive with {@link focusRoot} — setting one clears the other.
   */
  reRoot: string | null;
  /** Re-root the tree at `id` (or `null` to restore the natural roots). Clears any active {@link focusRoot}. */
  setReRoot: (id: string | null) => void;
  multiParent: Set<string>;
  /**
   * Ephemeral correlation number for each duplicate (multi-parent) node id, shown on the "Duplicate" badge so a
   * user can tell which occurrences are the same node. Numbers are assigned monotonically the first time a node
   * is seen and never renumbered, so growing the graph can't shuffle labels a user is already reading. Reset per
   * graph.
   */
  duplicateGroupIds: Map<string, number>;
  /**
   * The duplicate node id whose highlight is currently pinned (survives mouse-leave), or `null`. A ref rather
   * than state so pinning costs zero re-renders and — critically — can be read synchronously inside a
   * mouse-leave handler that fires *before* a state update would commit (the jump-induced scroll fires
   * `mouseleave` before React flushes). Rows apply/clear the highlight class imperatively; a row re-mounted
   * after pinning (e.g. by a re-root) re-applies it on mount by reading this ref.
   */
  pinnedDuplicate: React.MutableRefObject<string | null>;
  presentKinds: NodeType[];
  tagOptions: TagOptions;
  groupOptions: string[];
  /** Node ids allowed by the active filter, or `null` when no filter is active (render everything). */
  visibleSet: Set<string> | null;
  /** Per-node subtree flag aggregate (count / max suspicion / max confidence); one memoized pass per graph. */
  flagStats: Map<string, FlagStat>;
  /** How rows at each level are sorted (Flags/Suspicion/Confidence, descending; the rest tiebreak). */
  sortMode: SortMode;
  /** Set the active {@link sortMode}. */
  setSortMode: (mode: SortMode) => void;
  /** Whether rows at a level are grouped by resource kind (with layer headers). Defaults to true. */
  groupByResource: boolean;
  /** Toggle {@link groupByResource}. */
  setGroupByResource: (on: boolean) => void;
  /** Layer policies + depth bound governing traversal/rendering. */
  traversalConfig: TraversalConfig;
  // omnibar + flagged state
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  flaggedOnly: boolean;
  setFlaggedOnly: (b: boolean) => void;
  // per-node hide (entities view only): hidden ids and their whole subtrees are dropped from the tree
  hiddenNodes: Set<string>;
  /** Hide a node (and its entire subtree) by id. */
  hideNode: (id: string) => void;
  /** Unhide a single previously-hidden node by id. */
  unhideNode: (id: string) => void;
  /** Clear all hidden nodes. */
  unhideAll: () => void;
  /** Human-readable label for a (possibly hidden) node id, for the hidden-nodes list. */
  labelForNode: (id: string) => string;
  // per-row expansion (keyed by a path-unique row key so DAG duplicates expand independently). A row is
  // expanded when explicitly expanded, OR auto-expanded because it's within the current depth (and not
  // explicitly collapsed) — so raising the depth reveals the nesting without a manual click per level.
  isChildrenExpanded: (rowKey: string, nodeId: string, viaReversed?: boolean, reverseDepth?: number) => boolean;
  setChildrenExpanded: (rowKey: string, expanded: boolean) => void;
  /** Bulk expand/collapse many rows at once (e.g. a layer header's "collapse subsection" control). */
  setManyChildrenExpanded: (rowKeys: string[], expanded: boolean) => void;
  /** Grow-once guard shared across rows (growth mutates the shared graph). */
  grownNodes: Set<string>;
}

const EntityBrowserContext = createContext<EntityBrowserContextValue | undefined>(undefined);

export const useEntityBrowser = (): EntityBrowserContextValue => {
  const ctx = useContext(EntityBrowserContext);
  if (ctx === undefined) {
    throw new Error('useEntityBrowser must be used within an EntityBrowserProvider');
  }
  return ctx;
};

interface EntityBrowserProviderProps {
  roots: RootSpec;
  defaultPolicies?: LayerPolicyMap;
  fallbackPolicy?: LayerPolicy;
  defaultDepth?: number;
  /**
   * Optional controlled omnibar clauses. When provided (with {@link setClauses}), the provider is controlled
   * and the caller owns the clause state (e.g. a URL-backed dashboard); when omitted it falls back to internal
   * `useState`. Standard controlled/uncontrolled pattern.
   */
  clauses?: Clause[];
  /** Setter for controlled {@link clauses}. Required for the clauses to be controlled. */
  setClauses?: (next: Clause[]) => void;
  /**
   * Optional controlled hidden-node set. When provided (with {@link onHiddenNodesChange}), the caller owns the
   * hidden set (e.g. the dashboard keeps it in the URL); when omitted it falls back to internal `useState`.
   */
  hiddenNodes?: Set<string>;
  /** Change handler for controlled {@link hiddenNodes}. Required for the set to be controlled. */
  onHiddenNodesChange?: (next: Set<string>) => void;
  /** Optional controlled flagged-only toggle; falls back to internal `useState` when omitted. */
  flaggedOnly?: boolean;
  /** Setter for controlled {@link flaggedOnly}. Required for the toggle to be controlled. */
  setFlaggedOnly?: (b: boolean) => void;
  /** Optional controlled sort mode; falls back to internal `useState` (default {@link SortMode.Flags}). */
  sortMode?: SortMode;
  /** Setter for controlled {@link sortMode}. Required for the sort mode to be controlled. */
  setSortMode?: (mode: SortMode) => void;
  /** Optional controlled group-by-resource toggle; falls back to internal `useState` (default `true`). */
  groupByResource?: boolean;
  /** Setter for controlled {@link groupByResource}. Required for the toggle to be controlled. */
  setGroupByResource?: (on: boolean) => void;
  /**
   * Optional controlled focus root (re-rooted subtree). When provided (with {@link onFocusRootChange}), the
   * caller owns it (e.g. the dashboard keeps it in the URL); when omitted it falls back to internal `useState`.
   */
  focusRoot?: string | null;
  /** Change handler for controlled {@link focusRoot}. Required for the focus root to be controlled. */
  onFocusRootChange?: (id: string | null) => void;
  /**
   * Optional controlled re-root node (gear). When provided (with {@link onReRootChange}), the caller owns it
   * (e.g. the dashboard keeps it in the URL); when omitted it falls back to internal `useState`.
   */
  reRoot?: string | null;
  /** Change handler for controlled {@link reRoot}. Required for the re-root to be controlled. */
  onReRootChange?: (id: string | null) => void;
  children: React.ReactNode;
}

/**
 * Holds the entity browser's UI state (omnibar clauses, flagged toggle, per-row expansion) and the
 * graph-derived structures (index, roots, options, filter set, traversal config) so nested levels/rows read
 * them from context instead of prop-drilling. Layer policies, filters, and depth are all *derived from the
 * omnibar clauses*. Raising the depth clause additively grows the shared graph via {@link useGraphData}
 * (never a full refetch).
 */
export const EntityBrowserProvider: React.FC<EntityBrowserProviderProps> = ({
  roots: rootSpec,
  defaultPolicies = {},
  fallbackPolicy = LayerPolicy.Show,
  defaultDepth = 0,
  clauses: controlledClauses,
  setClauses: controlledSetClauses,
  hiddenNodes: controlledHiddenNodes,
  onHiddenNodesChange,
  flaggedOnly: controlledFlaggedOnly,
  setFlaggedOnly: controlledSetFlaggedOnly,
  sortMode: controlledSortMode,
  setSortMode: controlledSetSortMode,
  groupByResource: controlledGroupByResource,
  setGroupByResource: controlledSetGroupByResource,
  focusRoot: controlledFocusRoot,
  onFocusRootChange,
  reRoot: controlledReRoot,
  onReRootChange,
  children,
}) => {
  const { graph, graphId, graphVersion, growToDepth, growable } = useGraphData();

  // clauses / flagged / hidden each follow the standard controlled-or-uncontrolled pattern: when the caller
  // supplies the value + setter the provider is controlled, otherwise it manages the state internally
  const [internalClauses, setInternalClauses] = useState<Clause[]>([]);
  const clauses = controlledClauses ?? internalClauses;
  const setClauses = controlledSetClauses ?? setInternalClauses;
  const [internalFlaggedOnly, setInternalFlaggedOnly] = useState(false);
  const flaggedOnly = controlledFlaggedOnly ?? internalFlaggedOnly;
  const setFlaggedOnly = controlledSetFlaggedOnly ?? setInternalFlaggedOnly;
  const [internalHiddenNodes, setInternalHiddenNodes] = useState<Set<string>>(new Set());
  const hiddenNodes = controlledHiddenNodes ?? internalHiddenNodes;
  // apply a hidden-set update to whichever store owns it (controlled callback or internal state)
  const applyHiddenNodes = useCallback(
    (updater: (prev: Set<string>) => Set<string>) => {
      if (onHiddenNodesChange) {
        onHiddenNodesChange(updater(controlledHiddenNodes ?? new Set()));
      } else {
        setInternalHiddenNodes((prev) => updater(prev));
      }
    },
    [onHiddenNodesChange, controlledHiddenNodes],
  );
  const hideNode = useCallback(
    (id: string) => {
      applyHiddenNodes((prev) => {
        const next = new Set(prev);
        next.add(id);
        return next;
      });
    },
    [applyHiddenNodes],
  );
  const unhideNode = useCallback(
    (id: string) => {
      applyHiddenNodes((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    },
    [applyHiddenNodes],
  );
  const unhideAll = useCallback(() => {
    applyHiddenNodes(() => new Set());
  }, [applyHiddenNodes]);
  // focus root and re-root each follow the same controlled-or-uncontrolled pattern. focus (bullseye) prunes the
  // tree to a subtree; re-root (gear) reorders the whole graph under a new root. They are two distinct views of
  // the same data, so they are mutually exclusive — activating one clears the other.
  const [internalFocusRoot, setInternalFocusRoot] = useState<string | null>(null);
  const focusRoot = controlledFocusRoot !== undefined ? controlledFocusRoot : internalFocusRoot;
  const [internalReRoot, setInternalReRoot] = useState<string | null>(null);
  const reRoot = controlledReRoot !== undefined ? controlledReRoot : internalReRoot;
  // Setting one clears the other (they can't both be active). In the CONTROLLED case the owner's single change
  // handler is responsible for clearing the counterpart *atomically* — the dashboard writes both `focus` and
  // `root` in one URL update, because two separate URL writes in a single click tick would race and the second
  // would clobber the first. In the UNCONTROLLED case we clear the counterpart via a second setState, which
  // React batches safely.
  const setFocusRoot = useCallback(
    (id: string | null) => {
      if (onFocusRootChange) {
        onFocusRootChange(id);
      } else {
        setInternalFocusRoot(id);
        if (id) setInternalReRoot(null);
      }
    },
    [onFocusRootChange],
  );
  const setReRoot = useCallback(
    (id: string | null) => {
      if (onReRootChange) {
        onReRootChange(id);
      } else {
        setInternalReRoot(id);
        if (id) setInternalFocusRoot(null);
      }
    },
    [onReRootChange],
  );
  // explicit user expands / collapses layered over the depth-driven auto-expand default
  const [expandedChildren, setExpandedChildren] = useState<Set<string>>(new Set());
  const [collapsedChildren, setCollapsedChildren] = useState<Set<string>>(new Set());
  const grownNodesRef = useRef<Set<string>>(new Set());
  // the currently pinned duplicate node id (see the context field docs for why this is a ref, not state)
  const pinnedDuplicateRef = useRef<string | null>(null);
  // persistent monotonic numbering for duplicate node ids: `map` holds the assigned numbers, `counter` the last
  // one handed out, and `graphId` guards a reset when the graph changes (numbers restart at 1 per graph)
  const duplicateGroupRef = useRef<{ map: Map<string, number>; counter: number; graphId: string | null }>({
    map: new Map(),
    counter: 0,
    graphId: null,
  });
  // sort/group controls follow the same controlled-or-uncontrolled pattern: the dashboard keeps them in the
  // URL (controlled), while the file-details tab manages them internally (session-only)
  const [internalSortMode, setInternalSortMode] = useState<SortMode>(SortMode.Flags);
  const sortMode = controlledSortMode ?? internalSortMode;
  const setSortMode = controlledSetSortMode ?? setInternalSortMode;
  const [internalGroupByResource, setInternalGroupByResource] = useState(true);
  const groupByResource = controlledGroupByResource ?? internalGroupByResource;
  const setGroupByResource = controlledSetGroupByResource ?? setInternalGroupByResource;

  const setChildrenExpanded = useCallback((rowKey: string, expanded: boolean) => {
    setExpandedChildren((prev) => {
      const next = new Set(prev);
      if (expanded) next.add(rowKey);
      else next.delete(rowKey);
      return next;
    });
    setCollapsedChildren((prev) => {
      const next = new Set(prev);
      if (expanded) next.delete(rowKey);
      else next.add(rowKey);
      return next;
    });
  }, []);
  // expand/collapse many rows in one state update (a layer header collapsing its whole subsection at once)
  const setManyChildrenExpanded = useCallback((rowKeys: string[], expanded: boolean) => {
    setExpandedChildren((prev) => {
      const next = new Set(prev);
      for (const rowKey of rowKeys) {
        if (expanded) next.add(rowKey);
        else next.delete(rowKey);
      }
      return next;
    });
    setCollapsedChildren((prev) => {
      const next = new Set(prev);
      for (const rowKey of rowKeys) {
        if (expanded) next.delete(rowKey);
        else next.add(rowKey);
      }
      return next;
    });
  }, []);

  // graph-derived structures; recomputed only when the shared graph version bumps
  // the tree index is derived ONCE in the shared layer and consumed here (and by the association-tree overlay)
  const { index } = useSharedTreeIndex();
  // structural key for the root spec so inline `{ kind: 'sha256', sha256 }` literals (a fresh object every
  // parent render) don't re-run the O(data_map) resolveRoots scan and churn the roots array identity
  const rootSpecKey = useMemo(() => {
    switch (rootSpec.kind) {
      case 'sha256':
        return `sha256:${rootSpec.sha256}`;
      case 'nodes':
        return `nodes:${rootSpec.roots.map((r) => r.id).join(',')}`;
      case 'initial':
        return 'initial';
    }
  }, [rootSpec]);
  // re-root (gear) takes precedence over focus (bullseye) when resolving the single active root — both can't be
  // active, but a stale URL could carry both params, so precedence keeps the derivations unambiguous. A root is
  // only "applied" once its node is actually loaded; a stale id for an absent node falls back to natural roots.
  const reRootApplied = !!(reRoot && reRoot in graph.data_map);
  const focusApplied = !reRootApplied && !!(focusRoot && focusRoot in graph.data_map);
  const activeRootId = reRootApplied ? reRoot : focusApplied ? focusRoot : null;
  // resolveRoots depends only on the spec's structural content (captured by rootSpecKey), the graph, and the
  // index; keying on rootSpecKey instead of the spec object avoids recomputes from inline-literal identity.
  // When a focus/re-root node is active (and loaded), the tree is rooted at that single node instead.
  const roots = useMemo(() => {
    if (activeRootId) {
      const node = graph.data_map[activeRootId];
      return [{ id: activeRootId, label: (node ? getNodeName(node, 80) : '') || activeRootId }];
    }
    return resolveRoots(graph, rootSpec, index);
  }, [graphVersion, rootSpecKey, index, activeRootId]);
  // the breadcrumb trail back up from a focus root (top→down, incl. the focus root); empty unless focused. Not
  // shown for re-root, which surfaces its "rooted at X" clear affordance in the omnibar instead.
  const focusAncestors = useMemo(
    () => (focusApplied && focusRoot ? focusBreadcrumb(graph, index, focusRoot) : []),
    [graphVersion, index, focusApplied, focusRoot],
  );
  const multiParent = useMemo(() => findMultiParentNodeIds(graph, index), [graphVersion, index]);
  // assign each duplicate node id a stable correlation number. A fresh graph invalidates the numbers (different
  // nodes entirely), so reset synchronously when graphId changes; otherwise only *new* ids get a number and
  // existing ones keep theirs, so growth never reshuffles labels. Returns a fresh snapshot so the value identity
  // changes when membership does (rows read via `.get`). Also clear any pin from the previous graph.
  const duplicateGroupIds = useMemo(() => {
    const store = duplicateGroupRef.current;
    const currentGraphId = graphId ?? null;
    if (store.graphId !== currentGraphId) {
      store.map = new Map();
      store.counter = 0;
      store.graphId = currentGraphId;
      pinnedDuplicateRef.current = null;
    }
    for (const nodeId of multiParent) {
      if (!store.map.has(nodeId)) {
        store.counter += 1;
        store.map.set(nodeId, store.counter);
      }
    }
    return new Map(store.map);
  }, [multiParent, graphId]);
  const distances = useMemo(() => computeDistances(graph), [graphVersion]);
  // when focused/re-rooted, measure auto-expand depth relative to the active root so its first levels expand
  // (distances from the far-away original seeds would otherwise leave the re-rooted view collapsed)
  const rootedDistances = useMemo(() => (activeRootId ? computeDistances(graph, [activeRootId]) : null), [graphVersion, activeRootId]);
  const effectiveDistances = rootedDistances ?? distances;
  // one pass per graph version yields BOTH the flagged set (for the Flagged-Only filter) and the per-node
  // subtree flag stats (for the flag-count badge and sorting) — no per-render tree crawls
  const flagAgg = useMemo(() => computeFlagStats(graph, index), [graphVersion, index]);
  const flaggedNodes = flagAgg.flagged;
  const flagStats = flagAgg.stats;
  const tagOptions = useMemo(() => collectTagOptions(graph), [graphVersion]);
  const groupOptions = useMemo(() => collectGroupOptions(graph), [graphVersion]);
  const presentKinds = useMemo(() => {
    const kinds = new Set<NodeType>();
    for (const nodeId of Object.keys(graph.data_map ?? {})) {
      kinds.add(nodeTypeOf(nodeId, graph));
    }
    return Array.from(kinds);
  }, [graphVersion]);

  // clause-derived filter/policy inputs
  const layerConfig = useMemo(() => getEntityLayerConfigFromClauses(clauses), [clauses]);
  // 0 => no explicit depth clause => no depth bound (show everything pulled)
  const depthClause = useMemo(() => getDepthFromClauses(clauses, 0), [clauses]);
  const maxDepth = depthClause > 0 ? depthClause : null;
  // while focused/re-rooted the depth bound is lifted so the whole loaded graph under the active root is
  // browsable (the bound is distance-from-original-seeds, which would otherwise prune the re-rooted view)
  const effectiveMaxDepth = activeRootId ? null : maxDepth;
  // rows within this many hops of the seeds (or the focus root, when focused) auto-expand so the loaded
  // nesting shows. The explicit depth clause takes precedence; otherwise the component's `defaultDepth`.
  const autoExpandDepth = maxDepth ?? defaultDepth;
  const isChildrenExpanded = useCallback(
    (rowKey: string, nodeId: string, viaReversed = false, reverseDepth = 0) => {
      if (collapsedChildren.has(rowKey)) return false;
      if (expandedChildren.has(rowKey)) return true;
      // a growable node with NO loaded children yet stays collapsed under auto-expand (nothing to show — the
      // grow affordance stands in for it). A growable node that ALREADY has loaded children still auto-expands,
      // otherwise its loaded descendants (e.g. a SigmaRule's Flag children) would be hidden while it remains
      // growable. The grow badge continues to signal more can be loaded. The child presence check honors the
      // row's arrival context so a reverse-reached node isn't judged by its (suppressed) forward children.
      if (growable.has(nodeId) && !hasContextualDisplayChildren(index, nodeId, DOWN_DEFAULT_CFG, viaReversed, reverseDepth)) return false;
      return autoExpandDepth > 0 && (effectiveDistances.get(nodeId) ?? Infinity) < autoExpandDepth;
    },
    [collapsedChildren, expandedChildren, growable, effectiveDistances, autoExpandDepth, index],
  );
  // resolve a display label for a (possibly hidden) node id so the hidden-nodes control can list what's hidden
  const labelForNode = useCallback(
    (id: string) => {
      const node = graph.data_map?.[id];
      return (node ? getNodeName(node, 80) : '') || id;
    },
    [graph],
  );
  const text = useMemo(() => getSearchTextFromClauses(clauses), [clauses]);
  const tags = useMemo(() => getTagsFromClauses(clauses), [clauses]);
  const groups = useMemo(() => getStringFieldListFromClauses(clauses, 'group'), [clauses]);

  const traversalConfig = useMemo<TraversalConfig>(
    () => ({
      clausePolicies: layerConfig.policies,
      includeSet: layerConfig.includeSet,
      defaultPolicies,
      fallback: fallbackPolicy,
      maxDepth: effectiveMaxDepth,
      distances: effectiveDistances,
      hiddenNodes,
      // the entity browser surfaces relationship (non-structural) associations against their stored direction
      // so e.g. a WindowsProcess shows its Flags and each Flag its SigmaRule; containment stays directional
      orientation: TreeOrientation.Down,
      bidirectional: defaultBidirectional,
      // re-root mode spans the whole component from the new root (keeps every node, re-nested); focus/default
      // keep the bounded, pruning descent
      spanning: reRootApplied,
    }),
    [layerConfig, defaultPolicies, fallbackPolicy, effectiveMaxDepth, effectiveDistances, hiddenNodes, reRootApplied],
  );

  const criteria = useMemo<FilterCriteria>(
    () => ({ text, tags, groups, flaggedOnly, flaggedNodes }),
    [text, tags, groups, flaggedOnly, flaggedNodes],
  );

  // depth bounding is applied in effectiveChildren (via traversalConfig), so the visible-set filter only
  // needs to reflect the client-side match criteria (text / tags / groups / flagged)
  const filterActive = text.trim().length > 0 || Object.keys(tags).length > 0 || groups.length > 0 || flaggedOnly;
  const visibleSet = useMemo(
    () =>
      filterActive
        ? filterTree(
            roots.map((r) => r.id),
            index,
            graph,
            criteria,
            traversalConfig,
          )
        : null,
    [graphVersion, index, roots, criteria, traversalConfig, filterActive],
  );

  // grow the shared graph to the deepest requested level (the explicit depth clause OR the component's initial
  // `defaultDepth`). The provider's `growToDepth` owns the raise-only, tree-scoped, success-gated guard, so a
  // redundant/lower request (or one racing the 3D graph's own trigger) is a no-op there — this effect just
  // forwards the target. Gated on `graphId` so we don't fire before the initial fetch lands.
  const growTarget = Math.max(maxDepth ?? 0, defaultDepth);
  useEffect(() => {
    if (!graphId || growTarget <= 1) return;
    void growToDepth(growTarget);
  }, [graphId, growTarget, growToDepth]);

  const value = useMemo<EntityBrowserContextValue>(
    () => ({
      index,
      roots,
      focusRoot,
      setFocusRoot,
      focusAncestors,
      reRoot,
      setReRoot,
      multiParent,
      duplicateGroupIds,
      pinnedDuplicate: pinnedDuplicateRef,
      presentKinds,
      tagOptions,
      groupOptions,
      visibleSet,
      flagStats,
      sortMode,
      setSortMode,
      groupByResource,
      setGroupByResource,
      traversalConfig,
      clauses,
      setClauses,
      flaggedOnly,
      setFlaggedOnly,
      hiddenNodes,
      hideNode,
      unhideNode,
      unhideAll,
      labelForNode,
      isChildrenExpanded,
      setChildrenExpanded,
      setManyChildrenExpanded,
      grownNodes: grownNodesRef.current,
    }),
    [
      index,
      roots,
      focusRoot,
      setFocusRoot,
      focusAncestors,
      reRoot,
      setReRoot,
      multiParent,
      duplicateGroupIds,
      presentKinds,
      tagOptions,
      groupOptions,
      visibleSet,
      flagStats,
      sortMode,
      groupByResource,
      traversalConfig,
      clauses,
      setClauses,
      flaggedOnly,
      setFlaggedOnly,
      hiddenNodes,
      hideNode,
      unhideNode,
      unhideAll,
      labelForNode,
      isChildrenExpanded,
      setChildrenExpanded,
      setManyChildrenExpanded,
    ],
  );

  return <EntityBrowserContext.Provider value={value}>{children}</EntityBrowserContext.Provider>;
};
