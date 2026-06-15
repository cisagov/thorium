// spec: ./EntityBrowser.spec.md

// project imports
import { TreeEdge, TreeOrientation } from '../treeHelpers';
import { RequestTags } from '@models/tags';
import { NodeType } from '@models/trees';

/**
 * How a given node "layer" (a `NodeType`) is treated by the browser.
 * - `Show`: render the node and let the user explore its children.
 * - `PassThrough`: don't render the node, but traverse it and graft its qualifying descendants onto the
 *   parent (e.g. hide a `WindowsProcessTree` but surface the `WindowsProcess`es inside it). Elided hops are
 *   breadcrumbed so the row still explains where the child came from.
 * - `Skip`: don't render the node and don't explore/expand it (prune the branch entirely).
 */
export enum LayerPolicy {
  Show = 'show',
  PassThrough = 'passThrough',
  Skip = 'skip',
}

/** Per-`NodeType` policy overrides; kinds not present fall back to {@link EntityBrowserProps.fallbackPolicy}. */
export type LayerPolicyMap = Partial<Record<NodeType, LayerPolicy>>;

/** How rows at a level are sorted. The selected mode leads; the others tiebreak in {@link SORT_PRIORITY} order. */
export enum SortMode {
  /** By number of flags in the node's subtree (most first). */
  Flags = 'flags',
  /** By the highest flag suspicion in the node's subtree. */
  Suspicion = 'suspicion',
  /** By the highest flag confidence in the node's subtree. */
  Confidence = 'confidence',
}

/** The fixed sort priority: the selected {@link SortMode} leads, the rest tiebreak in this order. */
export const SORT_PRIORITY: readonly SortMode[] = [SortMode.Flags, SortMode.Suspicion, SortMode.Confidence];

/**
 * Aggregate flag significance for a node's whole subtree, used for the flag-count badge and sorting. Computed
 * once per graph version (see `computeFlagStats`) and read O(1) per row — never re-crawled per render.
 */
export interface FlagStat {
  /**
   * Number of distinct `Flag` entities that propagate to this node — a flag counts on itself, on the entity it
   * flags, and up that entity's containing spine (part → whole). A `SigmaRule` never receives a flag count (it
   * shows `0`); flag counts flow to the flag's subject, not back to the rule that created it.
   */
  flags: number;
  /** The highest `Flag.suspicion` at or beneath this node. */
  suspicion: number;
  /** The highest `Flag.confidence` (as an ordinal, higher = more confident) at or beneath this node. */
  confidence: number;
  /** Number of danger-classified tag pairs on this node and everything beneath it (aggregated up the spine). */
  dangerTags: number;
}

/** A resolved root of the tree: a graph node id plus an optional display label. */
export interface RootDescriptor {
  id: string;
  label?: string;
}

/**
 * How the browser determines its root artifacts.
 * - `sha256`: the graph node for a file (resolved via {@link findFileNodeHash}).
 * - `nodes`: explicit node ids (with optional labels) — supports a future heterogeneous incident view.
 * - `initial`: the graph's seed nodes, ascended to their tree roots.
 */
export type RootSpec = { kind: 'sha256'; sha256: string } | { kind: 'nodes'; roots: RootDescriptor[] } | { kind: 'initial' };

export interface EntityBrowserProps {
  /** How to determine the tree roots. */
  roots: RootSpec;
  /** When false, the browser renders nothing (used to skip work while its tab is hidden). Defaults to true. */
  inView?: boolean;
  /** Initial per-kind layer policies (also adjustable via the toolbar). */
  defaultPolicies?: LayerPolicyMap;
  /** Policy for kinds not in {@link defaultPolicies}. Defaults to {@link LayerPolicy.Show}. */
  fallbackPolicy?: LayerPolicy;
  /**
   * When true (default), each root is rendered as its own expandable row. When false, the roots' children
   * are rendered directly as the top level — used by the file-details tab, where the file itself is implicit.
   */
  showRootNodes?: boolean;
  /**
   * Depth to additively grow the shared graph to when the browser mounts, so nested structures load without
   * a manual expand per level. Does not bound the view (that's the omnibar `depth` clause); it only ensures
   * data is present. Omit to load nothing beyond the provider's initial fetch.
   */
  defaultDepth?: number;
}

/**
 * A child produced by {@link effectiveChildren}: the underlying edge plus, when the child was surfaced
 * through one or more `PassThrough` layers, the labels of those elided layers (nearest-parent first).
 */
export interface EffectiveChild {
  edge: TreeEdge;
  breadcrumb?: string[];
  /** Arrival context for expanding THIS child: whether it was reached via a reversed edge. */
  viaReversed?: boolean;
  /** Arrival context for expanding THIS child: how many reversed hops preceded it. */
  reverseDepth?: number;
}

/** A group of sibling children sharing a `NodeType`, for a layer header. */
export interface KindGroup {
  nodeType: NodeType;
  children: EffectiveChild[];
}

/** Layer policy derived from the omnibar `Show`/`Hide`/`Exclude`/`Include` clauses. */
export interface EntityLayerConfig {
  /** Explicit per-kind policies from `Show`/`Hide`/`Exclude`. */
  policies: LayerPolicyMap;
  /** Whitelist from `Include` (only these render as Show; others elide); null when no `Include` clause. */
  includeSet: Set<NodeType> | null;
}

/**
 * Everything that governs traversal/rendering of the tree: layer policies (clause + component defaults +
 * whitelist), plus the depth bound. Passed to {@link effectiveChildren} and {@link filterTree}.
 */
export interface TraversalConfig {
  /** Explicit clause policies (Show/Hide/Exclude). */
  clausePolicies: LayerPolicyMap;
  /** Whitelist from `Include`, or null. */
  includeSet: Set<NodeType> | null;
  /** Component-supplied defaults (e.g. file-tab `Tag: Skip`). */
  defaultPolicies: LayerPolicyMap;
  /** Policy for kinds not otherwise resolved. */
  fallback: LayerPolicy;
  /** Hide nodes farther than this many hops from the graph seeds; null = no bound. */
  maxDepth: number | null;
  /** BFS distance from `graph.initial` per node (for the depth bound). */
  distances: Map<string, number>;
  /**
   * Node ids the user has explicitly hidden (entities view only). Any child whose id is in this set — and its
   * whole subtree — is dropped from the rendered tree regardless of its {@link LayerPolicy}. Hiding is by node
   * id, so a DAG node duplicated under multiple parents disappears under all of them.
   */
  hiddenNodes?: Set<string>;
  /** Descent orientation for the display resolver (default {@link TreeOrientation.Down}). Enables a future parents-above view. */
  orientation?: TreeOrientation;
  /** Predicate for which edges surface against their stored direction (default: none — directional). */
  bidirectional?: (edge: TreeEdge) => boolean;
  /**
   * Re-root ("gear") mode: traverse the graph as **undirected and unbounded** from the root — every edge
   * surfaces both ways, with no reverse-depth cap and no reverse-arrival suppression. Combined with the
   * per-path cycle guard this yields a spanning tree that re-nests *every* connected node under the new root
   * (former ancestors become descendants), rather than the default bounded/directional descent that prunes to
   * a subtree. Distinct from the bullseye focus, which keeps the default (pruning) traversal.
   */
  spanning?: boolean;
}

/** The active client-side match criteria (AND across categories; OR within a multi-value category). */
export interface FilterCriteria {
  /** Case-insensitive substring on the node name. */
  text: string;
  /** Tag key → any-of values (case-insensitive). */
  tags: RequestTags;
  /** Node must belong to one of these groups. */
  groups: string[];
  /** When true, only flagged nodes match. */
  flaggedOnly: boolean;
  /** Precomputed flagged node ids (danger-tagged or with a Flag reachable within the pulled tree). */
  flaggedNodes: Set<string>;
}
