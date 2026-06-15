// project imports
import { classifyNode } from '../graph/data';
import { getEdgeLabel, getNodeName, getOrCreate } from '../utilities';
import { CONTAINER_ASSOCIATION_KINDS, NON_STRUCTURAL_ASSOCIATION_KINDS } from '@models/associations';
import { entityLabel } from '@models/entities';
import { BranchNode, Direction, Graph, TreeRelationships } from '@models/trees';

// spec: ./AssociationTree.spec.md

/**
 * A single parent→child edge in a {@link TreeIndex}, carrying the graph's relationship metadata so
 * consumers (the entity browser's relationship badges) can describe *how* two nodes relate, not just that
 * they do. `direction` is the raw branch direction; `relationship` is the tagged relationship union
 * (Initial / Origin / Tags / Association); `label` is the pre-formatted display string.
 */
export interface TreeEdge {
  /** The child node id this edge points to. */
  id: string;
  /** The raw branch direction (To / From / Bidirectional). */
  direction: Direction;
  /** The relationship union for this edge (Association kind / Origin / Tags / Initial). */
  relationship: TreeRelationships;
  /** Pre-formatted, human-readable edge label (via {@link getEdgeLabel}). */
  label: string;
  /**
   * For containment ("… In …") associations, the container's `name kind` (e.g. "somefolder Folder"),
   * appended after {@link label} in the relationship badge. Undefined for non-containment edges.
   */
  containerLabel?: string;
  /**
   * True when this edge is being surfaced *against* its stored direction (a relationship child shown under
   * the node it points *to*). Reverse edges reuse the primary edge's stored `label` verbatim — nothing is
   * re-formatted. Undefined/false for primary-direction edges. Set only on copies returned by
   * {@link displayChildren}, never on the stored index edges.
   */
  reversed?: boolean;
}

export interface TreeIndex {
  /** Ordered, edge-carrying children per node. Use {@link childIdsOf} for a bare unique-id view. */
  childrenOf: Map<string, TreeEdge[]>;
  /** Ordered, edge-carrying parents per node (`edge.id` = the parent id). Use {@link parentIdsOf} for a bare unique-id view. */
  parentsOf: Map<string, TreeEdge[]>;
}

/**
 * Build a direction-aware parent/child index from a graph's branches.
 *
 * Each branch is resolved to a parent→child edge by direction (To/Bidirectional ⇒ owner→node;
 * From ⇒ node→owner) and stored with its relationship metadata. Edges are deduped per parent by
 * (childId, relationship_hash): this collapses the reverse-pair a directed association produces (stored as
 * `To` on one endpoint and `From` on the other with the same hash) into a single edge, while still allowing
 * genuinely distinct relationships between the same two nodes to coexist. Bidirectional edges intentionally
 * yield *mutual* parent/child entries (A is a child of B and vice-versa); the browser's per-path cycle guard
 * keeps that from rendering forever.
 *
 * @param graph - The graph to index.
 * @returns An index of edge-carrying `childrenOf` and `parentsOf`.
 */
export function buildTreeIndex(graph: Graph): TreeIndex {
  const childrenOf = new Map<string, TreeEdge[]>();
  const parentsOf = new Map<string, TreeEdge[]>();
  // dedupe keys per parent so reverse-pair branches don't produce duplicate edges
  const edgeKeys = new Map<string, Set<string>>();
  // dedupe parent edges per child by (parentId, relationship_hash), mirroring edgeKeys, so the same parent
  // linked via distinct relationships coexists while reverse-pair branches collapse
  const parentEdgeKeys = new Map<string, Set<string>>();
  // built once and reused so per-edge classifyNode() doesn't reallocate the growable/initial sets
  const precomputed = {
    growableSet: new Set(graph.growable.map((n) => n.toString())),
    initialSet: new Set(graph.initial.map((n) => n.toString())),
  };

  const addEdge = (parent: string, child: string, branch: BranchNode) => {
    const key = `${child}-${branch.relationship_hash ?? ''}`;
    const keys = getOrCreate(edgeKeys, parent, () => new Set<string>());
    if (!keys.has(key)) {
      keys.add(key);
      const edges = getOrCreate(childrenOf, parent, () => [] as TreeEdge[]);
      // For containment ("… In …") associations the `parent` here is the association source = the container
      // (folder→file, filesystem→folder, file→filesystem — true for both To and From resolutions), so name it.
      const assocKind = branch.relationship.Association?.kind;
      let containerLabel: string | undefined;
      if (assocKind && CONTAINER_ASSOCIATION_KINDS.has(assocKind) && graph.data_map[parent]) {
        const name = getNodeName(graph.data_map[parent], 40);
        if (name) containerLabel = `${name} ${entityLabel(classifyNode(parent, graph, precomputed).nodeType)}`;
      }
      edges.push({
        id: child,
        direction: branch.direction,
        relationship: branch.relationship,
        // getEdgeLabel formats the label from the branch's relationship; the branch's target node is `child`
        label: getEdgeLabel(child, parent, branch, graph),
        containerLabel,
      });
    }

    // parentsOf mirrors childrenOf but keyed on the child: the parent edge's `id` is the PARENT node id.
    // Dedup by (parentId, relationship_hash). Do NOT copy `containerLabel` — that names the container for a
    // *containment* (directional) edge and would be wrong on a reverse/parent edge.
    const parentKey = `${parent}-${branch.relationship_hash ?? ''}`;
    const pKeys = getOrCreate(parentEdgeKeys, child, () => new Set<string>());
    if (!pKeys.has(parentKey)) {
      pKeys.add(parentKey);
      const parents = getOrCreate(parentsOf, child, () => [] as TreeEdge[]);
      parents.push({
        id: parent,
        direction: branch.direction,
        relationship: branch.relationship,
        label: getEdgeLabel(child, parent, branch, graph),
      });
    }
  };

  if (graph.branches) {
    for (const [nodeId, branches] of Object.entries(graph.branches)) {
      for (const branch of branches) {
        if (branch.direction === Direction.To || branch.direction === Direction.Bidirectional) {
          addEdge(nodeId, branch.node, branch);
        } else if (branch.direction === Direction.From) {
          addEdge(branch.node, nodeId, branch);
        }
      }
    }
  }
  return { childrenOf, parentsOf };
}

/**
 * Collect the unique `edge.id`s of an edge list in first-seen order, optionally restricted to edges passing a
 * filter. Shared by the bare-id views ({@link childIdsOf}/{@link parentIdsOf}/{@link structuralParentIdsOf}).
 *
 * @param edges - The edges to read (may be undefined when the node has none).
 * @param filter - Optional predicate; only edges it accepts contribute their id.
 * @returns The distinct edge ids, in edge order.
 */
function uniqueEdgeIds(edges: TreeEdge[] | undefined, filter?: (edge: TreeEdge) => boolean): string[] {
  if (!edges) return [];
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const edge of edges) {
    if (filter && !filter(edge)) continue;
    if (!seen.has(edge.id)) {
      seen.add(edge.id);
      ids.push(edge.id);
    }
  }
  return ids;
}

/**
 * Bare, unique child-id view of a node's edges (preserves first-seen order).
 *
 * @param index - The tree index to read.
 * @param nodeId - The parent node id.
 * @returns The parent's child ids, each once, in edge order.
 */
export function childIdsOf(index: TreeIndex, nodeId: string): string[] {
  return uniqueEdgeIds(index.childrenOf.get(nodeId));
}

/**
 * Bare, unique parent-id view of a node's parent edges (preserves first-seen order). Mirrors {@link childIdsOf}.
 *
 * @param index - The tree index to read.
 * @param nodeId - The child node id.
 * @returns The node's distinct parent ids, in edge order.
 */
export function parentIdsOf(index: TreeIndex, nodeId: string): string[] {
  return uniqueEdgeIds(index.parentsOf.get(nodeId));
}

/** The primary direction a view descends the tree index. */
export enum TreeOrientation {
  /** Descend childrenOf (parent → child). */
  Down = 'down',
  /** Descend parentsOf (child → parent). */
  Up = 'up',
}

/** How a view descends the index, and which edges it additionally surfaces against their stored direction. */
export interface DisplayChildrenConfig {
  /** Primary descent direction: `Down` = childrenOf, `Up` = parentsOf. */
  orientation: TreeOrientation;
  /** Given an edge, whether it should ALSO be surfaced in the opposite (reverse) direction. */
  bidirectional: (edge: TreeEdge) => boolean;
}

/**
 * Default bidirectional predicate: a *relationship* association edge (an Association whose kind IS in
 * {@link NON_STRUCTURAL_ASSOCIATION_KINDS}) surfaces both ways; every other association kind, plus
 * non-Association relationships (Origin / Tags / Initial), stays directional.
 *
 * @param edge - The edge to classify.
 * @returns True when the edge should also be shown against its stored direction.
 */
export function defaultBidirectional(edge: TreeEdge): boolean {
  const kind = edge.relationship.Association?.kind;
  return kind != null && NON_STRUCTURAL_ASSOCIATION_KINDS.has(kind);
}

/**
 * Whether an edge forms *hierarchy* — the backbone the overlay tree hoists as an ancestor spine and descends
 * as structure. An edge is structural unless it is a non-Association relationship that never forms hierarchy
 * (Tags), or an Association whose kind is in {@link NON_STRUCTURAL_ASSOCIATION_KINDS}. `Origin` (the sample a
 * node was derived/carved from) and `Initial` are genuine hierarchy, so they are structural.
 *
 * @param edge - The edge to classify.
 * @returns True when the edge is part of the structural hierarchy.
 */
export function isStructuralEdge(edge: TreeEdge): boolean {
  // Tags never form hierarchy — a tag hoisted as a spine ancestor above a file would invert the annotation.
  if (edge.relationship.Tags) return false;
  const kind = edge.relationship.Association?.kind;
  if (kind != null) return !NON_STRUCTURAL_ASSOCIATION_KINDS.has(kind);
  // Origin / Initial (and any other non-Association, non-Tags relationship) are directional hierarchy.
  return true;
}

/**
 * Bare, unique parent-id view restricted to structural edges (the hoisting spine). Mirrors
 * {@link parentIdsOf} but drops non-structural (relationship) and Tags parents.
 *
 * @param index - The tree index to read.
 * @param nodeId - The child node id.
 * @returns The node's distinct *structural* parent ids, in edge order.
 */
export function structuralParentIdsOf(index: TreeIndex, nodeId: string): string[] {
  return uniqueEdgeIds(index.parentsOf.get(nodeId), isStructuralEdge);
}

/**
 * Ascend a first-parent chain from `start`, following whichever parent `pickParent` selects at each hop, and
 * stopping on a cycle. The single source of the cycle-guarded first-parent walk shared by the browser's root
 * resolution / focus breadcrumb (which pick {@link parentIdsOf}) and the overlay's spine ({@link
 * structuralParentIdsOf}); callers reverse or slice the returned chain as they need.
 *
 * @param index - The tree index to ascend.
 * @param start - The node to ascend from (included as the first element of the chain).
 * @param pickParent - Given the tree index and a node, the parent to ascend to next (or null to stop).
 * @returns The chain `[start, parent, …, top]`, nearest-first, each id visited at most once.
 */
export function ascendFirstParent(
  index: TreeIndex,
  start: string,
  pickParent: (index: TreeIndex, nodeId: string) => string | null,
): string[] {
  const chain: string[] = [];
  const visited = new Set<string>();
  let current: string | null = start;
  while (current && !visited.has(current)) {
    visited.add(current);
    chain.push(current);
    current = pickParent(index, current);
  }
  return chain;
}

/** A down + relationship-bidirectional config — the entity browser's default view policy. */
export const DOWN_DEFAULT_CFG: DisplayChildrenConfig = { orientation: TreeOrientation.Down, bidirectional: defaultBidirectional };

/**
 * Resolve a (possibly partial) view config to a {@link DisplayChildrenConfig}. Callers that omit the fields
 * get **directional** behavior (`bidirectional: () => false`) so legacy/test paths are unchanged; a view opts
 * into bidirectionality by supplying its own predicate (the entity browser sets {@link defaultBidirectional}).
 *
 * @param cfg - A config carrying optional `orientation`/`bidirectional`.
 * @returns A fully-resolved {@link DisplayChildrenConfig}.
 */
export function toDisplayCfg(cfg: { orientation?: TreeOrientation; bidirectional?: (edge: TreeEdge) => boolean }): DisplayChildrenConfig {
  return { orientation: cfg.orientation ?? TreeOrientation.Down, bidirectional: cfg.bidirectional ?? (() => false) };
}

/**
 * The display children of a node under a view orientation: primary-direction edges first (index order), then
 * opposite-direction edges the config marks bidirectional (returned as **copies** flagged `reversed: true`, so
 * the stored index edges are never mutated), deduped by `edge.id` — primary wins. Pure and path-free; cycle
 * guarding and the reverse-depth bound are the caller's job.
 *
 * @param index - The tree index.
 * @param nodeId - The node whose display children to resolve.
 * @param cfg - Orientation + bidirectional predicate.
 * @returns Display edges (each `edge.id` is the node to render), primary before reversed, unique by id.
 */
export function displayChildren(index: TreeIndex, nodeId: string, cfg: DisplayChildrenConfig): TreeEdge[] {
  const primaryMap = cfg.orientation === TreeOrientation.Down ? index.childrenOf : index.parentsOf;
  const reverseMap = cfg.orientation === TreeOrientation.Down ? index.parentsOf : index.childrenOf;
  const out: TreeEdge[] = [];
  const seen = new Set<string>();
  for (const edge of primaryMap.get(nodeId) ?? []) {
    if (seen.has(edge.id)) continue;
    seen.add(edge.id);
    out.push(edge);
  }
  for (const edge of reverseMap.get(nodeId) ?? []) {
    if (seen.has(edge.id)) continue;
    if (!cfg.bidirectional(edge)) continue;
    seen.add(edge.id);
    out.push({ ...edge, reversed: true });
  }
  return out;
}

/**
 * Default max reverse (against-stored-direction) hops the tree follows from a primary node. Tuned for the
 * `WindowsProcess →(rev) Flag →(rev) SigmaRule` chain (2 hops) and used by the entity browser. The overlay
 * passes `Infinity` (follow the full reverse chain) and relies on its per-path cycle guard instead. The
 * primary noise control is that a node reached via a reversed edge only surfaces further reverse edges (never
 * its forward children) — see {@link contextualDisplayEdges}.
 */
export const REVERSE_MAX_DEPTH = 2;

/**
 * The display edges of a node given its *arrival context*, applying the reverse-traversal rules that bound the
 * bidirectional fan-out:
 * - a node reached via a reversed edge (`viaReversed`) surfaces ONLY further reverse edges, never its forward
 *   children (suppresses e.g. a reverse-reached SigmaRule re-listing all its other Flags);
 * - reverse edges are only followed while `reverseDepth < REVERSE_MAX_DEPTH`.
 *
 * @param index - The tree index.
 * @param nodeId - The node whose contextual display edges to resolve.
 * @param cfg - The display config.
 * @param viaReversed - Whether this node was itself reached via a reversed edge.
 * @param reverseDepth - How many reversed hops preceded this node.
 * @param maxReverseDepth - Max reversed hops to follow (default {@link REVERSE_MAX_DEPTH}; overlay passes `Infinity`).
 * @returns The edges to render as this node's children (each carries `reversed`).
 */
export function contextualDisplayEdges(
  index: TreeIndex,
  nodeId: string,
  cfg: DisplayChildrenConfig,
  viaReversed: boolean,
  reverseDepth: number,
  maxReverseDepth: number = REVERSE_MAX_DEPTH,
): TreeEdge[] {
  return displayChildren(index, nodeId, cfg).filter((edge) => {
    const isReverse = !!edge.reversed;
    if (viaReversed && !isReverse) return false;
    if (isReverse && reverseDepth >= maxReverseDepth) return false;
    return true;
  });
}

/**
 * True when a node has any contextual display child (drives the expand affordance).
 *
 * @param index - The tree index.
 * @param nodeId - The node id.
 * @param cfg - The display config.
 * @param viaReversed - Whether this node was reached via a reversed edge.
 * @param reverseDepth - Reversed hops preceding this node.
 * @param maxReverseDepth - Max reversed hops to follow (default {@link REVERSE_MAX_DEPTH}; overlay passes `Infinity`).
 * @returns Whether the node has any child to render in this context.
 */
export function hasContextualDisplayChildren(
  index: TreeIndex,
  nodeId: string,
  cfg: DisplayChildrenConfig,
  viaReversed: boolean,
  reverseDepth: number,
  maxReverseDepth: number = REVERSE_MAX_DEPTH,
): boolean {
  return contextualDisplayEdges(index, nodeId, cfg, viaReversed, reverseDepth, maxReverseDepth).length > 0;
}

/**
 * Node ids that have more than one distinct parent (rendered with a "Duplicate" badge). Counts distinct parent
 * ids (parent edges can carry >1 edge to the same parent after the `(parentId, relationship_hash)` dedup).
 *
 * @param graph - The graph (used to build an index if one isn't supplied).
 * @param index - Optional prebuilt index.
 * @returns The set of multi-parent node ids.
 */
export function findMultiParentNodeIds(graph: Graph, index?: TreeIndex): Set<string> {
  const idx = index ?? buildTreeIndex(graph);
  const multiParent = new Set<string>();
  for (const nodeId of idx.parentsOf.keys()) {
    if (parentIdsOf(idx, nodeId).length > 1) multiParent.add(nodeId);
  }
  return multiParent;
}
