// project imports
import {
  ascendFirstParent,
  contextualDisplayEdges,
  DOWN_DEFAULT_CFG,
  isStructuralEdge,
  structuralParentIdsOf,
  TreeIndex,
} from './treeHelpers';
import { Graph } from '@models/trees';

// spec: ./AssociationTree.spec.md

/**
 * The overlay tree is a 2D projection of a DAG. Nodes on the *structural backbone* (the hoisted ancestor
 * spine and its structural descendants) are keyed by their RAW node id — so focus sync, duplicate highlight,
 * `expandedItems`, the re-root effect, and grow-once tracking all keep working on node ids as before. Every
 * *off-backbone* occurrence (a node reached via a non-structural forward edge, or against its stored direction)
 * is keyed by an **anchor-encoded composite id** carrying the full path from its backbone anchor, so each
 * occurrence is unique and the path doubles as a cycle guard (killing both echoes and infinite reverse loops).
 */

/**
 * Unit Separator (U+001F). Node ids are u64-hash strings / sha256 / UUIDs (`[0-9a-f-]`), so this control
 * character can never appear inside one — making it a collision-proof segment separator in composite ids.
 */
const SEP = '';
/** Prefix marking a composite (off-backbone) overlay item id. */
const PREFIX = `off${SEP}`;

/** A parsed overlay item id: the real node id plus the reverse arrival context and the cycle-guard path. */
export interface OverlayItemRef {
  /** The real graph node id (for `data_map`/name/icon/focus/duplicate/grow lookups). */
  nodeId: string;
  /** The anchor node id plus every hop's node id, inclusive — the per-occurrence cycle guard. */
  path: string[];
  /** Whether this occurrence was reached via a reversed (against-stored-direction) edge on its last hop. */
  viaReversed: boolean;
  /** How many reversed hops preceded this occurrence. */
  reverseDepth: number;
}

/**
 * Whether an item id is a composite (off-backbone) id rather than a raw node id.
 *
 * @param itemId - The overlay item id.
 * @returns True when the id is composite.
 */
export function isCompositeItemId(itemId: string): boolean {
  return itemId.startsWith(PREFIX);
}

/**
 * Parse an overlay item id into its node id, arrival context, and cycle-guard path.
 *
 * @param itemId - A raw node id or an anchor-encoded composite id.
 * @returns The decoded {@link OverlayItemRef}.
 */
export function parseItemId(itemId: string): OverlayItemRef {
  // a raw node id is its own anchor: no hops, forward arrival, guard is just itself
  if (!isCompositeItemId(itemId)) {
    return { nodeId: itemId, path: [itemId], viaReversed: false, reverseDepth: 0 };
  }
  // composite layout: PREFIX + anchor + (SEP + dir + SEP + node) per hop
  const parts = itemId.slice(PREFIX.length).split(SEP);
  const anchor = parts[0];
  const path = [anchor];
  let viaReversed = false;
  let reverseDepth = 0;
  // each hop is a (direction, node) pair after the anchor
  for (let i = 1; i + 1 < parts.length; i += 2) {
    const dir = parts[i];
    const node = parts[i + 1];
    path.push(node);
    viaReversed = dir === 'r';
    if (dir === 'r') reverseDepth += 1;
  }
  return { nodeId: path[path.length - 1], path, viaReversed, reverseDepth };
}

/**
 * The real node id for an overlay item id (cheap; avoids a full parse).
 *
 * @param itemId - A raw node id or a composite id.
 * @returns The underlying graph node id.
 */
export function itemNodeId(itemId: string): string {
  if (!isCompositeItemId(itemId)) return itemId;
  // the last segment is always the node id
  return itemId.slice(itemId.lastIndexOf(SEP) + 1);
}

/**
 * Build the composite child item id for a hop off `parentItemId` to `childNodeId`.
 *
 * @param parentItemId - The parent occurrence's item id (raw = anchor, or an existing composite).
 * @param childNodeId - The child node id.
 * @param reversed - Whether the hop follows a reversed (against-stored-direction) edge.
 * @returns The composite child item id.
 */
export function makeChildItemId(parentItemId: string, childNodeId: string, reversed: boolean): string {
  const dir = reversed ? 'r' : 'f';
  // a raw parent becomes the anchor of a new composite; a composite parent is extended
  const base = isCompositeItemId(parentItemId) ? parentItemId : `${PREFIX}${parentItemId}`;
  return `${base}${SEP}${dir}${SEP}${childNodeId}`;
}

/**
 * Ascend from `nodeId` to its topmost STRUCTURAL ancestor via first-parent links, returning the ancestor
 * chain nearest-first (excluding `nodeId`). Follows only structural (hierarchy) parent edges — relationship
 * and Tags parents never form the spine. Visited-guarded against structural cycles.
 *
 * @param index - The tree index.
 * @param nodeId - The node to ascend from.
 * @returns The structural ancestors nearest-first up to the topmost, or `[]` if none.
 */
export function ancestorChain(index: TreeIndex, nodeId: string): string[] {
  // ascend the structural spine, then drop `nodeId` itself (the shared walk includes the start)
  return ascendFirstParent(index, nodeId, (idx, id) => structuralParentIdsOf(idx, id)[0] ?? null).slice(1);
}

/**
 * The overlay's top-level roots: each seed ascended to its topmost STRUCTURAL ancestor (so hierarchy parents
 * render above the seed), deduped.
 *
 * @param graph - The shared graph (for the seed ids).
 * @param index - The tree index.
 * @returns Distinct root ids (topmost structural ancestor per seed, or the seed itself if it has none).
 */
export function buildTreeRoots(graph: Graph, index: TreeIndex): string[] {
  const roots: string[] = [];
  for (const initialId of graph.initial) {
    const start = initialId.toString();
    const chain = ancestorChain(index, start);
    const top = chain.length ? chain[chain.length - 1] : start;
    if (!roots.includes(top)) roots.push(top);
  }
  return roots;
}

/**
 * Per-index memo of computed child item ids. Keyed by index identity (a fresh index object per graph), so it
 * is naturally invalidated when the graph changes and GC'd with the old index — no manual clearing needed.
 */
const childIdsCache = new WeakMap<TreeIndex, Map<string, string[]>>();

/**
 * The overlay child item ids of an item, applying the 2D placement rule: forward structural children of a
 * backbone (raw) item stay raw; every other rendered edge (non-structural forward children, and reverse
 * relationship edges) becomes an anchor-encoded composite child. A reverse-reached item surfaces ONLY further
 * reverse edges (forward fan-out suppressed by {@link contextualDisplayEdges}); the reverse chain is unbounded
 * (`Infinity`) and terminated by the per-path cycle guard. Memoized per index identity.
 *
 * @param index - The tree index.
 * @param itemId - The overlay item id whose children to resolve.
 * @returns The child item ids (raw for backbone, composite otherwise).
 */
export function overlayChildItemIds(index: TreeIndex, itemId: string): string[] {
  let cache = childIdsCache.get(index);
  if (!cache) {
    cache = new Map();
    childIdsCache.set(index, cache);
  }
  const hit = cache.get(itemId);
  if (hit) return hit;
  const { nodeId, path, viaReversed, reverseDepth } = parseItemId(itemId);
  const composite = isCompositeItemId(itemId);
  const edges = contextualDisplayEdges(index, nodeId, DOWN_DEFAULT_CFG, viaReversed, reverseDepth, Infinity);
  const out: string[] = [];
  for (const edge of edges) {
    // per-path cycle guard: never re-render a node already on this occurrence's path (kills echoes + loops)
    if (path.includes(edge.id)) continue;
    // a forward structural edge off a raw backbone item stays on the backbone (raw); everything else is a
    // per-occurrence composite so it carries its own guard path
    if (isStructuralEdge(edge) && !edge.reversed && !composite) {
      out.push(edge.id);
    } else {
      out.push(makeChildItemId(itemId, edge.id, !!edge.reversed));
    }
  }
  cache.set(itemId, out);
  return out;
}

/**
 * Whether an overlay item has any children to render (context-aware: a reverse-reached leaf with no further
 * reverse edges is not a folder). Growability is layered on by the caller for forward occurrences only.
 *
 * @param index - The tree index.
 * @param itemId - The overlay item id.
 * @returns True when the item has at least one child.
 */
export function overlayIsFolder(index: TreeIndex, itemId: string): boolean {
  return overlayChildItemIds(index, itemId).length > 0;
}

/**
 * Locate a rendered occurrence of `nodeId` in the overlay tree for graph-driven focus.
 *
 * Fast path: a node whose structural ancestor chain tops out at a current backbone root is placed at its raw
 * id, expanded via that chain. Otherwise a breadth-first search over the ACTUAL render relation
 * ({@link overlayChildItemIds}) from the roots finds the nearest occurrence — guaranteeing the returned item
 * id truly renders — bounded by a per-node visited set.
 *
 * @param index - The tree index.
 * @param roots - The current backbone roots (from {@link buildTreeRoots}).
 * @param nodeId - The node to locate.
 * @returns The occurrence's `itemId` and the ancestor item ids to expand (topmost-first), or `null` if unreachable.
 */
export function overlayItemPathForNode(index: TreeIndex, roots: string[], nodeId: string): { itemId: string; expandIds: string[] } | null {
  const rootsSet = new Set(roots);
  // fast path: a structural node whose spine tops out at a rendered root is placed at its raw id
  const chain = ancestorChain(index, nodeId);
  const top = chain.length ? chain[chain.length - 1] : nodeId;
  if (rootsSet.has(top)) {
    return { itemId: nodeId, expandIds: [...chain].reverse() };
  }
  // otherwise BFS the real render relation to the nearest occurrence
  const parentOf = new Map<string, string>();
  const visitedNodes = new Set<string>();
  const queue: string[] = [];
  for (const r of roots) {
    queue.push(r);
    visitedNodes.add(r);
  }
  let head = 0;
  while (head < queue.length) {
    const parentItemId = queue[head];
    head += 1;
    for (const childItemId of overlayChildItemIds(index, parentItemId)) {
      const childNode = itemNodeId(childItemId);
      parentOf.set(childItemId, parentItemId);
      if (childNode === nodeId) {
        // reconstruct the ancestor item ids from root down to this occurrence's parent
        const expandIds: string[] = [];
        let p: string | undefined = parentItemId;
        while (p !== undefined) {
          expandIds.push(p);
          p = parentOf.get(p);
        }
        expandIds.reverse();
        return { itemId: childItemId, expandIds };
      }
      // bound the search: expand each distinct node once (any occurrence suffices to locate the target)
      if (!visitedNodes.has(childNode)) {
        visitedNodes.add(childNode);
        queue.push(childItemId);
      }
    }
  }
  return null;
}
