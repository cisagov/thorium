// project imports
import { getOrCreate } from '../utilities';
import { canonicalizeGraphOrder } from '@thorpi/trees';
import { Graph } from '@models/trees';

// spec: ./GraphDataContext.spec.md

/**
 * Merge a grow response into the current graph, producing a new {@link Graph}.
 *
 * Nodes from `grown.data_map` overwrite the initial entries (newer info wins). Branches are
 * appended per source, deduped by `(node, direction, relationship_hash)` so a repeated edge is
 * never drawn twice. The growable frontier becomes the previously-growable nodes minus the ones
 * just grown, plus any newly growable nodes the response reports.
 *
 * Because the merge rebuilds `data_map` and appends to branch arrays, the result would lose the
 * canonical ordering its inputs carry, so the merged graph is re-canonicalized before returning
 * (see {@link canonicalizeGraphOrder}) — a grown tree stays as deterministic as a freshly fetched one.
 *
 * @param initial - The current graph to merge into (not mutated).
 * @param grown - The grow response to fold in.
 * @param grownNodeIds - The node ids that were grown in this request (removed from the frontier).
 * @returns A new merged graph; `initial` and `grown` are left unchanged.
 */
export function mergeGrowthInto(initial: Graph, grown: Graph, grownNodeIds: string[]): Graph {
  const mergedDataMap = { ...initial.data_map };
  // merge nodes from grown graph into initial
  // override in case we get info for an existing node
  if (grown.data_map) {
    for (const nodeId of Object.keys(grown.data_map)) {
      mergedDataMap[nodeId] = grown.data_map[nodeId];
    }
  }
  // merge branches
  const mergedBranches = { ...initial.branches };
  if (grown.branches) {
    for (const source of Object.keys(grown.branches)) {
      if (source in mergedBranches) {
        const existing = mergedBranches[source];
        const existingKeys = new Set(existing.map((b) => `${b.node}-${b.direction}-${b.relationship_hash ?? ''}`));
        const newBranches = grown.branches[source].filter((branch) => {
          const key = `${branch.node}-${branch.direction}-${branch.relationship_hash ?? ''}`;
          return !existingKeys.has(key);
        });
        if (newBranches.length > 0) {
          mergedBranches[source] = [...existing, ...newBranches];
        }
      } else {
        mergedBranches[source] = grown.branches[source];
      }
    }
  }
  // get unique set of nodes that are still growable
  const grownSet = new Set(grownNodeIds);
  const remaining = initial.growable.filter((id) => !grownSet.has(id));
  if (grown.growable) {
    remaining.push(...grown.growable);
  }
  // re-canonicalize: the rebuilt data_map keys and appended branch arrays would otherwise be ordered
  // by merge sequence rather than by node id
  return canonicalizeGraphOrder({
    ...initial,
    data_map: mergedDataMap,
    branches: mergedBranches,
    growable: remaining,
  });
}

/**
 * Compute the shortest hop distance from a set of seed nodes to every reachable node.
 *
 * Runs a BFS from the seeds (each at distance 0) over an undirected adjacency built from
 * `graph.branches`, so distance ignores edge direction. `growToDepth` uses the `graph.initial`-seeded
 * result to pick which growable frontier nodes still need growing; the entity browser also seeds it from
 * a single **focus root** so auto-expand depth is measured relative to a re-rooted subtree.
 *
 * @param graph - The graph to traverse.
 * @param seeds - The nodes to measure distance from (each at distance 0). Defaults to `graph.initial`.
 * @returns A map of node id to its shortest distance from any seed; unreachable nodes are absent.
 */
export function computeDistances(graph: Graph, seeds?: Array<string | number>): Map<string, number> {
  const distances = new Map<string, number>();
  const queue: [string, number][] = [];
  // seed distances with 0 (from the given seeds, or the graph's own initial seeds by default)
  for (const id of seeds ?? graph.initial) {
    const nodeId = id.toString();
    if (!distances.has(nodeId)) {
      distances.set(nodeId, 0);
      queue.push([nodeId, 0]);
    }
  }
  // build undirected adjacency list from branches
  const adjacency = new Map<string, Set<string>>();
  const addEdge = (a: string, b: string) => {
    getOrCreate(adjacency, a, () => new Set<string>()).add(b);
    getOrCreate(adjacency, b, () => new Set<string>()).add(a);
  };
  for (const branchStartNode of Object.keys(graph.branches)) {
    for (const branchNode of graph.branches[branchStartNode]) {
      addEdge(branchStartNode, branchNode.node.toString());
    }
  }
  // BFS traversal
  let idx = 0;
  while (idx < queue.length) {
    const [current, dist] = queue[idx++];
    const neighbors = adjacency.get(current);
    if (!neighbors) continue;
    for (const neighbor of neighbors) {
      if (!distances.has(neighbor)) {
        distances.set(neighbor, dist + 1);
        queue.push([neighbor, dist + 1]);
      }
    }
  }
  return distances;
}
