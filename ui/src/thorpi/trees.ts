import { BranchNode, Direction, Graph, Seed } from '@models/trees';
import client, { parseRequestError } from './client';
import JSONBigInt from 'json-bigint';

// this will force big ints to be strings
const JSONBigString = JSONBigInt({ storeAsString: true });

/**
 * Normalize every node-id value in a parsed graph to a string.
 *
 * The API serializes u64 node ids as JSON numbers in `initial`/`growable`/`sent` and in each branch's
 * `node` field, while `data_map`/`branches` object keys are always strings. `json-bigint` (storeAsString)
 * only converts values longer than 15 digits to strings, so smaller hashes arrive as JS numbers. Downstream
 * code keys `Set`/`Map` lookups by string ids, so a numeric id silently misses (duplicate/short-hash nodes
 * then show no children and can't be grown). Coercing every id to a string here keeps the whole app
 * consistent regardless of hash magnitude.
 *
 * @param graph - The freshly parsed graph (mutated in place).
 * @returns The same graph with all node ids as strings.
 */
export const normalizeGraphIds = (graph: Graph): Graph => {
  const toStr = (id: unknown): string => String(id);
  if (Array.isArray(graph.initial)) graph.initial = graph.initial.map(toStr);
  if (Array.isArray(graph.growable)) graph.growable = graph.growable.map(toStr);
  if (Array.isArray(graph.sent)) graph.sent = graph.sent.map(toStr);
  const normBranches = (branches?: { [nodeId: string]: BranchNode[] }): void => {
    if (!branches) return;
    for (const key of Object.keys(branches)) {
      for (const branch of branches[key]) {
        branch.node = toStr(branch.node);
      }
    }
  };
  normBranches(graph.branches);
  normBranches(graph.hint_branches);
  return graph;
};

/**
 * Compare two node/relationship ids as numbers, not as strings.
 *
 * Node ids and relationship hashes are u64s that arrive as decimal strings (see
 * {@link normalizeGraphIds}). A plain string compare orders them lexically (`"9"` after `"10"`),
 * so canonicalization must compare their numeric value. `BigInt` is used because the values
 * routinely exceed `Number.MAX_SAFE_INTEGER`. Any id that isn't a valid integer literal (should
 * not happen for real graph ids) falls back to a lexical compare so the ordering stays total and
 * never throws.
 *
 * @param a - The first id string.
 * @param b - The second id string.
 * @returns Negative if `a < b`, positive if `a > b`, `0` if equal.
 */
export const compareNumericIds = (a: string, b: string): number => {
  try {
    const bigA = BigInt(a);
    const bigB = BigInt(b);
    if (bigA < bigB) return -1;
    if (bigA > bigB) return 1;
    return 0;
  } catch {
    // non-numeric id (unexpected): fall back to a stable lexical order
    return a < b ? -1 : a > b ? 1 : 0;
  }
};

/**
 * Impose a deterministic, content-derived order on a graph's `HashMap`-serialized collections.
 *
 * The API serializes `data_map`/`branches`/`hint_branches` from Rust `HashMap`s, whose iteration
 * order is non-deterministic, and JS iterates these huge-integer string keys in insertion order
 * (they exceed 2³² so they are not treated as array indices). The same seed+depth therefore renders
 * in different orders run-to-run. Rewriting each collection into a stable order here — object keys by
 * numeric id, each branch array by `(node, direction, relationship_hash)`, `growable` by numeric id —
 * makes every downstream deriver (`buildTreeIndex`, the 3D graph, the overlay) deterministic without
 * per-view sorting. `initial` is intentionally left untouched: it is the caller's seed/request order,
 * already deterministic and semantically the root order.
 *
 * @param graph - The graph to canonicalize (mutated in place).
 * @returns The same graph with canonical collection ordering.
 */
export const canonicalizeGraphOrder = (graph: Graph): Graph => {
  // rebuild an id-keyed object with its keys in numeric order (JS preserves this insertion order for
  // the huge-integer string keys, which are not array-index keys)
  const sortObjectKeys = <T>(obj?: { [key: string]: T }): { [key: string]: T } | undefined => {
    if (!obj) return obj;
    const sorted: { [key: string]: T } = {};
    for (const key of Object.keys(obj).sort(compareNumericIds)) {
      sorted[key] = obj[key];
    }
    return sorted;
  };
  // total order over a node's branches: target node, then direction, then the unique relationship hash
  const compareBranches = (a: BranchNode, b: BranchNode): number => {
    const byNode = compareNumericIds(a.node, b.node);
    if (byNode !== 0) return byNode;
    const dirOrder: Record<Direction, number> = { [Direction.To]: 0, [Direction.From]: 1, [Direction.Bidirectional]: 2 };
    const byDir = (dirOrder[a.direction] ?? 3) - (dirOrder[b.direction] ?? 3);
    if (byDir !== 0) return byDir;
    return compareNumericIds(a.relationship_hash ?? '', b.relationship_hash ?? '');
  };
  // rebuild a branches map with keys in numeric order and each branch array replaced by a sorted COPY,
  // so callers that share array references with an untouched input graph (e.g. mergeGrowthInto) are never
  // mutated
  const canonicalBranches = (branches?: { [nodeId: string]: BranchNode[] }): { [nodeId: string]: BranchNode[] } | undefined => {
    if (!branches) return branches;
    const sorted: { [nodeId: string]: BranchNode[] } = {};
    for (const key of Object.keys(branches).sort(compareNumericIds)) {
      sorted[key] = [...branches[key]].sort(compareBranches);
    }
    return sorted;
  };
  const sortedBranches = canonicalBranches(graph.branches);
  if (sortedBranches) graph.branches = sortedBranches;
  const sortedHintBranches = canonicalBranches(graph.hint_branches);
  if (sortedHintBranches) graph.hint_branches = sortedHintBranches;
  const sortedDataMap = sortObjectKeys(graph.data_map);
  if (sortedDataMap) graph.data_map = sortedDataMap;
  if (Array.isArray(graph.growable)) graph.growable = [...graph.growable].sort(compareNumericIds);
  return graph;
};

/**
 * Build the initial association graph from a set of seed nodes (`POST /trees/`).
 *
 * The response is parsed with a BigInt-aware JSON parser (storing big integers as strings) so
 * large numeric ids survive the round-trip without precision loss.
 *
 * @param data - The {@link Seed} describing the starting node(s) for the graph.
 * @param filterChildless - When `true`, omit nodes that have no children from the result.
 * @param limit - Maximum number of nodes to expand per growable node.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @returns The built {@link Graph}, or `null` if the request failed.
 */
export const getInitialTree = async (
  data: Seed,
  filterChildless: boolean,
  limit: number,
  errorHandler: (error: string) => void,
): Promise<Graph | null> => {
  const url = '/trees/';
  const params: { filter_childless?: boolean; limit: number } = { limit };
  if (filterChildless) {
    params['filter_childless'] = filterChildless;
  }
  return client
    .post<string>(url, { ...data }, { transformResponse: [(data: string) => data], params: params })
    .then((res) => {
      if (res && res.status && res.status == 200) {
        // normalize all node ids to strings, then impose a canonical order so the same graph always
        // renders identically regardless of the API's HashMap serialization order
        return canonicalizeGraphOrder(normalizeGraphIds(JSONBigString.parse(res.data) as Graph));
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Build Tree');
      return null;
    });
};

/**
 * Expand an existing association graph by growing from the given nodes (`PATCH /trees/{id}`).
 *
 * Like {@link getInitialTree}, the response is parsed with a BigInt-aware JSON parser to
 * preserve large numeric ids.
 *
 * @param id - The id of the existing graph/tree to grow.
 * @param nodes - The ids of the growable nodes to expand.
 * @param errorHandler - Called with a formatted message if the request fails.
 * @param limit - Maximum number of children to expand per node (defaults to 1).
 * @returns The updated {@link Graph}, or `null` if the request failed.
 */
export const growTree = async (id: string, nodes: string[], errorHandler: (error: string) => void, limit = 1): Promise<Graph | null> => {
  const url = `/trees/${id}`;
  const params: { limit: number } = { limit };
  return client
    .patch<string>(url, { growable: nodes }, { transformResponse: [(data: string) => data], params: params })
    .then((res) => {
      if (res?.status && res.status == 200 && res.data) {
        // normalize all node ids to strings, then impose a canonical order so the same graph always
        // renders identically regardless of the API's HashMap serialization order
        return canonicalizeGraphOrder(normalizeGraphIds(JSONBigString.parse(res.data) as Graph));
      }
      return null;
    })
    .catch((error: unknown) => {
      parseRequestError(error, errorHandler, 'Grow Tree');
      return null;
    });
};
