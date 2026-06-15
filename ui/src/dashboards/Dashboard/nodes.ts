import { useMemo } from 'react';

// spec: ./SPEC.md

// project imports
import { useEntityBrowser } from '@components/associations/browsing/EntityBrowser/EntityBrowserContext';
import { useGraphData } from '@components/associations/data/GraphDataContext';
import type { TreeNode } from '@models/trees';

/**
 * Resolve the dashboard's counted node set from a graph's `data_map` and the entity browser's visible-id set.
 *
 * Both the stats charts and the tags tile must count over the **same** node set for their downselect to
 * agree (a SPEC invariant): the visible ids when a filter is active, otherwise every node in the graph.
 * Centralizing the resolution here keeps the two tiles from drifting apart.
 *
 * @param dataMap - The graph's `data_map` (`{ [nodeId]: TreeNode }`); may be undefined before the graph loads.
 * @param visibleSet - The entity browser's visible-id set when a filter is active, or `null`/`undefined` for
 *   "no filter" (count the whole `data_map`).
 * @returns The node objects to count over, with any ids missing from `data_map` dropped.
 */
export function visibleNodes(dataMap: Record<string, TreeNode> | undefined, visibleSet: Set<string> | null | undefined): TreeNode[] {
  const map = dataMap ?? {};
  // when a filter is active, resolve only the visible ids (dropping any that aren't in the map yet);
  // otherwise count every node in the graph
  return visibleSet
    ? Array.from(visibleSet, (id) => map[id]).filter((node): node is NonNullable<typeof node> => node !== undefined)
    : Object.values(map);
}

/**
 * Resolve the dashboard's counted node set from the graph and entity-browser contexts, memoized so the
 * two tiles that count over it share one identity-stable array and cannot drift on the memo/deps.
 *
 * Wraps {@link visibleNodes} with the SPEC "same node set" invariant: the memo recomputes only when the
 * graph's `data_map` changes (signaled by `graphVersion`) or the browser's `visibleSet` toggles/changes.
 * `graph` is a stable ref, so it is listed alongside `graphVersion` only to satisfy the exhaustive-deps
 * lint without affecting when the memo re-runs.
 *
 * @returns The node objects the stats charts and tags tile both count over.
 */
export function useVisibleNodes(): TreeNode[] {
  const { graph, graphVersion } = useGraphData();
  const { visibleSet } = useEntityBrowser();
  return useMemo(() => visibleNodes(graph.data_map, visibleSet), [graph, graphVersion, visibleSet]);
}
