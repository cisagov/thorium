// project imports
import { getNodeTags } from '@components/associations/browsing/EntityBrowser/browserHelpers';
import type { TreeNode } from '@models/trees';

// spec: ./SPEC.md

/**
 * Tag keys that are **never** shown in the tags filter, regardless of the omnibar's `hidden tags` clause.
 *
 * Each of these keys carries a distinct hash per folder, so its value cardinality is effectively unbounded:
 * listing them as clickable chips is useless and floods the tile. Unlike the omnibar-derived hidden keys
 * (which the user controls), these are excluded unconditionally. Add further high-cardinality keys here —
 * this constant is the single place that keeps such keys out of the tags filter.
 */
export const ALWAYS_HIDDEN_TAG_KEYS = ['FolderAllSha256', 'FolderDataSha256', 'FolderNamesSha256'];

/// A single tag value and how many nodes in the counted set carry it under a given key.
export interface TagValueCount {
  /// The tag value.
  value: string;
  /// The number of nodes that carry this value under the key.
  count: number;
}

/**
 * Tally the tag values across a set of nodes, grouped by tag key.
 *
 * Walks each node's tags via {@link getNodeTags} (normalizing Sample/Repo/Entity/Tag shapes) and counts,
 * per tag key, how many nodes carry each value — one increment per (node, key, value) so a node listing
 * the same value twice under a key still counts once. Keys in `hiddenKeys` are skipped entirely so the
 * dashboard's display-hidden tags (e.g. `Results`/`Parent`/`submitter`) never surface. Each key's value
 * list is returned sorted by descending count, then value ascending, so the highest-signal chips lead and
 * ties render in a stable order.
 *
 * The function is pure (no React, no DOM) so the tags tile can memoize over it.
 *
 * @param nodes - The nodes to count over (the dashboard's visible/renderable set).
 * @param hiddenKeys - Tag keys to exclude from the tally.
 * @returns A map of tag key to its sorted value+count list; keys with no surviving values are omitted.
 */
export function collectTagCounts(nodes: TreeNode[], hiddenKeys: string[]): Map<string, TagValueCount[]> {
  const hidden = new Set(hiddenKeys);
  // key -> (value -> count) accumulator, filled by walking every node's normalized tags
  const counts = new Map<string, Map<string, number>>();
  for (const node of nodes) {
    const tags = getNodeTags(node);
    for (const [key, valueMap] of Object.entries(tags)) {
      if (hidden.has(key)) {
        continue;
      }
      let valueCounts = counts.get(key);
      if (valueCounts === undefined) {
        valueCounts = new Map<string, number>();
        counts.set(key, valueCounts);
      }
      // one increment per (node, key, value); Object.keys dedupes repeated values within a node
      for (const value of Object.keys(valueMap)) {
        valueCounts.set(value, (valueCounts.get(value) ?? 0) + 1);
      }
    }
  }
  // materialize each key's value map into a list sorted by descending count then value ascending
  const result = new Map<string, TagValueCount[]>();
  for (const [key, valueCounts] of counts) {
    const list = Array.from(valueCounts, ([value, count]) => ({ value, count })).sort(
      (a, b) => b.count - a.count || a.value.localeCompare(b.value),
    );
    result.set(key, list);
  }
  return result;
}
