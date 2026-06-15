// project imports
import { getNodeTags } from '@components/associations/browsing/EntityBrowser/browserHelpers';
import { Entities, entityLabel } from '@models/entities';
import { TreeNode, TreeNodeKey } from '@models/trees';

// spec: ./SPEC.md

/**
 * One bar of the dashboard's "types" chart: a node-type/entity-kind and how many nodes carry it.
 *
 * `kind` is the raw {@link Entities} value (`File`/`Repo`/`Folder`/…), used both as the stable bar id and
 * to build the `Include` whitelist clause a click injects; `label` is the human-readable display name.
 */
export interface TypeCount {
  /// The raw entity kind — the bar id and the value of the `Include` clause a click injects.
  kind: Entities;
  /// The human-readable label shown under the bar.
  label: string;
  /// The number of nodes of this kind in the counted set.
  value: number;
}

/**
 * One bar of the dashboard's "tag-values" chart: a tag value and how many nodes carry it under the key.
 */
export interface TagValueBar {
  /// The tag value — the bar id and the value toggled into the key's is-one-of filter on click.
  value: string;
  /// The number of nodes carrying this value under the counted key.
  count: number;
}

/**
 * Count the nodes in a set by node type, blending files, repos, and each distinct entity kind.
 *
 * Walks every node once, discriminating on {@link TreeNodeKey}:
 *
 * - **Sample** nodes count toward {@link Entities.File}.
 * - **Repo** nodes count toward {@link Entities.Repo}.
 * - **Entity** nodes count toward their `Entity.kind` (one bar per kind, e.g. `Folder`/`Device`/
 *   `FileSystem`), falling back to {@link Entities.Other} for a kind-less node.
 * - **Tag** nodes are not a node type on this chart and are skipped.
 *
 * The result is sorted by descending count, then kind ascending, so the highest-signal bars lead and
 * ties render in a stable order. The function is pure (no React, no DOM) so the chart can memoize over it.
 *
 * @param nodes - The nodes to count over (the dashboard's visible/renderable set).
 * @returns The per-type counts, highest count first.
 */
export function collectTypeCounts(nodes: TreeNode[]): TypeCount[] {
  // kind -> running count, filled by walking every node's discriminant
  const counts = new Map<Entities, number>();
  const bump = (kind: Entities): void => {
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
  };
  for (const node of nodes) {
    if (node[TreeNodeKey.Sample]) {
      bump(Entities.File);
    } else if (node[TreeNodeKey.Repo]) {
      bump(Entities.Repo);
    } else if (node[TreeNodeKey.Entity]) {
      // Entity.kind is the raw entity kind; a kind-less/edge node falls back to Other
      bump(node[TreeNodeKey.Entity].kind ?? Entities.Other);
    }
  }
  return Array.from(counts, ([kind, value]) => ({ kind, label: entityLabel(kind), value })).sort(
    (a, b) => b.value - a.value || String(a.kind).localeCompare(String(b.kind)),
  );
}

/**
 * Count how many nodes in a set carry each value under a single tag key.
 *
 * Reads each node's normalized tags via {@link getNodeTags} and, for the requested `key`, increments once
 * per (node, value) — a node listing the same value twice under the key still counts once. Nodes lacking
 * the key contribute nothing. The result is sorted by descending count, then value ascending, so the
 * highest-signal bars lead and ties render in a stable order. Pure (no React, no DOM) so the chart can
 * memoize over it.
 *
 * @param nodes - The nodes to count over (the dashboard's visible/renderable set).
 * @param key - The tag key whose values to tally (e.g. `FileType`, `FileTypeExtension`).
 * @returns The value counts for `key`, highest count first; empty when no node carries the key.
 */
export function countTagKey(nodes: TreeNode[], key: string): TagValueBar[] {
  // value -> running count for the requested key only
  const counts = new Map<string, number>();
  for (const node of nodes) {
    const valueMap = getNodeTags(node)[key];
    if (valueMap === undefined) {
      continue;
    }
    // Object.keys dedupes repeated values within a node so each node counts once per value
    for (const value of Object.keys(valueMap)) {
      counts.set(value, (counts.get(value) ?? 0) + 1);
    }
  }
  return Array.from(counts, ([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
}
