// spec: ./SPEC.md

// project imports
import { ClauseCondition, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { TreeNode, TreeNodeKey } from '@models/trees';

/**
 * The omnibar clause `field`/`category` that whitelists (narrows to) an entity/node kind.
 *
 * Matches the entity-layer lexicon added by `addEntityLayerOptions` and read back by
 * `getEntityLayerConfigFromClauses`/`getStringFieldListFromClauses(clauses, 'Include')`; the value is a
 * raw {@link NodeType} enum value so it maps straight to a layer-policy key.
 */
const INCLUDE_FIELD = 'Include';

/**
 * Build the `Include` (kind-whitelist) omnibar clause for a node kind.
 *
 * The single source of the Types-click clause contract, imported by the stats panel so the clause shape
 * can't drift between the callers. Narrowing requires an `Include` (whitelist) rather than a `Show`,
 * because the browser's default `Show` fallback already renders every kind.
 *
 * @param kind - The raw entity/node kind to whitelist (the clause value).
 * @returns The `Include` clause for `kind`.
 */
export function makeIncludeClause(kind: string): Clause {
  return {
    category: INCLUDE_FIELD,
    field: INCLUDE_FIELD,
    condition: ClauseCondition.Is,
    value: { value: kind },
  };
}

/**
 * Collect the deduped sha256 of every file (`Sample`) node in a graph's `data_map`.
 *
 * Walks the graph's `Sample` nodes, reading each `.Sample.sha256`, and returns them de-duplicated in
 * first-seen order. Pure — it neither mutates its input nor reads context. Used by the dashboard's
 * Analysis Status panel to fan out reaction lookups over the dashboard's files.
 *
 * @param dataMap - The graph's `data_map` (`{ [nodeId]: TreeNode }`).
 * @returns The distinct file sha256s present in the graph, in first-seen order.
 */
export function collectSampleSha256s(dataMap: Record<string, TreeNode>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const node of Object.values(dataMap)) {
    if (TreeNodeKey.Sample in node && node[TreeNodeKey.Sample]) {
      const sha256 = node[TreeNodeKey.Sample].sha256;
      if (sha256 && !seen.has(sha256)) {
        seen.add(sha256);
        out.push(sha256);
      }
    }
  }
  return out;
}
