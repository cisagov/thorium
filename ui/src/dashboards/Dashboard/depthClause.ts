// spec: ./SPEC.md

// project imports
import { ClauseCondition, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';

/**
 * The omnibar category/field that carries the graph traversal depth.
 *
 * Matches the `depth` option added by `addDepthOptions` and read back by `getDepthFromClauses`
 * (via `getStringFieldListFromClauses(clauses, 'depth')`), so a clause built here is both rendered
 * by the omnibar and picked up as the active depth.
 */
const DEPTH_CATEGORY = 'depth';

/**
 * Build a single omnibar depth {@link Clause} for the given crawl depth.
 *
 * The shape mirrors the `depth` option the browser omnibar renders (`category`/`field` `depth`, an
 * `Is` condition, the numeric value stringified) so it round-trips through the omnibar and is read
 * back by `getDepthFromClauses`.
 *
 * @param depth - The crawl depth to encode.
 * @returns The depth clause.
 */
export function makeDepthClause(depth: number): Clause {
  return {
    category: DEPTH_CATEGORY,
    field: DEPTH_CATEGORY,
    condition: ClauseCondition.Is,
    value: { value: String(depth) },
  };
}

/**
 * Return `clauses` with every existing depth clause removed and a single fresh depth clause appended.
 *
 * Used by "Grow Level" so raising the depth never duplicates the depth field: any prior depth clause
 * (regardless of position) is dropped and the new one is appended at the end, while all non-depth
 * clauses are preserved in their original order.
 *
 * @param clauses - The current clause list.
 * @param depth - The new crawl depth to encode.
 * @returns A new clause list with exactly one depth clause (the new one) appended last.
 */
export function withDepthClause(clauses: Clause[], depth: number): Clause[] {
  const withoutDepth = clauses.filter((clause) => clause.category !== DEPTH_CATEGORY);
  return [...withoutDepth, makeDepthClause(depth)];
}
