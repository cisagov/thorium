// project imports
import { ClauseCondition, ClauseIsMulti, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';

// spec: ./SPEC.md

/**
 * The omnibar clause `category` for a tag filter.
 *
 * Tag clauses use `category: 'tag'` with `field` set to the tag key; the dashboard merges same-key
 * clicks into a single `IsOneOf` clause under this category.
 */
const TAG_CATEGORY = 'tag';

/**
 * The omnibar clause `category` that carries the graph traversal depth.
 *
 * A filter reset keeps depth clauses so raising/lowering the crawl depth survives clearing filters.
 */
const DEPTH_CATEGORY = 'depth';

/**
 * The omnibar clause `field` for the display-only hidden-tags filter.
 *
 * A filter reset keeps this field so the default hidden tags (`Results`/`Parent`/`submitter`) stay
 * hidden after clearing the user's filters.
 */
const HIDDEN_TAGS_FIELD = 'hidden tags';

/**
 * Whether a clause is any tag clause targeting the given key, regardless of its condition.
 *
 * Matching ignores the exact condition because the omnibar's URL round-trip re-decodes a tag clause
 * with whatever condition its value count implies (`urlState.ts` emits `Is` for a single value and
 * `IsOneOf` for several), so a match keyed on a fixed condition would miss a round-tripped clause and
 * append a duplicate on the next click. Restricting to `category === 'tag'` naturally excludes the
 * `hidden tags` field (category `'hidden tags'`).
 *
 * @param clause - The clause to test.
 * @param key - The tag key the clause's `field` must equal.
 * @returns Whether `clause` is a tag clause targeting `key`.
 */
function isTagClauseForKey(clause: Clause, key: string): boolean {
  return clause.category === TAG_CATEGORY && clause.field === key;
}

/**
 * Toggle `value` for the tag `key`, collapsing every existing tag clause for that key into one clause.
 *
 * The dashboard funnels every tag-value click (stats bars and tag chips) through this helper. Because the
 * omnibar URL round-trip can re-decode a tag clause with a different condition (single-value `Is` vs
 * multi-value `IsOneOf`) and can even leave more than one clause for the same key, this helper is
 * round-trip-robust: it gathers the values from ALL tag clauses for `key` (a single `Is` contributes its
 * one value; a multi clause contributes its whole list), toggles the clicked value across that set (adding
 * it when absent, removing it when present), and rebuilds a single `{category:'tag', field:key,
 * condition:IsOneOf, value:{values:[...]}}` clause. Toggling behaviour:
 *
 * - No clause for `key` yet → a fresh single-value `IsOneOf` clause is appended (order preserved).
 * - One or more clauses for `key` → they collapse into one clause at the position of the FIRST such
 *   clause, with the clicked value toggled; any further duplicate clauses for `key` are dropped.
 * - The toggle empties the value set → the clause is removed entirely (no empty `IsOneOf` lingers).
 *
 * All non-matching clauses (and their order) are preserved.
 *
 * @param clauses - The current clause list.
 * @param key - The tag key (the clause's `field`).
 * @param value - The tag value to toggle for `key`.
 * @returns A new clause list with `value` toggled for `key`.
 */
export function toggleTagValue(clauses: Clause[], key: string, value: string): Clause[] {
  const firstMatchIndex = clauses.findIndex((clause) => isTagClauseForKey(clause, key));
  // no clause for this key yet: append a fresh single-value IsOneOf clause, preserving order
  if (firstMatchIndex === -1) {
    const created: Clause = {
      category: TAG_CATEGORY,
      field: key,
      condition: ClauseCondition.IsOneOf,
      value: { values: [value] },
    };
    return [...clauses, created];
  }
  // gather the values from every tag clause for this key (single Is -> [value], multi -> values),
  // deduping so a value carried by several round-tripped clauses is counted once
  const values: string[] = [];
  for (const clause of clauses) {
    if (!isTagClauseForKey(clause, key)) {
      continue;
    }
    const clauseValues = ClauseIsMulti(clause) ? clause.value.values : [clause.value.value];
    for (const clauseValue of clauseValues) {
      if (!values.includes(clauseValue)) {
        values.push(clauseValue);
      }
    }
  }
  // toggle the clicked value across the merged set: remove it if present, otherwise add it
  const nextValues = values.includes(value) ? values.filter((existing) => existing !== value) : [...values, value];
  // rebuild the collapsed clause at the first match's position, dropping any later duplicate clauses for
  // the key; an empty value set removes the clause entirely
  const result: Clause[] = [];
  clauses.forEach((clause, index) => {
    if (!isTagClauseForKey(clause, key)) {
      result.push(clause);
      return;
    }
    // only the first match slot is (potentially) rewritten; later duplicates are dropped
    if (index === firstMatchIndex && nextValues.length > 0) {
      result.push({ category: TAG_CATEGORY, field: key, condition: ClauseCondition.IsOneOf, value: { values: nextValues } });
    }
  });
  return result;
}

/**
 * Return only the clauses to keep when the user resets the dashboard filters.
 *
 * A reset clears the user's downselecting filters (tag filters, `Include` kind whitelists, free text,
 * etc.) but keeps two structural clauses so the graph shape and default hidden tags survive: any `depth`
 * clause (the crawl depth) and the `hidden tags` clause (the default display-hidden tag keys). Clause
 * order among the kept clauses is preserved.
 *
 * @param clauses - The current clause list.
 * @returns The subset of `clauses` to retain after a reset.
 */
export function resetFilterClauses(clauses: Clause[]): Clause[] {
  return clauses.filter((clause) => clause.category === DEPTH_CATEGORY || clause.field === HIDDEN_TAGS_FIELD);
}
