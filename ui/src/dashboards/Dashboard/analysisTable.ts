// spec: ./SPEC.md

// project imports
import { Reaction } from '@models/reactions';

/**
 * The sortable columns of the Analysis Status table.
 *
 * A string enum (not a union) so the panel's header map and the sort comparator get exhaustiveness
 * checking. The values are stable keys used as React keys and comparison discriminants.
 */
export enum AnalysisSortColumn {
  /// The reaction's pipeline name.
  Pipeline = 'pipeline',
  /// The reaction's file (its first sample sha256).
  File = 'file',
  /// The reaction's group.
  Group = 'group',
  /// The reaction's status.
  Status = 'status',
}

/**
 * The direction a sort is applied in.
 */
export enum SortDirection {
  /// Ascending (A→Z, 0→9).
  Asc = 'asc',
  /// Descending (Z→A, 9→0).
  Desc = 'desc',
}

/**
 * The default multi-key sort priority: file hash, then pipeline, then status.
 *
 * Applied when the user has not selected a primary column, and as the tiebreaker order after any
 * user-selected primary column.
 */
export const DEFAULT_SORT_PRIORITY: AnalysisSortColumn[] = [
  AnalysisSortColumn.File,
  AnalysisSortColumn.Pipeline,
  AnalysisSortColumn.Status,
];

/**
 * Extract the string a reaction sorts by for a given column.
 *
 * Lower-cased so sorting is case-insensitive; the File column sorts by the reaction's first sample
 * sha256 (the table renders each reaction's samples, and reactions in this view are file-scoped).
 *
 * # Arguments
 *
 * * `reaction` - The reaction to read the sort value from.
 * * `column` - The column whose value to extract.
 *
 * @returns The comparable, lower-cased sort value.
 */
export function reactionSortValue(reaction: Reaction, column: AnalysisSortColumn): string {
  switch (column) {
    case AnalysisSortColumn.Pipeline:
      return reaction.pipeline.toLowerCase();
    case AnalysisSortColumn.File:
      return (reaction.samples[0] ?? '').toLowerCase();
    case AnalysisSortColumn.Group:
      return reaction.group.toLowerCase();
    case AnalysisSortColumn.Status:
      return reaction.status.toLowerCase();
  }
}

/**
 * Sort reactions by a user-selected primary column, then by the default priority order.
 *
 * When `primary` is null the rows follow {@link DEFAULT_SORT_PRIORITY} (file → pipeline → status), all
 * ascending. When the user selects a primary column it becomes the first sort key (honoring
 * `direction`), and the remaining default-priority columns tiebreak after it, ascending — so the
 * user's choice leads and the standard ordering fills in beneath it. Pure and total (returns a new
 * array, never mutates its input) so it can be unit-tested directly.
 *
 * # Arguments
 *
 * * `reactions` - The reactions to sort.
 * * `primary` - The user-selected primary sort column, or null for the default order.
 * * `direction` - The direction to apply to the primary column (ignored when `primary` is null).
 *
 * @returns A new, sorted array of reactions.
 */
export function sortReactions(reactions: Reaction[], primary: AnalysisSortColumn | null, direction: SortDirection): Reaction[] {
  // the primary column leads, then the default priority columns tiebreak beneath it (deduped)
  const order = primary ? [primary, ...DEFAULT_SORT_PRIORITY.filter((column) => column !== primary)] : [...DEFAULT_SORT_PRIORITY];
  const sorted = [...reactions];
  sorted.sort((a, b) => {
    for (let i = 0; i < order.length; i += 1) {
      const column = order[i];
      // only the user-selected primary key honors the chosen direction; tiebreakers stay ascending
      const columnDirection = i === 0 && primary ? direction : SortDirection.Asc;
      const comparison = reactionSortValue(a, column).localeCompare(reactionSortValue(b, column));
      if (comparison !== 0) {
        return columnDirection === SortDirection.Asc ? comparison : -comparison;
      }
    }
    return 0;
  });
  return sorted;
}
