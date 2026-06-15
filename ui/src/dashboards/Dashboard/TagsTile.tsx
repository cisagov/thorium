import React, { useCallback, useMemo, useRef, useState } from 'react';
import { FaChevronDown, FaChevronUp } from 'react-icons/fa6';

// spec: ./SPEC.md

// project imports
import { useVisibleNodes } from './nodes';
import { ALWAYS_HIDDEN_TAG_KEYS, collectTagCounts, type TagValueCount } from './tagCounts';
import { EmptyTags, TagChip, TagChipCount, TagGroup, TagGroupKey, TagGroupValues, TagValuesToggle } from './styles';
import { toggleTagValue } from './tagFilter';
import Collapsible, { TogglePosition } from '@components/shared/info/Collapsible';
import { BalancedColumns, BalanceStrategy } from '@components/shared/layout/BalancedColumns';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { ClauseCondition, ClauseIsMulti, type Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { getHiddenTagsFromClauses } from '@components/shared/inputs/omnibar/utils';
import { spacers } from '@styles';

/// Props for {@link TagsTile}.
export interface TagsTileProps {
  /**
   * The shared clause state — the same value/setter handed to the surrounding `EntityBrowserProvider`
   * (URL-backed). Kept as props (rather than read from context) so the tile, the omnibar strip, and the
   * stats-bar clicks all mutate one authoritative clause list.
   */
  clauses: Clause[];
  /** Setter for the shared {@link clauses}. */
  setClauses: (next: Clause[]) => void;
}

/// The tag keys that are always hidden even when no `hidden tags` clause is present.
const DEFAULT_HIDDEN_KEYS = ['Results', 'Parent', 'submitter'];

/// The minimum width (px) of a per-key tag tile; `BalancedColumns` derives the column count from this so
/// wide screens pack more (narrower) columns while each tile stays only as tall as its content. Tiles are
/// balanced tallest-first (`LongestFirst`) on their collapsed heights and the layout re-balances only on a
/// data change (`layoutKey`), so expanding one tile lengthens its own column in place instead of
/// reshuffling every tile.
const TAG_TILE_MIN_WIDTH = 200;

/// A stable empty value list shared as the fallback for a key with no counted values, so the memoized
/// {@link TagKeyTile} isn't handed a fresh `[]` (which would defeat its shallow-prop memoization).
const EMPTY_TAG_VALUES: TagValueCount[] = [];

/// The `maxPx` passed to the value `Collapsible`. The tile runs the `Collapsible` in controlled top-N mode
/// (truncation is by count, see {@link TAG_TILE_COLLAPSED_COUNT}), so this cap is unused for the tiles; it
/// is retained only as the component's required argument.
const TAG_TILE_VALUES_MAX_PX = 120;

/// The number of value chips mounted per tag tile while collapsed. The remaining values are not mounted
/// until the tile is expanded (via the caret), keeping the DOM small for high-cardinality keys on large
/// dashboards. Values are count-sorted with active (already-filtered) ones first, so the preview keeps the
/// most relevant values visible.
const TAG_TILE_COLLAPSED_COUNT = 12;

/// Render the caret-only toggle for a tag tile's value collapse: a down chevron while collapsed (expand)
/// and an up chevron while expanded (collapse). The visible glyph is icon-only, so the accessible name is
/// carried by the wrapper's `aria-label` while the icon is `aria-hidden`.
function tagValuesToggleLabel(collapsed: boolean): React.ReactNode {
  return (
    <TagValuesToggle aria-label={collapsed ? 'Show all values' : 'Show fewer values'}>
      {collapsed ? <FaChevronDown size={12} aria-hidden /> : <FaChevronUp size={12} aria-hidden />}
    </TagValuesToggle>
  );
}

/**
 * Precompute the set of active (already-filtered) tag values per key from the clause list.
 *
 * Building this once per clause change is O(clauses) total, versus the previous per-chip
 * `isTagValueActive` scan that was O(clauses) *per rendered chip* (and called twice per chip). Lookups
 * against the returned map are then O(1) while rendering.
 *
 * @param clauses - The current clause list.
 * @returns A map from tag key to the set of its active `IsOneOf` values.
 */
function activeTagValues(clauses: Clause[]): Map<string, Set<string>> {
  const active = new Map<string, Set<string>>();
  for (const clause of clauses) {
    if (clause.category === 'tag' && clause.condition === ClauseCondition.IsOneOf && ClauseIsMulti(clause)) {
      let values = active.get(clause.field);
      if (values === undefined) {
        values = new Set<string>();
        active.set(clause.field, values);
      }
      for (const value of clause.value.values) values.add(value);
    }
  }
  return active;
}

/// Props for {@link TagKeyTile}.
interface TagKeyTileProps {
  /// The tag key this tile represents.
  tagKey: string;
  /// The key's value+count list (descending count), as returned by `collectTagCounts`.
  values: TagValueCount[];
  /// The set of this key's active (already-filtered) values, or `undefined` when none are filtered.
  activeForKey?: Set<string>;
  /// Whether this tile's value list is collapsed (controlled by the parent so it survives remounts).
  collapsed: boolean;
  /// Toggle the tile's collapsed state; called with the tag key and the requested next collapsed value.
  onToggle: (tagKey: string, nextCollapsed: boolean) => void;
  /// Toggle a value into/out of the key's filter; called with the tag key and the clicked value.
  onChipClick: (tagKey: string, value: string) => void;
}

/**
 * A single tag-key tile: the key label plus its value chips, wrapped in a controlled top-N
 * {@link Collapsible}. Memoized so unrelated dashboard re-renders (e.g. a graph tick that leaves this
 * key's counts and active values untouched) don't rebuild its chip subtree — only a change to this
 * key's `values`, `activeForKey`, or `collapsed` re-renders it. The collapsed state is owned by the
 * parent (via {@link TagKeyTileProps.onToggle}) so it is preserved even if `BalancedColumns` re-parents
 * the tile into a different column.
 *
 * @param tagKey - The tag key this tile represents.
 * @param values - The key's value+count list (descending count).
 * @param activeForKey - The key's active (already-filtered) values, if any.
 * @param collapsed - Whether the value list is collapsed.
 * @param onToggle - Toggles the collapsed state (key, next collapsed).
 * @param onChipClick - Toggles a value filter (key, value).
 * @returns The tag-key tile.
 */
const TagKeyTile: React.FC<TagKeyTileProps> = React.memo(({ tagKey, values, activeForKey, collapsed, onToggle, onChipClick }) => {
  // order active (already-filtered) values first — each partition keeping its descending-count order — so
  // applied filters stay visible in the collapsed preview under the fade. Active-ness depends on the
  // clause-derived `activeForKey`, so it's partitioned here rather than baked into `collectTagCounts`
  const ordered = useMemo(() => {
    const active: TagValueCount[] = [];
    const inactive: TagValueCount[] = [];
    for (const entry of values) {
      (activeForKey?.has(entry.value) ? active : inactive).push(entry);
    }
    return [...active, ...inactive];
  }, [values, activeForKey]);
  return (
    <TagGroup>
      <TagGroupKey>{tagKey}</TagGroupKey>
      <Collapsible
        maxPx={TAG_TILE_VALUES_MAX_PX}
        hasMore={ordered.length > TAG_TILE_COLLAPSED_COUNT}
        collapsed={collapsed}
        onToggleCollapsed={(next) => onToggle(tagKey, next)}
        renderToggleLabel={tagValuesToggleLabel}
        // keep the caret pinned BELOW the chips in both states (not Adaptive, which jumps it to the top on
        // expand and shoves the already-shown chips down) so expanding a tile just appends the extra chips
        togglePosition={TogglePosition.Bottom}
      >
        {(isCollapsed) => (
          <TagGroupValues>
            {(isCollapsed ? ordered.slice(0, TAG_TILE_COLLAPSED_COUNT) : ordered).map(({ value, count }) => {
              const isActive = activeForKey?.has(value) ?? false;
              // tell the user a click filters the dashboard by this tag (and that an active chip removes it)
              const tip = isActive ? `Remove the ${tagKey}: ${value} filter` : `Filter the dashboard to items tagged ${tagKey}: ${value}`;
              return (
                <OverlayTipTop key={value} tip={tip}>
                  <TagChip type="button" $active={isActive} aria-pressed={isActive} onClick={() => onChipClick(tagKey, value)}>
                    {value}
                    <TagChipCount>({count})</TagChipCount>
                  </TagChip>
                </OverlayTipTop>
              );
            })}
          </TagGroupValues>
        )}
      </Collapsible>
    </TagGroup>
  );
});
TagKeyTile.displayName = 'TagKeyTile';

/**
 * The dashboard's tags tile: every tag present in the current (visible) node set, grouped by key, each
 * value a clickable chip showing its count.
 *
 * Counts over the entity browser's `visibleSet` when a filter is active (so the tile *downselects* — a
 * value that no longer appears in the filtered view drops out) and over the whole `data_map` otherwise.
 * The display-hidden tag keys (`getHiddenTagsFromClauses`, defaulting to `Results`/`Parent`/`submitter`)
 * are excluded, as are the always-hidden high-cardinality keys ({@link ALWAYS_HIDDEN_TAG_KEYS}) which are
 * dropped unconditionally. Clicking a chip toggles its value into the key's single is-one-of filter via
 * {@link toggleTagValue}; already-filtered values render active and sort **first** within their tile so they
 * stay visible when the tile is collapsed. Each tile's value chips are wrapped in a shared {@link Collapsible}
 * run in controlled top-N mode (default collapsed): only the first {@link TAG_TILE_COLLAPSED_COUNT} values
 * are mounted while collapsed and the rest are mounted on expand — so a high-cardinality key never mounts
 * thousands of chips just to clip them. A key with more values than the cap shows a caret-only toggle to
 * expand (revealing all values inline, since the tile does not scroll); a key with fewer shows no caret. The tally is
 * memoized on `[graphVersion, visibleSet, hiddenKeys]` so it recomputes only when the graph, the visible set,
 * or the hidden-key exclusion set change; the active-value lookup is precomputed once per clause change.
 *
 * @param clauses - The shared clause list (also given to the provider).
 * @param setClauses - Setter for the shared clause list.
 * @returns The tags tile.
 */
const TagsTile: React.FC<TagsTileProps> = ({ clauses, setClauses }) => {
  // node set = only the visible ids when a filter is active, otherwise every node in the graph (the same
  // downselect the stats charts use, via the shared useVisibleNodes hook)
  const nodes = useVisibleNodes();

  // the set of tag keys to exclude from the tally: the omnibar-derived hidden keys (or the default
  // fallback) plus the always-hidden high-cardinality keys. Derived in its own memo (on `clauses`) so the
  // count walk below recomputes only when the *hidden keys* change, not on every unrelated clause edit.
  const hiddenKeys = useMemo(() => {
    const fromClauses = getHiddenTagsFromClauses(clauses);
    return [...(fromClauses.length > 0 ? fromClauses : DEFAULT_HIDDEN_KEYS), ...ALWAYS_HIDDEN_TAG_KEYS];
  }, [clauses]);

  // nodes changes identity only when the graph's data_map or the visible set change; hiddenKeys drives the
  // exclusion set
  const tagCounts = useMemo(() => collectTagCounts(nodes, hiddenKeys), [nodes, hiddenKeys]);

  // active (already-filtered) values per key, computed once per clause change for O(1) chip lookups
  const activeValues = useMemo(() => activeTagValues(clauses), [clauses]);

  // which keys are expanded, lifted here (keyed by tag key) so a tile's expanded state survives a
  // `BalancedColumns` re-parent (a column move remounts the tile, which would otherwise reset a
  // collapse state living inside the tile). Absent from the layout key below, so expanding is not a
  // rebalance trigger.
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(() => new Set());
  const handleToggle = useCallback((key: string, nextCollapsed: boolean) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      // `nextCollapsed` is the requested state: collapsing drops the key, expanding adds it
      if (nextCollapsed) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);
  const handleChipClick = useCallback(
    (key: string, value: string) => {
      setClauses(toggleTagValue(clauses, key, value));
    },
    [clauses, setClauses],
  );

  // a monotonic key that changes only when the data driving tile *heights* changes — the counted values
  // (`tagCounts`) or the active-value ordering (`activeValues`). Both are already memoized on their exact
  // inputs, so a ref-identity change is the precise "data changed" signal. Comparing identities during
  // render (rather than a size/hash heuristic) avoids missing a same-count filter change. Expanding a tile
  // is deliberately NOT reflected here, so `BalancedColumns` holds its column assignment steady on expand.
  const layoutVersionRef = useRef(0);
  const prevLayoutInputsRef = useRef<{ counts: unknown; active: unknown } | null>(null);
  const prevLayoutInputs = prevLayoutInputsRef.current;
  if (prevLayoutInputs === null || prevLayoutInputs.counts !== tagCounts || prevLayoutInputs.active !== activeValues) {
    layoutVersionRef.current += 1;
    prevLayoutInputsRef.current = { counts: tagCounts, active: activeValues };
  }
  const layoutKey = layoutVersionRef.current;

  const keys = Array.from(tagCounts.keys()).sort((a, b) => a.localeCompare(b));

  if (keys.length === 0) {
    return <EmptyTags>No tags</EmptyTags>;
  }

  return (
    <BalancedColumns
      columnWidth={TAG_TILE_MIN_WIDTH}
      gap={spacers.two}
      balanceStrategy={BalanceStrategy.LongestFirst}
      layoutKey={layoutKey}
      items={keys.map((key) => (
        <TagKeyTile
          key={key}
          tagKey={key}
          values={tagCounts.get(key) ?? EMPTY_TAG_VALUES}
          activeForKey={activeValues.get(key)}
          collapsed={!expandedKeys.has(key)}
          onToggle={handleToggle}
          onChipClick={handleChipClick}
        />
      ))}
    />
  );
};

export default TagsTile;
