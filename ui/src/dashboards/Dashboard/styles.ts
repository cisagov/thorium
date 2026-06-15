import styled, { keyframes } from 'styled-components';

// project imports
import { scaling, spacers } from '@styles';

// spec: ./SPEC.md

/// A continuous rotation used to spin the Refresh icon while a refresh is in flight.
const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

/// Wraps the Refresh icon; spins it (1s linear loop) while `$spinning`, otherwise renders it static.
export const SpinningIcon = styled.span<{ $spinning: boolean }>`
  display: inline-flex;
  animation: ${({ $spinning }) => ($spinning ? spin : 'none')} 1s linear infinite;
`;

/**
 * The ultra-wide breakpoint at which the content row splits into two side-by-side columns.
 *
 * `scaling.fourxl` (2000px) is the plan's proposed ~2000px breakpoint (vs `xxxl` 1700 / `fivexl`
 * 2300); {@link useMediaQuery} in `Dashboard.tsx` reads the same value so the JS arrangement switch
 * and the CSS grid agree.
 */
export const ULTRA_WIDE_BREAKPOINT = scaling.fourxl;

/**
 * The dashboard page's vertical stack: stats, quick-action controls, and the omnibar strip across the top
 * (each sized to its content), then the content region.
 *
 * The content region differs by breakpoint and so is composed by {@link DashboardContent} rather than
 * baked into a fixed grid-row template: at/above the ultra-wide breakpoint it is a `BalancedColumns`
 * grid (browser tile fixed left, graph tile fixed right, remaining tiles flowed into the shorter
 * column); below it, a single-column {@link ContentRow} (tabbed tiles) followed by a full-width
 * {@link AnalysisRow}. A flex column keeps the top strips at their natural height while the content
 * region grows to fill the page.
 */
export const DashboardLayout = styled.div`
  display: flex;
  flex-direction: column;
  /* a touch more than spacers.four so stacked sections read as clearly separated panels on the
     dark/ocean themes, where adjacent panel backgrounds are otherwise low-contrast */
  gap: calc(${spacers.four} + 2px);
  padding-bottom: ${spacers.five};
`;

/// The full-width stats row wrapper (grid row 1).
export const StatsRow = styled.div`
  min-width: 0;
`;

/// The full-width quick-action controls row wrapper (grid row 2). Negative vertical margin trims the
/// stack gap so the floating controls sit closer to the stats above and the omnibar below.
export const ControlsRow = styled.div`
  min-width: 0;
  margin: -${spacers.three} 0;
`;

/// The full-width omnibar-strip row wrapper (grid row 3).
export const OmnibarRow = styled.div`
  min-width: 0;
`;

/// The full-width analysis-status row wrapper, rendered below the tabs in the narrow (tabbed) layout.
export const AnalysisRow = styled.div`
  min-width: 0;
`;

/**
 * The dashboard's quick-action controls bar: a horizontally-centered icon toolbar of quick-action buttons
 * (Grow, Reset filters, Refresh). The icons **float** — no panel background/border/card — matching the
 * entity browser's own floating controls (`HiddenNodesControl`/`FlaggedOnlyToggle`); it wraps so
 * additional actions flow onto a new line on narrow viewports.
 */
export const ControlsBar = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: ${spacers.three};
`;

/**
 * The dashboard's always-shown omnibar strip: the omnibar grows to fill the row while the hidden-nodes
 * and flagged-only controls sit beside it. Mirrors the entity browser's own `ToolbarBar` so the two
 * strips read identically, but is owned by the dashboard so it can live outside the browser body.
 */
export const OmnibarStrip = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: ${spacers.two};
  padding: ${spacers.two} ${spacers.three};
  background: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

/// Flex slot letting the omnibar grow to fill the strip while the toggles sit beside it.
export const OmnibarStripSlot = styled.div`
  flex: 1 1 320px;
  min-width: 220px;
`;

/**
 * A removable hidden-node chip in the omnibar strip: an eye-slash icon, the node's label (ellipsized when
 * long), and an X, reading as one "click to unhide" affordance. Matches the neutral chip chrome of the
 * strip's other controls; the whole chip is the button (its accessible name is `Unhide <label>`).
 */
export const HiddenChip = styled.button`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.two};
  max-width: 14rem;
  padding: ${spacers.one} ${spacers.three};
  border-radius: 12px;
  font-size: 0.8rem;
  cursor: pointer;
  background: var(--thorium-panel-bg);
  color: var(--thorium-secondary-text);
  border: 1px solid var(--thorium-panel-border);

  /* keep the label on one line and ellipsize it so a long name never stretches the strip */
  & > span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  &:hover {
    border-color: var(--thorium-highlight-panel-border);
    color: var(--thorium-text);
  }
`;

/**
 * A hidden-node tile rendered **inside** the omnibar entry field (as an `extraChips` element), styled to
 * match an `OmnibarClause` chip: an eye-slash logo, the node's (ellipsized) label, and an `×` that unhides
 * just that node — so hidden items read as first-class removable filter tiles alongside the clause chips.
 */
export const HiddenNodeTile = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 2px;
  max-width: 14rem;
  margin: 2px;
  padding: 2px 5px;
  border-radius: 5px;
  border: 1px solid var(--thorium-text);
  color: var(--thorium-text);
  /* mirror the OmnibarClause chip exactly: inherit the omnibar font (don't shrink it) and use content-box so
     the border sits outside the padding — otherwise these tiles render shorter with smaller text than the
     real clause chips beside them */
  box-sizing: content-box;
  letter-spacing: normal;

  & > span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
`;

/// The `×` unhide button inside a {@link HiddenNodeTile} (mirrors the omnibar clause delete button).
export const HiddenNodeDelete = styled.button`
  display: inline-flex;
  align-items: center;
  height: 100%;
  padding: 0 4px;
  border: 0;
  background-color: inherit;
  color: inherit;
  cursor: pointer;
`;

/**
 * The active re-root tile rendered **inside** the omnibar entry field (as an `extraChips` element): a gear
 * logo, the re-rooted node's (ellipsized) label, and an `×` that clears the re-root back to the natural roots.
 * Mirrors {@link HiddenNodeTile} but uses the highlight accent so the current view-root reads as distinct from
 * the hide tiles beside it. Reuses {@link HiddenNodeDelete} for its `×`.
 */
export const ReRootTile = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 2px;
  max-width: 14rem;
  margin: 2px;
  padding: 2px 5px;
  border-radius: 5px;
  border: 1px solid var(--thorium-highlight-text);
  color: var(--thorium-highlight-text);
  box-sizing: content-box;
  letter-spacing: normal;

  & > span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
`;

/// A labelled cluster of hidden-node chips that share one resource type, shown inline in the omnibar strip.
export const HiddenTypeGroup = styled.div`
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: ${spacers.two};
`;

/// The resource-type header preceding a hidden group's chips (e.g. "Files", "Devices").
export const HiddenTypeLabel = styled.span`
  color: var(--thorium-secondary-text);
  font-size: 0.75rem;
  font-weight: 600;
`;

/// The "Clear all" action beside the hidden groups that unhides every hidden node at once.
export const ClearHiddenButton = styled.button`
  padding: ${spacers.one} ${spacers.three};
  background: transparent;
  border: none;
  color: var(--thorium-link-text);
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;

  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

/**
 * The expandable "filters" area beneath the omnibar row, hosting the {@link TagsTile} inside a
 * `Collapsible`. Full-width so the tag groups wrap freely; a small top margin separates it from the
 * omnibar controls above.
 */
export const FiltersSection = styled.div`
  flex: 1 1 100%;
  min-width: 0;
  margin-top: ${spacers.two};
`;

/**
 * The filters expand/collapse toggle content: a chevron icon + "filters", horizontally centered and
 * vertically aligned. Slightly larger than the default `Collapsible` toggle text, and the chevron is a
 * sized icon (not a unicode glyph) so it matches the label's size and baseline.
 */
export const FiltersToggleLabel = styled.span`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.two};
  font-size: 0.88rem;
  line-height: 1;
`;

/// A single tag key's tile: a bordered/padded box with the key as its header and its value chips inside.
export const TagGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${spacers.two};
  padding: ${spacers.two} ${spacers.three};
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  min-width: 0;
`;

/// The tag key heading at the top of a tile, shown verbatim (keys are never uppercased).
export const TagGroupKey = styled.span`
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--thorium-secondary-text);
  overflow-wrap: anywhere;
`;

/// The wrapping row of value chips within a tag tile.
export const TagGroupValues = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${spacers.two};
`;

/**
 * The caret-only toggle for a tag tile's per-tile value collapse (inside `Collapsible`'s toggle button).
 * Icon-only — it carries the toggle's accessible name via `aria-label` while the icon itself is
 * `aria-hidden` — so the nameless icon still reads to assistive tech. Link-colored and vertically centered
 * to match the surrounding chrome.
 */
export const TagValuesToggle = styled.span`
  display: inline-flex;
  align-items: center;
  line-height: 1;
  color: var(--thorium-link-text);
`;

/**
 * A clickable tag-value chip showing the value and its count. `$active` marks a value already present in
 * the key's is-one-of filter (highlighted background + border), otherwise it reads as a neutral chip.
 */
export const TagChip = styled.button<{ $active: boolean }>`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.two};
  /* extra right padding so the parenthesized count is never clipped by the pill's rounded edge */
  padding: ${spacers.one} ${spacers.three} ${spacers.one} ${spacers.two};
  border-radius: 12px;
  font-size: 0.78rem;
  cursor: pointer;
  text-align: left;
  /* wrap long tag values at any character so a chip never overflows its key tile */
  max-width: 100%;
  min-width: 0;
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-all;
  background: ${({ $active }) => ($active ? 'var(--thorium-highlight-panel-bg)' : 'var(--thorium-panel-bg)')};
  color: var(--thorium-text);
  border: 1px solid ${({ $active }) => ($active ? 'var(--thorium-highlight-panel-border)' : 'var(--thorium-panel-border)')};

  &:hover {
    border-color: var(--thorium-highlight-panel-border);
  }
`;

/// The small parenthesized count shown after the value inside a {@link TagChip}. It never wraps (the
/// chip's break-all is for the value only) and stays a single, vertically-centered token beside the value.
export const TagChipCount = styled.span`
  flex: 0 0 auto;
  white-space: nowrap;
  word-break: normal;
  overflow-wrap: normal;
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--thorium-secondary-text);
`;

/// Empty-state message shown when the visible node set carries no (non-hidden) tags.
export const EmptyTags = styled.div`
  color: var(--thorium-secondary-text);
  font-size: 0.82rem;
`;

/**
 * The dashboard's content region: a stable N-column grid hosting the browser pane and the graph pane as
 * **fixed sibling grid items in a constant source order**, so switching arrangement never re-parents them
 * (preserving the browser's scroll/expansion state and the graph's WebGL canvas). Every arrangement is
 * expressed purely as CSS/props over this one tree:
 *
 * - **split** (ultra-wide, no pane focused): `$columns=2` — browser left, graph right.
 * - **expanded** (a pane focused via the ⤢ toggle): `$columns=1` — the focused pane fills, the other is
 *   `display:none` (still mounted, its `active` gated off so the graph pays no WebGL cost).
 * - **tabs** (narrow): `$columns=1` — only the active-tab pane is shown; the other is `display:none`.
 *
 * `minmax(0, 1fr)` tracks let a column shrink so the 3D canvas / browser table never force an overflow.
 */
export const ContentGrid = styled.div<{ $columns: number }>`
  display: grid;
  grid-template-columns: repeat(${({ $columns }) => $columns}, minmax(0, 1fr));
  align-items: start;
  /* keep the horizontal column gap at spacers.four, but give stacked tiles a touch more vertical
     separation (matching DashboardLayout) so single-column tiles read as distinct panels */
  column-gap: ${spacers.four};
  row-gap: calc(${spacers.four} + 2px);
  min-height: 0;
`;

/**
 * Horizontal-scroll safety valve wrapping the browser tree inside its pane: for a pathologically wide/deep
 * subtree that even the frozen indent can't keep within the pane, this scrolls sideways rather than crushing
 * rows. No vertical scrollbar appears because the pane isn't height-capped (the page grows), so the
 * `overflow-y` that `overflow-x` forces to `auto` never has capped content to scroll.
 */
export const PaneScroll = styled.div`
  min-width: 0;
  overflow-x: auto;
`;

/**
 * A content-tile wrapper (browser or graph). `$hidden` toggles visibility with `display: none` so a
 * tile stays mounted (and its state/canvas preserved) while its tab is inactive. `$order` sets its grid
 * `order` so the focused pane can sit on top when the region stacks into one column (expanded mode) without
 * re-parenting the tile (which would remount it).
 */
export const ContentTile = styled.div<{ $hidden?: boolean; $order?: number; $col?: string; $row?: string }>`
  display: ${({ $hidden }) => ($hidden ? 'none' : 'flex')};
  order: ${({ $order }) => $order ?? 0};
  grid-column: ${({ $col }) => $col ?? 'auto'};
  grid-row: ${({ $row }) => $row ?? 'auto'};
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  overflow: hidden;
`;

/**
 * The graph tile. By default it renders as a **square** (`aspect-ratio: 1 / 1`) deriving its height from the
 * width of its {@link ColWrap} column — the side-by-side (split) and tabbed layouts. When **`$fill`** is set
 * (the expanded, single-column stack) the square would be as tall as the full page width, so it instead takes a
 * viewport-bounded height (`70vh`) as a large landscape graph. A `min-height` floor keeps the canvas usable on
 * narrow viewports.
 *
 * The graph body (`AssociationGraph`'s own `GraphWindow` root) is stretched to `flex: 1` so it fills the tile
 * beneath {@link TileHeader}; the tile's `overflow: hidden` clips the graph's internally fixed-height canvas.
 */
export const GraphContentTile = styled(ContentTile)<{ $fill?: boolean }>`
  ${({ $fill }) => ($fill ? 'height: 70vh;' : 'aspect-ratio: 1 / 1;')}
  min-height: 320px;

  /* stretch the lazily-loaded graph root (and its Suspense fallback) to fill the tile under the header */
  & > div:last-child {
    flex: 1;
    min-height: 0;
  }
`;

/**
 * A content **column wrapper** (a grid item in {@link ContentGrid}) that flex-stacks its pane and — in the
 * split layout — the reactions tile beneath it. Because each column is its own flex stack, the reactions
 * tile sits **flush** under the (shorter) column's pane, unlike a shared grid row whose tracks align across
 * columns and leave a gap. Holds the browser pane in one column and the graph pane in the other, always —
 * so switching split ⇄ expanded ⇄ tabs never re-parents (remounts) them. `$order` puts the focused pane on
 * top when the region stacks into one column (expanded); `$hidden` collapses the column for the narrow tabs
 * layout (kept mounted).
 */
export const ColWrap = styled.div<{ $hidden?: boolean; $order?: number }>`
  display: ${({ $hidden }) => ($hidden ? 'none' : 'flex')};
  flex-direction: column;
  /* match the dashboard's vertical section gap so a tile and the reactions tile stacked beneath it read
     as clearly separated panels on the dark/ocean themes */
  gap: calc(${spacers.four} + 2px);
  min-width: 0;
  min-height: 0;
  order: ${({ $order }) => $order ?? 0};
`;

/// Header strip shown at the top of a stats/content tile. A flex box that vertically centers its single
/// title child, so the tight vertical padding still reads as balanced rather than top-heavy.
export const TileHeader = styled.div`
  display: flex;
  align-items: center;
  padding: ${spacers.two} ${spacers.four};
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--thorium-text);
  border-bottom: 1px solid var(--thorium-panel-border);
  background: var(--thorium-secondary-panel-bg);

  /* let the single header child (the title row / seed summary) fill the width so its own
     space-between / wrapping layout is preserved while this flex box centers it vertically */
  & > * {
    flex: 1;
    min-width: 0;
  }
`;

/**
 * A tile header laid out as a row: the title on the left and an actions cluster (e.g. a refresh
 * button) pushed to the right. Used by tiles that need a header-level control alongside the title.
 */
export const TileHeaderRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${spacers.three};
`;

/**
 * The right-hand actions cluster of a {@link TileHeaderRow}: keeps header-level controls (e.g. the entity
 * sort/group controls) grouped together immediately before the pane expand toggle, rather than being spread
 * apart by the row's `space-between`.
 */
export const TileHeaderActions = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.three};
`;

/// The stats tile container (a panel that always spans the full stats row).
export const StatsTile = styled.div`
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  overflow: hidden;
`;

/// The stats tile body: hosts one bar-chart cluster per series.
export const StatsBody = styled.div`
  padding: ${spacers.three} ${spacers.four};
`;

/// A lightweight, non-blocking "updating…" indicator shown while data is loading.
export const UpdatingIndicator = styled.span`
  font-size: 0.75rem;
  font-style: italic;
  color: var(--thorium-secondary-text);
`;

/**
 * The stats tile header rendered as a wrapping row: the "Seeded by" prefix, the seed-item chips, and the
 * "updating…" indicator flow together and wrap onto new lines when the summary is long.
 */
export const SeedSummaryHeader = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: ${spacers.two};
`;

/// The muted "Seeded by" prefix shown before the seed-item chips in the stats header.
export const SeedSummaryPrefix = styled.span`
  color: var(--thorium-secondary-text);
  font-weight: 600;
`;

/**
 * A single seed-item chip in the stats header, showing one human-readable seed resource (file/entity/repo
 * name or `key: value` tag). A compact, non-interactive pill matching the tags-tile chip styling.
 */
export const SeedSummaryChip = styled.span`
  display: inline-flex;
  align-items: center;
  max-width: 20rem;
  padding: ${spacers.one} ${spacers.three};
  border-radius: 999px;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  color: var(--thorium-text);
  font-size: 0.78rem;
  font-weight: 400;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

/// Placeholder shown when there are no stats to render (empty graph).
export const EmptyStats = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 80px;
  color: var(--thorium-secondary-text);
  font-size: 0.9rem;
`;

/// Centered container for the lazy-graph loading fallback.
export const GraphFallbackContainer = styled.div`
  display: flex;
  flex: 1;
  align-items: center;
  justify-content: center;
  min-height: 200px;
`;

/// The Analysis Status section container (a full-width panel below the content row).
export const AnalysisTile = styled.div`
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  overflow: hidden;
`;

/// The Analysis Status body: padding around the summary, table, and state banners.
export const AnalysisBody = styled.div`
  padding: ${spacers.three} ${spacers.four};
`;

/// The compact status-count summary row shown above the reactions table.
export const AnalysisSummary = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: ${spacers.three};
  margin-bottom: ${spacers.three};
  color: var(--thorium-secondary-text);
  font-size: 0.85rem;
`;

/**
 * A clickable per-status count chip that toggles the table's status filter. `$active` outlines the
 * currently-selected status; unselected chips are borderless so the row reads as a set of toggles.
 */
export const AnalysisFilterChip = styled.button<{ $active: boolean }>`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.two};
  padding: ${spacers.one} ${spacers.two};
  border: 1px solid ${({ $active }) => ($active ? 'var(--thorium-highlight-panel-border)' : 'transparent')};
  border-radius: 999px;
  background: ${({ $active }) => ($active ? 'var(--thorium-highlight-panel-bg)' : 'transparent')};
  color: inherit;
  font: inherit;
  cursor: pointer;

  &:hover {
    border-color: var(--thorium-highlight-panel-border);
  }
`;

/**
 * A max-height scroll container around the reactions table so a large fetched set (hundreds of rows once
 * "Load more" has run several times) stays bounded in the layout instead of stretching the tile arbitrarily
 * tall. The table's `thead` sticks to the top of the scroll area so column headers stay visible.
 */
export const AnalysisTableScroll = styled.div`
  max-height: 60vh;
  overflow-y: auto;

  thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--thorium-panel-bg);
  }
`;

/// The reactions table (Pipeline / File / Group / Status).
export const AnalysisTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;

  th,
  td {
    text-align: left;
    padding: ${spacers.two} ${spacers.three};
    border-bottom: 1px solid var(--thorium-panel-border);
    vertical-align: top;
  }

  th {
    color: var(--thorium-secondary-text);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.7rem;
  }

  /* the final row is the list's end — no divider beneath it */
  tbody tr:last-child td {
    border-bottom: none;
  }

  td a {
    color: var(--thorium-link-text);
    font-family: monospace;
  }
`;

/**
 * A clickable column header that sorts the table by its column. Resets native button chrome to inherit
 * the `th`'s uppercase label styling, adds a pointer cursor, and keeps the label and sort caret on one
 * line without wrapping/selecting on repeated clicks.
 */
export const SortHeaderButton = styled.button`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.two};
  padding: 0;
  border: none;
  background: none;
  color: inherit;
  font: inherit;
  letter-spacing: inherit;
  text-transform: inherit;
  cursor: pointer;
  user-select: none;
`;

/// The ascending/descending caret shown on the active sort column header.
export const SortCaret = styled.span`
  font-size: 0.7rem;
  color: var(--thorium-highlight-text);
`;

/**
 * A small pill showing a reaction status, colored by outcome. Uses CSS variables per status keyword so a
 * single styled component covers every status without a stylesheet explosion.
 */
export const StatusPill = styled.span<{ $bg: string; $fg: string }>`
  display: inline-block;
  padding: ${spacers.one} ${spacers.three};
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  background: ${({ $bg }) => $bg};
  color: ${({ $fg }) => $fg};
`;

/// The File cell's link cluster: wraps each sample sha256 link with a small gap so a multi-sample
/// reaction's links read as distinct tokens instead of one run-together monospace string.
export const FileLinks = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${spacers.one} ${spacers.three};
`;

/**
 * The footer row for the analysis fan-out: a three-cell grid so the "Load more" action is centered in
 * the row while the "Showing N of M files" count sits left-justified on the same line (the empty third
 * cell balances the centered middle cell).
 */
export const AnalysisFooter = styled.div`
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  margin-top: ${spacers.three};
  color: var(--thorium-secondary-text);
  font-size: 0.8rem;
`;
