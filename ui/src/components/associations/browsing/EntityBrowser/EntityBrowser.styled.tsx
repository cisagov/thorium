import styled, { css, keyframes } from 'styled-components';

// project imports
import { ExpandToggle } from '@components/shared/buttons/ExpandToggle';
import { BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

/** Tiny inline spinner shown on a row while its node is being grown. */
export const RowSpinner = styled.span`
  flex: 0 0 auto;
  width: 12px;
  height: 12px;
  border: 2px solid var(--thorium-panel-border);
  border-top-color: var(--thorium-highlight-text);
  border-radius: 50%;
  animation: ${spin} 0.7s linear infinite;
`;

/** Outer container for the whole browser. */
export const BrowserRoot = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  /* horizontal inset applied once here (so the header and every row share it and the rows don't butt against
     the tile edge) plus a small end gap so the last tree item doesn't sit flush against the container bottom */
  padding: 0 12px 10px;
  /* establish a size container so nested Level indentation adapts to the tile's OWN width (narrow dashboard
     column vs. full-width tab vs. expanded) rather than the viewport — see the container queries on Level */
  container-type: inline-size;
  container-name: entitybrowser;
`;

// --- toolbar ---

export const ToolbarBar = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  background: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

/** Flex slot that lets the omnibar grow to fill the toolbar while the Flagged toggle sits beside it. */
export const OmnibarSlot = styled.div`
  flex: 1 1 320px;
  min-width: 220px;
`;

/**
 * A pill toggle. When active it fills with its tone's color: `danger` (default — the red used by the "Flagged
 * Only" toggle) or `accent` (the per-theme highlight color, for neutral on/off toggles like "Group by Type").
 */
export const ToggleChip = styled.button<{ $active?: boolean; $tone?: 'danger' | 'accent' }>`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  font-size: 0.8rem;
  font-weight: 600;
  border-radius: 12px;
  cursor: pointer;
  color: ${({ $active }) => ($active ? 'var(--thorium-button-text)' : 'var(--thorium-secondary-text)')};
  ${({ $active, $tone }) => {
    // active fill picks the tone's color; inactive is the plain panel chip regardless of tone
    const activeColor = $tone === 'accent' ? 'var(--thorium-highlight-text)' : 'var(--thorium-danger-bg)';
    return css`
      background: ${$active ? activeColor : 'var(--thorium-panel-bg)'};
      border: 1px solid ${$active ? activeColor : 'var(--thorium-panel-border)'};
    `;
  }}

  &:hover {
    border-color: var(--thorium-highlight-panel-border);
  }
`;

/**
 * The browser's own header row (distinct from the omnibar filters line): holds the sort/group controls,
 * right-aligned, directly above the entity list. The horizontal inset comes from `BrowserRoot` (shared with the
 * rows); the top padding and bottom margin keep the controls off the section above and give a clear gap before
 * the first entity.
 */
export const BrowserHeader = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  padding: 8px 0 0;
  /* clear separation between the controls and the first entity below (adds to BrowserRoot's row gap) */
  margin-bottom: 8px;
`;

/** Inline cluster holding the sort-mode dropdown and the group-by toggle. */
export const SortControls = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
`;

/** Muted lead-in label ("Sort") sitting before the sort dropdown. */
export const SortLabel = styled.span`
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--thorium-secondary-text);
`;

/** Compact pill-shaped dropdown for choosing the flag-stat sort mode. */
export const SortSelect = styled.select`
  padding: 4px 8px;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--thorium-text);
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 12px;
  cursor: pointer;

  &:hover {
    border-color: var(--thorium-highlight-panel-border);
  }
`;

/** Positioning context for the hidden-nodes chip + its dropdown. */
export const HiddenControl = styled.div`
  position: relative;
  display: inline-flex;
`;

/** Popover listing hidden node labels for per-item unhide, anchored under the chip. */
export const HiddenMenu = styled.div`
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  z-index: 20;
  min-width: 220px;
  max-width: 320px;
  max-height: 280px;
  overflow-y: auto;
  padding: 4px;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
`;

/** A row within the hidden-nodes menu: the node label plus an unhide button. */
export const HiddenMenuItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 4px 6px;
  font-size: 0.8rem;
  color: var(--thorium-text);
  border-radius: 6px;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

/** The label span in a hidden-menu row (ellipsized). */
export const HiddenMenuLabel = styled.span`
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

/** A small text button used for per-item unhide and clear-all inside the hidden control. */
export const HiddenMenuAction = styled.button`
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: transparent;
  border: none;
  color: var(--thorium-highlight-text);
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 6px;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }
  &:focus-visible {
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
`;

/** A dividing header row inside the hidden-nodes menu (clear-all lives here). */
export const HiddenMenuHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 4px 6px;
  border-bottom: 1px solid var(--thorium-panel-border);
  margin-bottom: 4px;
  color: var(--thorium-secondary-text);
  font-size: 0.78rem;
  font-weight: 700;
`;

/** Slot that holds the hover/focus-revealed hide affordance inside the header's trailing rail. */
export const HideSlot = styled.span`
  flex: 0 0 auto;
  display: inline-flex;
`;

/** Hover/focus-revealed "hide this item" affordance in a row header (eye-slash). */
export const HideButton = styled.button`
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--thorium-secondary-text);
  cursor: pointer;
  /* revealed only on row hover / keyboard focus-within (see RowHeader) */
  opacity: 0;
  transition: opacity 0.12s ease;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }
  &:focus-visible {
    opacity: 1;
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
`;

/** Hover/focus-revealed "focus this subtree" (re-root) affordance in a row header. Mirrors {@link HideButton}. */
export const FocusButton = styled.button`
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--thorium-secondary-text);
  cursor: pointer;
  /* revealed only on row hover / keyboard focus-within (see RowHeader) */
  opacity: 0;
  transition: opacity 0.12s ease;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }
  &:focus-visible {
    opacity: 1;
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
`;

/** Hover/focus-revealed "re-root the view here" (gear) affordance in a row header. Mirrors {@link FocusButton}. */
export const ReRootButton = styled.button`
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--thorium-secondary-text);
  cursor: pointer;
  /* revealed only on row hover / keyboard focus-within (see RowHeader) */
  opacity: 0;
  transition: opacity 0.12s ease;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }
  &:focus-visible {
    opacity: 1;
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
`;

/**
 * The always-visible depth/focus pill shown once a row's nesting passes {@link INDENT_CAP} (where indentation
 * freezes and no longer conveys depth). It reports the depth **and** acts as the re-root trigger — clicking it
 * focuses the tree on that subtree — so the automatic indent cap and the user's focus action are one affordance.
 */
export const DepthPill = styled.button`
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 0.68rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  padding: 0 6px;
  height: 1.1rem;
  border-radius: 8px;
  background: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  color: var(--thorium-secondary-text);
  cursor: pointer;

  &:hover {
    border-color: var(--thorium-highlight-panel-border);
    color: var(--thorium-text);
  }
  &:focus-visible {
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
`;

/**
 * The focus breadcrumb bar shown above the tree while re-rooted: a clickable trail from "All" (clears the
 * focus) down through the ancestors to the current focus root, so the user can pop back out one level at a
 * time. Wraps on narrow tiles.
 */
export const FocusBar = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  font-size: 0.8rem;
  background: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

/** A clickable crumb in the {@link FocusBar} (re-roots at that ancestor, or clears focus for the "All" crumb). */
export const Crumb = styled.button`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  max-width: 14rem;
  padding: 1px 6px;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--thorium-link-text);
  font: inherit;
  cursor: pointer;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-highlight-text);
  }
  &:focus-visible {
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
`;

/** The current (last) crumb in the {@link FocusBar}: the focus root itself, shown bold and non-interactive. */
export const CurrentCrumb = styled.span`
  max-width: 18rem;
  padding: 1px 6px;
  font-weight: 700;
  color: var(--thorium-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

/** The `›` separator between focus crumbs. */
export const CrumbSep = styled.span`
  color: var(--thorium-secondary-text);
`;

// --- tree levels / rows ---

/**
 * The nesting depth past which per-level indentation is **frozen**: levels deeper than this add only a
 * hairline {@link FROZEN_INDENT} step (keeping the guide-rule visible) instead of a full indent step, so an
 * arbitrarily deep tree can never march its rows off the right edge. Exported so the row can surface the
 * depth/focus affordance (re-root) at exactly the depth where indentation stops conveying nesting.
 */
export const INDENT_CAP = 6;

/** The minimal per-level indent applied past {@link INDENT_CAP} — just enough to keep nested guide-rules apart. */
const FROZEN_INDENT = '4px';

/**
 * The per-level indent for a level at `$depth`: nothing at the root, one adaptive `--indent-step` for the
 * first {@link INDENT_CAP} levels, then a frozen hairline step. The step itself is set by the container
 * queries below off the browser tile's own width, so a narrow column compresses the indent automatically.
 */
function levelIndent($depth: number): string {
  if ($depth <= 0) return '0';
  return $depth <= INDENT_CAP ? 'var(--indent-step)' : FROZEN_INDENT;
}

/** A nested level; the left guide-rule keeps deep DAGs readable. */
export const Level = styled.div<{ $depth: number }>`
  display: flex;
  flex-direction: column;
  /* very small margin between listed entities to keep the tree compact */
  gap: 2px;
  /* per-level indent step, adapted to the browser tile's own width (container queries below) and frozen past
     INDENT_CAP so deep rows keep usable header width; the guide-rule is retained at every non-root level */
  --indent-step: 10px;
  @container entitybrowser (max-width: 480px) {
    & {
      --indent-step: 6px;
    }
  }
  @container entitybrowser (min-width: 900px) {
    & {
      --indent-step: 14px;
    }
  }
  margin-left: ${({ $depth }) => levelIndent($depth)};
  padding-left: ${({ $depth }) => levelIndent($depth)};
  border-left: ${({ $depth }) => ($depth > 0 ? '1px solid var(--thorium-panel-border)' : 'none')};
`;

/**
 * A hover/focus-revealed icon button on a layer (kind-group) header — used for both the "collapse this
 * subsection" and "hide this whole kind" affordances (grouped in {@link GroupHeaderActions}).
 */
export const GroupHeaderButton = styled.button`
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--thorium-secondary-text);
  cursor: pointer;
  /* revealed only on group-header hover / keyboard focus-within (see GroupHeaderRow) */
  opacity: 0;
  transition: opacity 0.12s ease;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }
  &:focus-visible {
    opacity: 1;
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
`;

/** Right-aligned cluster of a layer header's action buttons (collapse subsection, hide kind). */
export const GroupHeaderActions = styled.span`
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 2px;
`;

export const GroupHeaderRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  color: var(--thorium-secondary-text);
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;

  /* reveal the header action buttons when the group header is hovered or has keyboard focus */
  &:hover ${GroupHeaderButton}, &:focus-within ${GroupHeaderButton} {
    opacity: 1;
  }
`;

export const GroupCount = styled.span`
  font-variant-numeric: tabular-nums;
  color: var(--thorium-secondary-text);
`;

/** Wraps a row's header + expanded body so they group as a single flex item within a level. */
export const RowContainer = styled.div`
  display: flex;
  flex-direction: column;
  /* same small gap between a node's info box and its nested children level as between sibling entities */
  gap: 2px;
`;

/** A single entity/file/repo "info box": header + metadata preview grouped in one bordered card. */
export const InfoBox = styled.div`
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  background: var(--thorium-panel-bg);
  overflow: hidden;
`;

/**
 * The row header row: a fixed {@link Chevron} rail, a growing {@link HeaderLead} (identity + badges that
 * wrap as a unit), and a fixed {@link HeaderTrail} (spinner + hide). `align-items: flex-start` keeps the
 * chevron and trailing rail aligned to the FIRST line even when the lead wraps its badges onto later lines.
 */
export const RowHeader = styled.div`
  /* INVARIANT: this component must take NO dynamic ($-prefixed) props. The duplicate-group highlight class is
     toggled on this node imperatively (see duplicateHighlight.ts); if styled-components had to emit a
     prop-derived dynamic class, React would rewrite className on re-render and silently wipe that highlight. */
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 10px;
  cursor: pointer;
  color: var(--thorium-text);

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
  &:focus-visible {
    outline: 2px solid var(--thorium-highlight-text);
    outline-offset: -2px;
  }
  /* every visible occurrence of the hovered/pinned duplicate node lights up together. Declared after :hover so
     it stays visible on hover; inset outline avoids being clipped by InfoBox's overflow:hidden + radius. */
  &.duplicate-highlight {
    outline: 2px dashed var(--thorium-warning-bar-bg);
    outline-offset: -2px;
    background: color-mix(in srgb, var(--thorium-warning-bar-bg) 15%, transparent);
  }
  &.duplicate-highlight:hover {
    background: color-mix(in srgb, var(--thorium-warning-bar-bg) 24%, transparent);
  }
  /* reveal the hide + focus + re-root affordances when the row is hovered or anything inside it has keyboard focus */
  &:hover
    ${HideButton},
    &:focus-within
    ${HideButton},
    &:hover
    ${FocusButton},
    &:focus-within
    ${FocusButton},
    &:hover
    ${ReRootButton},
    &:focus-within
    ${ReRootButton} {
    opacity: 1;
  }
`;

/** The shared height of the header's first line, so the chevron / trailing rail center-align to the name row. */
const HEADER_LINE = '1.4rem';

export const Chevron = styled.span<{ $expanded: boolean; $hidden?: boolean }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 12px;
  height: ${HEADER_LINE};
  flex: 0 0 auto;
  color: var(--thorium-secondary-text);
  visibility: ${({ $hidden }) => ($hidden ? 'hidden' : 'visible')};
  transform: rotate(${({ $expanded }) => ($expanded ? '90deg' : '0deg')});
  transition: transform 0.15s ease;
`;

/**
 * The header's growing region: the identity (icon + name) followed by the structural badges and tag chips.
 * A wrapping flex row so, as the row narrows, the whole badge/tag cluster **floats onto the next line under
 * the name** rather than crushing the name — the name keeps the first line, badges/tags flow after it. The
 * row/column gaps are tight (2px vertical, 6px horizontal) so wrapped lines stay compact.
 */
export const HeaderLead = styled.div`
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 2px 6px;
`;

/**
 * One flow holding both the structural {@link BadgeGroup} and the descriptive {@link TagBadgeGroup}, so every
 * chip — badges and tags alike — sits at the same tight gap. Without this, the two groups were direct
 * {@link HeaderLead} items and the boundary between them (e.g. a flag-count badge and the first tag chip)
 * inherited HeaderLead's wider name↔cluster column-gap, reading as an odd gap in the middle of the chip row.
 * Each group still wraps as its own unit; HeaderLead's larger gap now only separates the name from this cluster.
 */
export const HeaderChips = styled.span`
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: ${BUTTON_BAR_GAP};
  min-width: 0;
`;

/**
 * The identity cluster (type icon + name) kept together as one wrap-unit so the icon never detaches from the
 * name. `flex: 0 1 auto` lets it take only the width its (ellipsized) name needs — no longer *growing* to eat
 * the row and stranding the badges far to the right (the previous behavior); `min-width: 0` lets the name
 * ellipsize under pressure.
 */
export const HeaderIdentity = styled.span`
  flex: 0 1 auto;
  min-width: 0;
  min-height: ${HEADER_LINE};
  display: inline-flex;
  align-items: center;
  gap: 6px;
`;

/** The header's fixed trailing rail (grow spinner + hide affordance), aligned to the first line. */
export const HeaderTrail = styled.span`
  flex: 0 0 auto;
  min-height: ${HEADER_LINE};
  display: inline-flex;
  align-items: center;
  gap: 6px;
`;

/* Shared name styling: the name takes only the width it needs (no flex-grow), ellipsizing under pressure with
   a small floor so badges/tags sit right after it and wrap beneath it rather than the name eating the row. */
const identifierBase = css`
  flex: 0 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
`;

export const IdentifierLink = styled.a`
  ${identifierBase}
  color: var(--thorium-link-text);
  text-decoration: none;

  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

export const IdentifierText = styled.span`
  ${identifierBase}
  color: var(--thorium-text);
`;

/** Groups the header badges with the shared button-bar/badge gap between them (tighter than the header gap). */
// `display: contents` (not its own flex box): the badges become direct flex items of the parent HeaderChips so
// every chip — badges and tags — shares HeaderChips' single wrapping flow at one uniform gap. A nested flex box
// here would keep its width at its widest line while a short last line (e.g. ending in the flag-count badge)
// left trailing space before the next group, reading as an odd gap mid-row.
export const BadgeGroup = styled.span`
  display: contents;
`;

/**
 * Groups the header's descriptive tag chips. Sits after {@link BadgeGroup} in {@link HeaderLead} and wraps
 * with it, so tags flow right after the structural badges and drop to the next line together under pressure.
 */
// `display: contents` (see BadgeGroup): the tag chips join the parent HeaderChips flow directly, so the
// boundary between the last structural badge and the first tag chip carries the same uniform gap as the rest —
// no separate flex box that can leave trailing space when it wraps.
export const TagBadgeGroup = styled.span`
  display: contents;
`;

/**
 * A single descriptive tag chip (`key: value`) in the row header. Deliberately lighter than the structural
 * {@link BaseBadge} (dashed, transparent) so tags read as metadata rather than competing with the kind /
 * relationship badges. Capped in width and ellipsized so one long value can't stretch the header.
 */
export const TagBadge = styled.span`
  flex: 0 1 auto;
  min-width: 0;
  max-width: 100%;
  display: inline-flex;
  align-items: baseline;
  gap: 3px;
  font-size: 0.7rem;
  font-weight: 500;
  padding: 1px 6px;
  border-radius: 10px;
  border: 1px dashed var(--thorium-panel-border);
  color: var(--thorium-secondary-text);
  /* show the full value, wrapping at any character, so long tag values (e.g. a process command line or
     image path) read in full instead of truncating to an ellipsis */
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
`;

/** The key portion of a {@link TagBadge}, de-emphasized so the value stands out. */
export const TagBadgeKey = styled.span`
  flex: 0 0 auto;
  opacity: 0.75;
`;

/** The value portion of a {@link TagBadge}; wraps at any character so a long value shows in full. */
export const TagBadgeValue = styled.span`
  min-width: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
  color: var(--thorium-text);
`;

/** The `+N` chip shown when a node has more tags than the header cap; its title lists the overflowed tags. */
export const TagOverflowBadge = styled.span`
  flex: 0 0 auto;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 10px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-secondary-text);
  cursor: default;
`;

const BaseBadge = styled.span`
  flex: 0 0 auto;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 1px 7px;
  border-radius: 10px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-secondary-text);
`;

export const KindBadge = styled(BaseBadge)``;

export const RelationshipBadge = styled(BaseBadge)`
  background: var(--thorium-highlight-panel-bg);
  color: var(--thorium-highlight-text);
  // "… In <container> <Kind>" labels can get long — cap and ellipsize (full text in the title attr);
  // the percentage bound keeps the badge from eating narrow (e.g. two-column dashboard) headers
  max-width: min(20rem, 55%);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

export const ViaBadge = styled(BaseBadge)`
  background: transparent;
  border: 1px dashed var(--thorium-panel-border);
  font-style: italic;
  font-weight: 500;
  /* breadcrumbs grow with depth — cap and ellipsize (full path stays in the title attr) */
  max-width: 14rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

/**
 * The "Duplicate ·N" badge. Rendered as a `<button>` (via `as="button"`) because it is interactive: hovering it
 * highlights every visible occurrence of the same node, and clicking pins that highlight and jumps to the next
 * occurrence. `N` is the ephemeral correlation number ({@link DuplicateGroupNumber}).
 */
export const DuplicateBadge = styled(BaseBadge)`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  /* reset the inherited button chrome so it reads as a badge, not a form control */
  border: none;
  font: inherit;
  font-size: 0.72rem;
  font-weight: 600;
  text-align: inherit;
  cursor: pointer;
  background: var(--thorium-warning-secondary-bg);
  color: var(--thorium-warning-text);
  &:focus-visible {
    outline: 2px solid var(--thorium-warning-text);
    outline-offset: 1px;
  }
`;

/** The ephemeral correlation number inside a {@link DuplicateBadge} (e.g. the `·3` in "Duplicate ·3"). */
export const DuplicateGroupNumber = styled.span`
  font-variant-numeric: tabular-nums;
  opacity: 0.85;
`;

/** Indicates a node has more associations that can be fetched (grown) from the server. */
export const GrowBadge = styled(BaseBadge)`
  display: inline-flex;
  align-items: center;
  gap: 3px;
  background: var(--thorium-ok-bg);
  color: var(--thorium-button-text);
`;

/** Small aggregate badge on a layer header (e.g. danger / ATT&CK counts). */
export const AggBadge = styled(BaseBadge)<{ $danger?: boolean }>`
  background: ${({ $danger }) => ($danger ? 'var(--thorium-danger-bg)' : 'var(--thorium-secondary-panel-bg)')};
  color: ${({ $danger }) => ($danger ? 'var(--thorium-button-text)' : 'var(--thorium-secondary-text)')};
`;

/**
 * A prominent red badge on a resource header holding the significance counts within a node's subtree/branch:
 * the `Flag` entity count (flag icon) and/or the danger-classified tag count (tags icon), each rendered as a
 * {@link BadgeMetric}. Counts read from the precomputed `flagStats` map (no per-render tree crawl); replaces the
 * plain danger dot with an aggregated, labelled count.
 */
export const FlagBadge = styled(BaseBadge)`
  display: inline-flex;
  align-items: center;
  /* separation between the flag-count and danger-tag-count metrics (each metric keeps a tighter icon↔count gap) */
  gap: 6px;
  background: var(--thorium-danger-bg);
  color: var(--thorium-button-text);
`;

/** One icon+count metric inside a {@link FlagBadge} (the flag count or the danger-tag count). */
export const BadgeMetric = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 3px;
`;

/**
 * A thin vertical divider between the flag-count and danger-tag-count {@link BadgeMetric}s inside a
 * {@link FlagBadge}, rendered only when both metrics are present. `currentColor` tracks the badge's text color.
 */
export const BadgeDivider = styled.span`
  align-self: stretch;
  width: 1px;
  background: currentColor;
`;

// --- metadata (condensed under the header, inside the info box) ---

/** The metadata region under the header: a subtle divider + secondary background, tight vertically. */
export const MetadataSection = styled.div`
  border-top: 1px solid var(--thorium-panel-border);
  background: var(--thorium-secondary-panel-bg);
  padding: 0 12px;
`;

/** The condensed "details" caret row — minimal vertical footprint when collapsed. */
export const MetadataToggleRow = styled.div`
  display: flex;
  justify-content: center;

  /* the details toggle reads like the dashboard's "filters" expand: hover changes only the text color, with no
     background fill (overrides the shared ExpandToggle's default hover, which also tints the background) */
  ${ExpandToggle}:hover {
    background: transparent;
    color: var(--thorium-highlight-text);
  }
`;

/** The revealed metadata body (only rendered when the details caret is expanded). */
export const MetadataContent = styled.div`
  padding: 2px 0 6px;
`;

export const ShowMoreRow = styled.div`
  display: flex;
  justify-content: center;
  padding: 2px 0;
`;

export const ShowMoreButton = styled.button`
  background: transparent;
  border: none;
  color: var(--thorium-highlight-text);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  padding: 4px 12px;
  border-radius: 6px;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }
`;
