// spec: ./SPEC.md

import styled from 'styled-components';

// project imports
import { BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';
import { spacers } from '@styles';

/**
 * The builder page grid: stacked full-width rows — selected-resources tile, the depth/create menu,
 * then the browse list (which carries its own resource-type dropdown). Everything is a single
 * column; the page is an authoring surface rather than a data dashboard, so no responsive column
 * split is needed.
 */
export const BuilderLayout = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${spacers.four};
  padding-bottom: ${spacers.five};
`;

/**
 * The pinned summary tile at the top of the builder holding the selected-resources chips in a themed
 * panel. A flex column so future summary rows stack with a consistent gap.
 */
export const SummaryTile = styled.section`
  display: flex;
  flex-direction: column;
  gap: ${spacers.four};
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  padding: ${spacers.four};
  min-width: 0;
`;

/**
 * The depth/create menu row: the depth control and the Create button centered together with no
 * card/background fill, ordered Depth then Create. A borderless flex row so it reads as a light
 * action bar rather than a panel; wraps on narrow viewports.
 */
export const BuilderMenu = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${spacers.three};
  flex-wrap: wrap;
`;

/**
 * A titled section wrapper (type picker, browse list, selection panel) with a themed panel look.
 *
 * The trailing `EntityList` pagination (a react-bootstrap `<Pagination>` in a `Row.mt-3`) carries its own
 * bottom margin, which leaves Back/Next floating too far above the card's bottom edge. Zeroing the
 * pagination's bottom margin here — scoped to the builder so other browse pages are unaffected — lets it
 * sit a normal gap above the card edge.
 */
export const BuilderSection = styled.section`
  /* no card fill/border — the add-resources area reads as a plain section under the controls divider */
  min-width: 0;

  & .pagination {
    margin-bottom: 0;
  }
`;

/// The divider between the controls (depth / view) and the add-resources section, in place of the card edge.
export const SectionHr = styled.hr`
  margin: ${spacers.three} 0;
  border: none;
  border-top: 1px solid var(--thorium-panel-border);
`;

/// A small uppercase section label shown above a section's content.
export const SectionLabel = styled.h3`
  margin: 0 0 ${spacers.three} 0;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--thorium-secondary-text);
`;

/**
 * The controls row placing the resource-type picker inline, directly in front of the filters omnibar (or
 * the Tag-mode key/value entry). The picker keeps its own compact width; the LAST child — a wrapper around
 * the omnibar/tags-select — takes the omnibar's max width (1000px) as its basis but can shrink, so the two
 * sit as one tight `[picker][filters]` group. The row centers that group with a small gap between the two.
 * A bottom margin (matching the divider spacing above the section) separates it from the resource list
 * below so the controls read as distinct from the list/tabs.
 */
export const BrowseControls = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${spacers.three};
  margin-bottom: ${spacers.four};
  min-width: 0;

  & > *:last-child {
    /* basis of the omnibar's own max width so the filters area doesn't grow to fill the whole row (which
       would push the omnibar away from the picker and defeat the centered grouping); still shrinks when
       the viewport is narrow */
    flex: 0 1 1000px;
    min-width: 0;
  }

  /* the omnibar wrapper carries a shared 20px bottom margin meant for standalone search-bar layouts;
     cancel it here so the omnibar vertically centers against the picker in this inline row */
  & .col > div {
    margin-bottom: 0;
  }
`;

/// Groups the depth label and its selector in the footer.
export const DepthGroup = styled.div`
  display: flex;
  align-items: center;
  gap: ${spacers.three};
  color: var(--thorium-text);
  font-size: 0.9rem;
`;

/**
 * The custom "Create dashboard" button.
 *
 * A styled-component (not the legacy react-bootstrap `<Button>`) so the builder stays react-bootstrap
 * free. Uses the neutral gray panel-highlight fill (matching the builder's Pager/add controls) rather
 * than a prominent primary/ok tone, since it opens the assembled dashboard rather than committing a
 * create. Disabled styling dims the button and blocks pointer events; the caller wraps it in a focusable
 * span so the overlay tip still fires while disabled.
 */
export const CreateButton = styled.button`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.three};
  padding: ${spacers.three} ${spacers.five};
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  background: var(--thorium-highlight-panel-bg);
  color: var(--thorium-text);
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;

  &:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

/**
 * The Back/Next pager for a browse subsection (the Files tabs). A centered, react-bootstrap-free pair of
 * buttons matching the shared entity-list pagination affordance without pulling in react-bootstrap.
 */
export const Pager = styled.div`
  display: flex;
  justify-content: center;
  gap: ${BUTTON_BAR_GAP};
  margin-top: ${spacers.four};
`;

/// A single Back/Next control in the {@link Pager}; disabled at the list ends.
export const PagerButton = styled.button`
  display: inline-flex;
  align-items: center;
  gap: ${spacers.two};
  padding: ${spacers.two} ${spacers.four};
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  background: var(--thorium-highlight-panel-bg);
  color: var(--thorium-text);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;

  &:hover:not(:disabled) {
    filter: brightness(1.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

/**
 * A single browse row: the rendered entity on the left, the add/added control on the right.
 */
export const BrowseRow = styled.div`
  display: flex;
  align-items: center;
  gap: ${spacers.three};
  min-width: 0;

  & > *:first-child {
    flex: 1;
    min-width: 0;
  }
`;

/**
 * The per-row add/remove control. Rendered as a square icon button (`$square`) so the icon sits
 * centered in a square rather than a wide pill. `$remove` styles the already-added state with a solid
 * danger (red) fill at rest — so an already-added resource reads as removable at a glance — carrying an
 * X, versus the neutral add (plus) state. Both brighten on hover to signal they're clickable.
 */
export const AddButton = styled.button<{ $remove?: boolean; $square?: boolean }>`
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  ${({ $square }) => ($square ? 'width: 1.75rem; height: 1.75rem; padding: 0;' : `padding: ${spacers.two} ${spacers.four};`)}
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
  ${({ $remove }) =>
    $remove
      ? 'background: var(--thorium-danger-bg); color: var(--thorium-button-text); border-color: var(--thorium-danger-bg);'
      : 'background: var(--thorium-highlight-panel-bg); color: var(--thorium-text);'}
  cursor: pointer;

  &:hover:not(:disabled) {
    filter: brightness(1.15);
  }

  &:disabled {
    cursor: default;
  }
`;

/// The key/value entry row used in Tag mode (a TagSelect plus an Add button), vertically centered so
/// the add button lines up with the tag input.
export const TagModeRow = styled.div`
  display: flex;
  align-items: center;
  gap: ${spacers.three};
  min-width: 0;

  & > *:first-child {
    flex: 1;
    min-width: 0;
  }
`;

/// A muted placeholder shown when a browse subsection has no results.
export const EmptyBrowse = styled.div`
  padding: ${spacers.four};
  text-align: center;
  color: var(--thorium-secondary-text);
  font-size: 0.85rem;
`;
