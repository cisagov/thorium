import React, { useMemo } from 'react';
import { FaChevronDown, FaChevronUp, FaEyeSlash, FaGear, FaXmark } from 'react-icons/fa6';

// spec: ./SPEC.md

// project imports
import {
  ClearHiddenButton,
  FiltersSection,
  FiltersToggleLabel,
  HiddenNodeDelete,
  HiddenNodeTile,
  OmnibarStrip,
  OmnibarStripSlot,
  ReRootTile,
} from './styles';
import TagsTile from './TagsTile';
import { useEntityBrowser } from '@components/associations/browsing/EntityBrowser/EntityBrowserContext';
import FlaggedOnlyToggle from '@components/associations/browsing/EntityBrowser/FlaggedOnlyToggle';
import { buildBrowserOmnibarOptions } from '@components/associations/browsing/EntityBrowser/omnibarOptions';
import Collapsible, { TogglePosition } from '@components/shared/info/Collapsible';
import Omnibar from '@components/shared/inputs/omnibar/Omnibar';
import type { Clause } from '@components/shared/inputs/omnibar/ClauseTypes';

/// The collapsed height (px) of the expandable filters section: a scrollable, bottom-faded window onto
/// the tags tile, tall enough to show a couple of tag-key tiles before the "filters" toggle expands it.
const FILTERS_COLLAPSED_MAX_PX = 200;

/// Render the filters-section toggle label: a sized chevron icon + "filters", flipping direction with the
/// collapsed state. The icon (not a unicode glyph) matches the label's size/baseline and stays centered.
function filtersToggleLabel(collapsed: boolean): React.ReactNode {
  return <FiltersToggleLabel>{collapsed ? <FaChevronDown size={12} /> : <FaChevronUp size={12} />}filters</FiltersToggleLabel>;
}

/// Props for {@link DashboardOmnibar}.
export interface DashboardOmnibarProps {
  /**
   * The shared clause state — the same value/setter handed to the surrounding {@link EntityBrowserProvider}
   * (URL-backed via `useOmnibarUrlState`). Kept as props (rather than read from context) so the dashboard's
   * stats-bar click handler and this strip mutate one authoritative clause list.
   */
  clauses: Clause[];
  /** Setter for the shared {@link clauses}. */
  setClauses: (next: Clause[]) => void;
}

/**
 * The dashboard's always-shown filter strip, sitting between the stats panel and the content tiles.
 *
 * Renders an {@link Omnibar} bound to the shared clause state, with the active re-root and the hidden nodes
 * surfaced **inside the omnibar entry field** as removable tiles (via `extraChips`). The re-root tile (gear +
 * label + `×`) leads and clears the view back to the natural roots; each hidden tile (eye-slash + label + `×`)
 * unhides just that node — so both read as first-class filter tiles alongside the clause chips (kind-level
 * hides come from the `Exclude` verb, which already renders as its own removable clause tile). A single "Clear
 * all" tile follows when more than one node is hidden. The omnibar's option lexicon
 * (text/tag/group/Show/Hide/Exclude/Include/depth) is built from the graph-derived
 * `presentKinds`/`tagOptions`/`groupOptions` read from the surrounding {@link EntityBrowserProvider}. The
 * {@link FlaggedOnlyToggle} sits beside it (also context-backed, so this strip must render inside the
 * provider); sort/group controls live in the entity browser's own header, not in this filter strip. Below the
 * controls sits an expandable "filters" section (a shared {@link Collapsible}) hosting
 * the {@link TagsTile}, collapsed by default: the capped area **scrolls** with a **static bottom fade**. Its
 * toggle sits at the **top**; when the tags fit and no toggle is shown, no slot is reserved, so the tag tiles
 * sit flush under the omnibar rather than leaving an empty gap.
 *
 * @param clauses - The shared clause list (also given to the provider).
 * @param setClauses - Setter for the shared clause list.
 * @returns The omnibar strip.
 */
const DashboardOmnibar: React.FC<DashboardOmnibarProps> = ({ clauses, setClauses }) => {
  const { presentKinds, tagOptions, groupOptions, hiddenNodes, unhideNode, unhideAll, labelForNode, reRoot, setReRoot } =
    useEntityBrowser();
  const dropdownOptions = useMemo(
    () => buildBrowserOmnibarOptions(presentKinds, tagOptions, groupOptions),
    [presentKinds, tagOptions, groupOptions],
  );
  // the active re-root and any hidden ids are URL-backed; render them as removable tiles INSIDE the omnibar
  // (passed as `extraChips`) so they read like the depth/tag/exclude clause tiles. The re-root tile clears the
  // view back to the natural roots; each hidden tile unhides just that node (plus a Clear-all when >1 hidden)
  const extraChips = useMemo(() => {
    const ids = Array.from(hiddenNodes);
    if (!reRoot && ids.length === 0) return undefined;
    const reRootLabel = reRoot ? labelForNode(reRoot) : '';
    return (
      <>
        {reRoot && (
          <ReRootTile data-testid="entity-browser-reroot">
            {/* gear + label + × mirrors the hidden-node tile so it sits flush with the clause chips */}
            <FaGear aria-hidden />
            <span title={`Re-rooted at ${reRootLabel}`}>Rooted: {reRootLabel}</span>
            <HiddenNodeDelete
              type="button"
              data-testid="entity-browser-reroot-clear"
              aria-label={`Clear re-root at ${reRootLabel}`}
              onClick={() => setReRoot(null)}
            >
              <FaXmark size={12} aria-hidden />
            </HiddenNodeDelete>
          </ReRootTile>
        )}
        {ids.map((id) => {
          const label = labelForNode(id);
          return (
            <HiddenNodeTile key={id}>
              {/* icons sized like the clause chip's CategoryLogo (1em) and delete (FaX size 12) so the tile
                  matches the surrounding clause chips in height */}
              <FaEyeSlash aria-hidden />
              <span title={label}>{label}</span>
              <HiddenNodeDelete type="button" aria-label={`Unhide ${label}`} onClick={() => unhideNode(id)}>
                <FaXmark size={12} aria-hidden />
              </HiddenNodeDelete>
            </HiddenNodeTile>
          );
        })}
        {ids.length > 1 && (
          <ClearHiddenButton type="button" onClick={unhideAll}>
            Clear all
          </ClearHiddenButton>
        )}
      </>
    );
  }, [reRoot, setReRoot, hiddenNodes, labelForNode, unhideNode, unhideAll]);
  return (
    <OmnibarStrip>
      <OmnibarStripSlot>
        <Omnibar
          clauses={clauses}
          setClauses={setClauses}
          dropdownOptions={dropdownOptions}
          placeholder="Filter entities…"
          extraChips={extraChips}
        />
      </OmnibarStripSlot>
      <FlaggedOnlyToggle />
      <FiltersSection>
        <Collapsible
          maxPx={FILTERS_COLLAPSED_MAX_PX}
          renderToggleLabel={filtersToggleLabel}
          togglePosition={TogglePosition.Top}
          scrollWhenCollapsed
        >
          <TagsTile clauses={clauses} setClauses={setClauses} />
        </Collapsible>
      </FiltersSection>
    </OmnibarStrip>
  );
};

export default DashboardOmnibar;
