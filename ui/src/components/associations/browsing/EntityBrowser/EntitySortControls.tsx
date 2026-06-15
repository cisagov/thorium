// spec: ./EntityBrowser.spec.md
import React from 'react';
import { FaLayerGroup } from 'react-icons/fa6';

// project imports
import { useEntityBrowser } from './EntityBrowserContext';
import { SortControls, SortLabel, SortSelect, ToggleChip } from './EntityBrowser.styled';
import { SortMode } from './types';

/** Sort-mode dropdown choices, in the order they appear in the selector. */
const SORT_OPTIONS: readonly { value: SortMode; label: string }[] = [
  { value: SortMode.Flags, label: 'Flags' },
  { value: SortMode.Suspicion, label: 'Suspicion' },
  { value: SortMode.Confidence, label: 'Confidence' },
];

/**
 * Standalone sort/group controls for the entity browser, reading state from {@link useEntityBrowser}: a
 * dropdown selecting the primary flag-stat sort field (Flags / Suspicion / Confidence — the unselected two act
 * as descending tiebreakers) and an on-by-default "Group by Type" toggle that groups each level by node kind
 * under {@link LayerHeader}s (off renders one flat, sorted list). Rendered in the browser's own header row
 * (`BrowserHeader`) so it sits directly above the list, shared by the file-details tab and the dashboard.
 */
const EntitySortControls: React.FC = () => {
  const { sortMode, setSortMode, groupByResource, setGroupByResource } = useEntityBrowser();
  return (
    <SortControls>
      <SortLabel>Sort</SortLabel>
      <SortSelect
        aria-label="Sort entities by"
        data-testid="entity-sort-mode"
        value={sortMode}
        onChange={(e) => setSortMode(e.target.value as SortMode)}
      >
        {SORT_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </SortSelect>
      <ToggleChip
        type="button"
        $active={groupByResource}
        $tone="accent"
        data-testid="entity-group-by-resource"
        aria-pressed={groupByResource}
        onClick={() => setGroupByResource(!groupByResource)}
      >
        <FaLayerGroup size={12} /> Group by Type
      </ToggleChip>
    </SortControls>
  );
};

export default EntitySortControls;
