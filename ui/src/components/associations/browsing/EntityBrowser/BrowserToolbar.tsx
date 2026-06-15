// spec: ./EntityBrowser.spec.md
import React, { useMemo } from 'react';

// project imports
import { useEntityBrowser } from './EntityBrowserContext';
import { OmnibarSlot, ToolbarBar } from './EntityBrowser.styled';
import FlaggedOnlyToggle from './FlaggedOnlyToggle';
import HiddenNodesControl from './HiddenNodesControl';
import { buildBrowserOmnibarOptions } from './omnibarOptions';
import Omnibar from '@components/shared/inputs/omnibar/Omnibar';

/**
 * Omnibar-driven filter bar: text (name), tags, groups, the `Show`/`Hide`/`Exclude`/`Include` entity-layer
 * lexicon, and a traversal `depth`. Tag/group options come from the pulled graph (no extra request). The
 * standalone {@link FlaggedOnlyToggle} and {@link HiddenNodesControl} (shared with the dashboard strip) sit
 * beside the omnibar. Sort/group controls live in the browser's own header (`BrowserHeader`), not here.
 */
const BrowserToolbar: React.FC = () => {
  const { clauses, setClauses, presentKinds, tagOptions, groupOptions } = useEntityBrowser();

  const dropdownOptions = useMemo(
    () => buildBrowserOmnibarOptions(presentKinds, tagOptions, groupOptions),
    [tagOptions, groupOptions, presentKinds],
  );

  return (
    <ToolbarBar>
      <OmnibarSlot>
        <Omnibar clauses={clauses} setClauses={setClauses} dropdownOptions={dropdownOptions} placeholder="Filter entities…" />
      </OmnibarSlot>
      <HiddenNodesControl />
      <FlaggedOnlyToggle />
    </ToolbarBar>
  );
};

export default BrowserToolbar;
