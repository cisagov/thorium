// spec: ./EntityBrowser.spec.md
import React from 'react';
import { FaFlag } from 'react-icons/fa6';

// project imports
import { useEntityBrowser } from './EntityBrowserContext';
import { ToggleChip } from './EntityBrowser.styled';

/**
 * Standalone "Flagged Only" toggle chip, reading flagged state from the {@link useEntityBrowser} context. A
 * node is flagged if it is danger-tagged or has a `Flag` entity reachable within the pulled tree. Extracted so
 * both {@link BrowserToolbar} and a dashboard omnibar strip can render the same control.
 */
const FlaggedOnlyToggle: React.FC = () => {
  const { flaggedOnly, setFlaggedOnly } = useEntityBrowser();
  return (
    <ToggleChip
      type="button"
      $active={flaggedOnly}
      data-testid="entity-browser-flagged"
      aria-pressed={flaggedOnly}
      onClick={() => setFlaggedOnly(!flaggedOnly)}
    >
      <FaFlag size={12} /> Flagged Only
    </ToggleChip>
  );
};

export default FlaggedOnlyToggle;
