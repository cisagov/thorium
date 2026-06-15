import React from 'react';
import { FaCompress, FaExpand } from 'react-icons/fa';

// spec: ./SPEC.md

// project imports
import { IconButton } from '@components/shared/buttons';
import { OverlayTipBottom } from '@components/shared/overlay/tips';

/// The tip and accessible name shared by both panes when expanded — collapsing always restores the split view.
const RESTORE_LABEL = 'Restore the side-by-side view';

/// Props for {@link PaneExpandToggle}.
export interface PaneExpandToggleProps {
  /// Whether this pane is currently the focused (expanded) one, so the toggle shows the collapse affordance.
  expanded: boolean;
  /// Called when the toggle is clicked (focus this pane / restore the split view).
  onToggle?: () => void;
  /// The tooltip shown while collapsed (the affordance to expand this specific pane).
  expandTip: string;
  /// The accessible name applied to the button while collapsed (a fuller phrasing than the tip).
  expandAriaLabel: string;
}

/**
 * The tile-header focus toggle shared by the dashboard's Entities and Graph panes: an icon button whose
 * icon and labels swap between "expand this pane" and "restore the split view".
 *
 * Only meaningful in the two-column ultra-wide layout; the composing panes render it inside their own
 * tile header rows so this component owns just the toggle affordance (icon, tip, aria-label), keeping the
 * expand/collapse contract in one place across both panes.
 *
 * @param expanded - Whether this pane is currently focused (toggle shows the collapse icon and restore labels).
 * @param onToggle - Called when the toggle is clicked.
 * @param expandTip - The tooltip shown while collapsed.
 * @param expandAriaLabel - The accessible name applied while collapsed.
 * @returns The pane focus toggle.
 */
const PaneExpandToggle: React.FC<PaneExpandToggleProps> = ({ expanded, onToggle, expandTip, expandAriaLabel }) => (
  <OverlayTipBottom tip={expanded ? RESTORE_LABEL : expandTip}>
    <IconButton onClick={onToggle} aria-label={expanded ? RESTORE_LABEL : expandAriaLabel}>
      {expanded ? <FaCompress size={15} /> : <FaExpand size={15} />}
    </IconButton>
  </OverlayTipBottom>
);

export default PaneExpandToggle;
