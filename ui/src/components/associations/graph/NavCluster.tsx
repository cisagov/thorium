import React from 'react';
import { FaExpand, FaHome, FaSearchMinus, FaSearchPlus } from 'react-icons/fa';

// project imports
import { NavClusterContainer } from './Shared';
import { ToolbarIconButton } from './controls/Toolbar.styled';
import { OverlayTipLeft } from '@components/shared/overlay/tips';

// spec: ./AssociationGraph.spec.md

interface NavClusterProps {
  /** Dolly the camera toward the orbit target */
  onZoomIn: () => void;
  /** Dolly the camera away from the orbit target */
  onZoomOut: () => void;
  /** Frame the whole graph in view */
  onFitAll: () => void;
  /** Restore the camera to the saved "home" view */
  onResetView: () => void;
}

/**
 * Always-visible camera navigation cluster pinned to the graph window's bottom-right
 * corner: zoom in/out, fit-all, and reset-to-home buttons. Tooltips include the
 * matching keyboard shortcut; tooltips open to the left so they never clip against
 * the window edge or cover the button above.
 *
 * @param onZoomIn - Called when the zoom-in button is clicked.
 * @param onZoomOut - Called when the zoom-out button is clicked.
 * @param onFitAll - Called when the fit-all button is clicked.
 * @param onResetView - Called when the reset-view button is clicked.
 */
const NavCluster: React.FC<NavClusterProps> = ({ onZoomIn, onZoomOut, onFitAll, onResetView }) => (
  <NavClusterContainer>
    <OverlayTipLeft tip="Zoom in (+)">
      <ToolbarIconButton aria-label="Zoom in" onClick={onZoomIn}>
        <FaSearchPlus size={14} />
      </ToolbarIconButton>
    </OverlayTipLeft>
    <OverlayTipLeft tip="Zoom out (-)">
      <ToolbarIconButton aria-label="Zoom out" onClick={onZoomOut}>
        <FaSearchMinus size={14} />
      </ToolbarIconButton>
    </OverlayTipLeft>
    <OverlayTipLeft tip="Fit all (f)">
      <ToolbarIconButton aria-label="Fit all" onClick={onFitAll}>
        <FaExpand size={14} />
      </ToolbarIconButton>
    </OverlayTipLeft>
    <OverlayTipLeft tip="Reset view (r)">
      <ToolbarIconButton aria-label="Reset view" onClick={onResetView}>
        <FaHome size={14} />
      </ToolbarIconButton>
    </OverlayTipLeft>
  </NavClusterContainer>
);

export default NavCluster;
