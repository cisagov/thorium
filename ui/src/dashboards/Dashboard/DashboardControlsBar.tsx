import React from 'react';
import { FaSearchPlus } from 'react-icons/fa';
import { FaArrowsRotate, FaFilterCircleXmark } from 'react-icons/fa6';

// spec: ./SPEC.md

// project imports
import { ControlsBar, SpinningIcon } from './styles';
import { ButtonToolbar, IconButton } from '@components/shared/buttons';
import { OverlayTipBottom } from '@components/shared/overlay/tips';

/// Props for {@link DashboardControlsBar}.
export interface DashboardControlsBarProps {
  /// The current graph crawl depth (drives the Grow button's at-max disabled state and tip).
  currentDepth: number;
  /// The maximum crawl depth; the Grow action disables at/above this bound.
  maxDepth: number;
  /// Called to increase the crawl depth by one level.
  onGrowLevel: () => void;
  /// Called to clear all filter clauses (keeping the depth and hidden-tags clauses).
  onResetFilters: () => void;
  /// Called to refresh the dashboard data (graph reload + Analysis Status reactions).
  onRefresh: () => void;
  /// Whether a refresh is currently in flight (spins the Refresh icon and disables the button).
  refreshing: boolean;
  /// Whether a grow/growToDepth is in flight; disables the Grow button so rapid clicks don't queue work.
  growing: boolean;
}

/**
 * The dashboard's quick-action controls bar, sitting between the stats panel and the omnibar strip.
 *
 * Renders an icon toolbar (matching the entity-details toolbar) of quick actions, each wrapped in an
 * {@link OverlayTipBottom}:
 * - **Grow** — a {@link FaSearchPlus} button that raises the graph crawl depth by one; disabled (with an
 *   at-max tip) once the depth reaches `maxDepth`. The current depth is surfaced in the omnibar, so no
 *   depth label is repeated here.
 * - **Reset filters** — a {@link FaFilterCircleXmark} button clearing all filter clauses.
 * - **Refresh data** — a {@link FaArrowsRotate} button reloading the graph and the Analysis Status
 *   reactions; it spins and disables while a refresh is in flight.
 *
 * @param currentDepth - The current crawl depth.
 * @param maxDepth - The maximum crawl depth (Grow disables at/above it).
 * @param onGrowLevel - Called to increase the crawl depth by one level.
 * @param onResetFilters - Called to clear all filter clauses.
 * @param onRefresh - Called to refresh the dashboard data.
 * @param refreshing - Whether a refresh is currently in flight.
 * @param growing - Whether a grow/growToDepth is in flight (disables the Grow button and swaps its tip).
 * @returns The controls bar.
 */
const DashboardControlsBar: React.FC<DashboardControlsBarProps> = ({
  currentDepth,
  maxDepth,
  onGrowLevel,
  onResetFilters,
  onRefresh,
  refreshing,
  growing,
}) => {
  const atMax = currentDepth >= maxDepth;
  // disable while a grow is in flight so rapid clicks can't queue redundant depth grows
  const growDisabled = atMax || growing;
  const growTip = atMax
    ? `Maximum search depth (${maxDepth}) reached`
    : growing
      ? 'Growing…'
      : 'Increase the search depth of the dashboard';
  return (
    <ControlsBar>
      <ButtonToolbar>
        <OverlayTipBottom tip={growTip}>
          {/* wrap in a span so the tip still shows while the button is disabled (disabled elements swallow hover) */}
          <span>
            <IconButton onClick={onGrowLevel} disabled={growDisabled} aria-label="Increase the search depth of the dashboard">
              <FaSearchPlus size={18} />
            </IconButton>
          </span>
        </OverlayTipBottom>
        <OverlayTipBottom tip="Clear all filters">
          <IconButton onClick={onResetFilters} aria-label="Clear all filters">
            <FaFilterCircleXmark size={18} />
          </IconButton>
        </OverlayTipBottom>
        <OverlayTipBottom tip="Refresh the dashboard data">
          {/* wrap in a span so the tip still shows while the button is disabled during a refresh */}
          <span>
            <IconButton onClick={onRefresh} disabled={refreshing} aria-label="Refresh the dashboard data">
              <SpinningIcon $spinning={refreshing}>
                <FaArrowsRotate size={18} />
              </SpinningIcon>
            </IconButton>
          </span>
        </OverlayTipBottom>
      </ButtonToolbar>
    </ControlsBar>
  );
};

export default DashboardControlsBar;
