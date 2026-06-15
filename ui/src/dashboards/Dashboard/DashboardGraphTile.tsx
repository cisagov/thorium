import React, { Suspense } from 'react';

// spec: ./SPEC.md

// project imports
import PaneExpandToggle from './PaneExpandToggle';
import { GraphContentTile, GraphFallbackContainer, TileHeader, TileHeaderRow } from './styles';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';

/**
 * The association graph, lazily loaded so its heavy 3D bundle is only fetched when a dashboard that
 * shows the graph mounts.
 */
const AssociationGraph = React.lazy(() => import('@components/associations/graph/AssociationGraph'));

/// Centered spinner shown while the lazy graph chunk loads.
const GraphFallback = (
  <GraphFallbackContainer>
    <LoadingSpinner loading />
  </GraphFallbackContainer>
);

/// Props for {@link DashboardGraphTile}.
export interface DashboardGraphTileProps {
  /**
   * Whether the tile is currently shown (its tab is active, or it is always visible on ultra-wide
   * layouts). Drives `AssociationGraph`'s `inView` so the 3D canvas never renders behind a hidden
   * tab — the plan prefers this explicit signal over `useInView({ triggerOnce: true })`.
   */
  active: boolean;
  /** Whether the focus (expand-to-fill) toggle is offered — only meaningful in the two-column ultra-wide layout. */
  canExpand?: boolean;
  /** Whether this pane is currently the focused (expanded) one, so the toggle shows the collapse affordance. */
  expanded?: boolean;
  /** Called when the focus toggle is clicked (focus this pane / restore the split view). */
  onToggleExpand?: () => void;
  /**
   * Whether the tile is in the single-column stacked (expanded) layout: it then renders at a viewport-bounded
   * height instead of a column-width square (which would be far too tall at full page width).
   */
  fill?: boolean;
}

/**
 * The dashboard's graph tile: a lazily-loaded {@link AssociationGraph} whose rendering is gated on
 * the `active` prop.
 *
 * The tile itself stays mounted (its wrapper is toggled with `display: none` by the composing page so
 * layout/state is preserved), but `AssociationGraph` only mounts its 3D scene while `active` is true;
 * `AssociationGraph` internally renders nothing when `inView` is false, so an inactive tab pays no
 * WebGL cost. When `canExpand` is set the header carries a focus toggle that expands this pane to fill the
 * content region (and collapses back to the split view).
 *
 * @param active - Whether the graph should render (visible tab / ultra-wide layout).
 * @param canExpand - Whether to show the focus (expand-to-fill) toggle.
 * @param expanded - Whether this pane is currently focused (toggle shows the collapse icon).
 * @param onToggleExpand - Called when the focus toggle is clicked.
 * @param fill - Whether to render at a viewport-bounded height for the single-column stacked (expanded) layout.
 * @returns The graph tile.
 */
const DashboardGraphTile: React.FC<DashboardGraphTileProps> = ({
  active,
  canExpand = false,
  expanded = false,
  onToggleExpand,
  fill = false,
}) => (
  <GraphContentTile $fill={fill}>
    <TileHeader>
      <TileHeaderRow>
        <span>Association Graph</span>
        {canExpand && (
          <PaneExpandToggle
            expanded={expanded}
            onToggle={onToggleExpand}
            expandTip="Expand graph (stack entities below)"
            expandAriaLabel="Expand graph and stack entities below"
          />
        )}
      </TileHeaderRow>
    </TileHeader>
    <Suspense fallback={GraphFallback}>
      <AssociationGraph inView={active} />
    </Suspense>
  </GraphContentTile>
);

export default DashboardGraphTile;
