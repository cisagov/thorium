import React from 'react';

// spec: ./SPEC.md

// project imports
import { EntityBrowserBody } from '@components/associations/browsing/EntityBrowser/EntityBrowser';

/**
 * The dashboard's entity-browser tile.
 *
 * Renders {@link EntityBrowserBody} directly (no nested {@link EntityBrowserProvider}) so it relies on
 * the provider composed at the dashboard level — the Phase 1 provider seam — sharing the URL-backed
 * clause/hidden/flagged state with the dashboard's own omnibar strip. The dashboard supplies that strip
 * separately, so the body renders no toolbar (`toolbar={null}`).
 *
 * `roots={{ kind: 'initial' }}` is configured on the surrounding provider, not here; the body reads the
 * resolved roots from context. Root nodes are shown as their own expandable rows (`showRootNodes`) because
 * a dashboard has heterogeneous seeds with no single implicit artifact (unlike the file-details tab).
 *
 * @param showSortControls - Whether the body renders its own sort/group header. The ultra-wide dashboard
 *   passes `false` and hosts those controls in the "Entities" tile header instead; the narrow/tabs layout
 *   passes `true` (no tile header there), so the controls stay in the browser body.
 * @returns The entity-browser tile body.
 */
interface BrowserTileProps {
  /** Whether the browser body renders its own sort/group header (see the component doc). */
  showSortControls: boolean;
}

const BrowserTile: React.FC<BrowserTileProps> = ({ showSortControls }) => (
  <EntityBrowserBody showRootNodes toolbar={null} showSortControls={showSortControls} />
);

export default React.memo(BrowserTile);
