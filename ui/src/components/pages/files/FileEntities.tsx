import React from 'react';

// project imports
import EntityBrowser from '@components/associations/browsing/EntityBrowser/EntityBrowser';

// spec: ./files.spec.md

// Depth to grow the shared graph to when the tab opens so nested structures (e.g. a memory dump's
// filesystem → folders → files) load and nest automatically rather than one manual expand per level.
// Growth is additive (no refetch) and converges with the Associations graph; users can go deeper via the
// omnibar `depth` clause. All layers default to "shown" — nothing is skipped/hidden initially.
const FILE_TAB_DEPTH = 3;

interface FileEntitiesProps {
  sha256: string;
  /** Whether the Entities tab is active; used only to skip rendering while hidden. */
  inView: boolean;
}

/**
 * File-details "Entities" tab: a generic, graph-driven browser rooted at this file. Reuses the shared
 * association graph via the surrounding {@link GraphDataProvider} (the same data the Associations tab uses),
 * so no extra request is made and expanding here converges with the association graph. The file itself is
 * implicit (`showRootNodes={false}`), so its associations render directly as the top level.
 */
const FileEntities: React.FC<FileEntitiesProps> = ({ sha256, inView }) => (
  <EntityBrowser roots={{ kind: 'sha256', sha256 }} inView={inView} defaultDepth={FILE_TAB_DEPTH} showRootNodes={false} />
);

export default FileEntities;
