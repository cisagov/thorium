// spec: ../ToolResult.spec.md
import React from 'react';
import styled, { keyframes } from 'styled-components';

// project imports
import { ChildrenFetchStatus } from './useChildrenMetadata';
import { ToolResultTabProps } from './types';
import { getNodeName } from '@components/associations/utilities';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';
import EntitySummaryHover from '@components/shared/info/EntitySummaryHover';
import { SummaryPart, treeNodeToInfo } from '@components/shared/info/info';
import { Sample } from '@models/files';
import { TreeNodeKey } from '@models/trees';

// display cap for a resolved child name; long names are also visually clipped via CSS ellipsis
const MAX_NAME_LENGTH = 120;

const ChildList = styled.div`
  display: flex;
  flex-direction: column;
`;

// children sit directly on the result tile (no card background/border), matching the Files tab; a
// divider only separates stacked rows — the last row has none so it doesn't double the tile's edge
const ChildRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;

  &:not(:last-child) {
    border-bottom: 1px solid var(--thorium-panel-border);
  }
`;

const ChildLink = styled.a`
  display: block;
  flex: 1 1 auto;
  min-width: 0;
  padding: 6px 4px;
  font-family: var(--bs-font-monospace, monospace);
  font-size: 0.9rem;
  color: var(--thorium-link-text);
  text-decoration: none;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

// compact inline progress line shown while child metadata is being fetched (the shared LoadingSpinner
// is a large centered block, unsuited to a one-line status)
const LoadingRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 4px;
  color: var(--thorium-secondary-text);
  font-size: 0.85rem;
`;

const MiniSpinner = styled.span`
  width: 14px;
  height: 14px;
  border: 2px solid var(--thorium-panel-border);
  border-top-color: var(--thorium-highlight-text);
  border-radius: 50%;
  animation: ${spin} 0.7s linear infinite;
  flex: 0 0 auto;
`;

/** A single child row: the resolved file name (with hover info) once loaded, else the raw SHA256 link. */
const ChildEntry: React.FC<{ sha256: string; sample?: Sample }> = ({ sha256, sample }) => {
  // until the file details resolve (or if the fetch failed), fall back to the raw sha256 link
  if (!sample) {
    return (
      <ChildRow>
        <ChildLink href={`/file/${sha256}`} title={sha256}>
          {sha256}
        </ChildLink>
      </ChildRow>
    );
  }
  const node = { [TreeNodeKey.Sample]: sample };
  const name = getNodeName(node, MAX_NAME_LENGTH) || sha256;
  const info = treeNodeToInfo(node);
  const link = (
    <ChildLink href={`/file/${sha256}`} title={name}>
      {name}
    </ChildLink>
  );
  return (
    <ChildRow>
      {/* treeNodeToInfo is nullable in general, but always non-null for a Sample node. The popover
          opens to the right so it doesn't cover the file name being hovered. */}
      {info ? (
        <EntitySummaryHover model={info} placement="right" exclude={[SummaryPart.Title]}>
          {link}
        </EntitySummaryHover>
      ) : (
        link
      )}
    </ChildRow>
  );
};

interface ChildrenTabProps extends ToolResultTabProps {
  /** Resolved child file details keyed by SHA256 (from `useChildrenMetadata`). */
  samples: Record<string, Sample>;
  /** Fetch progress for the child metadata. */
  status: ChildrenFetchStatus;
  /** Number of children whose fetch has been attempted so far. */
  loaded: number;
  /** Total child count. */
  total: number;
}

/** The "Children" tab body: the child files (sha256) discovered by the tool, resolved to names. */
const ChildrenTab: React.FC<ChildrenTabProps> = ({ result, samples, status, loaded, total }) => {
  const children = Object.keys(result.children);
  if (children.length === 0) {
    return <RenderErrorAlert page={false} message="This result has no children." />;
  }
  return (
    <>
      {status === 'loading' && (
        <LoadingRow>
          <MiniSpinner aria-hidden="true" />
          Loaded {loaded} / {total}
        </LoadingRow>
      )}
      <ChildList>
        {children.map((child) => (
          <ChildEntry key={child} sha256={child} sample={samples[child]} />
        ))}
      </ChildList>
    </>
  );
};

export default ChildrenTab;
