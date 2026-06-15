import React, { useMemo } from 'react';
import styled from 'styled-components';

// project imports
import { decodeHexValues } from './decode';
import { HexByteStatus, HexSelection } from './types';
// spec: ../SPEC.md

const Panel = styled.div`
  flex: 0 0 auto;
  min-width: 220px;
  max-width: 320px;
  padding: 10px 12px;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  font-size: 0.82rem;
  align-self: flex-start;
`;

const Heading = styled.div`
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--thorium-text);
  display: flex;
  align-items: center;
  gap: 8px;
`;

// long file names have no spaces, so break mid-word to keep them inside the panel
const HeadingText = styled.span`
  min-width: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
`;

const SourceDot = styled.span<{ $status: HexByteStatus }>`
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: ${({ $status }) =>
    $status === HexByteStatus.Added
      ? 'var(--thorium-ok-bg)'
      : $status === HexByteStatus.Removed
        ? 'var(--thorium-danger-bg)'
        : 'var(--thorium-secondary-panel-bg)'};
`;

const Entry = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 2px 0;
  border-bottom: 1px solid var(--thorium-panel-border);

  /* the last row sits just above the panel's own border — drop its divider to avoid a doubled line */
  &:last-child {
    border-bottom: none;
  }
`;

const Label = styled.span`
  color: var(--thorium-secondary-text);
  flex: 0 0 auto;
`;

const ValueText = styled.span`
  font-family: var(--bs-font-monospace, monospace);
  color: var(--thorium-text);
  word-break: break-all;
  text-align: right;
`;

const Empty = styled.div`
  color: var(--thorium-secondary-text);
  font-style: italic;
`;

export interface HexValueInspectorProps {
  bytes: Uint8Array;
  selection: HexSelection | null;
  /** Optional title (e.g. the file name or "Base"/"Compare" in diff mode). */
  title?: string;
  /** Optional diff-source indicator shown next to the title. */
  sourceStatus?: HexByteStatus;
}

/**
 * Side panel that decodes the currently-selected hex bytes into typed interpretations. Reused by
 * the standalone hex viewer and both sides of the hex diff (where `sourceStatus` colors the dot).
 */
const HexValueInspector: React.FC<HexValueInspectorProps> = ({ bytes, selection, title = 'Value', sourceStatus }) => {
  const entries = useMemo(() => decodeHexValues(bytes, selection), [bytes, selection]);
  return (
    <Panel>
      <Heading>
        {sourceStatus && <SourceDot $status={sourceStatus} />}
        <HeadingText>{title}</HeadingText>
      </Heading>
      {entries.length === 0 ? (
        <Empty>Select bytes to inspect their value.</Empty>
      ) : (
        entries.map((e) => (
          <Entry key={e.label}>
            <Label>{e.label}</Label>
            <ValueText>{e.value}</ValueText>
          </Entry>
        ))
      )}
    </Panel>
  );
};

export default HexValueInspector;
