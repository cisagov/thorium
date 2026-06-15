// spec: ../ToolResult.spec.md
import React, { useCallback, useMemo, useState } from 'react';
import styled from 'styled-components';

// project imports
import { computeByteDiff, mapSelectionAcross } from './byteDiff';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { HexByteStatus, HexRenderer, HexSelection, HexValueInspector } from '@components/shared/renderers';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const Layout = styled.div`
  display: flex;
  gap: 12px;
  align-items: flex-start;
`;

const Side = styled.div`
  flex: 1 1 0;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const SideTitle = styled.div`
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--thorium-text);
`;

export interface HexDiffViewerProps {
  base: ArrayBuffer;
  compare: ArrayBuffer;
  baseTitle?: string;
  compareTitle?: string;
}

/** The status of a selected region: the strongest non-Same status, else Same. */
function regionStatus(status: HexByteStatus[], sel: HexSelection | null, diffStatus: HexByteStatus): HexByteStatus {
  if (!sel) return HexByteStatus.Same;
  for (let i = sel.offset; i < sel.offset + sel.length; i++) {
    if (status[i] === diffStatus) return diffStatus;
  }
  return HexByteStatus.Same;
}

/**
 * Side-by-side hex diff of two byte buffers.
 *
 * Bytes are colored by an LCS byte diff (green added / red removed / theme background for
 * unchanged). Selecting a range in one pane mirrors the selection onto the aligned range in the
 * other pane, and each side's value inspector indicates the diff source via a colored dot.
 *
 * For pathologically large inputs the diff is skipped (see `computeByteDiff`); the panes then render
 * as plain, uncolored hex with a "diff truncated" banner so the tab can never freeze.
 */
const HexDiffViewer: React.FC<HexDiffViewerProps> = ({ base, compare, baseTitle = 'Base', compareTitle = 'Compare' }) => {
  const baseBytes = useMemo(() => new Uint8Array(base), [base]);
  const compareBytes = useMemo(() => new Uint8Array(compare), [compare]);
  const diff = useMemo(() => computeByteDiff(baseBytes, compareBytes), [baseBytes, compareBytes]);

  const [baseSel, setBaseSel] = useState<HexSelection | null>(null);
  const [compareSel, setCompareSel] = useState<HexSelection | null>(null);

  // selecting in one pane mirrors onto the aligned range of the other
  const onBaseSelect = useCallback(
    (sel: HexSelection | null) => {
      setBaseSel(sel);
      setCompareSel(sel ? mapSelectionAcross(sel, diff.baseToCompare) : null);
    },
    [diff.baseToCompare],
  );
  const onCompareSelect = useCallback(
    (sel: HexSelection | null) => {
      setCompareSel(sel);
      setBaseSel(sel ? mapSelectionAcross(sel, diff.compareToBase) : null);
    },
    [diff.compareToBase],
  );

  const baseStatusFn = useCallback((i: number) => diff.baseStatus[i], [diff.baseStatus]);
  const compareStatusFn = useCallback((i: number) => diff.compareStatus[i], [diff.compareStatus]);

  return (
    <Container>
      {diff.truncated && (
        <AlertBanner severity={Severity.Warning}>
          Diff truncated — files too large to diff; showing plain hex without diff coloring.
        </AlertBanner>
      )}
      <Layout>
        <Side>
          <SideTitle>{baseTitle}</SideTitle>
          <HexRenderer bytes={baseBytes} selection={baseSel} onSelectionChange={onBaseSelect} byteStatus={baseStatusFn} />
          <HexValueInspector
            bytes={baseBytes}
            selection={baseSel}
            title={baseTitle}
            sourceStatus={regionStatus(diff.baseStatus, baseSel, HexByteStatus.Removed)}
          />
        </Side>
        <Side>
          <SideTitle>{compareTitle}</SideTitle>
          <HexRenderer bytes={compareBytes} selection={compareSel} onSelectionChange={onCompareSelect} byteStatus={compareStatusFn} />
          <HexValueInspector
            bytes={compareBytes}
            selection={compareSel}
            title={compareTitle}
            sourceStatus={regionStatus(diff.compareStatus, compareSel, HexByteStatus.Added)}
          />
        </Side>
      </Layout>
    </Container>
  );
};

export default HexDiffViewer;
