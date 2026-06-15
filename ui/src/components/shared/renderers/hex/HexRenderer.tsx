import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';

// project imports
import { HexByteStatus, HexSelection } from './types';
// spec: ../SPEC.md

const BYTES_PER_ROW = 16;
const DEFAULT_MAX_BYTES = 16 * 1024;

const Container = styled.div`
  font-family: var(--bs-font-monospace, monospace);
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--thorium-text);
  padding: 8px 4px;
  // rows are fixed-width (nowrap) so scroll them horizontally within this pane; vertical scrolling
  // of the many rows is owned by the surrounding container so scrollbars don't stack
  overflow-x: auto;
  overflow-y: hidden;
  user-select: none;
  white-space: nowrap;
`;

const Row = styled.div`
  display: flex;
  gap: 16px;
`;

const Offset = styled.span`
  color: var(--thorium-secondary-text);
  flex: 0 0 auto;
`;

const HexCols = styled.span`
  flex: 0 0 auto;
`;

const AsciiCols = styled.span`
  flex: 0 0 auto;
  color: var(--thorium-secondary-text);
`;

const Notice = styled.div`
  margin-top: 8px;
  color: var(--thorium-secondary-text);
  font-style: italic;
  white-space: normal;
`;

// per-byte cell; status drives diff coloring, $selected the cross-highlight
const Cell = styled.span<{ $status?: HexByteStatus; $selected: boolean }>`
  padding: 0 1px;
  border-radius: 2px;
  cursor: text;
  background: ${({ $status, $selected }) =>
    $selected
      ? 'var(--thorium-link-text)'
      : $status === HexByteStatus.Added
        ? 'var(--thorium-ok-bg)'
        : $status === HexByteStatus.Removed
          ? 'var(--thorium-danger-bg)'
          : 'transparent'};
  color: ${({ $selected }) => ($selected ? 'var(--thorium-button-text)' : 'inherit')};
`;

const hex2 = (n: number) => n.toString(16).padStart(2, '0');

interface HexRowProps {
  data: Uint8Array;
  rowStart: number;
  count: number;
  byteStatus?: (index: number) => HexByteStatus | undefined;
  /** Selection bounds intersected with this row: [selFrom, selTo). `-1` means no selection here. */
  selFrom: number;
  selTo: number;
}

/**
 * A single hex row, memoized so a selection change only re-renders rows whose selection
 * intersection actually changed (instead of rebuilding every byte cell on each mouse move).
 */
const HexRow = React.memo(function HexRow({ data, rowStart, count, byteStatus, selFrom, selTo }: HexRowProps) {
  const hexCells: React.ReactNode[] = [];
  const asciiCells: React.ReactNode[] = [];
  for (let col = 0; col < BYTES_PER_ROW; col++) {
    const index = rowStart + col;
    if (col >= count) {
      hexCells.push(<span key={`pad-${col}`}>{'   '}</span>);
      continue;
    }
    const b = data[index];
    const status = byteStatus?.(index);
    const selected = index >= selFrom && index < selTo;
    hexCells.push(
      <Cell key={`h-${index}`} data-hex-index={index} $status={status} $selected={selected}>
        {hex2(b)}
        {col === BYTES_PER_ROW - 1 ? '' : ' '}
      </Cell>,
    );
    const printable = b >= 0x20 && b <= 0x7e;
    asciiCells.push(
      <Cell key={`a-${index}`} data-hex-index={index} $status={status} $selected={selected}>
        {printable ? String.fromCharCode(b) : '.'}
      </Cell>,
    );
  }
  return (
    <Row>
      <Offset>{rowStart.toString(16).padStart(8, '0')}</Offset>
      <HexCols>{hexCells}</HexCols>
      <AsciiCols>{asciiCells}</AsciiCols>
    </Row>
  );
});

export interface HexRendererProps {
  bytes: ArrayBuffer | Uint8Array;
  /** Controlled selection; when provided the component reflects it instead of internal state. */
  selection?: HexSelection | null;
  /** Called when the user changes the selection by clicking/dragging. */
  onSelectionChange?: (selection: HexSelection | null) => void;
  /** Optional per-byte diff status for coloring (index -> status). */
  byteStatus?: (index: number) => HexByteStatus | undefined;
  /** Max bytes to render before truncating (defaults to 16 KiB). */
  maxBytes?: number;
}

function toUint8(bytes: ArrayBuffer | Uint8Array): Uint8Array {
  return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
}

/**
 * Interactive, read-only hex dump (offset | hex | ascii). Supports click/drag byte selection
 * (controllable for cross-pane sync in the diff view) and optional per-byte diff coloring.
 *
 * Rows are split into memoized {@link HexRow}s keyed by offset so that updating the selection
 * during a drag only re-renders the handful of rows it touches, not the whole buffer.
 */
const HexRenderer: React.FC<HexRendererProps> = ({ bytes, selection, onSelectionChange, byteStatus, maxBytes = DEFAULT_MAX_BYTES }) => {
  const data = useMemo(() => toUint8(bytes), [bytes]);
  const shown = Math.min(data.length, maxBytes);
  const truncated = data.length > shown;

  const [internalSel, setInternalSel] = useState<HexSelection | null>(null);
  const sel = selection !== undefined ? selection : internalSel;

  const anchorRef = useRef<number | null>(null);
  const draggingRef = useRef(false);

  const updateSelection = useCallback(
    (next: HexSelection | null) => {
      if (selection === undefined) setInternalSel(next);
      onSelectionChange?.(next);
    },
    [selection, onSelectionChange],
  );

  // end any in-progress drag on a global mouseup
  useEffect(() => {
    const onUp = () => {
      draggingRef.current = false;
    };
    window.addEventListener('mouseup', onUp);
    return () => window.removeEventListener('mouseup', onUp);
  }, []);

  const indexFromEvent = (e: React.MouseEvent): number | null => {
    const target = e.target as HTMLElement;
    const el = target.closest<HTMLElement>('[data-hex-index]');
    if (!el) return null;
    const idx = Number(el.dataset.hexIndex);
    return Number.isNaN(idx) ? null : idx;
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const idx = indexFromEvent(e);
    if (idx === null) return;
    e.preventDefault();
    anchorRef.current = idx;
    draggingRef.current = true;
    updateSelection({ offset: idx, length: 1 });
  };

  const handleMouseOver = (e: React.MouseEvent) => {
    if (!draggingRef.current || anchorRef.current === null) return;
    const idx = indexFromEvent(e);
    if (idx === null) return;
    const start = Math.min(anchorRef.current, idx);
    const end = Math.max(anchorRef.current, idx);
    updateSelection({ offset: start, length: end - start + 1 });
  };

  // stable list of row offsets; rebuilt only when the rendered byte count changes
  const rowStarts = useMemo(() => {
    const starts: number[] = [];
    for (let rowStart = 0; rowStart < shown; rowStart += BYTES_PER_ROW) starts.push(rowStart);
    return starts;
  }, [shown]);

  const selStart = sel ? sel.offset : -1;
  const selEnd = sel ? sel.offset + sel.length : -1;

  return (
    <Container onMouseDown={handleMouseDown} onMouseOver={handleMouseOver}>
      {rowStarts.map((rowStart) => {
        // intersect the selection with this row so only touched rows re-render
        const from = sel ? Math.max(selStart, rowStart) : -1;
        const to = sel ? Math.min(selEnd, rowStart + BYTES_PER_ROW) : -1;
        const hasSel = from < to;
        return (
          <HexRow
            key={rowStart}
            data={data}
            rowStart={rowStart}
            count={Math.min(BYTES_PER_ROW, shown - rowStart)}
            byteStatus={byteStatus}
            selFrom={hasSel ? from : -1}
            selTo={hasSel ? to : -1}
          />
        );
      })}
      {truncated && (
        <Notice>
          Showing the first {shown.toLocaleString()} of {data.length.toLocaleString()} bytes.
        </Notice>
      )}
    </Container>
  );
};

export default HexRenderer;
