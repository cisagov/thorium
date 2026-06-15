import React, { useMemo, useState } from 'react';
import styled from 'styled-components';

// project imports
import HexRenderer from './HexRenderer';
import HexValueInspector from './HexValueInspector';
import { HexSelection } from './types';
import { FileRendererProps } from '../types';
// spec: ../SPEC.md

// wrap so the value inspector drops below the dump when the container is too narrow to hold both
// side by side (instead of overlapping it); at wide widths they sit side by side
const Layout = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: flex-start;
`;

// min-width sets the wrap threshold: the pane won't shrink below this, so once dump + inspector no
// longer fit the inspector wraps to the next line. The dump itself scrolls horizontally inside.
const HexPane = styled.div`
  flex: 1 1 auto;
  min-width: 22rem;
`;

/**
 * Standalone binary file view: an interactive hex dump alongside a value inspector that decodes
 * the current selection. Used by the {@link FileRenderer} for `RenderKind.Hex`.
 */
const HexView: React.FC<FileRendererProps> = ({ input }) => {
  const data = useMemo(() => new Uint8Array(input.bytes), [input.bytes]);
  const [selection, setSelection] = useState<HexSelection | null>(null);
  return (
    <Layout>
      <HexPane>
        <HexRenderer bytes={data} selection={selection} onSelectionChange={setSelection} />
      </HexPane>
      <HexValueInspector bytes={data} selection={selection} title={input.fileName || 'Value'} />
    </Layout>
  );
};

export default HexView;
