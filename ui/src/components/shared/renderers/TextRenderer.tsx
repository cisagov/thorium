import React, { useMemo } from 'react';
import styled from 'styled-components';

// project imports
import { textOf } from './detect';
import { FileRendererProps } from './types';
// spec: ./SPEC.md

const Pre = styled.pre`
  margin: 0;
  padding: 8px;
  min-width: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--bs-font-monospace, monospace);
  font-size: 0.85rem;
  color: var(--thorium-text);
`;

/** Plain-text fallback renderer. */
const TextRenderer: React.FC<FileRendererProps> = ({ input }) => {
  const text = useMemo(() => textOf(input), [input.text, input.bytes]);
  return <Pre>{text}</Pre>;
};

export default TextRenderer;
