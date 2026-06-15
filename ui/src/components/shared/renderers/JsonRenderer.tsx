import React, { useMemo } from 'react';
import { JSONTree } from 'react-json-tree';
import styled from 'styled-components';

// project imports
import { textOf } from './detect';
import { OceanJsonTheme, useJsonTreeInvert } from './jsonTheme';
import TextRenderer from './TextRenderer';
import { FileRendererProps } from './types';

// spec: ./SPEC.md

const TreeWrapper = styled.div`
  padding: 8px 12px;
  min-width: 0;
  font-size: 0.9rem;
`;

/**
 * Render bytes as a collapsible JSON tree. Falls back to the plain text renderer when the
 * content does not parse as JSON.
 */
const JsonRenderer: React.FC<FileRendererProps> = ({ input }) => {
  // invert the dark token palette on light-background themes so the tree stays legible
  const invertTheme = useJsonTreeInvert();
  const parsed = useMemo<{ ok: boolean; value: unknown }>(() => {
    const text = textOf(input);
    try {
      return { ok: true, value: JSON.parse(text) };
    } catch {
      return { ok: false, value: null };
    }
  }, [input.text, input.bytes]);

  if (!parsed.ok) {
    return <TextRenderer input={input} />;
  }

  return (
    <TreeWrapper>
      <JSONTree
        data={parsed.value}
        shouldExpandNodeInitially={() => true}
        hideRoot={true}
        theme={OceanJsonTheme}
        invertTheme={invertTheme}
      />
    </TreeWrapper>
  );
};

export default JsonRenderer;
