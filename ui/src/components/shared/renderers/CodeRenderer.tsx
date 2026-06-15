import React, { useCallback, useMemo } from 'react';

// project imports
import { formatFromFileName, textOf } from './detect';
import { FileRendererProps } from './types';
import CodeEditor from '@components/shared/inputs/code/CodeEditor';
import { FormatType } from '@utilities/rules/types';

// spec: ./SPEC.md

export interface CodeRendererProps extends FileRendererProps {
  /** Force a specific editor format instead of inferring from the file name. */
  format?: FormatType;
  /** Editor height (CSS value). */
  height?: string;
}

/**
 * Render bytes as read-only, syntax-highlighted code by reusing the shared CodeEditor in
 * disabled (view-only) mode.
 */
const CodeRenderer: React.FC<CodeRendererProps> = ({ input, format, height = '100%' }) => {
  const text = useMemo(() => textOf(input), [input.text, input.bytes]);
  const resolvedFormat = format ?? formatFromFileName(input.fileName);
  // CodeEditor is controlled; in read-only mode changes can't happen, so onChange is a no-op
  const noop = useCallback(() => {}, []);
  return <CodeEditor value={text} onChange={noop} format={resolvedFormat} height={height} disabled={true} />;
};

export default CodeRenderer;
