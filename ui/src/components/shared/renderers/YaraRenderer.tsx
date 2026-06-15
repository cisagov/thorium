import React, { useCallback, useMemo } from 'react';

// project imports
import { textOf } from './detect';
import { FileRendererProps } from './types';
import CodeEditor from '@components/shared/inputs/code/CodeEditor';
import { FormatType } from '@utilities/rules/types';
import { YaraRuleChecker } from '@utilities/rules/yara';

// spec: ./SPEC.md

/**
 * Render bytes as a read-only YARA rule using the shared CodeEditor with the YARA language and
 * rule checker (view-only). The checker still surfaces lint diagnostics so the viewer matches
 * the authoring experience.
 */
const YaraRenderer: React.FC<FileRendererProps> = ({ input }) => {
  const text = useMemo(() => textOf(input), [input.text, input.bytes]);
  const checker = useMemo(() => new YaraRuleChecker(), []);
  const noop = useCallback(() => {}, []);
  return <CodeEditor value={text} onChange={noop} format={FormatType.YARA} checker={checker} height="100%" disabled={true} />;
};

export default YaraRenderer;
