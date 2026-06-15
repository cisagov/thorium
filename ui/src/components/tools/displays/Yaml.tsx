import React, { useCallback, useMemo } from 'react';
import { stringify as yamlStringify } from 'yaml';

// project imports
import { ResultRenderProps } from '../props';
import CodeEditor from '@components/shared/inputs/code/CodeEditor';
import { FormatType } from '@utilities/rules/types';

// spec: ../ToolResult.spec.md

/**
 * Render a tool result as read-only YAML using the shared CodeEditor (YAML syntax highlighting).
 *
 * String results are shown verbatim; structured results are serialized to YAML.
 */
const Yaml: React.FC<ResultRenderProps> = ({ result }) => {
  const text = useMemo(() => {
    const value = result.result;
    if (typeof value === 'string') return value;
    try {
      return yamlStringify(value);
    } catch {
      return JSON.stringify(value, null, 2) ?? '';
    }
  }, [result]);

  const noop = useCallback(() => {}, []);
  return <CodeEditor value={text} onChange={noop} format={FormatType.YAML} height="auto" disabled={true} />;
};

export default Yaml;
