import React, { useMemo } from 'react';

// project imports
import { ResultRenderProps } from '../props';
import CodeRenderer from '@components/shared/renderers/CodeRenderer';
import { stringToRenderableInput } from '@components/shared/renderers/detect';
import { FormatType } from '@utilities/rules/types';

// spec: ../ToolResult.spec.md

/**
 * Render a tool result as a read-only decompilation. Delegates to the shared {@link CodeRenderer}
 * (decomp syntax highlighting) so the viewer is identical to the entity view-only field and the
 * file-preview decomp option — no duplicated rendering logic.
 *
 * String results are shown verbatim; structured results are serialized to JSON text.
 */
const Decomposition: React.FC<ResultRenderProps> = ({ result }) => {
  const input = useMemo(() => {
    const value = result.result;
    const text = typeof value === 'string' ? value : (JSON.stringify(value, null, 2) ?? '');
    return stringToRenderableInput(text);
  }, [result]);

  return <CodeRenderer input={input} format={FormatType.Decomp} height="auto" />;
};

export default Decomposition;
