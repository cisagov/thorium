import React, { useMemo, useRef, useCallback } from 'react';
import CodeEditor from './CodeEditor';
import { toText } from './serialize';
import type { RuleChecker } from '@utilities/rules/types';
import { FormatType } from '@utilities/rules/types';

export interface ImagePipelineEditorProps {
  value: Record<string, unknown>;
  onChange: (obj: Record<string, unknown> | null) => void;
  checker: RuleChecker;
  format: FormatType;
  height?: string;
  disabled?: boolean;
}

const ImagePipelineEditor: React.FC<ImagePipelineEditorProps> = ({
  value,
  onChange,
  checker,
  format,
  height = '500px',
  disabled = false,
}) => {
  const lastSentRef = useRef<Record<string, unknown> | null>(null);
  const editorTextRef = useRef('');
  const lastFormatRef = useRef<FormatType>(format);

  const serialized = useMemo(() => {
    // If value is the same object we just sent from editor onChange, keep the
    // current editor text to avoid re-serialization which resets cursor position.
    if (value === lastSentRef.current && editorTextRef.current && format === lastFormatRef.current) {
      return editorTextRef.current;
    }
    const t = toText(value, format);
    editorTextRef.current = t;
    lastFormatRef.current = format;
    return t;
  }, [value, format]);

  const handleChange = useCallback(
    (text: string, parsed: unknown) => {
      editorTextRef.current = text;
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        const obj = parsed as Record<string, unknown>;
        lastSentRef.current = obj;
        onChange(obj);
      } else {
        lastSentRef.current = null;
        onChange(null);
      }
    },
    [onChange],
  );

  return <CodeEditor value={serialized} onChange={handleChange} checker={checker} format={format} height={height} disabled={disabled} />;
};

export default ImagePipelineEditor;
