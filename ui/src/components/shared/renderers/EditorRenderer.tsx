import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';

// project imports
import { formatFromFileName } from './detect';
import { prettifiedSeed } from './prettify';
import { FileRendererProps } from './types';
import CodeEditor from '@components/shared/inputs/code/CodeEditor';

// spec: ./SPEC.md

const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  // fill the preview body so the editor sizes to the window and owns its own scroll
  height: 100%;
  min-height: 0;
`;

/**
 * Editable raw-text view. Copy/Download of the edited buffer live in the {@link FilePreview}
 * toolbar, which drives this renderer as a **controlled** component (passing `value` +
 * `onTextChange`). When rendered without those props (e.g. forced via the registry), it falls back
 * to self-managed state seeded from the file's (prettified) text.
 *
 * Edits are local only — result files are immutable, so there is no save-back to Thorium.
 */
const EditorRenderer: React.FC<FileRendererProps> = ({ input, value, onTextChange, formatHint }) => {
  const controlled = value !== undefined;
  const format = formatHint ?? formatFromFileName(input.fileName);

  // uncontrolled fallback: seed once from the (prettified) file text and report changes upward
  const seed = useMemo(() => prettifiedSeed(input), [input.text, input.bytes, input.fileName]);
  const [internal, setInternal] = useState(seed);
  // track the seed the internal buffer was last synced to so a new input file resets it (the seed is
  // recomputed when input.bytes/text/fileName change; without this the uncontrolled editor would keep
  // showing the previous file's text)
  const seedRef = useRef(seed);

  // in the uncontrolled fallback, re-seed the buffer whenever the source file changes and report the
  // fresh text upward so a consumer's toolbar acts on the current file (also covers the initial mount)
  useEffect(() => {
    if (controlled) return;
    if (seedRef.current !== seed) {
      seedRef.current = seed;
      setInternal(seed);
    }
    onTextChange?.(seedRef.current);
  }, [controlled, seed, onTextChange]);

  const text = controlled ? value : internal;
  const handleChange = useCallback(
    (next: string) => {
      if (!controlled) setInternal(next);
      onTextChange?.(next);
    },
    [controlled, onTextChange],
  );

  return (
    <Wrapper>
      <CodeEditor value={text} onChange={handleChange} format={format} height="100%" disabled={false} />
    </Wrapper>
  );
};

export default EditorRenderer;
