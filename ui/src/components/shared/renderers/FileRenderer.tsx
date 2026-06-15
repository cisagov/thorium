import React, { useMemo } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

// project imports
import { detectRenderKind } from './detect';
import { RENDERERS } from './registry';
import { FileRendererProps, RenderKind } from './types';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';

// spec: ./SPEC.md

/**
 * The single public entry point for the modular renderer system: given raw bytes (and an
 * optional file name), detect the content kind and render it with the matching renderer.
 *
 * Callers may force a specific `kind` to bypass detection. Wrapped in an `ErrorBoundary` so a
 * malformed file can't crash the surrounding view.
 */
const FileRenderer: React.FC<FileRendererProps> = ({ input, kind, value, onTextChange, formatHint }) => {
  const resolvedKind = useMemo(() => kind ?? detectRenderKind(input.fileName ?? '', input.bytes), [kind, input.bytes, input.fileName]);
  // an empty file has nothing to render — tell the user rather than showing a blank pane
  if (input.bytes.byteLength === 0) {
    return <AlertBanner severity={Severity.Info}>This file is empty.</AlertBanner>;
  }
  const Renderer = RENDERERS[resolvedKind] ?? RENDERERS[RenderKind.Text];
  return (
    <ErrorBoundary fallback={<RenderErrorAlert page={false} message="Unable to render this file." />}>
      {/* value/onTextChange/formatHint are editor-only; other renderers ignore them */}
      <Renderer input={input} kind={resolvedKind} value={value} onTextChange={onTextChange} formatHint={formatHint} />
    </ErrorBoundary>
  );
};

export default FileRenderer;
