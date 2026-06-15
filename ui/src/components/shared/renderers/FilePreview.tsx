import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { FaCopy, FaDownload } from 'react-icons/fa';
import { toast } from 'react-toastify';
import styled from 'styled-components';

// project imports
import { editorFormatHint, textOf } from './detect';
import FileRenderer from './FileRenderer';
import { detectRenderGroup, RENDER_LABELS } from './groups';
import { prettifiedSeed } from './prettify';
import { RenderableInput, RenderKind } from './types';
import { IconButton } from '@components/shared/buttons';
import { ButtonSize } from '@components/shared/buttons/types';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { downloadBlob } from '@utilities/download';

// spec: ./SPEC.md

const PreviewWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  // fill the overlay body and let PreviewBody (below) own scrolling
  flex: 1 1 auto;
  min-height: 0;
  min-width: 0;
`;

const Toolbar = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex: 0 0 auto;
`;

const RendererSelect = styled.select`
  padding: 4px 8px;
  font-size: 12px;
  color: var(--thorium-text);
  background: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 3px;
  cursor: pointer;
`;

// the single scroll owner for the preview content — every renderer below grows to its intrinsic
// size and is scrolled here (renderers no longer own overflow), so scrollbars don't stack
const PreviewBody = styled.div`
  flex: 1 1 auto;
  min-height: 0;
  min-width: 0;
  overflow: auto;
`;

export interface FilePreviewProps {
  input: RenderableInput;
  /** Download the original, unmodified file bytes. */
  onDownloadOriginal: () => void;
}

/**
 * Preview wrapper shown inside the result-file overlay window: a toolbar to pick which renderer to
 * view the file with (from {@link detectRenderGroup}) plus combined Copy/Download actions.
 *
 * The Copy/Download buttons toggle by active renderer: in the Raw Editor they act on the edited
 * buffer (held here so the toolbar can reach it); in every other view Download fetches the original
 * file and Copy copies the file's text. Copy is hidden for binary (Image/Hex) views.
 */
const FilePreview: React.FC<FilePreviewProps> = ({ input, onDownloadOriginal }) => {
  const group = useMemo(() => detectRenderGroup(input.fileName ?? '', input.bytes), [input.fileName, input.bytes]);
  const [selected, setSelected] = useState<RenderKind>(group.default);

  // keep the selection valid if the underlying file (and thus its group) changes
  const active = group.options.includes(selected) ? selected : group.default;
  const isEditor = active === RenderKind.Editor;
  const fileName = input.fileName || 'result.txt';

  // editor highlight language + the prettified seed (computed synchronously so the toolbar can
  // Copy/Download the editor content on the first click, before any edit event fires)
  const editorHint = useMemo(() => editorFormatHint(input.fileName ?? '', input.bytes), [input.fileName, input.bytes]);
  const seed = useMemo(() => prettifiedSeed(input), [input.text, input.bytes, input.fileName]);

  // the edited buffer lives here so it survives toggling Editor <-> other views; reset per file
  const [editedText, setEditedText] = useState<string | null>(null);
  useEffect(() => setEditedText(null), [input.bytes]);
  const editorText = editedText ?? seed;

  const showCopy = active !== RenderKind.Image && active !== RenderKind.Hex;

  const handleDownload = useCallback(() => {
    if (isEditor) downloadBlob(editorText, fileName);
    else onDownloadOriginal();
  }, [isEditor, editorText, fileName, onDownloadOriginal]);

  const handleCopy = useCallback(() => {
    const content = isEditor ? editorText : textOf(input);
    void navigator.clipboard.writeText(content);
    toast(isEditor ? 'Copied edited content to clipboard!' : 'Copied file content to clipboard!');
  }, [isEditor, editorText, input.text, input.bytes]);

  const downloadLabel = isEditor ? 'Download edited content' : 'Download original file';
  const copyLabel = isEditor ? 'Copy edited content' : 'Copy file content';

  return (
    <PreviewWrapper>
      <Toolbar>
        {group.options.length > 1 && (
          <RendererSelect aria-label="Select renderer" value={active} onChange={(e) => setSelected(e.target.value as RenderKind)}>
            {group.options.map((kind) => (
              <option key={kind} value={kind}>
                {RENDER_LABELS[kind]}
              </option>
            ))}
          </RendererSelect>
        )}
        {showCopy && (
          <OverlayTipTop tip={copyLabel}>
            <IconButton size={ButtonSize.XSmall} aria-label={copyLabel} onClick={handleCopy}>
              <FaCopy />
            </IconButton>
          </OverlayTipTop>
        )}
        <OverlayTipTop tip={downloadLabel}>
          <IconButton size={ButtonSize.XSmall} aria-label={downloadLabel} onClick={handleDownload}>
            <FaDownload />
          </IconButton>
        </OverlayTipTop>
      </Toolbar>
      <PreviewBody>
        <FileRenderer
          input={input}
          kind={active}
          value={isEditor ? editorText : undefined}
          onTextChange={isEditor ? setEditedText : undefined}
          formatHint={isEditor ? editorHint : undefined}
        />
      </PreviewBody>
    </PreviewWrapper>
  );
};

export default FilePreview;
