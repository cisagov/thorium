// spec: ../ToolResult.spec.md
import React, { useCallback, useState } from 'react';
import { FaDownload, FaEye } from 'react-icons/fa';
import styled from 'styled-components';

// project imports
import { ToolResultTabProps } from './types';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';
import { IconButton } from '@components/shared/buttons';
import { ButtonSize } from '@components/shared/buttons/types';
import { FilePreview } from '@components/shared/renderers';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { OverlayWindow, PositionType, Placement } from '@components/shared/windows';
import { downloadBlob } from '@utilities/download';
import { fetchResultFileCached } from '@utilities/resultFiles';
import { getCachedResultFile } from '@utilities/resultFileCache';
import { useAuth } from '@utilities/auth';

const FileList = styled.div`
  display: flex;
  flex-direction: column;
`;

// files sit directly on the result tile (no card background/border); a divider only separates
// stacked rows — the last row has none so it doesn't double up with the tile's own edge
const FileRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 4px;

  &:not(:last-child) {
    border-bottom: 1px solid var(--thorium-panel-border);
  }
`;

const FileName = styled.span`
  font-family: var(--bs-font-monospace, monospace);
  font-size: 0.9rem;
  color: var(--thorium-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const RowActions = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 0 0 auto;
`;

/**
 * The "Files" tab body: lists a result's files with per-file download and render actions.
 *
 * Rendering is deferred and user-initiated — clicking the eye downloads the bytes into the shared
 * result-file cache (see {@link fetchResultFileCached}) and opens a floating {@link OverlayWindow}
 * that renders the file via the modular {@link FileRenderer}. Closing a window removes it from view
 * but the shared cache keeps the bytes (subject to its LRU budget), so re-opening is usually instant.
 *
 * Bytes for currently-open windows are also held in local `openBytes` state so an open preview keeps
 * rendering even if the shared cache evicts that file to stay within its memory budget.
 */
const FilesTab: React.FC<ToolResultTabProps> = ({ result, sha256, tool }) => {
  const { checkCookie } = useAuth();
  const [openBytes, setOpenBytes] = useState<Record<string, ArrayBuffer>>({});
  const [openWindows, setOpenWindows] = useState<string[]>([]);
  const [loading, setLoading] = useState<string[]>([]);

  const errorHandler = useCallback(() => void checkCookie(), [checkCookie]);

  // fetch a file's bytes via the shared cache (keyed by this result's unique id)
  const fetchBytes = useCallback(
    (name: string): Promise<ArrayBuffer | null> => fetchResultFileCached(sha256, tool, result.id, name, errorHandler),
    [sha256, tool, result.id, errorHandler],
  );

  const handleRender = useCallback(
    async (name: string) => {
      // guard against a rapid double-click issuing two fetches before the first setLoading commits
      if (loading.includes(name)) return;
      let bytes: ArrayBuffer | null = openBytes[name] ?? getCachedResultFile(result.id, name) ?? null;
      if (!bytes) {
        setLoading((prev) => [...prev, name]);
        bytes = await fetchBytes(name);
        setLoading((prev) => prev.filter((n) => n !== name));
        if (!bytes) return;
      }
      // hold a reference for the open window so it renders even if the shared cache later evicts it
      const resolved = bytes;
      setOpenBytes((prev) => (prev[name] ? prev : { ...prev, [name]: resolved }));
      setOpenWindows((prev) => (prev.includes(name) ? prev : [...prev, name]));
    },
    [openBytes, loading, fetchBytes, result.id],
  );

  const handleDownload = useCallback(
    async (name: string) => {
      const bytes = await fetchBytes(name);
      if (bytes) downloadBlob(bytes, name);
    },
    [fetchBytes],
  );

  const closeWindow = useCallback((name: string) => {
    setOpenWindows((prev) => prev.filter((n) => n !== name));
    // drop our render-time reference; the shared cache keeps the bytes for instant re-open
    setOpenBytes((prev) => {
      if (!(name in prev)) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  }, []);

  const files = result.files ?? [];
  if (files.length === 0) {
    return <RenderErrorAlert page={false} message="This result has no files." />;
  }

  return (
    <>
      <FileList>
        {files.map((name) => (
          <FileRow key={name}>
            <FileName title={name}>{name}</FileName>
            <RowActions>
              <OverlayTipTop tip="Preview File">
                <IconButton
                  size={ButtonSize.XSmall}
                  aria-label={`View ${name}`}
                  disabled={loading.includes(name)}
                  onClick={() => void handleRender(name)}
                >
                  <FaEye />
                </IconButton>
              </OverlayTipTop>
              <OverlayTipTop tip="Download Result File">
                <IconButton size={ButtonSize.XSmall} aria-label={`Download ${name}`} onClick={() => void handleDownload(name)}>
                  <FaDownload />
                </IconButton>
              </OverlayTipTop>
            </RowActions>
          </FileRow>
        ))}
      </FileList>
      {openWindows.map((name) => (
        <OverlayWindow
          key={name}
          id={`result-file-${tool}-${result.id}-${name}`}
          show={true}
          title={name}
          width={760}
          height={560}
          positioning={PositionType.Fixed}
          placement={Placement.Center}
          onHide={() => closeWindow(name)}
        >
          {openBytes[name] && (
            <FilePreview input={{ bytes: openBytes[name], fileName: name }} onDownloadOriginal={() => void handleDownload(name)} />
          )}
        </OverlayWindow>
      ))}
    </>
  );
};

export default FilesTab;
