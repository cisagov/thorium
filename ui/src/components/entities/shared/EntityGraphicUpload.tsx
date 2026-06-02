import React, { useMemo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';
import { FaTimes } from 'react-icons/fa';

const ACCEPTED_TYPES = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/gif': ['.gif'],
  'image/bmp': ['.bmp'],
  'image/svg+xml': ['.svg'],
};

const MAX_SIZE = 5 * 1024 * 1024;

const Container = styled.div`
  width: 100%;
`;

const DropArea = styled.div<{ $active?: boolean; $reject?: boolean }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 16px;
  min-height: 120px;
  border: 2px dashed
    ${(props) =>
      props.$reject ? 'var(--thorium-danger-bg)' : props.$active ? 'var(--thorium-info-secondary-bg)' : 'var(--thorium-panel-border)'};
  border-radius: 6px;
  background-color: var(--thorium-secondary-panel-bg);
  color: var(--thorium-secondary-text);
  cursor: pointer;
  transition: border-color 0.2s;
  font-size: 13px;

  &:hover {
    border-color: var(--thorium-info-secondary-bg);
  }
`;

const PreviewContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  background-color: var(--thorium-secondary-panel-bg);
`;

const PreviewImage = styled.img`
  width: 64px;
  height: 64px;
  object-fit: contain;
  border-radius: 4px;
  border: 1px solid var(--thorium-panel-border);
  background-color: var(--thorium-panel-bg);
`;

const FileInfo = styled.div`
  flex: 1;
  font-size: 12px;
  color: var(--thorium-text);
  overflow: hidden;

  span {
    display: block;
    color: var(--thorium-secondary-text);
    font-size: 11px;
  }
`;

const RemoveButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  color: var(--thorium-secondary-text);
  cursor: pointer;
  padding: 4px 8px;
  font-size: 11px;
  gap: 4px;

  &:hover {
    color: var(--thorium-danger-bg);
    border-color: var(--thorium-danger-bg);
  }
`;

const ErrorText = styled.div`
  color: var(--thorium-danger-bg);
  font-size: 11px;
  margin-top: 4px;
`;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface EntityGraphicUploadProps {
  file: File | null;
  onChange: (file: File | null) => void;
  existingImageUrl?: string | null;
  onClearExisting?: () => void;
}

const EntityGraphicUpload: React.FC<EntityGraphicUploadProps> = ({ file, onChange, existingImageUrl, onClearExisting }) => {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) {
        onChange(accepted[0]);
      }
    },
    [onChange],
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject, fileRejections } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    maxSize: MAX_SIZE,
    multiple: false,
  });

  const previewUrl = useMemo(() => {
    if (file) return URL.createObjectURL(file);
    return null;
  }, [file]);

  const displayUrl = previewUrl ?? existingImageUrl;
  const displayName = file?.name ?? null;
  const displaySize = file ? formatSize(file.size) : null;

  const handleRemove = () => {
    if (file) {
      onChange(null);
    } else if (existingImageUrl && onClearExisting) {
      onClearExisting();
    }
  };

  const rejectionError = fileRejections.length > 0 ? fileRejections[0].errors.map((e) => e.message).join(', ') : null;

  if (displayUrl) {
    return (
      <Container>
        <PreviewContainer>
          <PreviewImage src={displayUrl} alt="Entity graphic" />
          <FileInfo>
            {displayName && <div>{displayName}</div>}
            {displaySize && <span>{displaySize}</span>}
            {!displayName && <div>Current graphic</div>}
          </FileInfo>
          <RemoveButton onClick={handleRemove} title={file ? 'Remove selected file' : 'Clear existing graphic'}>
            <FaTimes size={10} />
            {file ? 'Remove' : 'Clear'}
          </RemoveButton>
        </PreviewContainer>
        {rejectionError && <ErrorText>{rejectionError}</ErrorText>}
      </Container>
    );
  }

  return (
    <Container>
      <DropArea {...getRootProps()} $active={isDragActive} $reject={isDragReject}>
        <input {...getInputProps()} />
        <div>Drop an image here or click to select</div>
        <span style={{ fontSize: '11px', marginTop: 4 }}>PNG, JPEG, GIF, BMP, SVG — max 5 MB</span>
      </DropArea>
      {rejectionError && <ErrorText>{rejectionError}</ErrorText>}
    </Container>
  );
};

export default EntityGraphicUpload;
