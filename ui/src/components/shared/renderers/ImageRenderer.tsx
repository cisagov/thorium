import React, { useEffect, useState } from 'react';
import styled from 'styled-components';

// project imports
import { imageMimeForName } from './detect';
import { FileRendererProps } from './types';
// spec: ./SPEC.md

const ImageWrapper = styled.div`
  display: flex;
  justify-content: center;
  align-items: flex-start;
`;

const Img = styled.img`
  max-width: 100%;
  height: auto;
  /* checkerboard so transparent images are visible against any theme */
  background-image:
    linear-gradient(45deg, var(--thorium-secondary-panel-bg) 25%, transparent 25%),
    linear-gradient(-45deg, var(--thorium-secondary-panel-bg) 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, var(--thorium-secondary-panel-bg) 75%),
    linear-gradient(-45deg, transparent 75%, var(--thorium-secondary-panel-bg) 75%);
  background-size: 16px 16px;
  background-position:
    0 0,
    0 8px,
    8px -8px,
    -8px 0;
  border-radius: 4px;
`;

/** Render image bytes via a blob URL `<img>`. Handles PNG/JPEG/GIF/WEBP/SVG/etc. */
const ImageRenderer: React.FC<FileRendererProps> = ({ input }) => {
  const [url, setUrl] = useState('');

  // Create the blob URL in an effect (not useMemo) and revoke it on cleanup. Keying on the stable
  // bytes/fileName — and creating a fresh URL each effect run — avoids the StrictMode double-invoke
  // race that revoked the URL before the <img> could load it.
  useEffect(() => {
    const mime = imageMimeForName(input.fileName);
    const blob = new Blob([input.bytes], mime ? { type: mime } : undefined);
    const objectUrl = URL.createObjectURL(blob);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [input.bytes, input.fileName]);

  if (!url) return null;
  return (
    <ImageWrapper>
      <Img src={url} alt={input.fileName || 'result image'} />
    </ImageWrapper>
  );
};

export default ImageRenderer;
