import styled from 'styled-components';
import { Popover } from 'react-bootstrap';

export const PreviewPopover = styled(Popover)`
  --bs-popover-max-width: 360px;
  --bs-popover-bg: var(--thorium-secondary-panel-bg);
  --bs-popover-border-color: var(--thorium-panel-border);
  --bs-popover-body-color: var(--thorium-text);
  --bs-popover-arrow-border: var(--thorium-panel-border);
  overflow: hidden;

  .popover-body {
    padding: 10px 14px;
    font-size: 0.82rem;
    overflow: hidden;
  }

  .preview-type {
    font-weight: 600;
    margin-bottom: 4px;
    color: var(--thorium-text);
  }

  .preview-field {
    margin-bottom: 2px;
    color: var(--thorium-secondary-text, var(--thorium-text));
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .preview-field strong {
    color: var(--thorium-text);
  }

  .preview-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
    overflow: hidden;
    max-width: 100%;
  }

  .preview-duplicate-warn {
    margin-top: 6px;
    padding: 3px 6px;
    font-size: 0.7rem;
    border-radius: 4px;
    background-color: rgba(232, 168, 56, 0.15);
    color: #b07d1a;
  }
`;
