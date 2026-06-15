import styled from 'styled-components';

// project imports
// ExpandToggle/ToggleRow live in shared/buttons; re-export from the barrel for existing importers.
export { ExpandToggle, ToggleRow } from '@components/shared/buttons';

/** Themed container for a single tool-result tile (replaces the legacy react-bootstrap card). */
export const ToolResultCard = styled.div`
  margin-top: 0.75rem;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

export const CardHeader = styled.div`
  /* tight top padding; no bottom padding so the inline tab underline sits on the header divider */
  padding: 3px 14px;
  border-bottom: 1px solid var(--thorium-panel-border);

  background-color: var(--thorium-panel-bg);
  border-top-right-radius: 8px;
  border-top-left-radius: 8px;

  position: sticky;
  top: 47px; //header size
  z-index: 20;
`;

/** Single-row header: title group, inline tabs, and right-aligned controls. */
export const TitleRow = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
`;

export const TitleGroup = styled.div`
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
`;

/** Wrapper that lets the inline Tabs grow and scroll between the title and controls. */
export const HeaderTabs = styled.div`
  flex: 1 1 auto;
  min-width: 0;
`;

export const TitleLink = styled.a`
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: var(--thorium-text);
  text-decoration: none;
  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

export const ToolName = styled.span`
  font-size: 1.1rem;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

export const ToolVersion = styled.span`
  font-size: 0.85rem;
  color: var(--thorium-secondary-text);
  font-family: var(--bs-font-monospace, monospace);
`;

export const HeaderControls = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
`;

/** Compact themed native select used for the version + diff version pickers. */
export const VersionSelect = styled.select`
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  padding: 5px 8px;
  font-size: 0.85rem;
  cursor: pointer;
  max-width: 220px;

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--thorium-link-text);
  }
`;

export const CardBody = styled.div`
  padding: 6px 12px;
`;

/**
 * Non-scrolling wrapper that positions the fade overlay. The fade must live here (not inside the
 * scroll area) so it stays pinned to the bottom of the collapsed viewport instead of scrolling away.
 */
export const ClipViewport = styled.div`
  position: relative;
`;

/**
 * The body's scroll area. When collapsed it's capped to a default height and scrolls internally so
 * long content can be previewed without expanding the tile; when expanded it grows to full height.
 */
export const ScrollArea = styled.div<{ $collapsed: boolean; $maxHeight: number }>`
  ${({ $collapsed, $maxHeight }) =>
    $collapsed
      ? `
      max-height: ${$maxHeight}px;
      overflow-y: auto;
`
      : `
    max-height: none;
    overflow: visible;
`}
`;

/** Fade-out gradient shown at the bottom of a collapsed body to hint at more content. */
export const FadeOverlay = styled.div`
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 48px;
  pointer-events: none;
  background: linear-gradient(to bottom, transparent, var(--thorium-panel-bg));
`;
