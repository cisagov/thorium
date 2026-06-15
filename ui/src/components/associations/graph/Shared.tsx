import styled from 'styled-components';

// spec: ./AssociationGraph.spec.md

export const GraphWindow = styled.div<{ $bordered?: boolean }>`
  position: relative;
  background-color: var(--thorium-panel-bg);
  ${({ $bordered = true }) =>
    $bordered &&
    `
    border: 1px solid var(--thorium-panel-border);
    border-radius: 4px;
  `}
  /* default viewport-relative height for the standalone graph page and entity/file/repo detail views,
     which render the graph directly with no sizing parent. In the dashboard the tile stretches this to
     flex: 1 (flex-basis 0 overrides this height) so the canvas fills the square tile instead. */
  height: 90vh;
  overflow: hidden;

  /* the window is focusable so the keyboard shortcuts stay scoped to the graph; show a
     ring only for keyboard focus so mouse clicks on the canvas don't draw an outline */
  &:focus-visible {
    outline: 2px solid var(--thorium-highlight-panel-border);
    outline-offset: -2px;
  }
`;

export const GraphDiv = styled.div`
  z-index: 200;
  overflow: hidden;
  /* fill the GraphWindow so the ForceGraph3D canvas matches its container's real height (the tile in the
     dashboard, or GraphWindow's 90vh default elsewhere). min-height: 0 lets it shrink inside a flex tile
     shorter than 90vh instead of overflowing and being clipped from the bottom. */
  width: 100%;
  height: 100%;
  min-height: 0;
`;

export const TreeOverlayPanel = styled.div`
  position: absolute;
  z-index: 300;
  top: 8px;
  left: 8px;
  max-width: 30%;
  max-height: 60vh;
  min-width: 200px;
  background: color-mix(in srgb, var(--thorium-secondary-panel-bg) 82%, transparent);
  backdrop-filter: blur(8px);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;

  @media (max-width: 768px) {
    max-width: calc(100% - 16px);
  }
`;

export const TreeOverlayHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px 4px;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--thorium-text);
  position: sticky;
  top: 0;
  background: inherit;
  z-index: 1;
`;

export const LoadingOverlay = styled.div`
  position: absolute;
  inset: 0;
  z-index: 400;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
`;

// Muted, centered message shown over the graph canvas for empty ("No graph data") or error states.
export const GraphOverlayMessage = styled.div`
  color: var(--thorium-secondary-text);
  font-size: 0.9rem;
`;

export const PreviewContainer = styled.div`
  position: absolute;
  z-index: 300;
  top: 8px;
  right: 8px;
  background: color-mix(in srgb, var(--thorium-secondary-panel-bg) 90%, transparent);
  backdrop-filter: blur(8px);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
  width: fit-content;
  min-width: 240px;
  max-width: min(400px, 35vw);
`;

// scrollable content region of the preview overlay. Scrolling lives here (not on PreviewContainer) so
// the absolutely-positioned collapse button stays pinned to the corner instead of scrolling with content
export const PreviewScroll = styled.div`
  padding: 2px;
  max-height: 30vh;
  overflow-y: auto;
  overflow-x: auto;

  /* always reserve the scrollbar's gutter so the collapse button can sit just inside it (clear of the
     scrollbar) without the layout shifting when details overflow */
  scrollbar-gutter: stable;

  /* thin, themed scrollbar so it reads as part of the panel rather than a heavy native bar */
  scrollbar-width: thin;
  scrollbar-color: var(--thorium-panel-border) transparent;
  &::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  &::-webkit-scrollbar-thumb {
    background: var(--thorium-panel-border);
    border-radius: 4px;
  }
`;

// collapse control floating over the overlay's top-right corner, level with the summary's first line
// (kind/title). `right` clears the reserved scrollbar gutter so the button never overlaps the scrollbar.
export const PreviewCollapseButton = styled.button`
  position: absolute;
  /* aligns with the summary's first subtitle, which is inset by the EntitySummary wrapper's top margin */
  top: 8px;
  right: 10px;
  z-index: 1;
  width: 20px;
  height: 20px;
  background: none;
  border: none;
  color: var(--thorium-secondary-text, var(--thorium-text));
  cursor: pointer;
  padding: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: background 0.15s;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

export const PreviewToggleButton = styled.button`
  position: absolute;
  z-index: 300;
  top: 8px;
  right: 8px;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid var(--thorium-panel-border);
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background 0.15s;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

export const TreeOverlayToggle = styled.button`
  position: absolute;
  z-index: 300;
  top: 8px;
  left: 8px;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: 1px solid var(--thorium-panel-border);
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background 0.15s;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

// vertical stack for the always-visible camera navigation buttons, pinned bottom-right to
// mirror the toolbar's bottom-left placement without covering the top-right data preview
export const NavClusterContainer = styled.div`
  position: absolute;
  z-index: 500;
  bottom: 12px;
  right: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

export const MinimizeButton = styled.button`
  background: none;
  border: none;
  color: var(--thorium-secondary-text, var(--thorium-text));
  cursor: pointer;
  padding: 2px;
  display: flex;
  align-items: center;
  border-radius: 4px;
  transition: background 0.15s;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;
