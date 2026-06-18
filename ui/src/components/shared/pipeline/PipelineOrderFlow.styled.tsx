import styled from 'styled-components';

// project imports
import { BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';
import { NODE_HEIGHT } from './order';

export const FlowContainer = styled.div<{ $height: number }>`
  position: relative;
  width: 100%;
  height: ${(props) => props.$height}px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  background-color: var(--thorium-secondary-panel-bg);
  overflow: hidden;

  .react-flow {
    --xy-node-border-default: none;
    --xy-node-border-selected-default: 1px solid var(--thorium-highlight-panel-border);
    --xy-handle-background-color-default: var(--thorium-info-secondary-bg);
    --xy-edge-stroke-default: var(--thorium-secondary-text);
    --xy-edge-stroke-selected-default: var(--thorium-info-secondary-bg);
  }

  .react-flow__background {
    background-color: transparent;
  }

  [theme='Crab'] & .react-flow__edge.animated path {
    stroke-dasharray: 4 6;
    stroke-opacity: 0.3;
    animation: none;
  }
`;

// Drag handle on the bottom edge of the canvas; the user drags it vertically to resize the diagram
// height (clamped between a min and max in the component).
export const ResizeHandle = styled.div`
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 14px;
  cursor: ns-resize;
  color: var(--thorium-secondary-text);
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-bottom: none;
  border-radius: 6px 6px 0 0;
  font-size: 10px;
  user-select: none;
  touch-action: none;

  &:hover {
    background-color: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }

  svg {
    pointer-events: none;
  }
`;

export const StepNodeWrapper = styled.div<{ $parallel?: boolean; $banned?: boolean }>`
  display: flex;
  align-items: center;
  height: ${NODE_HEIGHT}px;
  box-sizing: border-box;
  padding: 8px 16px;
  background-color: ${(props) => (props.$banned ? 'var(--thorium-danger-bg)' : 'var(--thorium-panel-bg)')};
  border: 1px solid ${(props) => (props.$banned ? 'var(--thorium-danger, #e74c3c)' : 'var(--thorium-panel-border)')};
  border-left: 3px solid
    ${(props) => {
      if (props.$banned) return 'var(--thorium-danger, #e74c3c)';
      return props.$parallel ? 'var(--thorium-warning-bg)' : 'var(--thorium-info-secondary-bg)';
    }};
  border-radius: 4px;
  font-family: monospace;
  font-size: 12px;
  font-weight: 600;
  color: var(--thorium-text);
  white-space: nowrap;
  /* The node's box width is set per-stage on the ReactFlow node (uniform stage width for handle
     alignment); fill it so the visible node matches the box and the right-edge handle sits flush. */
  width: 100%;
  min-width: 120px;
  max-width: 240px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);

  .react-flow__handle {
    width: 6px;
    height: 6px;
    min-width: 6px;
    min-height: 6px;
    border: none;
    opacity: 0.6;
  }
`;

export const StepLabel = styled.span`
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  overflow: hidden;

  > span {
    overflow: hidden;
    text-overflow: ellipsis;
  }
`;

// Hover tooltip showing a node's full image name. Rendered via a portal to document.body so it
// escapes the diagram's `overflow: hidden` and the ReactFlow viewport's zoom transform; positioned
// with fixed coords from the node's screen rect. Themed to match the app's tooltips.
export const NodeTooltip = styled.div`
  position: fixed;
  transform: translate(-50%, calc(-100% - 8px));
  z-index: 2000;
  pointer-events: none;
  background-color: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-highlight-panel-border);
  border-radius: 6px;
  padding: 4px 8px;
  font-family: monospace;
  font-size: 12px;
  white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
`;

export const BanIcon = styled.span`
  color: var(--thorium-danger, #e74c3c);
  font-size: 10px;
  display: flex;
  align-items: center;
`;

export const TerminalNodeWrapper = styled.div`
  width: 12px;
  height: ${NODE_HEIGHT}px;
  display: flex;
  align-items: center;
  justify-content: center;

  &::before {
    content: '';
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background-color: var(--thorium-secondary-text);
  }

  .react-flow__handle {
    width: 1px;
    height: 1px;
    min-width: 1px;
    min-height: 1px;
    border: none;
    opacity: 0;
  }
`;

export const BarrierNodeWrapper = styled.div<{ $height: number }>`
  width: 2px;
  height: ${(props) => props.$height}px;
  background-color: var(--thorium-secondary-text);
  opacity: 0.5;

  .react-flow__handle {
    width: 1px;
    height: 1px;
    min-width: 1px;
    min-height: 1px;
    border: none;
    opacity: 0;
  }
`;

export const OrderChangeBar = styled.div`
  display: flex;
  justify-content: center;
  gap: ${BUTTON_BAR_GAP};
  padding: 8px 0 0;
`;

const OrderBtn = styled.button`
  padding: 4px 16px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition:
    background 0.15s,
    border-color 0.15s;
`;

export const ApplyBtn = styled(OrderBtn)`
  background: var(--thorium-ok-bg);
  color: var(--thorium-button-text);
  border: 1px solid var(--thorium-ok-bg);

  &:hover {
    filter: brightness(1.1);
  }
`;

export const DiscardBtn = styled(OrderBtn)`
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-secondary-text);
  border: 1px solid var(--thorium-panel-border);

  &:hover {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
    border-color: var(--thorium-highlight-panel-border);
  }
`;

export const ContextMenuOverlay = styled.div<{ $top: number; $left: number }>`
  position: fixed;
  top: ${(props) => props.$top}px;
  left: ${(props) => props.$left}px;
  z-index: 1000;
  min-width: 140px;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
  padding: 4px 0;
`;

export const ContextMenuItem = styled.div`
  padding: 6px 14px;
  font-size: 12px;
  color: var(--thorium-text);
  cursor: pointer;
  white-space: nowrap;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

export const ImageSelectOverlay = styled.div<{ $top: number; $left: number }>`
  position: fixed;
  top: ${(props) => props.$top}px;
  left: ${(props) => props.$left}px;
  z-index: 1000;
  width: 220px;
`;
