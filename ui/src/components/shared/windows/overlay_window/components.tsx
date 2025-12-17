import styled from 'styled-components';

// project imports
import { Bounds, CanvasType } from '../bounds';

// window wrapper, we don't use styled components here because of the repetitive setting of element size/width which would result in many style sheets being created
export const OverlayWrapper: React.FC<{
  children: any;
  className?: string;
  show: boolean;
  size: { width: number; height: number };
  position: { top: number | string; left: number | string };
  bounds: Bounds;
  onMouseDown: () => void;
  zIndex: number;
  ref: (node: HTMLDivElement | null) => void;
}> = ({ children, className, show, size, position, bounds, onMouseDown, zIndex, ref }) => {
  return (
    <div
      ref={ref}
      onMouseDown={onMouseDown}
      style={{
        display: show ? 'flex' : 'none',
        position: bounds.type === CanvasType.Viewport ? 'fixed' : 'absolute',
        top: `${position.top}px`,
        left: `${position.left}px`,
        width: `${size.width}px`,
        height: `${size.height}px`,
        overflowX: 'hidden',
        overflowY: 'auto',
        borderRadius: '5px',
        zIndex: zIndex,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.9)',
        backgroundColor: 'var(--thorium-panel-bg)',
        color: 'var(--thorium-text)',
      }}
      className={className}
    >
      {children}
    </div>
  );
};

export const OverlayHeader = styled.div<{ $zindex: number }>`
  display: flex;
  cursor: move;
  position: absolute;
  padding: 4px 4px 4px 10px;
  max-height: 36px; // maxHeight forces wrapping of extra things instead of increase in header height
  word-break: break-all; // breakup text at any point to ensure content doesn't overlap
  overflow-y: clip; // hide wrapped header content
  width: 100%; // width should match window dimensions
  font-size: 20px;
  border-bottom: solid 1px var(--thorium-panel-border);
  z-index: ${(props) => props.$zindex || 'auto'};
`;

export const OverlayBody = styled.div<{ $zindex: number }>`
  margin-top: 40px;
  padding: 4px;
  z-index: ${(props) => props.$zindex || 'auto'};
  position: absolute;
  overflow-y: auto;
  overflow-x: hidden;
  width: 100%;
  height: 100%;
  max-height: calc(100% - 40px);
`;

export const CloseButton = styled.button`
  width: 20px;
  height: 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: auto;
  background: transparent;
  border: none;
  font-size: 15px;
  color: var(--thorium-text);
  margin-left: 4px;
  margin-right: 4px;
  &:hover {
    color: var(--thorium-highlight-text);
  }
`;
