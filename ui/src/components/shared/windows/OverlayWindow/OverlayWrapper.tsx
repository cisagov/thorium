// project imports
import { Bounds, CanvasType } from '../bounds';

// window wrapper, we don't use styled components here because of the repetitive setting of element size/width which would result in many style sheets being created
const OverlayWrapper: React.FC<{
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

export default OverlayWrapper;
