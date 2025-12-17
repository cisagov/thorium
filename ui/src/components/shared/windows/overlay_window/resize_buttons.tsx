// project imports
import { Bounds } from '../bounds';
import { Placement } from '../placement';
import { getResizeCursorType } from '../resize';

// Button bumpers to enable triggering window resizing
export const ResizeButtons: React.FC<{
  size: { height: number; width: number };
  onResizeMouseDown: (e: React.MouseEvent<HTMLButtonElement>, clickedLocation: Placement) => void;
  zIndex: number | string;
  position: { top: number; left: number };
  bounds: Bounds;
  cornerDepth?: number;
  edgeDepth?: number;
}> = ({ size, onResizeMouseDown, zIndex, position, bounds, cornerDepth = 12, edgeDepth = 2 }) => {
  // describe all eight handles with their placement and style-patch
  const bumperButtons: Array<{
    placement: Placement;
    style: React.CSSProperties;
  }> = [
    // corners
    {
      placement: Placement.BottomRight,
      style: { width: cornerDepth, height: cornerDepth, right: 0, bottom: 0 },
    },
    {
      placement: Placement.BottomLeft,
      style: { width: cornerDepth, height: cornerDepth, right: size.width - cornerDepth, bottom: 0 },
    },
    {
      placement: Placement.TopLeft,
      style: { width: cornerDepth, height: cornerDepth, right: size.width - cornerDepth, bottom: size.height - cornerDepth },
    },
    {
      placement: Placement.TopRight,
      style: { width: cornerDepth, height: cornerDepth, right: 0, bottom: size.height - cornerDepth },
    },
    // edges
    {
      placement: Placement.Bottom,
      style: { width: size.width - cornerDepth * 2, height: edgeDepth, right: cornerDepth, bottom: 0 },
    },
    {
      placement: Placement.Top,
      style: { width: size.width - cornerDepth * 2, height: edgeDepth, right: cornerDepth, bottom: size.height - edgeDepth },
    },
    {
      placement: Placement.Right,
      style: { width: edgeDepth, height: size.height - cornerDepth * 2, right: 0, bottom: cornerDepth },
    },
    {
      placement: Placement.Left,
      style: { width: edgeDepth, height: size.height - cornerDepth * 2, right: size.width - edgeDepth, bottom: cornerDepth },
    },
  ];
  return (
    <>
      {bumperButtons.map(({ placement, style }) => (
        <button
          key={placement}
          onMouseDown={(e) => onResizeMouseDown(e, placement)}
          style={{
            position: 'absolute',
            backgroundColor: 'transparent',
            border: 0,
            zIndex,
            cursor: getResizeCursorType(placement, position, size, bounds),
            ...style,
          }}
        />
      ))}
    </>
  );
};
