import { useRef, useState, useEffect, useLayoutEffect } from 'react';

// project imports
import { boundElementSize } from '../utilities';
import { calculateCanvasResizeWindowPosition, calculateWindowResizeState, ElementSize, getMarginRatios } from '../resize';
import { getCanvasType, getCanvasBounds, Padding, PositionType } from '../bounds';
import { ElementPosition, getInitialPosition, Placement } from '../placement';
import { clamp } from '../utilities';
import { useWindowNode } from '../WindowManager/use_window_node';
import CloseButton from './CloseButton';
import OverlayBody from './OverlayBody';
import OverlayHeader from './OverlayHeader';
import OverlayWrapper from './OverlayWrapper';
import ResizeButtons from './ResizeButtons';

const DEFAULT_EDGE_PADDING = { start: 4, end: 4, top: 4, bottom: 4 };

const OverlayWindow: React.FC<{
  className?: string; // any classes to apply to Modal wrapper
  children: React.ReactNode; // children within box
  show: boolean; // whether box is visible
  title?: string; // optional modal header title
  placement?: Placement | string; // initial placement of box
  fixed?: boolean; // box is draggable
  width?: number; // initial width of box
  height?: number; // initial height of box
  positioning?: PositionType;
  onHide?: () => void; // callback for when box is "exited"
  padding?: Padding; // padding between window and relative placed element
  parentRef?: React.RefObject<HTMLElement | null>; // optional parent ref to override default relative positioning for absolute positioned windows
  id?: string; // string id for use when window is rendered directly by window manager
}> = ({
  className,
  children,
  show,
  title,
  width = 500,
  height = 500,
  fixed = false,
  positioning = PositionType.Absolute,
  placement = Placement.Bottom, // initial placement compared to ancestor
  onHide,
  padding = DEFAULT_EDGE_PADDING,
  parentRef = undefined,
  id,
}) => {
  const { nodeRef, windowRef, windowZRange, onWindowClick, onWindowClose, margin: managerMargin } = useWindowNode<HTMLDivElement>(id);
  const [bounds, setBounds] = useState(() => getCanvasBounds(getCanvasType(positioning), managerMargin));
  // position is relative to next ancestor when PositionType is not fixed to viewport or when using Fixed positioning with a passed in valid parentRef
  const isInitialRelative = positioning == PositionType.Fixed && parentRef == undefined ? false : true;
  const [position, setPosition] = useState<ElementPosition>(() =>
    getInitialPosition({ width, height }, placement, bounds, padding, isInitialRelative, parentRef, nodeRef),
  );
  const [size, setSize] = useState<ElementSize>(boundElementSize({ width, height }, bounds, padding));
  const [resizing, setResizing] = useState(false);
  const [dragging, setDragging] = useState(false);
  // start ref position info for moving and resizing
  const windowStateRef = useRef({
    size, // current size of window
    position, // current position of window
    marginRatio: getMarginRatios(position, size), // margin ratios at start of render
    clickedLocation: Placement.Center, // position of element initiating window resize
    start: { width: 0, height: 0, top: position.top, left: position.left, pointerX: 0, pointerY: 0 }, // start dimensions, position and mouse location before actions
    bounds: bounds,
    resized: false, // window has changed size
    moved: false, // window has been repositioned
  });

  // force repositioning after parentRef current set on initial render
  useLayoutEffect(() => {
    setPosition(getInitialPosition({ width, height }, placement, bounds, padding, isInitialRelative, parentRef, nodeRef));
  }, []);

  const onDragMouseDown = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (resizing || fixed) return;
    // Only start drag when clicking the header area
    setDragging(true);
    // Start values change only when mousing down
    windowStateRef.current = {
      ...windowStateRef.current,
      start: {
        ...windowStateRef.current.start,
        pointerX: e.clientX,
        pointerY: e.clientY,
        top: position.top,
        left: position.left,
        width: size.width,
        height: size.height,
      },
    };
  };

  const onResizeMouseDown = (e: React.MouseEvent<HTMLButtonElement>, clickedLocation: Placement) => {
    e.stopPropagation();
    // disable resize when dragging or when configured for a fixed position
    if (dragging || fixed) return;
    setResizing(true);
    // start values change only when mousing down
    windowStateRef.current = {
      ...windowStateRef.current,
      clickedLocation: clickedLocation,
      start: {
        ...windowStateRef.current.start,
        pointerX: e.clientX,
        pointerY: e.clientY,
        top: position.top,
        left: position.left,
        width: size.width,
        height: size.height,
      },
    };
  };

  const handleWindowDrag = (dx: number, dy: number) => {
    const { start, bounds, size: windowSize } = windowStateRef.current;
    // we no longer want to change location with window resizing
    windowStateRef.current.moved = true;
    const updatedPosition = {
      top: clamp(start.top - dy, bounds.top, bounds.height - start.height),
      left: clamp(start.left - dx, bounds.start, bounds.width - start.width),
    };
    // apply bounds of document
    setPosition(updatedPosition);
    windowStateRef.current = {
      ...windowStateRef.current,
      position: updatedPosition,
      marginRatio: getMarginRatios(updatedPosition, windowSize),
    };
  };

  const handleWindowResize = (dx: number, dy: number) => {
    const { start, bounds, clickedLocation } = windowStateRef.current;
    // calculate resized window dimensions and location
    const {
      position: updatedPosition,
      size: updatedSize,
      resized,
    } = calculateWindowResizeState(start, { width: width, height: height }, dx, dy, bounds, clickedLocation);
    // if window dimensions changed, it was been resized
    windowStateRef.current.resized = resized;
    // update position and changed ref values
    setPosition(updatedPosition);
    setSize(updatedSize);
    windowStateRef.current = {
      ...windowStateRef.current,
      size: updatedSize,
      position: updatedPosition,
      marginRatio: getMarginRatios(updatedPosition, updatedSize),
    };
  };

  // Handle mouse moves that occur when dragging or resizing the target window
  const handleMouseMove = (e: MouseEvent) => {
    // calculate the change in X/Y for cursor: down/right positive, up/left negative
    const dx = windowStateRef.current.start.pointerX - e.clientX;
    const dy = windowStateRef.current.start.pointerY - e.clientY;
    if (dragging) {
      handleWindowDrag(dx, dy);
    } else if (resizing) {
      handleWindowResize(dx, dy);
    }
  };

  // Resize and reposition the window based on changes to the display canvas size (viewport or document)
  const handleCanvasResize = () => {
    // get updated canvas (viewport or document) size
    const canvasBounds = getCanvasBounds(getCanvasType(positioning), managerMargin);
    // bound element size to updated canvas dimensions
    const updatedSize = boundElementSize(
      windowStateRef.current.resized ? windowStateRef.current.size : { width: width, height: height },
      canvasBounds,
      padding,
    );
    // default position is the initial position when not moved
    let updatedPosition = getInitialPosition(updatedSize, placement, canvasBounds, padding, isInitialRelative, parentRef, nodeRef);
    // only update location on resize when window was not initially moved or resized
    if (windowStateRef.current.moved) {
      updatedPosition = calculateCanvasResizeWindowPosition(
        windowStateRef.current.bounds,
        canvasBounds,
        updatedSize,
        padding,
        windowStateRef.current.marginRatio,
        windowStateRef.current.position,
      );
    }
    // update window ref state
    windowStateRef.current = {
      ...windowStateRef.current,
      position: updatedPosition,
      size: updatedSize,
      bounds: canvasBounds,
    };
    setSize(updatedSize);
    setPosition(updatedPosition);
    setBounds(canvasBounds);
  };

  useEffect(() => {
    window.visualViewport?.addEventListener('resize', handleCanvasResize);
    return () => {
      window.visualViewport?.removeEventListener('resize', handleCanvasResize);
    };
  }, []);

  // Attach global listeners while dragging/resizing
  useEffect(() => {
    // Reset actions when mouse is no longer clicked
    const onMouseUp = () => {
      setResizing(false);
      setDragging(false);
    };
    if (dragging || resizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', onMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };
    }
  }, [dragging, resizing]);

  return (
    <OverlayWrapper
      show={show}
      ref={windowRef}
      zIndex={windowZRange.start}
      className={`${className ? className : ''}`}
      position={position}
      bounds={bounds}
      size={size}
      onMouseDown={onWindowClick}
    >
      <ResizeButtons
        bounds={bounds}
        zIndex={windowZRange.start + 3 * windowZRange.step}
        onResizeMouseDown={onResizeMouseDown}
        size={size}
        position={position}
      />
      <OverlayHeader
        $zindex={windowZRange.start + 2 * windowZRange.step}
        onClick={onWindowClick}
        onMouseDown={(e: any) => {
          onDragMouseDown(e);
          onWindowClick();
        }}
      >
        {title ? <div style={{ width: 'calc(100% - 30px)' }}>{title}</div> : null}
        <CloseButton
          onClick={() => {
            // Reset position back to initial
            const updatedPosition = getInitialPosition(size, placement, bounds, padding, isInitialRelative, parentRef, nodeRef);
            setPosition(updatedPosition);
            // Update window ref state when resetting back to original position
            windowStateRef.current = {
              ...windowStateRef.current,
              ...updatedPosition,
            };
            onWindowClose();
            // Execute user callback when hiding
            if (onHide) onHide();
          }}
        >
          X
        </CloseButton>{' '}
      </OverlayHeader>
      <OverlayBody $zindex={windowZRange.start + 1 * windowZRange.step}>{children}</OverlayBody>
    </OverlayWrapper>
  );
};

export default OverlayWindow;
