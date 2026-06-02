import { useCallback, useRef, useState, useEffect, useLayoutEffect } from 'react';
import styled from 'styled-components';

// project imports
import { boundElementSize, clamp } from '../utilities';
import { calculateCanvasResizeWindowPosition, calculateWindowResizeState, ElementSize, getMarginRatios } from '../resize';
import { getCanvasType, getCanvasBounds, Padding, PositionType } from '../bounds';
import { ElementPosition, getInitialPosition, Placement } from '../placement';
import { useWindowNode } from '../WindowManager/use_window_node';
import CloseButton from './CloseButton';
import OverlayBody from './OverlayBody';
import OverlayHeader from './OverlayHeader';
import OverlayWrapper from './OverlayWrapper';
import ResizeButtons from './ResizeButtons';

// Floating close button for headerless windows
const FloatingCloseButton = styled(CloseButton)<{ $zindex: number }>`
  position: absolute;
  top: 4px;
  right: 4px;
  z-index: ${(props) => props.$zindex};
  background-color: var(--thorium-panel-bg);
  border-radius: 4px;
  width: 24px;
  height: 24px;
  align-items: center;
`;

const DEFAULT_EDGE_PADDING: Padding = { start: 4, end: 4, top: 4, bottom: 4 };

const OverlayWindow: React.FC<{
  className?: string;
  children: React.ReactNode;
  show: boolean;
  title?: string;
  placement?: Placement;
  locked?: boolean;
  width?: number;
  height?: number;
  positioning?: PositionType;
  customPosition?: ElementPosition;
  onHide?: () => void;
  onClose?: () => void;
  padding?: Padding;
  parentRef?: React.RefObject<HTMLElement | null>;
  id?: string;
}> = ({
  className,
  children,
  show,
  title,
  width = 500,
  height = 500,
  locked = false,
  positioning = PositionType.Absolute,
  placement = Placement.Bottom,
  customPosition,
  onHide,
  onClose,
  padding = DEFAULT_EDGE_PADDING,
  parentRef,
  id,
}) => {
  const { nodeRef, windowRef, windowZRange, onWindowClick, onWindowClose, margin: managerMargin } = useWindowNode<HTMLDivElement>(id);
  const [bounds, setBounds] = useState(() => getCanvasBounds(getCanvasType(positioning), managerMargin));
  const usesReferenceElement = positioning !== PositionType.Fixed || parentRef !== undefined;
  const [position, setPosition] = useState<ElementPosition>(() => {
    if (placement === Placement.Custom && customPosition) {
      return customPosition;
    }
    return getInitialPosition({ width, height }, placement, bounds, padding, usesReferenceElement, parentRef, nodeRef);
  });
  const [size, setSize] = useState<ElementSize>(boundElementSize({ width, height }, bounds, padding));
  const [resizing, setResizing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const hasHeader = title !== undefined;
  const windowStateRef = useRef({
    size,
    position,
    marginRatio: getMarginRatios(position, size),
    clickedLocation: Placement.Center,
    start: { width: 0, height: 0, top: position.top, left: position.left, pointerX: 0, pointerY: 0 },
    bounds: bounds,
    resized: false,
    moved: false,
  });

  // Deregister from WindowManager on unmount only (not on close button click, which preserves registration for reopen)
  useEffect(() => {
    return () => {
      onWindowClose();
    };
  }, [onWindowClose]);

  // Close on click-outside or Escape when onClose is provided
  useEffect(() => {
    if (!show || !onClose) return;
    const handlePointerDown = (e: PointerEvent) => {
      if (nodeRef.current && !nodeRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [show, onClose]);

  // Reposition after parentRef.current is set on initial render (custom placement uses explicit coordinates)
  useLayoutEffect(() => {
    if (placement === Placement.Custom && customPosition) return;
    setPosition(getInitialPosition({ width, height }, placement, bounds, padding, usesReferenceElement, parentRef, nodeRef));
  }, []);

  const onDragMouseDown = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (resizing || locked) return;
    setDragging(true);
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
    if (dragging || locked) return;
    setResizing(true);
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

  const handleWindowDrag = useCallback((dx: number, dy: number) => {
    const { start, bounds: currentBounds } = windowStateRef.current;
    windowStateRef.current.moved = true;
    const updatedPosition = {
      top: clamp(start.top - dy, currentBounds.top, currentBounds.height - start.height),
      left: clamp(start.left - dx, currentBounds.start, currentBounds.width - start.width),
    };
    setPosition(updatedPosition);
    windowStateRef.current = {
      ...windowStateRef.current,
      position: updatedPosition,
      marginRatio: getMarginRatios(updatedPosition, windowStateRef.current.size),
    };
  }, []);

  const handleWindowResize = useCallback(
    (dx: number, dy: number) => {
      const { start, bounds: currentBounds, clickedLocation } = windowStateRef.current;
      const {
        position: updatedPosition,
        size: updatedSize,
        resized,
      } = calculateWindowResizeState(start, { width, height }, dx, dy, currentBounds, clickedLocation);
      windowStateRef.current.resized = resized;
      setPosition(updatedPosition);
      setSize(updatedSize);
      windowStateRef.current = {
        ...windowStateRef.current,
        size: updatedSize,
        position: updatedPosition,
        marginRatio: getMarginRatios(updatedPosition, updatedSize),
      };
    },
    [width, height],
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      const dx = windowStateRef.current.start.pointerX - e.clientX;
      const dy = windowStateRef.current.start.pointerY - e.clientY;
      if (dragging) {
        handleWindowDrag(dx, dy);
      } else if (resizing) {
        handleWindowResize(dx, dy);
      }
    },
    [dragging, resizing, handleWindowDrag, handleWindowResize],
  );

  const handleCanvasResize = useCallback(() => {
    const canvasBounds = getCanvasBounds(getCanvasType(positioning), managerMargin);
    const updatedSize = boundElementSize(
      windowStateRef.current.resized ? windowStateRef.current.size : { width, height },
      canvasBounds,
      padding,
    );
    let updatedPosition =
      placement === Placement.Custom && customPosition
        ? customPosition
        : getInitialPosition(updatedSize, placement, canvasBounds, padding, usesReferenceElement, parentRef, nodeRef);
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
    windowStateRef.current = {
      ...windowStateRef.current,
      position: updatedPosition,
      size: updatedSize,
      bounds: canvasBounds,
    };
    setSize(updatedSize);
    setPosition(updatedPosition);
    setBounds(canvasBounds);
  }, [positioning, managerMargin, width, height, padding, placement, customPosition, usesReferenceElement, parentRef, nodeRef]);

  useEffect(() => {
    window.visualViewport?.addEventListener('resize', handleCanvasResize);
    return () => {
      window.visualViewport?.removeEventListener('resize', handleCanvasResize);
    };
  }, [handleCanvasResize]);

  // Attach global listeners while dragging/resizing
  useEffect(() => {
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
  }, [dragging, resizing, handleMouseMove]);

  // Reset position and notify parent on close
  const handleClose = () => {
    const updatedPosition =
      placement === Placement.Custom && customPosition
        ? customPosition
        : getInitialPosition(size, placement, bounds, padding, usesReferenceElement, parentRef, nodeRef);
    setPosition(updatedPosition);
    windowStateRef.current = {
      ...windowStateRef.current,
      position: updatedPosition,
    };
    if (onHide) onHide();
  };

  return (
    <OverlayWrapper
      show={show}
      ref={windowRef}
      zIndex={windowZRange.start}
      className={className}
      position={position}
      bounds={bounds}
      size={size}
      onMouseDown={onWindowClick}
    >
      {!locked && (
        <ResizeButtons
          bounds={bounds}
          zIndex={windowZRange.start + 3 * windowZRange.step}
          onResizeMouseDown={onResizeMouseDown}
          size={size}
          position={position}
        />
      )}
      {hasHeader ? (
        <OverlayHeader
          $zindex={windowZRange.start + 2 * windowZRange.step}
          $locked={locked}
          onClick={onWindowClick}
          onMouseDown={(e: React.MouseEvent<HTMLDivElement>) => {
            onDragMouseDown(e as unknown as React.MouseEvent<HTMLButtonElement>);
            onWindowClick();
          }}
        >
          {title ? <div style={{ width: 'calc(100% - 30px)' }}>{title}</div> : null}
          <CloseButton onClick={handleClose}>X</CloseButton>
        </OverlayHeader>
      ) : onHide ? (
        <FloatingCloseButton $zindex={windowZRange.start + 2 * windowZRange.step} onClick={handleClose}>
          X
        </FloatingCloseButton>
      ) : null}
      <OverlayBody $zindex={windowZRange.start + windowZRange.step} $hasHeader={hasHeader}>
        {children}
      </OverlayBody>
    </OverlayWrapper>
  );
};

export default OverlayWindow;
