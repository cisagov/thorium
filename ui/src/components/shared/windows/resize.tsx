import { clamp, getDocumentDimensions } from './utilities';
import { ElementPosition, Placement } from './placement';
import { Bounds, Padding } from './bounds';

export type ElementSize = {
  width: number;
  height: number;
};

export type MarginRatios = {
  start: number;
  end: number;
  top: number;
  bottom: number;
};

export type StartPosition = {
  width: number;
  height: number;
  top: number;
  left: number;
  pointerX: number;
  pointerY: number;
};

// Calculate fractional margin ratios for proportional repositioning on canvas resize
export function getMarginRatios(position: ElementPosition, size: ElementSize): MarginRatios {
  const { docWidth, docHeight } = getDocumentDimensions();
  const leftMargin = position.left;
  const rightMargin = docWidth - position.left - size.width;
  const topMargin = position.top;
  const bottomMargin = docHeight - position.top - size.height;
  let leftMarginRatio = leftMargin / (leftMargin + rightMargin);
  let rightMarginRatio = rightMargin / (leftMargin + rightMargin);
  let topMarginRatio = topMargin / (topMargin + bottomMargin);
  let bottomMarginRatio = bottomMargin / (topMargin + bottomMargin);
  // Both margins zero produces NaN — default to centered
  if (leftMargin <= 0 && rightMargin <= 0) {
    leftMarginRatio = 0.5;
    rightMarginRatio = 0.5;
  }
  if (topMargin <= 0 && bottomMargin <= 0) {
    topMarginRatio = 0.5;
    bottomMarginRatio = 0.5;
  }
  return { start: leftMarginRatio, end: rightMarginRatio, top: topMarginRatio, bottom: bottomMarginRatio };
}

// Calculate new window size and position after a resize handle drag
export function calculateWindowResizeState(
  start: StartPosition,
  minSize: ElementSize,
  dx: number,
  dy: number,
  bounds: Bounds,
  clickedLocation: Placement,
): { position: ElementPosition; size: ElementSize; resized: boolean } {
  let resizeTop = start.top;
  let resizeLeft = start.left;
  let resizeWidth = start.width;
  let resizeHeight = start.height;
  const maxDecreaseHeight = Math.max(start.height - minSize.height, 0);
  const maxDecreaseWidth = Math.max(start.width - minSize.width, 0);
  let yChange = 0;
  let xChange = 0;
  // Vertical resize from bottom or top edge
  if ([Placement.Bottom, Placement.BottomLeft, Placement.BottomRight].includes(clickedLocation)) {
    yChange = clamp(dy, -(bounds.height - start.top - start.height - bounds.top - bounds.bottom), maxDecreaseHeight);
    resizeHeight = start.height - yChange;
  } else if ([Placement.Top, Placement.TopLeft, Placement.TopRight].includes(clickedLocation)) {
    yChange = clamp(-dy, -(start.top - bounds.top - bounds.bottom), maxDecreaseHeight);
    resizeTop = start.top + yChange;
    resizeHeight = start.height - yChange;
  }
  // Horizontal resize from left or right edge
  if ([Placement.Left, Placement.BottomLeft, Placement.TopLeft].includes(clickedLocation)) {
    xChange = clamp(-dx, -(start.left - bounds.end - bounds.start), maxDecreaseWidth);
    resizeLeft = start.left + xChange;
    resizeWidth = start.width - xChange;
  } else if ([Placement.Right, Placement.BottomRight, Placement.TopRight].includes(clickedLocation)) {
    xChange = clamp(dx, -(bounds.width - start.left - start.width - bounds.end - bounds.start), maxDecreaseWidth);
    resizeWidth = start.width - xChange;
  }
  return {
    position: { top: resizeTop, left: resizeLeft },
    size: { width: resizeWidth, height: resizeHeight },
    resized: xChange !== 0 || yChange !== 0,
  };
}

// Calculate updated position after the canvas (viewport/document) changes size
export function calculateCanvasResizeWindowPosition(
  startBounds: Bounds,
  updatedBounds: Bounds,
  size: ElementSize,
  padding: Padding,
  marginRatio: MarginRatios,
  startPosition: ElementPosition,
) {
  const dx = updatedBounds.width - startBounds.width;
  const dy = updatedBounds.height - startBounds.height;
  const changeLeft = Math.round(marginRatio.start * dx);
  const changeTop = Math.round(marginRatio.top * dy);
  let left = startPosition.left + changeLeft;
  let top = startPosition.top + changeTop;
  if (left <= 0) left = padding.start;
  if (left + size.width > updatedBounds.width) left = updatedBounds.width - size.width - padding.end;
  if (top <= 0) top = padding.top;
  if (top + size.height > updatedBounds.height) top = updatedBounds.height - size.height - padding.bottom;
  return { top: Math.max(top, padding.top), left: Math.max(left, padding.start) };
}

export function isCursorDisabled(
  placement: Placement,
  position: { top: number; left: number },
  size: { height: number; width: number },
  bounds: Bounds,
) {
  if (placement === Placement.Center) return false;
  const { docHeight, docWidth } = getDocumentDimensions();
  const atTop = position.top <= 0;
  const atBottom = position.top + size.height >= docHeight;
  const atStart = position.left <= 0;
  const atEnd = position.left + size.width >= docWidth;
  const atMinHeight = bounds.height >= size.height;
  const atMinWidth = bounds.width >= size.width;
  if (placement === Placement.Bottom && atBottom && atMinHeight) return true;
  if (placement === Placement.Top && atTop && atMinHeight) return true;
  if (placement === Placement.Right && atEnd && atMinWidth) return true;
  if (placement === Placement.Left && atStart && atMinWidth) return true;
  if (placement === Placement.TopLeft && atStart && atTop && atMinHeight && atMinWidth) return true;
  if (placement === Placement.TopRight && atEnd && atTop && atMinHeight && atMinWidth) return true;
  if (placement === Placement.BottomLeft && atStart && atBottom && atMinHeight && atMinWidth) return true;
  if (placement === Placement.BottomRight && atEnd && atBottom && atMinHeight && atMinWidth) return true;
  return false;
}

export function getResizeCursorType(
  placement: Placement,
  position: { top: number; left: number },
  size: { height: number; width: number },
  bounds: Bounds,
) {
  if (isCursorDisabled(placement, position, size, bounds)) return 'auto';
  switch (placement) {
    case Placement.Bottom:
    case Placement.Top:
      return 'ns-resize';
    case Placement.Left:
    case Placement.Right:
      return 'ew-resize';
    case Placement.BottomLeft:
    case Placement.TopRight:
      return 'nesw-resize';
    case Placement.TopLeft:
    case Placement.BottomRight:
      return 'nwse-resize';
  }
  return 'auto';
}
