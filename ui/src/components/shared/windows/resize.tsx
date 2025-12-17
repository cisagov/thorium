import React from 'react';
import { clamp, getDocumentDimensions } from './utilities';
import { ElementPosition, Placement } from './placement';
import { Bounds, Padding } from './bounds';

// Dimensions of HTML element
export type ElementSize = {
  width: number; // element width
  height: number; // element height
};

// Ratios of margins between canvas and element
export type MarginRatios = {
  start: number; // left margin ratio
  end: number; // right margin ratio
  top: number; // top margin ratio
  bottom: number; // bottom margin ratio
};

// State info at start of resize or reposition action (canvas or element)
export type StartPosition = {
  width: number; // starting width of window
  height: number; // start height of window
  top: number; // start top location of window
  left: number; // start left location of window
  pointerX: number; // start cursor pointer X position
  pointerY: number; // start cursor pointer Y position
};

// Get fractional margin ratios for use in proportional resizing of elements
export function getMarginRatios(position: ElementPosition, size: ElementSize): MarginRatios {
  const { docWidth, docHeight } = getDocumentDimensions();
  const leftMargin = position.left;
  const rightMargin = docWidth - position.left - size.width;
  const topMargin = position.top;
  const bottomMargin = docHeight - position.top - size.height;
  // calculate margin ratios as fractions of 1.0
  let leftMarginRatio = leftMargin / (leftMargin + rightMargin);
  let rightMarginRatio = rightMargin / (leftMargin + rightMargin);
  let topMarginRatio = topMargin / (topMargin + bottomMargin);
  let bottomMarginRatio = bottomMargin / (topMargin + bottomMargin);
  // margins calculate to NaN (infinite) when opposite sides are zero
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

// Calculate window size and position after window resize
export function calculateWindowResizeState(
  start: StartPosition,
  minSize: ElementSize,
  dx: number,
  dy: number,
  bounds: Bounds,
  clickedLocation: Placement,
): { position: ElementPosition; size: ElementSize; resized: boolean } {
  // resizing defaults are to remain same
  let resizeTop = start.top;
  let resizeLeft = start.left;
  let resizeWidth = start.width;
  let resizeHeight = start.height;
  // max window height/width is specified by the user or the document size, whichever is smaller
  // we use this min value to calculate the max we can decrease the dimension
  const maxDecreaseHeight = Math.max(start.height - minSize.height, 0);
  const maxDecreaseWidth = Math.max(start.width - minSize.width, 0);
  let yChange = 0;
  let xChange = 0;
  // position is the location of bumper being clicked to changes size, determines y/x dimension and size changes
  // X changes
  if ([Placement.Bottom, Placement.BottomLeft, Placement.BottomRight].includes(clickedLocation)) {
    // max shift down is lesser value of 1) margin below window or 2) max change in height bounded by min height for window
    yChange = clamp(dy, -(bounds.height - start.top - start.height - bounds.top - bounds.bottom), maxDecreaseHeight);
    // calculate new height from change in Y
    resizeHeight = start.height - yChange;
  } else if ([Placement.Top, Placement.TopLeft, Placement.TopRight].includes(clickedLocation)) {
    // max shift up is lesser value of 1) margin above window or 2) max change in height bounded by min height for the window
    yChange = clamp(-dy, -(start.top - bounds.top - bounds.bottom), maxDecreaseHeight);
    resizeTop = start.top + yChange;
    // calculate new height from change in Y
    resizeHeight = start.height - yChange;
  }
  // Y changes
  if ([Placement.Left, Placement.BottomLeft, Placement.TopLeft].includes(clickedLocation)) {
    // max shift left is lesser value of 1) margin in front window or 2) max change in height bounded by min height for the window
    xChange = clamp(-dx, -(start.left - bounds.end - bounds.start), maxDecreaseWidth);
    resizeLeft = start.left + xChange;
    // calculate new width from change in X
    resizeWidth = start.width - xChange;
  } else if ([Placement.Right, Placement.BottomRight, Placement.TopRight].includes(clickedLocation)) {
    // max shift right is lesser value of 1) margin to the right of the window or 2) max change in width bounded by min width for window
    xChange = clamp(dx, -(bounds.width - start.left - start.width - bounds.end - bounds.start), maxDecreaseWidth);
    // calculate new width from change in X
    resizeWidth = start.width - xChange;
  }
  return {
    position: { top: resizeTop, left: resizeLeft },
    size: { width: resizeWidth, height: resizeHeight },
    resized: xChange != 0 || yChange != 0,
  };
}

// Calculate updated position after canvas size change
export function calculateCanvasResizeWindowPosition(
  startBounds: Bounds,
  updatedBounds: Bounds,
  size: ElementSize,
  padding: Padding,
  marginRatio: MarginRatios,
  startPosition: ElementPosition,
) {
  // calculate amount of movement in each direction
  const dx = updatedBounds.width - startBounds.width;
  const dy = updatedBounds.height - startBounds.height;
  // calculate change in position
  const changeLeft = Math.round(marginRatio.start * dx);
  const changeTop = Math.round(marginRatio.top * dy);
  // calculate resize position
  let left = startPosition.left + changeLeft;
  let top = startPosition.top + changeTop;
  // calculate resize size, we can't exceed size of display canvas
  // window location bounded by left side of document
  if (left <= 0) left = padding.start;
  // window location bounded by right side of document
  if (left + size.width > updatedBounds.width) left = updatedBounds.width - size.width - padding.end;
  // window location bounded by top of document
  if (top <= 0) top = padding.top;
  // window location bounded by bottom of document
  if (top + size.height > updatedBounds.height) top = updatedBounds.height - size.height - padding.bottom;
  return { top: Math.max(top, padding.top), left: Math.max(left, padding.start) };
}

// Determine if resize cursor is disabled based on absolute position of window and whether window is at minSize
export function isCursorDisabled(
  placement: Placement,
  position: { top: number; left: number },
  size: { height: number; width: number },
  bounds: Bounds,
) {
  if (placement == Placement.Center) return false;
  const { docHeight, docWidth } = getDocumentDimensions();
  // get touching document edge position booleans
  const atTop = position.top <= 0;
  const atBottom = position.top + size.height >= docHeight;
  const atStart = position.left <= 0;
  const atEnd = position.left + size.width >= docWidth;
  const atMinHeight = bounds.height >= size.height;
  const atMinWidth = bounds.height >= size.width;
  if (placement == Placement.Bottom && atBottom && atMinHeight) return true;
  if (placement == Placement.Top && atTop && atMinHeight) return true;
  if (placement == Placement.Right && atEnd && atMinWidth) return true;
  if (placement == Placement.Left && atStart && atMinWidth) return true;
  if (placement == Placement.TopLeft && atStart && atTop && atMinHeight && atMinWidth) return true;
  if (placement == Placement.TopRight && atEnd && atTop && atMinHeight && atMinWidth) return true;
  if (placement == Placement.BottomLeft && atStart && atBottom && atMinHeight && atMinWidth) return true;
  if (placement == Placement.BottomRight && atEnd && atBottom && atMinHeight && atMinWidth) return true;
  return false;
}

// Get cursor type based on element placement and whether cursor is disabled due to size/position of parent
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
