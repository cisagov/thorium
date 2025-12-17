// project imports
import { Bounds, Padding } from './bounds';
import { clamp } from './utilities';
import { ElementSize } from './resize';

// Placement position compared to relative ("thorium" or parent or specified ref)
//   - Top/Tight/Left/Center are placed in the middle of that edge of the relative element
//   - Center: over the center of the relative element
//   - BottomRight/BottomLeft/TopRight/TopLeft are placed at one of the four corners of the relative element
export enum Placement {
  Center = 'Center',
  BottomRight = 'BottomRight',
  BottomLeft = 'BottomLeft',
  TopLeft = 'TopLeft',
  TopRight = 'TopRight',
  Left = 'Left',
  Right = 'Right',
  Top = 'Top',
  Bottom = 'Bottom',
}

// Position of element on screen defined by top/left edge of the element
export type ElementPosition = {
  top: number;
  left: number;
};

// Get initial position for fixed position elements (aka fixed to visible viewport)
function getFixedInitialPosition(
  size: { width: number; height: number }, // size of self
  placement: Placement | string, // initial placement position within the viewport
  bounds: Bounds, //viewport bounds
  padding: Padding, // padding between window placement and viewport edge
) {
  // use either Thorium or element for page dimensions
  const thoriumElement = document.getElementById('thorium');
  const element = thoriumElement != null ? thoriumElement : document.documentElement;
  // on initial page load thorium element is sometimes > 0 width
  const elementWidth = element.clientWidth > 0 ? element.clientWidth : window.innerWidth;
  const elementHeight = element.clientHeight > 0 ? element.clientHeight : window.innerHeight;
  // find smallest calculated window width which handles viewport containing elements with scroll
  const clientWidth = Math.min(elementWidth, window.innerWidth);
  const clientHeight = Math.min(elementHeight, window.innerHeight);
  let top = 0;
  let left = 0;
  if (placement == Placement.Top) {
    top = bounds.top + padding.top;
    left = clientWidth / 2 - size.width / 2;
  } else if (placement == Placement.Bottom) {
    top = clientHeight - size.height - bounds.bottom - padding.bottom;
    left = clientWidth / 2 - size.width / 2;
  } else if (placement == Placement.Center) {
    top = clientHeight / 2 - size.height / 2;
    left = clientWidth / 2 - size.width / 2;
  } else if (placement == Placement.Left) {
    top = clientHeight / 2 - size.height / 2;
    left = bounds.start + padding.start;
  } else if (placement == Placement.Right) {
    top = clientHeight / 2 - size.height / 2;
    left = clientWidth - size.width - bounds.end - padding.end;
  } else if (placement == Placement.BottomRight) {
    top = clientHeight - size.height - bounds.bottom - padding.bottom;
    left = clientWidth - size.width - bounds.end - padding.end;
  } else if (placement == Placement.TopLeft) {
    top = bounds.top + padding.top;
    left = bounds.start + padding.start;
  } else if (placement == Placement.TopRight) {
    top = bounds.top + padding.top;
    left = clientWidth - size.width - bounds.end - padding.end;
  } else if (placement == Placement.BottomLeft) {
    top = clientHeight - size.height - bounds.bottom - padding.bottom;
    left = bounds.start + padding.start;
  }
  // we must clamp initial position to the viewport bounds and viewport padding
  return {
    top: clamp(top, bounds.top, bounds.height - size.height - Math.abs(padding.bottom)),
    left: clamp(left, bounds.start, bounds.width - size.width - Math.abs(padding.end)),
  };
}

// Get initial position for Absolute  position elements (aka fixed to document position)
function getRelativeInitialPosition<T extends HTMLElement>(
  size: { width: number; height: number }, // size of self
  placement: Placement | string, // placement compared to relative
  bounds: Bounds, // bounds of the document itself including non-visible elements (scroll)
  padding: Padding, // padding from relative element when placing self
  parentRef: React.RefObject<T | null> | undefined, // element to reference when building initial relative position
  selfRef: React.RefObject<T | null> | undefined, // element to reference when building initial relative position
) {
  let top = 0;
  let left = 0;
  if (parentRef?.current !== null || selfRef?.current != null) {
    const parent = parentRef?.current != undefined ? parentRef.current : (selfRef?.current?.parentNode as HTMLElement | null);
    if (parent) {
      // Get parent dimensions and destructure
      const { top: pTop, left: pLeft, height: pHeight, width: pWidth } = parent.getBoundingClientRect();
      const pTopWithScroll = pTop + window.scrollY;
      const pLeftWithScroll = pLeft + window.scrollX;
      if (placement == Placement.Top) {
        top = pTopWithScroll - size.height - padding.top;
        left = pLeftWithScroll + pWidth / 2 - size.width / 2;
      } else if (placement == Placement.Bottom) {
        top = pTopWithScroll + pHeight + padding.bottom;
        left = pLeftWithScroll + pWidth / 2 - size.width / 2;
      } else if (placement == Placement.Center) {
        top = pTopWithScroll + pHeight / 2 - size.height / 2;
        left = pLeftWithScroll + pWidth / 2 - size.width / 2;
      } else if (placement == Placement.Left) {
        top = pTopWithScroll + pHeight / 2 - size.height / 2;
        left = pLeftWithScroll - size.width - padding.start;
      } else if (placement == Placement.Right) {
        top = pTopWithScroll + pHeight / 2 - size.height / 2;
        left = pLeftWithScroll + pWidth + padding.end;
      } else if (placement == Placement.BottomLeft) {
        top = pTopWithScroll + pHeight + padding.bottom;
        left = pLeftWithScroll - size.width - padding.start;
      } else if (placement == Placement.TopLeft) {
        top = pTopWithScroll - size.height - padding.top;
        left = pLeftWithScroll - size.width - padding.start;
      } else if (placement == Placement.TopRight) {
        top = pTopWithScroll - size.height - padding.top;
        left = pLeftWithScroll + pWidth + padding.end;
      } else if (placement == Placement.BottomRight) {
        top = pTopWithScroll + pHeight + padding.bottom;
        left = pLeftWithScroll + pWidth + padding.end;
      } else {
        top = pTopWithScroll + pHeight / 2 - size.height / 2;
        left = pLeftWithScroll + pWidth / 2 - size.width / 2;
      }
    }
  }
  return {
    top: clamp(top, bounds.top, bounds.height - size.height - Math.abs(padding.bottom)),
    left: clamp(left, bounds.start, bounds.width - size.width - Math.abs(padding.end)),
  };
}

export function getInitialPosition<T extends HTMLElement>(
  size: ElementSize,
  placement: Placement | string,
  bounds: Bounds,
  padding: Padding,
  isRelativePosition: boolean, // is positioning relative to an element or fixed on the screen
  parentRef: React.RefObject<T | null> | undefined, // reference element if relative positioned and not defaulting to nearest ancestor
  selfRef: React.RefObject<T | null> | undefined, // reference element if positioned relative to nearest ancestor
) {
  if (isRelativePosition) {
    return getRelativeInitialPosition(size, placement, bounds, padding, parentRef, selfRef);
  } else {
    return getFixedInitialPosition(size, placement, bounds, padding);
  }
}
