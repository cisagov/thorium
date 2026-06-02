// project imports
import { Bounds, Padding } from './bounds';
import { clamp } from './utilities';
import { ElementSize } from './resize';

// Placement relative to a reference element or viewport region
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
  Custom = 'Custom',
}

export type ElementPosition = {
  top: number;
  left: number;
};

// Calculate initial position for viewport-fixed windows
function getFixedInitialPosition(size: ElementSize, placement: Placement, bounds: Bounds, padding: Padding) {
  const thoriumElement = document.getElementById('thorium');
  const element = thoriumElement ?? document.documentElement;
  // Thorium element can report 0 width before first paint
  const elementWidth = element.clientWidth > 0 ? element.clientWidth : window.innerWidth;
  const elementHeight = element.clientHeight > 0 ? element.clientHeight : window.innerHeight;
  // Viewport may be smaller than element due to scrollbars
  const clientWidth = Math.min(elementWidth, window.innerWidth);
  const clientHeight = Math.min(elementHeight, window.innerHeight);
  let top = 0;
  let left = 0;
  switch (placement) {
    case Placement.Top:
      top = bounds.top + padding.top;
      left = clientWidth / 2 - size.width / 2;
      break;
    case Placement.Bottom:
      top = clientHeight - size.height - bounds.bottom - padding.bottom;
      left = clientWidth / 2 - size.width / 2;
      break;
    case Placement.Center:
      top = clientHeight / 2 - size.height / 2;
      left = clientWidth / 2 - size.width / 2;
      break;
    case Placement.Left:
      top = clientHeight / 2 - size.height / 2;
      left = bounds.start + padding.start;
      break;
    case Placement.Right:
      top = clientHeight / 2 - size.height / 2;
      left = clientWidth - size.width - bounds.end - padding.end;
      break;
    case Placement.BottomRight:
      top = clientHeight - size.height - bounds.bottom - padding.bottom;
      left = clientWidth - size.width - bounds.end - padding.end;
      break;
    case Placement.TopLeft:
      top = bounds.top + padding.top;
      left = bounds.start + padding.start;
      break;
    case Placement.TopRight:
      top = bounds.top + padding.top;
      left = clientWidth - size.width - bounds.end - padding.end;
      break;
    case Placement.BottomLeft:
      top = clientHeight - size.height - bounds.bottom - padding.bottom;
      left = bounds.start + padding.start;
      break;
    case Placement.Custom:
      return { top: 0, left: 0 };
  }
  return {
    top: clamp(top, bounds.top, bounds.height - size.height - Math.abs(padding.bottom)),
    left: clamp(left, bounds.start, bounds.width - size.width - Math.abs(padding.end)),
  };
}

// Calculate initial position for document-absolute windows, relative to a parent element
function getRelativeInitialPosition<T extends HTMLElement>(
  size: ElementSize,
  placement: Placement,
  bounds: Bounds,
  padding: Padding,
  parentRef: React.RefObject<T | null> | undefined,
  selfRef: React.RefObject<T | null> | undefined,
) {
  let top = 0;
  let left = 0;
  if (parentRef?.current !== null || selfRef?.current !== null) {
    const parent = parentRef?.current ?? (selfRef?.current?.parentNode as HTMLElement | null);
    if (parent) {
      const { top: pTop, left: pLeft, height: pHeight, width: pWidth } = parent.getBoundingClientRect();
      const pTopWithScroll = pTop + window.scrollY;
      const pLeftWithScroll = pLeft + window.scrollX;
      switch (placement) {
        case Placement.Top:
          top = pTopWithScroll - size.height - padding.top;
          left = pLeftWithScroll + pWidth / 2 - size.width / 2;
          break;
        case Placement.Bottom:
          top = pTopWithScroll + pHeight + padding.bottom;
          left = pLeftWithScroll + pWidth / 2 - size.width / 2;
          break;
        case Placement.Center:
          top = pTopWithScroll + pHeight / 2 - size.height / 2;
          left = pLeftWithScroll + pWidth / 2 - size.width / 2;
          break;
        case Placement.Left:
          top = pTopWithScroll + pHeight / 2 - size.height / 2;
          left = pLeftWithScroll - size.width - padding.start;
          break;
        case Placement.Right:
          top = pTopWithScroll + pHeight / 2 - size.height / 2;
          left = pLeftWithScroll + pWidth + padding.end;
          break;
        case Placement.BottomLeft:
          top = pTopWithScroll + pHeight + padding.bottom;
          left = pLeftWithScroll - size.width - padding.start;
          break;
        case Placement.TopLeft:
          top = pTopWithScroll - size.height - padding.top;
          left = pLeftWithScroll - size.width - padding.start;
          break;
        case Placement.TopRight:
          top = pTopWithScroll - size.height - padding.top;
          left = pLeftWithScroll + pWidth + padding.end;
          break;
        case Placement.BottomRight:
          top = pTopWithScroll + pHeight + padding.bottom;
          left = pLeftWithScroll + pWidth + padding.end;
          break;
        case Placement.Custom:
          return { top: 0, left: 0 };
        default:
          top = pTopWithScroll + pHeight / 2 - size.height / 2;
          left = pLeftWithScroll + pWidth / 2 - size.width / 2;
          break;
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
  placement: Placement,
  bounds: Bounds,
  padding: Padding,
  usesReferenceElement: boolean,
  parentRef: React.RefObject<T | null> | undefined,
  selfRef: React.RefObject<T | null> | undefined,
) {
  if (usesReferenceElement) {
    return getRelativeInitialPosition(size, placement, bounds, padding, parentRef, selfRef);
  }
  return getFixedInitialPosition(size, placement, bounds, padding);
}
