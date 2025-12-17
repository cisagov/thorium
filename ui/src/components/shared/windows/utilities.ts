// project imports
import { Bounds, Padding } from './bounds';
import { ElementSize } from './resize';

/**
 * clamp a number to a specified range.
 *
 * @param value  – the input number to clamp
 * @param min    – the lower bound of the range
 * @param max    – the upper bound of the range
 * @returns      – if value < min, returns min;
 *                 if value > max, returns max;
 *                 otherwise returns value unchanged.
 */
export function clamp(value: number, min: number, max: number): number {
  // ensure value isn't below minimum
  const top = Math.max(value, min);
  // ensure upper bound is not exceeded
  return Math.min(top, max);
}

// Bound element size given canvas bounds and assigned canvas padding
export function boundElementSize(size: ElementSize, bounds: Bounds, padding: Padding) {
  return {
    height: Math.min(bounds.height - Math.abs(padding.bottom) - Math.abs(padding.top), size.height),
    width: Math.min(bounds.width - Math.abs(padding.start) - Math.abs(padding.end), size.width),
  };
}

// Get vertical scroll bare width if present
export function getVerticalScrollbarWidth() {
  const element =
    document.getElementById('thorium') !== null
      ? (document.getElementById('thorium') as HTMLElement)
      : (document.documentElement as HTMLElement);
  if (element.scrollHeight > element.clientHeight) return 0;
  return window.innerWidth - document.body.clientWidth;
}

// Get visible window dimensions
export function getViewPortDimensions() {
  return { docWidth: document.body.clientWidth, docHeight: document.body.clientHeight };
}

// Get full page dimensions including hidden content from scrolling
export function getDocumentDimensions() {
  return { docWidth: document.body.clientWidth, docHeight: document.body.scrollHeight };
}
