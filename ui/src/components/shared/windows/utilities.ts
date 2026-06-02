// project imports
import { Bounds, Padding } from './bounds';
import { ElementSize } from './resize';

// Clamp a number to [min, max]
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

// Constrain element size to fit within canvas bounds minus padding
export function boundElementSize(size: ElementSize, bounds: Bounds, padding: Padding) {
  return {
    height: Math.min(bounds.height - Math.abs(padding.bottom) - Math.abs(padding.top), size.height),
    width: Math.min(bounds.width - Math.abs(padding.start) - Math.abs(padding.end), size.width),
  };
}

export function getViewPortDimensions() {
  return { docWidth: document.body.clientWidth, docHeight: document.body.clientHeight };
}

export function getDocumentDimensions() {
  return { docWidth: document.body.clientWidth, docHeight: document.body.scrollHeight };
}
