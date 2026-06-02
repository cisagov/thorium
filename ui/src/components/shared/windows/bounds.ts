// project imports
import { getDocumentDimensions, getViewPortDimensions } from './utilities';

export enum CanvasType {
  Viewport = 'Viewport',
  Document = 'Document',
}

export enum PositionType {
  Absolute = 'Absolute',
  Fixed = 'Fixed',
}

export type Margin = {
  top: number;
  bottom: number;
  start: number;
  end: number;
};

// Padding is structurally identical to Margin but semantically distinct
export type Padding = Margin;

export type Bounds = {
  type: CanvasType;
  width: number;
  height: number;
} & Margin;

export function getCanvasType(type: PositionType) {
  return type === PositionType.Fixed ? CanvasType.Viewport : CanvasType.Document;
}

export function getCanvasBounds(type: CanvasType, margin: Margin = { top: 0, bottom: 0, start: 0, end: 0 }): Bounds {
  if (type === CanvasType.Viewport) {
    const size = getViewPortDimensions();
    return { type, width: size.docWidth, height: size.docHeight, ...margin };
  }
  const size = getDocumentDimensions();
  return { type, width: size.docWidth, height: size.docHeight, ...margin };
}
