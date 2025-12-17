// project imports
import { getDocumentDimensions, getViewPortDimensions } from './utilities';

// Canvas type for displaying elements:
export enum CanvasType {
  Viewport = 'Viewport', // Includes hidden content due to viewport scroll
  Document = 'Document', // Only what is visible within the visible browser window
}

// Numerical bounds definition for a display canvas
export type Bounds = {
  top: number;
  bottom: number;
  start: number;
  end: number;
  type: CanvasType;
  width: number;
  height: number;
};

// Window CSS positioning value, only absolute and fixed are currently supported
export enum PositionType {
  Absolute = 'Absolute',
  Fixed = 'Fixed',
}

// Space between element and its parent in all four directions
export type Margin = {
  top: number; // space above
  bottom: number; // space below
  start: number; // space to the left
  end: number; // space to the end
};

// Space between parent and its contents. Padding takes functionally the same properties as a Margin.
export type Padding = Margin;

// Get canvas type (full document or just visible viewport) based on CSS window positioning property
export function getCanvasType(type: PositionType) {
  return type === PositionType.Fixed ? CanvasType.Viewport : CanvasType.Document;
}

// Calculate the canvas bounds where canvas is either the viewport or full document
export function getCanvasBounds(type: CanvasType, margin = { top: 0, bottom: 0, start: 0, end: 0 }): Bounds {
  if (type == CanvasType.Viewport) {
    const size = getViewPortDimensions();
    return { type: type, width: size.docWidth, height: size.docHeight, ...margin };
  } else if (type == CanvasType.Document) {
    const size = getDocumentDimensions();
    return { type: type, width: size.docWidth, height: size.docHeight, ...margin };
  }
  // should never hit this since only two CanvasTypes are currently implemented
  return { type: type, width: 0, height: 0, ...margin };
}
