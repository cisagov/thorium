// spec: ./SPEC.md
export { default as FileRenderer } from './FileRenderer';
export { default as FilePreview } from './FilePreview';
export type { FilePreviewProps } from './FilePreview';
export { RENDERERS } from './registry';
export {
  detectRenderKind,
  decodeText,
  looksLikeText,
  formatFromFileName,
  extensionOf,
  isJsonText,
  stringToRenderableInput,
} from './detect';
export { detectRenderGroup, RENDER_LABELS } from './groups';
export type { RenderGroup } from './groups';
export { OceanJsonTheme, useJsonTreeInvert } from './jsonTheme';
export { RenderKind } from './types';
export type { FileRendererProps, RenderableInput } from './types';
// hex view internals reused by the diff feature
export { default as HexRenderer } from './hex/HexRenderer';
export { default as HexValueInspector } from './hex/HexValueInspector';
export { decodeHexValues } from './hex/decode';
export type { HexValueEntry } from './hex/decode';
export { HexByteStatus } from './hex/types';
export type { HexSelection } from './hex/types';
