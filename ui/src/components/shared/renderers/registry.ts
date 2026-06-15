import React from 'react';

// project imports
import CodeRenderer from './CodeRenderer';
import EditorRenderer from './EditorRenderer';
import ImageRenderer from './ImageRenderer';
import JsonRenderer from './JsonRenderer';
import MarkdownRenderer from './MarkdownRenderer';
import TextRenderer from './TextRenderer';
import YaraRenderer from './YaraRenderer';
import HexView from './hex/HexView';
import { FileRendererProps, RenderKind } from './types';
import { FormatType } from '@utilities/rules/types';
// spec: ./SPEC.md

/** Decompiled-source renderer: the shared read-only CodeRenderer forced to the decomp format. */
const DecompRenderer: React.FC<FileRendererProps> = (props) => React.createElement(CodeRenderer, { ...props, format: FormatType.Decomp });

/**
 * Maps each {@link RenderKind} to the component that renders it. Exported so other areas of the
 * UI can extend or override individual renderers when reusing the render system.
 */
export const RENDERERS: Record<RenderKind, React.FC<FileRendererProps>> = {
  [RenderKind.Json]: JsonRenderer,
  [RenderKind.Code]: CodeRenderer,
  [RenderKind.Yara]: YaraRenderer,
  [RenderKind.Image]: ImageRenderer,
  [RenderKind.Hex]: HexView,
  [RenderKind.Text]: TextRenderer,
  [RenderKind.Markdown]: MarkdownRenderer,
  [RenderKind.Decomp]: DecompRenderer,
  [RenderKind.Editor]: EditorRenderer,
};
