// project imports
import { FormatType } from '@utilities/rules/types';
// spec: ./SPEC.md

/** The kinds of content the modular renderer system knows how to display. */
export enum RenderKind {
  /** Pretty-printed, collapsible JSON tree. */
  Json = 'Json',
  /** Syntax-highlighted, read-only code (JSON/YAML/etc.). */
  Code = 'Code',
  /** YARA rule, rendered in the read-only YARA editor. */
  Yara = 'Yara',
  /** Image content (PNG/JPEG/GIF/...), rendered as an `<img>`. */
  Image = 'Image',
  /** Binary content, rendered as a hex dump. */
  Hex = 'Hex',
  /** Plain text fallback. */
  Text = 'Text',
  /** Markdown source, rendered to formatted HTML. */
  Markdown = 'Markdown',
  /** Decompiled source, rendered read-only with the decomp (C-like) syntax highlighter. */
  Decomp = 'Decomp',
  /** Editable raw text (copy/download the edited buffer). */
  Editor = 'Editor',
}

/**
 * A unit of content to render. Always carries the raw bytes; `fileName` (used for
 * extension-based detection) and a pre-decoded `text` are optional.
 */
export interface RenderableInput {
  bytes: ArrayBuffer;
  fileName?: string;
  text?: string;
}

/** Props shared by every renderer component. */
export interface FileRendererProps {
  input: RenderableInput;
  /** Optional explicit kind; when omitted the dispatcher detects it. */
  kind?: RenderKind;
  // Editor-only props (ignored by non-editable renderers). Forwarded through FileRenderer so the
  // FilePreview toolbar can drive the raw editor as a controlled component.
  /** Controlled editor text; when provided the editable renderer reflects it instead of self-managing. */
  value?: string;
  /** Called with the editor's current text on change (and its initial value). */
  onTextChange?: (text: string) => void;
  /** Force a specific editor language instead of inferring from the file name. */
  formatHint?: FormatType;
}
