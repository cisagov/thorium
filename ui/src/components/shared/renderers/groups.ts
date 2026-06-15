// project imports
import { detectRenderKind } from './detect';
import { RenderKind } from './types';

// spec: ./SPEC.md

/**
 * The set of renderers offered for a file, plus which one to show first. `options[0]` is always
 * the default, so `default` is a convenience alias for it.
 */
export interface RenderGroup {
  /** Renderers the user may switch between, most-appropriate first. */
  options: RenderKind[];
  /** The renderer shown initially (always `options[0]`). */
  default: RenderKind;
}

/**
 * Maps each detected primary {@link RenderKind} to the group of renderers a user may switch
 * between. Binary content only supports the hex dump; textual/structured content additionally
 * offers the editable raw {@link RenderKind.Editor}; images offer the hex dump as an alternative.
 */
const GROUP_OPTIONS: Record<RenderKind, RenderKind[]> = {
  [RenderKind.Image]: [RenderKind.Image, RenderKind.Hex],
  [RenderKind.Hex]: [RenderKind.Hex],
  [RenderKind.Json]: [RenderKind.Json, RenderKind.Editor],
  [RenderKind.Code]: [RenderKind.Code, RenderKind.Decomp, RenderKind.Editor],
  [RenderKind.Yara]: [RenderKind.Yara, RenderKind.Editor],
  [RenderKind.Markdown]: [RenderKind.Markdown, RenderKind.Editor],
  [RenderKind.Text]: [RenderKind.Text, RenderKind.Decomp, RenderKind.Editor],
  // Decomp has no natural file extension, so it's never a detected primary; it appears only as a
  // manual alternative on Code/Text. Map to itself for direct/forced use.
  [RenderKind.Decomp]: [RenderKind.Decomp, RenderKind.Editor],
  // Editor is only ever a manual alternative, never a detected primary kind; map to itself.
  [RenderKind.Editor]: [RenderKind.Editor],
};

/** Human-readable labels for each renderer, used in the preview's renderer dropdown. */
export const RENDER_LABELS: Record<RenderKind, string> = {
  [RenderKind.Json]: 'JSON',
  [RenderKind.Code]: 'Code',
  [RenderKind.Yara]: 'YARA',
  [RenderKind.Image]: 'Image',
  [RenderKind.Hex]: 'Hex',
  [RenderKind.Text]: 'Text',
  [RenderKind.Markdown]: 'Markdown',
  [RenderKind.Decomp]: 'Decomp',
  [RenderKind.Editor]: 'Raw Editor',
};

/**
 * Determine the group of renderers to offer for a downloaded file: detect the primary kind, then
 * look up its allowed alternatives.
 *
 * @param fileName - The file name (used for extension hints); may be empty.
 * @param bytes - The raw file bytes.
 * @returns The {@link RenderGroup} of selectable renderers with the default first.
 */
export function detectRenderGroup(fileName: string, bytes: ArrayBuffer): RenderGroup {
  const primary = detectRenderKind(fileName, bytes);
  const options = GROUP_OPTIONS[primary] ?? [primary];
  return { options, default: options[0] };
}
