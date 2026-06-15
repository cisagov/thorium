import { parseDocument } from 'yaml';

// project imports
import { detectRenderKind, extensionOf, textOf } from './detect';
import { RenderableInput, RenderKind } from './types';
import { FormatType } from '@utilities/rules/types';

// spec: ./SPEC.md

/**
 * Best-effort pretty-print of text for the raw editor's "prettify on open" behavior.
 *
 * Only JSON and YAML are reformatted; every other (or absent) format is returned unchanged. All
 * reformatting is best-effort: unparseable input, empty input, and non-object YAML are returned
 * verbatim so nothing is ever mangled.
 *
 * @param text - The raw text to prettify.
 * @param format - The format to prettify as (usually from {@link prettifyFormatFor}); when omitted
 *   or not JSON/YAML the text is returned unchanged.
 * @returns The prettified text, or the original text when it can't be safely reformatted.
 */
export function prettify(text: string, format?: FormatType): string {
  if (!text.trim()) return text;
  switch (format) {
    case FormatType.JSON:
      try {
        return JSON.stringify(JSON.parse(text), null, 2);
      } catch {
        return text;
      }
    case FormatType.YAML:
      try {
        const doc = parseDocument(text);
        if (doc.errors.length > 0) return text;
        const js: unknown = doc.toJS();
        // only reformat structured documents; plain scalars/strings are left exactly as-is
        if (js === null || typeof js !== 'object') return text;
        // stringify the Document (not the plain JS) so comments and anchors are preserved
        return doc.toString({ lineWidth: 0 });
      } catch {
        return text;
      }
    default:
      return text;
  }
}

/**
 * Decide which format (if any) a file's content should be prettified as when opened in the raw
 * editor. Intentionally stricter than {@link editorFormatHint}: it only returns JSON/YAML when we
 * are confident, so arbitrary code files (which default to YAML *highlighting*) are never run
 * through the YAML prettifier.
 *
 * @param fileName - The file name (used for extension hints); may be empty.
 * @param bytes - The raw file bytes (used for content-based JSON detection).
 * @returns `FormatType.JSON`/`FormatType.YAML` when confident, otherwise `undefined` (no prettify).
 */
export function prettifyFormatFor(fileName: string, bytes: ArrayBuffer): FormatType | undefined {
  const ext = extensionOf(fileName);
  if (ext === 'json' || detectRenderKind(fileName, bytes) === RenderKind.Json) return FormatType.JSON;
  if (ext === 'yaml' || ext === 'yml') return FormatType.YAML;
  return undefined;
}

/**
 * Compute the text the raw editor is seeded with when a file opens: the input's text (see
 * {@link textOf}) pretty-printed with the format {@link prettifyFormatFor} deems safe. Shared by
 * {@link FilePreview} and {@link EditorRenderer} so both seed the editor identically.
 *
 * @param input - The renderable input to seed the editor from.
 * @returns The prettified seed text.
 */
export function prettifiedSeed(input: RenderableInput): string {
  return prettify(textOf(input), prettifyFormatFor(input.fileName ?? '', input.bytes));
}
