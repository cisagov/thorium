// project imports
import { RenderableInput, RenderKind } from './types';
import { FormatType } from '@utilities/rules/types';
// spec: ./SPEC.md

/** Extensions that map directly to the read-only YARA editor. */
const YARA_EXTENSIONS = new Set(['yar', 'yara']);

/** Extensions rendered as formatted markdown. */
const MARKDOWN_EXTENSIONS = new Set(['md', 'markdown']);

/** Image extensions rendered via an `<img>` element. */
const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg', 'svgz', 'apng', 'avif', 'ico', 'tif', 'tiff']);

/** MIME types keyed by image extension, used when building the blob URL. */
const IMAGE_MIME: Record<string, string> = {
  png: 'image/png',
  apng: 'image/apng',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  gif: 'image/gif',
  webp: 'image/webp',
  bmp: 'image/bmp',
  svg: 'image/svg+xml',
  svgz: 'image/svg+xml',
  avif: 'image/avif',
  ico: 'image/x-icon',
  tif: 'image/tiff',
  tiff: 'image/tiff',
};

/** Extensions we treat as syntax-highlightable code (handled by CodeRenderer). */
const CODE_EXTENSIONS = new Set([
  'yml',
  'yaml',
  'xml',
  'toml',
  'ini',
  'cfg',
  'conf',
  'c',
  'h',
  'cpp',
  'cc',
  'hpp',
  'rs',
  'go',
  'py',
  'js',
  'ts',
  'tsx',
  'jsx',
  'java',
  'sh',
  'bash',
  'ps1',
  'sql',
  'css',
  'scss',
  'html',
  'htm',
]);

/** Extensions that are plain text. */
const TEXT_EXTENSIONS = new Set(['txt', 'log', 'csv', 'text']);

/** Lower-cased file extension (without the dot), or empty string when none. */
export function extensionOf(fileName?: string): string {
  if (!fileName) return '';
  const dot = fileName.lastIndexOf('.');
  if (dot === -1 || dot === fileName.length - 1) return '';
  return fileName.slice(dot + 1).toLowerCase();
}

/**
 * Pick a CodeEditor {@link FormatType} from a file name; defaults to YAML highlighting.
 *
 * Shared by the read-only {@link RenderKind.Code} renderer and the editable
 * {@link RenderKind.Editor} renderer so both infer the same language.
 *
 * @param fileName - The file name (used for its extension); may be undefined.
 * @returns The matching {@link FormatType}.
 */
export function formatFromFileName(fileName?: string): FormatType {
  switch (extensionOf(fileName)) {
    case 'json':
      return FormatType.JSON;
    case 'yar':
    case 'yara':
      return FormatType.YARA;
    default:
      return FormatType.YAML;
  }
}

/**
 * Pick the CodeEditor {@link FormatType} to use for *syntax highlighting* in the raw editor,
 * accounting for content when the extension is missing/wrong (e.g. JSON detected by content).
 *
 * This drives highlighting only — the stricter prettify gating lives in `prettifyFormatFor`.
 *
 * @param fileName - The file name (used for extension hints); may be empty.
 * @param bytes - The raw file bytes (used to detect content-only JSON).
 * @returns The {@link FormatType} for editor highlighting.
 */
export function editorFormatHint(fileName: string, bytes: ArrayBuffer): FormatType {
  const kind = detectRenderKind(fileName, bytes);
  if (kind === RenderKind.Json) return FormatType.JSON;
  if (kind === RenderKind.Yara) return FormatType.YARA;
  return formatFromFileName(fileName);
}

/**
 * Decode bytes to a UTF-8 string. Uses the non-fatal decoder so partial/garbled binary
 * still produces a string (callers that care about "is this text" should use
 * {@link looksLikeText} instead of inspecting the decoded output).
 *
 * @param bytes - The raw bytes to decode.
 * @returns The decoded text.
 */
export function decodeText(bytes: ArrayBuffer): string {
  return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
}

/**
 * Resolve a {@link RenderableInput} to its text, encoding the shared contract "prefer the
 * pre-decoded `text`, otherwise decode the raw `bytes`". Every text-oriented renderer uses this so
 * the contract lives in one place.
 *
 * @param input - The renderable input to read text from.
 * @returns The input's pre-decoded text, or the UTF-8 decoding of its bytes.
 */
export function textOf(input: RenderableInput): string {
  return input.text ?? decodeText(input.bytes);
}

/**
 * Wrap an in-memory string as a {@link RenderableInput} so the byte-oriented renderers (e.g. the
 * shared CodeRenderer) can display text that never came from a file. Carries both the encoded `bytes`
 * (to satisfy the contract) and the original `text` (which renderers prefer when present).
 *
 * @param text - The text to render.
 * @returns A RenderableInput carrying the text and its UTF-8 bytes.
 */
export function stringToRenderableInput(text: string): RenderableInput {
  return { bytes: new TextEncoder().encode(text).buffer, text };
}

/**
 * Heuristic: does this byte buffer look like human-readable text rather than binary?
 *
 * Samples up to the first 4096 bytes. A NUL byte is a strong binary signal; otherwise the
 * ratio of non-printable (excluding common whitespace) bytes must stay below a threshold.
 *
 * @param bytes - The raw bytes to inspect.
 * @returns `true` if the content appears to be text.
 */
export function looksLikeText(bytes: ArrayBuffer): boolean {
  const view = new Uint8Array(bytes);
  const sample = Math.min(view.length, 4096);
  if (sample === 0) return true;
  let nonPrintable = 0;
  for (let i = 0; i < sample; i++) {
    const b = view[i];
    // a NUL byte is a strong binary signal
    if (b === 0) return false;
    // allow tab(9), LF(10), CR(13), and the printable ASCII range; bytes >=0x80 may be valid UTF-8
    const printable = b === 9 || b === 10 || b === 13 || (b >= 0x20 && b <= 0x7e) || b >= 0x80;
    if (!printable) nonPrintable++;
  }
  return nonPrintable / sample < 0.1;
}

/**
 * Detect common image formats by their magic bytes (for files whose extension is missing/wrong).
 *
 * @param bytes - The raw file bytes.
 * @returns `true` if the leading bytes match a known image signature.
 */
function hasImageMagic(bytes: ArrayBuffer): boolean {
  const b = new Uint8Array(bytes);
  if (b.length < 4) return false;
  // PNG: 89 50 4E 47
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return true;
  // JPEG: FF D8 FF
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return true;
  // GIF: "GIF8"
  if (b[0] === 0x47 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x38) return true;
  // BMP: "BM"
  if (b[0] === 0x42 && b[1] === 0x4d) return true;
  // TIFF: II*\0 or MM\0*
  if (
    (b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2a && b[3] === 0x00) ||
    (b[0] === 0x4d && b[1] === 0x4d && b[2] === 0x00 && b[3] === 0x2a)
  )
    return true;
  // RIFF....WEBP
  if (
    b.length >= 12 &&
    b[0] === 0x52 &&
    b[1] === 0x49 &&
    b[2] === 0x46 &&
    b[3] === 0x46 &&
    b[8] === 0x57 &&
    b[9] === 0x45 &&
    b[10] === 0x42 &&
    b[11] === 0x50
  )
    return true;
  // AVIF/HEIF: "....ftyp" with an image brand
  if (b.length >= 12 && b[4] === 0x66 && b[5] === 0x74 && b[6] === 0x79 && b[7] === 0x70) {
    const brand = String.fromCharCode(b[8], b[9], b[10], b[11]);
    if (brand === 'avif' || brand === 'avis' || brand === 'heic' || brand === 'heix' || brand === 'mif1') return true;
  }
  return false;
}

/** True if a file should be rendered as an image (by extension or magic bytes). */
export function isImageFile(fileName: string, bytes: ArrayBuffer): boolean {
  const ext = extensionOf(fileName);
  if (IMAGE_EXTENSIONS.has(ext)) return true;
  return hasImageMagic(bytes);
}

/**
 * Best-effort MIME type for an image file name (used when constructing a blob URL).
 *
 * @param fileName - The image file name.
 * @returns The MIME type, or `undefined` when the extension is unknown.
 */
export function imageMimeForName(fileName?: string): string | undefined {
  return IMAGE_MIME[extensionOf(fileName)];
}

/** True if the decoded text parses as a JSON object/array. */
export function isJsonText(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed || (trimmed[0] !== '{' && trimmed[0] !== '[')) return false;
  try {
    JSON.parse(trimmed);
    return true;
  } catch {
    return false;
  }
}

/**
 * Determine how a downloaded result file should be rendered.
 *
 * Detection order: explicit extension (yara/json/markdown/code/text) first, then content
 * heuristics (binary => Hex, valid JSON => Json, otherwise => Text). Result-file links carry no
 * MIME metadata, so this is intentionally best-effort.
 *
 * @param fileName - The file name (used for extension hints); may be empty.
 * @param bytes - The raw file bytes.
 * @returns The {@link RenderKind} to use.
 */
export function detectRenderKind(fileName: string, bytes: ArrayBuffer): RenderKind {
  const ext = extensionOf(fileName);

  // images (by extension or magic bytes) render as <img>, before the binary/hex fallback
  if (isImageFile(fileName, bytes)) return RenderKind.Image;

  // anything else that isn't textual is shown as hex regardless of extension
  if (!looksLikeText(bytes)) return RenderKind.Hex;

  if (YARA_EXTENSIONS.has(ext)) return RenderKind.Yara;
  if (ext === 'json') return RenderKind.Json;
  if (MARKDOWN_EXTENSIONS.has(ext)) return RenderKind.Markdown;
  if (CODE_EXTENSIONS.has(ext)) return RenderKind.Code;
  if (TEXT_EXTENSIONS.has(ext)) return RenderKind.Text;

  // no decisive extension: sniff the content
  const text = decodeText(bytes);
  if (isJsonText(text)) return RenderKind.Json;
  // YARA without an extension: a `rule <name> {` block, optionally preceded by imports and any
  // combination of the `private`/`global` rule modifiers
  if (/^\s*(import\s+"[^"]+"\s*)*((private|global)\s+)*rule\s+\w+/m.test(text)) return RenderKind.Yara;
  return RenderKind.Text;
}
