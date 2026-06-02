import { parseDocument, isMap, isPair, isScalar, isSeq, type Document, type YAMLMap } from 'yaml';
import { Severity, type Diagnostic } from './types';

interface LineIndex {
  offsets: number[];
}

/**
 * Build an index of line-start byte offsets for a document.
 *
 * Precomputing these offsets lets {@link offsetToLineCol} convert a character offset to a
 * line/column in O(log n) via binary search instead of rescanning the text each time.
 *
 * @param text - The full document text.
 * @returns A {@link LineIndex} whose `offsets[i]` is the start offset of line `i + 1`.
 */
function buildLineIndex(text: string): LineIndex {
  const offsets: number[] = [0];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') {
      offsets.push(i + 1);
    }
  }
  return { offsets };
}

/**
 * Convert a character offset into 1-based line and column numbers using a precomputed line index.
 *
 * @param index - The {@link LineIndex} produced by {@link buildLineIndex}.
 * @param offset - The character offset into the document.
 * @returns The 1-based `line` and `column` for that offset.
 */
function offsetToLineCol(index: LineIndex, offset: number): { line: number; column: number } {
  let low = 0;
  let high = index.offsets.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (index.offsets[mid] <= offset) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return { line: low + 1, column: offset - index.offsets[low] + 1 };
}

export interface YamlParseResult {
  doc: Document | null;
  value: unknown;
  diagnostics: Diagnostic[];
}

/**
 * Parse a YAML document and collect editor diagnostics (errors, warnings, duplicate keys).
 *
 * Parser errors and warnings are mapped to line/column ranges; duplicate keys (which the YAML
 * spec allows but are almost always mistakes here) are detected separately via
 * {@link detectDuplicateKeys}. The parsed JS `value` is only populated when there are no parse errors.
 *
 * @param text - The YAML source to parse.
 * @returns A {@link YamlParseResult} with the parsed document, its JS value (or `null`), and diagnostics.
 */
export function parseYaml(text: string): YamlParseResult {
  if (!text.trim()) {
    return { doc: null, value: null, diagnostics: [] };
  }

  const doc = parseDocument(text, { keepSourceTokens: true, uniqueKeys: false });
  const lineIndex = buildLineIndex(text);
  const diagnostics: Diagnostic[] = [];

  for (const err of doc.errors) {
    const pos = err.pos?.[0] ?? 0;
    const endPos = err.pos?.[1] ?? pos;
    const start = offsetToLineCol(lineIndex, pos);
    const end = offsetToLineCol(lineIndex, endPos);
    diagnostics.push({
      line: start.line,
      column: start.column,
      endLine: end.line,
      endColumn: end.column,
      severity: Severity.Error,
      message: err.message,
    });
  }

  for (const warn of doc.warnings) {
    const pos = warn.pos?.[0] ?? 0;
    const endPos = warn.pos?.[1] ?? pos;
    const start = offsetToLineCol(lineIndex, pos);
    const end = offsetToLineCol(lineIndex, endPos);
    diagnostics.push({
      line: start.line,
      column: start.column,
      endLine: end.line,
      endColumn: end.column,
      severity: Severity.Warning,
      message: warn.message,
    });
  }

  detectDuplicateKeys(doc, lineIndex, diagnostics);

  const value: unknown = doc.errors.length === 0 ? doc.toJS() : null;
  return { doc, value, diagnostics };
}

/**
 * Detect duplicate mapping keys anywhere in a YAML document, appending diagnostics for each.
 *
 * No-op unless the document root is a map; recurses into nested maps/sequences via {@link walkMap}.
 *
 * @param doc - The parsed YAML document.
 * @param lineIndex - Line index for resolving key positions.
 * @param diagnostics - Diagnostics array that detected duplicates are pushed onto (mutated in place).
 */
function detectDuplicateKeys(doc: Document, lineIndex: LineIndex, diagnostics: Diagnostic[]): void {
  if (!isMap(doc.contents)) return;
  walkMap(doc.contents, lineIndex, diagnostics);
}

/**
 * Recursively scan a YAML map for duplicate keys, flagging both the original and repeated entries.
 *
 * For each key seen more than once at the same level, an error diagnostic is pushed on the first
 * occurrence (once) and on every subsequent occurrence. Recurses into nested maps and maps inside
 * sequences.
 *
 * @param map - The YAML map node to scan.
 * @param lineIndex - Line index for resolving key positions.
 * @param diagnostics - Diagnostics array that duplicate-key errors are pushed onto (mutated in place).
 */
function walkMap(map: YAMLMap, lineIndex: LineIndex, diagnostics: Diagnostic[]): void {
  const seen = new Map<string, { line: number; column: number; keyLen: number; flagged: boolean }>();

  for (const item of map.items) {
    if (!isPair(item) || !isScalar(item.key)) continue;
    const key = String(item.key.value);
    const offset = item.key.range?.[0] ?? 0;
    const pos = offsetToLineCol(lineIndex, offset);

    const prev = seen.get(key);
    if (prev) {
      if (!prev.flagged) {
        diagnostics.push({
          line: prev.line,
          column: prev.column,
          endColumn: prev.column + prev.keyLen,
          severity: Severity.Error,
          message: `Duplicate key '${key}' (also defined on line ${pos.line})`,
        });
        prev.flagged = true;
      }
      diagnostics.push({
        line: pos.line,
        column: pos.column,
        endColumn: pos.column + key.length,
        severity: Severity.Error,
        message: `Duplicate key '${key}' (previously defined on line ${prev.line})`,
      });
    } else {
      seen.set(key, { line: pos.line, column: pos.column, keyLen: key.length, flagged: false });
    }

    if (isMap(item.value)) {
      walkMap(item.value, lineIndex, diagnostics);
    } else if (isSeq(item.value)) {
      for (const seqItem of item.value.items) {
        if (isMap(seqItem)) {
          walkMap(seqItem, lineIndex, diagnostics);
        }
      }
    }
  }
}

export { buildLineIndex, offsetToLineCol };
export type { LineIndex };
