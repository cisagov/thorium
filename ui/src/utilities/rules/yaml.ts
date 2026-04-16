import { parseDocument, isMap, isPair, isScalar, isSeq, type Document, type YAMLMap } from 'yaml';
import { Severity, type Diagnostic } from './types';

interface LineIndex {
  offsets: number[];
}

function buildLineIndex(text: string): LineIndex {
  const offsets: number[] = [0];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') {
      offsets.push(i + 1);
    }
  }
  return { offsets };
}

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

function detectDuplicateKeys(doc: Document, lineIndex: LineIndex, diagnostics: Diagnostic[]): void {
  if (!isMap(doc.contents)) return;
  walkMap(doc.contents, lineIndex, diagnostics);
}

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
