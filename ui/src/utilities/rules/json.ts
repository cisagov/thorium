import { Severity, type Diagnostic } from './types';
import { parseYaml, type YamlParseResult } from './yaml';

export type JsonParseResult = YamlParseResult;

/**
 * Pull the character offset out of a `JSON.parse` `SyntaxError` message.
 *
 * Matches the `position N` fragment that V8/most engines include; returns 0 when absent.
 *
 * @param message - The error message text.
 * @returns The parsed position offset, or 0 if none was found.
 */
function extractPosition(message: string): number {
  const match = message.match(/position\s+(\d+)/i);
  return match ? Number(match[1]) : 0;
}

/**
 * Convert a character offset into 1-based line/column numbers by scanning the text.
 *
 * Unlike the YAML variant this scans directly (no precomputed index), which is fine for the
 * single position JSON parsing needs.
 *
 * @param text - The full document text.
 * @param offset - The character offset to locate.
 * @returns The 1-based `line` and `column` for that offset.
 */
function offsetToLineCol(text: string, offset: number): { line: number; column: number } {
  let line = 1;
  let col = 1;
  for (let i = 0; i < offset && i < text.length; i++) {
    if (text[i] === '\n') {
      line++;
      col = 1;
    } else {
      col++;
    }
  }
  return { line, column: col };
}

/**
 * Detect duplicate object keys in a JSON document via a lightweight character scan.
 *
 * `JSON.parse` silently keeps the last duplicate key, so this hand-rolled scanner tracks a stack
 * of object scopes and flags any key that repeats within the same `{}` scope (reporting both the
 * earlier and later occurrence). Strings are skipped with escape handling so braces/quotes inside
 * values aren't mistaken for structure.
 *
 * @param text - The JSON source to scan.
 * @returns A diagnostic for each side of every duplicate-key collision found.
 */
function detectDuplicateJsonKeys(text: string): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const stack: Map<string, { line: number; column: number }>[] = [];
  let i = 0;

  while (i < text.length) {
    const ch = text[i];
    if (ch === '{') {
      stack.push(new Map());
      i++;
    } else if (ch === '}') {
      stack.pop();
      i++;
    } else if (ch === '"' && stack.length > 0) {
      const keyStart = i;
      i++;
      let key = '';
      while (i < text.length && text[i] !== '"') {
        if (text[i] === '\\') {
          key += text[i] + (text[i + 1] ?? '');
          i += 2;
        } else {
          key += text[i];
          i++;
        }
      }
      i++;

      let j = i;
      while (j < text.length && (text[j] === ' ' || text[j] === '\t' || text[j] === '\n' || text[j] === '\r')) j++;

      if (j < text.length && text[j] === ':') {
        const pos = offsetToLineCol(text, keyStart);
        const currentScope = stack[stack.length - 1];
        const prev = currentScope.get(key);
        if (prev) {
          diagnostics.push({
            line: prev.line,
            column: prev.column,
            severity: Severity.Error,
            message: `Duplicate key '${key}' (also defined on line ${pos.line})`,
          });
          diagnostics.push({
            ...pos,
            severity: Severity.Error,
            message: `Duplicate key '${key}' (previously defined on line ${prev.line})`,
          });
        }
        currentScope.set(key, pos);
        i = j + 1;
      }
    } else {
      i++;
    }
  }

  return diagnostics;
}

/**
 * Parse a JSON document and collect editor diagnostics.
 *
 * Empty input yields an empty result. A `JSON.parse` failure produces a single error diagnostic
 * located at the reported position. On success it reuses {@link parseYaml} (a superset of JSON) to
 * obtain the parsed value and structural diagnostics, then layers in JSON-specific duplicate-key
 * detection ({@link detectDuplicateJsonKeys}) while dropping YAML's own duplicate-key diagnostics
 * to avoid double-reporting.
 *
 * @param text - The JSON source to parse.
 * @returns A {@link JsonParseResult} with the parsed document, its value (or `null`), and diagnostics.
 */
export function parseJson(text: string): JsonParseResult {
  if (!text.trim()) {
    return { doc: null, value: null, diagnostics: [] };
  }

  try {
    JSON.parse(text);
  } catch (e) {
    const msg = e instanceof SyntaxError ? e.message : 'Invalid JSON';
    const offset = extractPosition(msg);
    const pos = offsetToLineCol(text, offset);
    return {
      doc: null,
      value: null,
      diagnostics: [
        {
          ...pos,
          severity: Severity.Error,
          message: msg,
        },
      ],
    };
  }

  const yamlResult = parseYaml(text);
  const duplicates = detectDuplicateJsonKeys(text);

  return {
    doc: yamlResult.doc,
    value: yamlResult.value,
    diagnostics: [...duplicates, ...yamlResult.diagnostics.filter((d) => !d.message.includes('Duplicate key'))],
  };
}
