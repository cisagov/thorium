// spec: ../ToolResult.spec.md
import { DiffMethod } from 'react-diff-viewer-continued';
import { stringify as yamlStringify } from 'yaml';

// project imports
import { extensionOf, isJsonText } from '@components/shared/renderers';

/** Pretty-print a value as JSON, falling back to `String()` if it can't be serialized. */
export function jsonPretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/** Re-indent JSON text so changes localize per line; returns the original text if it isn't JSON. */
export function prettyJsonText(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

/**
 * Pick the diff inputs + method for a result value.
 *
 * Structured values (objects/arrays) are pretty-printed and diffed with the JSON method (one key
 * per line, so the changed key is obvious). Strings and scalars are diffed as raw text by lines so
 * their real newlines — not escaped `\n` — localize the change.
 */
export function diffForValue(base: unknown, compare: unknown): { oldValue: string; newValue: string; method: DiffMethod } {
  const structured = (v: unknown): v is object => v != null && typeof v === 'object';
  if (structured(base) && structured(compare)) {
    return { oldValue: jsonPretty(base), newValue: jsonPretty(compare), method: DiffMethod.JSON };
  }
  const serialize = (v: unknown) => (typeof v === 'string' ? v : v == null ? '' : jsonPretty(v));
  return { oldValue: serialize(base), newValue: serialize(compare), method: DiffMethod.LINES };
}

/** Render a result value as YAML text for the YAML differ (strings are used verbatim). */
export function resultToYaml(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value == null) return '';
  try {
    return yamlStringify(value);
  } catch {
    return jsonPretty(value);
  }
}

/**
 * Pick diff inputs + method for a result *file* by extension/content (caller handles binary).
 *
 * `.yaml`/`.yml` use the YAML differ; `.json` (or extension-less content that parses as JSON on both
 * sides) uses the JSON differ with re-indented inputs; everything else diffs by lines.
 */
export function diffForFile(
  name: string,
  baseText: string,
  compareText: string,
): { oldValue: string; newValue: string; method: DiffMethod } {
  const ext = extensionOf(name);
  if (ext === 'yaml' || ext === 'yml') {
    return { oldValue: baseText, newValue: compareText, method: DiffMethod.YAML };
  }
  if (ext === 'json' || (isJsonText(baseText) && isJsonText(compareText))) {
    return { oldValue: prettyJsonText(baseText), newValue: prettyJsonText(compareText), method: DiffMethod.JSON };
  }
  return { oldValue: baseText, newValue: compareText, method: DiffMethod.LINES };
}
