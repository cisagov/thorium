import { isMap, isPair, isScalar } from 'yaml';

// project imports
import { includes, Severity, type Diagnostic, type Suggestion, type FieldSchema } from '../types';
import { offsetToLineCol, type LineIndex } from '../yaml';

export function nodeLineCol(
  node: { range?: [number, number, number] | [number, number] | null | undefined },
  lineIndex: LineIndex,
): { line: number; column: number } {
  const offset = node.range?.[0] ?? 0;
  return offsetToLineCol(lineIndex, offset);
}

export function findMapKey(map: unknown, key: string) {
  if (!isMap(map)) return null;
  for (const item of map.items) {
    if (isPair(item) && isScalar(item.key) && item.key.value === key) {
      return item.key;
    }
  }
  return null;
}

export function findMapValue(map: unknown, key: string) {
  if (!isMap(map)) return null;
  for (const item of map.items) {
    if (isPair(item) && isScalar(item.key) && item.key.value === key) {
      return item.value;
    }
  }
  return null;
}

export function nodePosition(node: unknown, lineIndex: LineIndex): { line: number; column: number } {
  if (node && typeof node === 'object' && 'range' in node) {
    const range = (node as Record<string, unknown>).range;
    if (Array.isArray(range) && typeof range[0] === 'number') {
      return offsetToLineCol(lineIndex, range[0]);
    }
  }
  return { line: 1, column: 1 };
}

export function validateUnknownFields(
  map: unknown,
  knownFields: readonly string[],
  parentLabel: string,
  parsed: Record<string, unknown>,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
) {
  const fieldList = knownFields.join(', ');
  for (const key of Object.keys(parsed)) {
    const fieldName: string = key;
    if (!includes(knownFields, key)) {
      const node = findMapKey(map, fieldName);
      const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
      diagnostics.push({
        ...pos,
        severity: Severity.Warning,
        message: `Unknown ${parentLabel} field: '${fieldName}'. Known fields: ${fieldList}`,
      });
    }
  }
}

export function validateEnumField(
  map: unknown,
  parsed: Record<string, unknown>,
  field: string,
  values: readonly string[],
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
  nullable = false,
) {
  if (!(field in parsed)) return;
  const val = parsed[field];
  if (val === null) {
    if (nullable) return;
    const valueList = values.join(', ');
    const node = findMapValue(map, field);
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'${field}' must be one of: ${valueList}`,
    });
    return;
  }
  if (typeof val === 'string') {
    const strVal: string = val;
    if (!includes(values, val)) {
      const valueList = values.join(', ');
      const node = findMapValue(map, field);
      const pos = nodePosition(node, lineIndex);
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `Invalid ${field} value: '${strVal}'. Must be one of: ${valueList}`,
      });
    }
  }
}

export function validateNumberField(
  map: unknown,
  parsed: Record<string, unknown>,
  field: string,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
  nullable = false,
) {
  if (!(field in parsed)) return;
  if (parsed[field] === null && nullable) return;
  if (typeof parsed[field] !== 'number') {
    const node = findMapValue(map, field);
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'${field}' must be a number`,
    });
  }
}

export function validateBooleanField(
  map: unknown,
  parsed: Record<string, unknown>,
  field: string,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
) {
  if (!(field in parsed)) return;
  if (typeof parsed[field] !== 'boolean') {
    const node = findMapValue(map, field);
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'${field}' must be a boolean (true/false)`,
    });
  }
}

export function validateStringField(
  map: unknown,
  parsed: Record<string, unknown>,
  field: string,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
  nullable = false,
) {
  if (!(field in parsed)) return;
  if (parsed[field] === null && nullable) return;
  if (typeof parsed[field] !== 'string') {
    const node = findMapValue(map, field);
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'${field}' must be a string`,
    });
  }
}

export function validateObjectField(
  map: unknown,
  parsed: Record<string, unknown>,
  field: string,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
): Record<string, unknown> | null {
  if (!(field in parsed)) return null;
  const val = parsed[field];
  if (val === null) return null;
  if (typeof val !== 'object' || Array.isArray(val)) {
    const node = findMapValue(map, field);
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'${field}' must be an object`,
    });
    return null;
  }
  return val as Record<string, unknown>;
}

export function validateSubObject(
  parentMap: unknown,
  parsed: Record<string, unknown>,
  field: string,
  knownFields: readonly string[],
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
) {
  const obj = validateObjectField(parentMap, parsed, field, lineIndex, diagnostics);
  if (!obj) return;
  const subMap = findMapValue(parentMap, field);
  validateUnknownFields(subMap, knownFields, field, obj, lineIndex, diagnostics);
}

/**
 * Validates a field whose value is an externally-tagged enum (e.g. KwargDependency, AutoTagLogic).
 * Accepts a bare string for unit variants (those mapped to `null` in `schema.variants`), or a
 * single-key object `{ <Variant>: value }` for variants that carry a payload.
 */
export function validateVariantField(
  map: unknown,
  parsed: Record<string, unknown>,
  field: string,
  schema: FieldSchema,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
  nullable = false,
) {
  if (!(field in parsed)) return;
  const val = parsed[field];
  if (val === null && nullable) return;

  const variants = schema.variants ?? {};
  const variantNames = Object.keys(variants);
  const unitNames = variantNames.filter((n) => variants[n] === null);
  const node = findMapValue(map, field);
  const pos = nodePosition(node, lineIndex);

  if (typeof val === 'string') {
    if (!unitNames.includes(val)) {
      const allowed = unitNames.length ? `'${unitNames.join("', '")}'` : 'a variant object';
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `Invalid '${field}' value: '${val}'. Use ${allowed} or { <variant>: ... } with one of: ${variantNames.join(', ')}`,
      });
    }
    return;
  }
  if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
    const keys = Object.keys(val);
    if (keys.length !== 1 || !variantNames.includes(keys[0])) {
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `'${field}' must be a single-key object { <variant>: ... } with one of: ${variantNames.join(', ')}`,
      });
    }
    return;
  }
  diagnostics.push({
    ...pos,
    severity: Severity.Error,
    message: `'${field}' must be a string or a { <variant>: ... } object`,
  });
}

/**
 * Removes duplicate suggestions for the same field + kind, keeping the first occurrence.
 * Conditional/specialized suggestions are generated before the generic field loops, so the
 * richer message and more specific schema win when a field is suggested from two code paths.
 */
export function dedupeSuggestions(suggestions: Suggestion[]): Suggestion[] {
  const seen = new Set<string>();
  const result: Suggestion[] = [];
  for (const s of suggestions) {
    const key = `${s.field}|${s.isRemoval ? 'r' : ''}${s.isMapEntry ? 'm' : ''}${s.isReplace ? 'p' : ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(s);
  }
  return result;
}
