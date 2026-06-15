import type { Document } from 'yaml';
import { isMap, isPair, isScalar, isSeq } from 'yaml';

// project imports
import { findMapKey, findMapValue, nodeLineCol, nodePosition } from '../tools/shared';
import { Severity, includes, type Diagnostic } from '../types';
import { buildLineIndex } from '../yaml';
import {
  REQUIRED_FIELDS,
  DETECTION_REQUIRED_FIELDS,
  STATUS_VALUES,
  LEVEL_VALUES,
  RELATED_TYPES,
  TITLE_MAX_LENGTH,
  NAME_MAX_LENGTH,
  TAXONOMY_MAX_LENGTH,
  DESCRIPTION_MAX_LENGTH,
  FALSEPOSITIVES_MIN_LENGTH,
  SCOPE_MIN_LENGTH,
  UUID_V4_PATTERN,
  DATE_PATTERN,
  TAG_PATTERN,
  VALUE_MODIFIERS,
  KNOWN_TOP_LEVEL_FIELDS,
} from './schema';

// spec: ./validate.spec.md

/**
 * Validate a parsed Sigma rule YAML document against the Sigma schema.
 *
 * Checks for missing required fields, validates the `detection` block and its condition, and reports
 * value/format issues (status, level, related types, taxonomy/description lengths, UUID/date/tag
 * patterns, value modifiers) as line/column-anchored diagnostics.
 *
 * @param doc - The parsed YAML document (positional info for diagnostics).
 * @param text - The raw YAML source, used to map node offsets to line/column.
 * @param parsed - The plain-object form of `doc` used for value checks.
 * @returns The list of {@link Diagnostic}s found (empty when the rule is valid).
 */
export function validateSigmaRule(doc: Document, text: string, parsed: Record<string, unknown>): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const lineIndex = buildLineIndex(text);

  const lastLine = lineIndex.offsets.length;
  for (const field of REQUIRED_FIELDS) {
    if (!(field in parsed)) {
      diagnostics.push({
        line: 1,
        column: 1,
        endLine: lastLine,
        severity: Severity.Error,
        message: `Missing required field: '${field}'`,
      });
    }
  }

  for (const key of Object.keys(parsed)) {
    if (!includes(KNOWN_TOP_LEVEL_FIELDS, key)) {
      const node = findMapKey(doc.contents, key);
      const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
      diagnostics.push({
        ...pos,
        severity: Severity.Info,
        message: `Unknown Sigma field: '${key}'. Known fields: ${KNOWN_TOP_LEVEL_FIELDS.join(', ')}`,
      });
    }
  }

  if (typeof parsed['title'] === 'string' && parsed['title'].length > TITLE_MAX_LENGTH) {
    const node = findMapKey(doc.contents, 'title');
    const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
    diagnostics.push({
      ...pos,
      severity: Severity.Warning,
      message: `Title exceeds maximum length of ${TITLE_MAX_LENGTH} characters (currently ${parsed['title'].length})`,
    });
  }

  if (typeof parsed['name'] === 'string' && parsed['name'].length > NAME_MAX_LENGTH) {
    const node = findMapKey(doc.contents, 'name');
    const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
    diagnostics.push({
      ...pos,
      severity: Severity.Warning,
      message: `Name exceeds maximum length of ${NAME_MAX_LENGTH} characters (currently ${parsed['name'].length})`,
    });
  }

  if (typeof parsed['taxonomy'] === 'string' && parsed['taxonomy'].length > TAXONOMY_MAX_LENGTH) {
    const node = findMapKey(doc.contents, 'taxonomy');
    const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
    diagnostics.push({
      ...pos,
      severity: Severity.Warning,
      message: `Taxonomy exceeds maximum length of ${TAXONOMY_MAX_LENGTH} characters (currently ${parsed['taxonomy'].length})`,
    });
  }

  if (typeof parsed['description'] === 'string' && parsed['description'].length > DESCRIPTION_MAX_LENGTH) {
    const node = findMapKey(doc.contents, 'description');
    const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
    diagnostics.push({
      ...pos,
      severity: Severity.Warning,
      message: `Description exceeds maximum length of ${DESCRIPTION_MAX_LENGTH} characters`,
    });
  }

  if ('id' in parsed && typeof parsed['id'] === 'string' && !UUID_V4_PATTERN.test(parsed['id'])) {
    const node = findMapValue(doc.contents, 'id');
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({
      ...pos,
      severity: Severity.Warning,
      message: `id should be a UUIDv4 (e.g. 929a690e-bef0-4204-a928-ef5e620d6fcc)`,
    });
  }

  if ('status' in parsed && typeof parsed['status'] === 'string') {
    if (!includes(STATUS_VALUES, parsed['status'])) {
      const node = findMapValue(doc.contents, 'status');
      const pos = nodePosition(node, lineIndex);
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `Invalid status value: '${parsed['status']}'. Must be one of: ${STATUS_VALUES.join(', ')}`,
      });
    }
  }

  if ('level' in parsed && typeof parsed['level'] === 'string') {
    if (!includes(LEVEL_VALUES, parsed['level'])) {
      const node = findMapValue(doc.contents, 'level');
      const pos = nodePosition(node, lineIndex);
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `Invalid level value: '${parsed['level']}'. Must be one of: ${LEVEL_VALUES.join(', ')}`,
      });
    }
  }

  for (const dateField of ['date', 'modified'] as const) {
    if (!(dateField in parsed)) continue;
    const rawValue = parsed[dateField];
    if (rawValue instanceof Date) {
      continue;
    }
    if (typeof rawValue === 'string') {
      if (!DATE_PATTERN.test(rawValue)) {
        const node = findMapValue(doc.contents, dateField);
        const pos = nodePosition(node, lineIndex);
        diagnostics.push({
          ...pos,
          severity: Severity.Error,
          message: `Invalid ${dateField} format: '${rawValue}'. Must be YYYY-MM-DD`,
        });
      }
    } else {
      const node = findMapValue(doc.contents, dateField);
      const pos = nodePosition(node, lineIndex);
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `${dateField} must be a string in YYYY-MM-DD format`,
      });
    }
  }

  if ('tags' in parsed && Array.isArray(parsed['tags'])) {
    const tagsNode = findMapValue(doc.contents, 'tags');
    if (isSeq(tagsNode)) {
      for (let i = 0; i < tagsNode.items.length; i++) {
        const item = tagsNode.items[i];
        if (isScalar(item) && typeof item.value === 'string' && !TAG_PATTERN.test(item.value)) {
          const pos = item.range ? nodeLineCol(item, lineIndex) : { line: 1, column: 1 };
          diagnostics.push({
            ...pos,
            severity: Severity.Warning,
            message: `Tag '${item.value}' does not match expected pattern (lowercase, dot-separated namespaces)`,
          });
        }
      }
    }
  }

  if ('related' in parsed && Array.isArray(parsed['related'])) {
    const relatedNode = findMapValue(doc.contents, 'related');
    if (isSeq(relatedNode)) {
      for (let i = 0; i < relatedNode.items.length; i++) {
        const entry = relatedNode.items[i];
        if (isMap(entry)) {
          let hasId = false;
          let hasType = false;
          for (const pair of entry.items) {
            if (isPair(pair) && isScalar(pair.key)) {
              if (pair.key.value === 'id') hasId = true;
              if (pair.key.value === 'type') {
                hasType = true;
                if (isScalar(pair.value) && typeof pair.value.value === 'string') {
                  if (!includes(RELATED_TYPES, pair.value.value)) {
                    const pos = pair.value.range ? nodeLineCol(pair.value, lineIndex) : { line: 1, column: 1 };
                    diagnostics.push({
                      ...pos,
                      severity: Severity.Error,
                      message: `Invalid related type: '${pair.value.value}'. Must be one of: ${RELATED_TYPES.join(', ')}`,
                    });
                  }
                }
              }
            }
          }
          if (!hasId || !hasType) {
            const pos = nodePosition(entry, lineIndex);
            if (!hasId) {
              diagnostics.push({ ...pos, severity: Severity.Error, message: `related entry is missing required field: 'id'` });
            }
            if (!hasType) {
              diagnostics.push({ ...pos, severity: Severity.Error, message: `related entry is missing required field: 'type'` });
            }
          }
        }
      }
    }
  }

  if ('logsource' in parsed && typeof parsed['logsource'] === 'object' && parsed['logsource'] !== null) {
    const ls = parsed['logsource'] as Record<string, unknown>;
    const hasIdentifier = 'category' in ls || 'product' in ls || 'service' in ls;
    if (!hasIdentifier) {
      const node = findMapKey(doc.contents, 'logsource');
      const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
      diagnostics.push({
        ...pos,
        severity: Severity.Warning,
        message: 'logsource should have at least one of: category, product, service',
      });
    }
  }

  if ('detection' in parsed && typeof parsed['detection'] === 'object' && parsed['detection'] !== null) {
    const det = parsed['detection'] as Record<string, unknown>;

    for (const field of DETECTION_REQUIRED_FIELDS) {
      if (!(field in det)) {
        const node = findMapKey(doc.contents, 'detection');
        const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
        diagnostics.push({
          ...pos,
          severity: Severity.Error,
          message: `detection is missing required field: '${field}'`,
        });
      }
    }

    if ('condition' in det && typeof det['condition'] === 'string') {
      const condition = det['condition'];
      const identifiers = Object.keys(det).filter((k) => k !== 'condition');

      const tokens = condition.match(/[a-zA-Z_][a-zA-Z0-9_]*/g) || [];
      const keywords = new Set(['and', 'or', 'not', 'of', 'them', 'all', '1']);
      for (const token of tokens) {
        if (keywords.has(token)) continue;
        if (/^\d+$/.test(token)) continue;

        const matchesIdentifier = identifiers.some((id) => {
          if (token === id) return true;
          if (token.includes('*')) {
            const pattern = new RegExp('^' + token.replace(/\*/g, '.*') + '$');
            return identifiers.some((ident) => pattern.test(ident));
          }
          return false;
        });

        if (!matchesIdentifier && token !== 'them') {
          const hasWildcardMatch = identifiers.some((id) => {
            const pattern = new RegExp('^' + token.replace(/\*/g, '.*') + '$');
            return pattern.test(id);
          });
          if (!hasWildcardMatch) {
            const detNode = findMapValue(doc.contents, 'detection');
            if (isMap(detNode)) {
              for (const pair of detNode.items) {
                if (isPair(pair) && isScalar(pair.key) && pair.key.value === 'condition') {
                  const pos = nodePosition(pair.value, lineIndex);
                  diagnostics.push({
                    ...pos,
                    severity: Severity.Warning,
                    message: `Condition references '${token}' which is not defined as a search identifier`,
                  });
                  break;
                }
              }
            }
          }
        }
      }
    }

    const detNode = findMapValue(doc.contents, 'detection');
    if (isMap(detNode)) {
      for (const pair of detNode.items) {
        if (isPair(pair) && isScalar(pair.key) && typeof pair.key.value === 'string' && pair.key.value !== 'condition') {
          const searchId = pair.key.value;
          if (isMap(pair.value)) {
            for (const fieldPair of pair.value.items) {
              if (isPair(fieldPair) && isScalar(fieldPair.key) && typeof fieldPair.key.value === 'string') {
                const fieldName = fieldPair.key.value;
                if (fieldName.includes('|')) {
                  const modifiers = fieldName.split('|').slice(1);
                  for (const mod of modifiers) {
                    if (!includes(VALUE_MODIFIERS, mod)) {
                      const pos = fieldPair.key.range ? nodeLineCol(fieldPair.key, lineIndex) : { line: 1, column: 1 };
                      diagnostics.push({
                        ...pos,
                        severity: Severity.Error,
                        message: `Unknown value modifier '${mod}' in search identifier '${searchId}'`,
                      });
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if ('falsepositives' in parsed && Array.isArray(parsed['falsepositives'])) {
    const fpNode = findMapValue(doc.contents, 'falsepositives');
    if (isSeq(fpNode)) {
      for (let i = 0; i < fpNode.items.length; i++) {
        const item = fpNode.items[i];
        if (isScalar(item) && typeof item.value === 'string' && item.value.length < FALSEPOSITIVES_MIN_LENGTH) {
          const pos = item.range ? nodeLineCol(item, lineIndex) : { line: 1, column: 1 };
          diagnostics.push({
            ...pos,
            severity: Severity.Warning,
            message: `False positive entry should be at least ${FALSEPOSITIVES_MIN_LENGTH} characters`,
          });
        }
      }
    }
  }

  if ('scope' in parsed && Array.isArray(parsed['scope'])) {
    const scopeNode = findMapValue(doc.contents, 'scope');
    if (isSeq(scopeNode)) {
      for (let i = 0; i < scopeNode.items.length; i++) {
        const item = scopeNode.items[i];
        if (isScalar(item) && typeof item.value === 'string' && item.value.length < SCOPE_MIN_LENGTH) {
          const pos = item.range ? nodeLineCol(item, lineIndex) : { line: 1, column: 1 };
          diagnostics.push({
            ...pos,
            severity: Severity.Warning,
            message: `Scope entry should be at least ${SCOPE_MIN_LENGTH} characters`,
          });
        }
      }
    }
  }

  if ('references' in parsed && !Array.isArray(parsed['references'])) {
    const node = findMapKey(doc.contents, 'references');
    const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'references' must be a list of strings`,
    });
  }

  if ('fields' in parsed && !Array.isArray(parsed['fields'])) {
    const node = findMapKey(doc.contents, 'fields');
    const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'fields' must be a list of strings`,
    });
  }

  return diagnostics;
}
