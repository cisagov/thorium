import type { Document } from 'yaml';
import { isMap, isPair, isScalar } from 'yaml';

// project imports
import type { Suggestion } from '../types';
import { buildLineIndex, offsetToLineCol, type LineIndex } from '../yaml';
import {
  STATUS_VALUES,
  LEVEL_VALUES,
  RELATED_TYPES,
  COMMON_LOGSOURCE_CATEGORIES,
  COMMON_LOGSOURCE_PRODUCTS,
  COMMON_LOGSOURCE_SERVICES,
} from './schema';

function findKeyLine(doc: Document, key: string, lineIndex: LineIndex): number {
  const contents = doc.contents;
  if (!isMap(contents)) return 1;
  for (const item of contents.items) {
    if (isPair(item) && isScalar(item.key) && item.key.value === key) {
      const offset = item.key.range?.[0] ?? 0;
      return offsetToLineCol(lineIndex, offset).line;
    }
  }
  return 1;
}

function lastDocLine(lineIndex: LineIndex): number {
  return lineIndex.offsets.length;
}

export function generateSuggestions(doc: Document, text: string, parsed: Record<string, unknown>): Suggestion[] {
  const suggestions: Suggestion[] = [];
  const lineIndex = buildLineIndex(text);

  if ('status' in parsed) {
    const line = findKeyLine(doc, 'status', lineIndex);
    if (typeof parsed['status'] !== 'string' || parsed['status'] === '') {
      suggestions.push({
        line,
        field: 'status',
        message: 'Valid status values',
        values: STATUS_VALUES,
      });
    }
  }

  if ('level' in parsed) {
    const line = findKeyLine(doc, 'level', lineIndex);
    if (typeof parsed['level'] !== 'string' || parsed['level'] === '') {
      suggestions.push({
        line,
        field: 'level',
        message: 'Valid level values',
        values: LEVEL_VALUES,
      });
    }
  }

  if ('logsource' in parsed && typeof parsed['logsource'] === 'object' && parsed['logsource'] !== null) {
    const ls = parsed['logsource'] as Record<string, unknown>;
    const logsourceLine = findKeyLine(doc, 'logsource', lineIndex);

    if (!('category' in ls) || ls['category'] === '' || ls['category'] === null) {
      suggestions.push({
        line: logsourceLine,
        field: 'logsource.category',
        message: 'Common logsource categories',
        values: COMMON_LOGSOURCE_CATEGORIES,
      });
    }

    if (!('product' in ls) || ls['product'] === '' || ls['product'] === null) {
      suggestions.push({
        line: logsourceLine,
        field: 'logsource.product',
        message: 'Common logsource products',
        values: COMMON_LOGSOURCE_PRODUCTS,
      });
    }

    if (!('service' in ls) || ls['service'] === '' || ls['service'] === null) {
      suggestions.push({
        line: logsourceLine,
        field: 'logsource.service',
        message: 'Common logsource services',
        values: COMMON_LOGSOURCE_SERVICES,
      });
    }
  }

  if ('related' in parsed && Array.isArray(parsed['related'])) {
    for (const entry of parsed['related']) {
      if (typeof entry === 'object' && entry !== null) {
        const rel = entry as Record<string, unknown>;
        if (!('type' in rel) || typeof rel['type'] !== 'string' || rel['type'] === '') {
          const line = findKeyLine(doc, 'related', lineIndex);
          suggestions.push({
            line,
            field: 'related.type',
            message: 'Valid related type values',
            values: RELATED_TYPES,
          });
        }
      }
    }
  }

  const endLine = lastDocLine(lineIndex);
  const missingOptional: Array<{ field: string; message: string; values?: readonly string[]; isList?: boolean }> = [];

  if (!('id' in parsed)) {
    missingOptional.push({ field: 'id', message: 'Consider adding a UUID v4 identifier for tracking' });
  }
  if (!('description' in parsed)) {
    missingOptional.push({ field: 'description', message: 'Consider adding a description of the detection' });
  }
  if (!('author' in parsed)) {
    missingOptional.push({ field: 'author', message: 'Consider adding an author attribution' });
  }
  if (!('date' in parsed)) {
    missingOptional.push({ field: 'date', message: 'Consider adding a creation date (YYYY-MM-DD)' });
  }
  if (!('modified' in parsed)) {
    missingOptional.push({ field: 'modified', message: 'Consider adding a modification date (YYYY-MM-DD)' });
  }
  if (!('level' in parsed)) {
    missingOptional.push({
      field: 'level',
      message: 'Consider adding a severity level',
      values: LEVEL_VALUES,
    });
  }
  if (!('status' in parsed)) {
    missingOptional.push({
      field: 'status',
      message: 'Consider adding a rule status',
      values: STATUS_VALUES,
    });
  }
  if (!('tags' in parsed)) {
    missingOptional.push({ field: 'tags', message: 'Consider adding tags (lowercase, dot-separated: e.g. attack.t1059)', isList: true });
  }
  if (!('falsepositives' in parsed)) {
    missingOptional.push({ field: 'falsepositives', message: 'Consider documenting known false positives', isList: true });
  }
  if (!('references' in parsed)) {
    missingOptional.push({ field: 'references', message: 'Consider adding reference URLs for context', isList: true });
  }

  for (const opt of missingOptional) {
    suggestions.push({
      line: endLine,
      field: opt.field,
      message: opt.message,
      values: opt.values,
      isList: opt.isList,
    });
  }

  return suggestions;
}
