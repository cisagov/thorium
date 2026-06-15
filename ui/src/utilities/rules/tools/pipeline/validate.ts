import type { Document } from 'yaml';
import { isSeq } from 'yaml';

// project imports
import { Severity, type Diagnostic } from '../../types';
import { buildLineIndex } from '../../yaml';
import {
  findMapValue,
  nodePosition,
  validateUnknownFields,
  validateStringField,
  validateNumberField,
  validateObjectField,
} from '../shared';
import { REQUIRED_PIPELINE_FIELDS, KNOWN_PIPELINE_FIELDS } from './schema';

// spec: ./validate.spec.md

/**
 * Validate a parsed pipeline-request YAML document against the pipeline schema.
 *
 * Reports missing required fields, unknown top-level fields, and per-field type errors, then checks
 * the `order` and `triggers` structures. When `validImageNames` is supplied, each image referenced
 * in `order` (either a bare stage entry or a nested parallel group) is verified against that set and
 * flagged if unknown.
 *
 * @param doc - The parsed YAML document (positional info for diagnostics).
 * @param text - The raw YAML source, used to map node offsets to line/column.
 * @param parsed - The plain-object form of `doc` used for value checks.
 * @param validImageNames - Optional set of image names valid for the pipeline's group; when given,
 *   unknown image references in `order` are reported. Omit or pass `null` to skip the check.
 * @returns The list of {@link Diagnostic}s found (empty when the document is valid).
 */
export function validatePipelineRequest(
  doc: Document,
  text: string,
  parsed: Record<string, unknown>,
  validImageNames?: Set<string> | null,
): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const lineIndex = buildLineIndex(text);
  const contents = doc.contents;

  const lastLine = lineIndex.offsets.length;
  for (const field of REQUIRED_PIPELINE_FIELDS) {
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

  validateUnknownFields(contents, KNOWN_PIPELINE_FIELDS, 'pipeline', parsed, lineIndex, diagnostics);

  validateStringField(contents, parsed, 'group', lineIndex, diagnostics);
  validateStringField(contents, parsed, 'name', lineIndex, diagnostics);
  validateStringField(contents, parsed, 'description', lineIndex, diagnostics);
  validateNumberField(contents, parsed, 'sla', lineIndex, diagnostics);

  if ('order' in parsed) {
    if (!Array.isArray(parsed['order'])) {
      const node = findMapValue(contents, 'order');
      const pos = nodePosition(node, lineIndex);
      diagnostics.push({ ...pos, severity: Severity.Error, message: "'order' must be an array" });
    } else {
      const orderNode = findMapValue(contents, 'order');
      const orderArr = parsed['order'] as unknown[];
      const group = typeof parsed['group'] === 'string' ? parsed['group'] : null;
      if (isSeq(orderNode)) {
        for (let i = 0; i < orderNode.items.length; i++) {
          const item = orderNode.items[i];
          const val = orderArr[i];
          if (typeof val === 'string') {
            if (validImageNames && !validImageNames.has(val)) {
              const pos = nodePosition(item, lineIndex);
              diagnostics.push({
                ...pos,
                severity: Severity.Error,
                message: `Image '${val}' not found in group '${group}'`,
              });
            }
            continue;
          }
          if (Array.isArray(val)) {
            for (let j = 0; j < val.length; j++) {
              if (typeof val[j] !== 'string') {
                const pos = nodePosition(item, lineIndex);
                diagnostics.push({
                  ...pos,
                  severity: Severity.Error,
                  message: `order[${i}][${j}] must be a string (image name)`,
                });
              } else if (validImageNames && !validImageNames.has(val[j] as string)) {
                const subNode = isSeq(item) && item.items[j] ? item.items[j] : item;
                const pos = nodePosition(subNode, lineIndex);
                diagnostics.push({
                  ...pos,
                  severity: Severity.Error,
                  message: `Image '${val[j]}' not found in group '${group}'`,
                });
              }
            }
          } else {
            const pos = nodePosition(item, lineIndex);
            diagnostics.push({
              ...pos,
              severity: Severity.Error,
              message: `order[${i}] must be a string or array of strings`,
            });
          }
        }
      }
    }
  }

  if ('triggers' in parsed) {
    const triggers = validateObjectField(contents, parsed, 'triggers', lineIndex, diagnostics);
    if (triggers) {
      const trigMap = findMapValue(contents, 'triggers');
      for (const [key, val] of Object.entries(triggers)) {
        if (typeof val === 'string') {
          if (val !== 'NewSample') {
            const node = findMapValue(trigMap, key);
            const pos = nodePosition(node, lineIndex);
            diagnostics.push({
              ...pos,
              severity: Severity.Error,
              message: `Invalid trigger value: '${val}'. Must be 'NewSample' or a Tag object`,
            });
          }
        } else if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
          const trigObj = val as Record<string, unknown>;
          if (!('Tag' in trigObj)) {
            const node = findMapValue(trigMap, key);
            const pos = nodePosition(node, lineIndex);
            diagnostics.push({
              ...pos,
              severity: Severity.Warning,
              message: `Trigger '${key}' object should contain a 'Tag' key`,
            });
          }
        } else {
          const node = findMapValue(trigMap, key);
          const pos = nodePosition(node, lineIndex);
          diagnostics.push({
            ...pos,
            severity: Severity.Error,
            message: `Trigger '${key}' must be 'NewSample' or a Tag object`,
          });
        }
      }
    }
  }

  return diagnostics;
}
