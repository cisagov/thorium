import type { Document } from 'yaml';
import { isMap, isPair, isScalar } from 'yaml';
import { FieldValueType, type Suggestion, type FieldSchema } from '../../types';
import { buildLineIndex, offsetToLineCol, type LineIndex } from '../../yaml';
import { includes } from '../../types';
import { dedupeSuggestions } from '../shared';
import {
  EVENT_TRIGGER_VALUES,
  KNOWN_PIPELINE_FIELDS,
  KNOWN_TAG_TRIGGER_FIELDS,
  PIPELINE_FIELD_SCHEMAS,
  PIPELINE_SECTION_ORDER,
  REQUIRED_PIPELINE_FIELDS,
  TAG_TRIGGER_SCHEMA,
  pipelineFieldCategory,
} from './schema';

function suggestNullReplace(
  field: string,
  parsed: Record<string, unknown>,
  line: number,
  schemas: Record<string, FieldSchema>,
  suggestions: Suggestion[],
): boolean {
  if (!(field in parsed) || parsed[field] !== null) return false;
  const schema = schemas[field];
  if (!schema) return false;
  suggestions.push({
    line,
    field,
    message: `Populate '${field}'`,
    schema,
    isReplace: true,
  });
  return true;
}

function lookupSchema(schemas: Record<string, FieldSchema>, dottedField: string): FieldSchema | undefined {
  const parts = dottedField.split('.');
  let schema = schemas[parts[0]];
  for (let i = 1; i < parts.length && schema; i++) {
    if (schema.type === FieldValueType.Object && schema.fields) {
      schema = schema.fields[parts[i]];
    } else {
      return undefined;
    }
  }
  return schema;
}

function findKeyLine(map: unknown, key: string, lineIndex: LineIndex): number {
  if (!isMap(map)) return 1;
  for (const item of map.items) {
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

export function generatePipelineSuggestions(
  doc: Document,
  text: string,
  parsed: Record<string, unknown>,
  imageNames?: Set<string> | null,
): Suggestion[] {
  const suggestions: Suggestion[] = [];
  const lineIndex = buildLineIndex(text);
  const endLine = lastDocLine(lineIndex);

  for (const field of KNOWN_PIPELINE_FIELDS) {
    if (!(field in parsed) || parsed[field] !== null) continue;
    // triggers is a map, not a struct — its population is handled by the trigger-name map entry below
    if (field === 'triggers') continue;
    const schema = lookupSchema(PIPELINE_FIELD_SCHEMAS, field);
    if (!schema) continue;
    const line = findKeyLine(doc.contents, field, lineIndex);
    if (schema.type === FieldValueType.Object || schema.type === FieldValueType.StringArray) {
      suggestNullReplace(field, parsed, line, PIPELINE_FIELD_SCHEMAS, suggestions);
    } else if (schema.enumValues) {
      suggestions.push({ line, field, message: `Populate '${field}'`, values: schema.enumValues, schema });
    } else {
      suggestions.push({ line, field, message: `Populate '${field}'`, schema, isReplace: true });
    }
  }

  // Empty `order: []` (e.g. a freshly created pipeline) — offer to populate it via the stage editor.
  // (A null `order` is already covered by the loop above; an empty array is not.)
  if (Array.isArray(parsed['order']) && (parsed['order'] as unknown[]).length === 0) {
    suggestions.push({
      line: findKeyLine(doc.contents, 'order', lineIndex),
      field: 'order',
      message: 'Populate order',
      schema: PIPELINE_FIELD_SCHEMAS.order,
      isReplace: true,
    });
  }

  // Trigger suggestions: fix invalid existing triggers + always offer "add new trigger".
  // `triggers` is a map (HashMap<String, EventTrigger>); a null value is treated the same as an
  // empty `{}` so population goes through the trigger-name map entry (not a struct populate).
  let trigLine: number | null = null;
  let triggersIsEmpty = false;
  if ('triggers' in parsed && parsed['triggers'] === null) {
    trigLine = findKeyLine(doc.contents, 'triggers', lineIndex);
    triggersIsEmpty = true;
  } else if (
    'triggers' in parsed &&
    typeof parsed['triggers'] === 'object' &&
    parsed['triggers'] !== null &&
    !Array.isArray(parsed['triggers'])
  ) {
    const triggers = parsed['triggers'] as Record<string, unknown>;
    trigLine = findKeyLine(doc.contents, 'triggers', lineIndex);
    triggersIsEmpty = Object.keys(triggers).length === 0;
    for (const [key, val] of Object.entries(triggers)) {
      if (
        val === '' ||
        (typeof val === 'string' && !includes(EVENT_TRIGGER_VALUES, val)) ||
        (typeof val !== 'string' && typeof val !== 'object')
      ) {
        suggestions.push({
          line: trigLine,
          field: `triggers.${key}`,
          message: `Set trigger type`,
          schema: PIPELINE_FIELD_SCHEMAS.triggers.fields!['trigger-name'],
        });
      } else if (typeof val === 'object' && val !== null) {
        // Object triggers are the `Tag` variant: { Tag: { tag_types, required, not } }.
        // Sub-fields live under the `Tag` key, so descend before checking/suggesting.
        const trigVal = val as Record<string, unknown>;
        const tagObj = typeof trigVal['Tag'] === 'object' && trigVal['Tag'] !== null ? (trigVal['Tag'] as Record<string, unknown>) : {};
        for (const f of KNOWN_TAG_TRIGGER_FIELDS) {
          if (!(f in tagObj)) {
            const trigSchema = TAG_TRIGGER_SCHEMA.fields?.[f];
            suggestions.push({
              line: trigLine,
              field: `triggers.${key}.Tag.${f}`,
              message: trigSchema?.description ?? `Consider adding '${f}'`,
              schema: trigSchema,
            });
          }
        }
      }
    }
  }

  const addTrigLine = trigLine ?? endLine;
  suggestions.push({
    line: addTrigLine,
    field: 'triggers.trigger-name',
    message: triggersIsEmpty ? 'Populate triggers' : 'Add trigger',
    schema: PIPELINE_FIELD_SCHEMAS.triggers.fields!['trigger-name'],
    isMapEntry: true,
    isReplace: triggersIsEmpty || undefined,
  });

  for (const field of KNOWN_PIPELINE_FIELDS) {
    // triggers is a map (HashMap<String, EventTrigger>), not a struct — adding it (even when the
    // key is absent) is handled by the trigger-name map entry above. A generic struct suggestion
    // here would render `trigger-name` as a literal field with no name/Tag editing.
    if (field === 'triggers') continue;
    if (!(field in parsed)) {
      const schema = lookupSchema(PIPELINE_FIELD_SCHEMAS, field);
      const isRequired = includes(REQUIRED_PIPELINE_FIELDS, field);
      suggestions.push({
        line: endLine,
        field,
        message: isRequired ? `Required: '${field}'` : (schema?.description ?? `Consider adding '${field}'`),
        values: schema?.enumValues,
        schema,
      });
    }
  }

  // Removal suggestions for unknown fields
  for (const key of Object.keys(parsed)) {
    if (!includes(KNOWN_PIPELINE_FIELDS, key)) {
      const keyLine = findKeyLine(doc.contents, key, lineIndex);
      suggestions.push({
        line: keyLine,
        field: key,
        message: `Remove unknown field '${key}'`,
        isRemoval: true,
        category: 'Unknown Fields',
      });
    }
  }

  // Surface the group's images to the order stage editor (rendered as select dropdowns) by
  // attaching them as the order schema's allowed values. Cleared from `values` so the suggestion
  // renders the stage editor, not value chips.
  if (imageNames && imageNames.size > 0) {
    const images = Array.from(imageNames).sort();
    for (const s of suggestions) {
      if (s.field === 'order' && s.schema) {
        s.schema = { ...s.schema, enumValues: images };
        s.values = undefined;
      }
    }
  }

  const deduped = dedupeSuggestions(suggestions);
  for (const s of deduped) s.category = s.category ?? pipelineFieldCategory(s.field);
  const sectionIndex = (cat: string) => {
    const idx = (PIPELINE_SECTION_ORDER as readonly string[]).indexOf(cat);
    return idx >= 0 ? idx : PIPELINE_SECTION_ORDER.length;
  };
  deduped.sort((a, b) => {
    const sa = sectionIndex(a.category!);
    const sb = sectionIndex(b.category!);
    if (sa !== sb) return sa - sb;
    return a.field.localeCompare(b.field);
  });
  return deduped;
}
