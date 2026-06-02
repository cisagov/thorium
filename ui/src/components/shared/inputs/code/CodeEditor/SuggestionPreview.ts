import { EditorView, Decoration, type DecorationSet, WidgetType } from '@codemirror/view';
import { EditorState, StateField, StateEffect } from '@codemirror/state';
import { FormatType, FieldValueType, type FieldSchema } from '@utilities/rules/types';

export interface PreviewProposal {
  field: string;
  value: string;
  format: FormatType;
  cursorLine?: number;
  isList?: boolean;
  isMapEntry?: boolean;
  isRemoval?: boolean;
  isReplace?: boolean;
  schema?: FieldSchema;
}

const INDENT = '  ';
const INDENT_SIZE = INDENT.length;

export const addPreview = StateEffect.define<PreviewProposal>();
export const clearPreview = StateEffect.define<void>();
export const acceptPreview = StateEffect.define<void>();
const updateInsertText = StateEffect.define<string>();

function lineOffset(lines: string[], lineIdx: number): number {
  let offset = 0;
  for (let k = 0; k < lineIdx; k++) {
    offset += lines[k].length + 1;
  }
  return offset;
}

function buildYaraInsertText(
  field: string,
  value: string,
  docText: string,
  lines: string[],
  cursorLine?: number,
): { text: string; pos: number; inline?: boolean } {
  if (field.startsWith('section.')) {
    const sectionName = field.slice('section.'.length);
    const sectionOrder = ['meta', 'strings', 'condition'];
    const sectionIdx = sectionOrder.indexOf(sectionName);

    const scaffolds: Record<string, string> = {
      meta: '    meta:\n        description = ""\n        author = ""\n',
      strings: '    strings:\n        $s1 = ""\n',
      condition: '    condition:\n        true\n',
    };

    let insertLineIdx = -1;
    for (let s = sectionIdx + 1; s < sectionOrder.length; s++) {
      const re = new RegExp(`^\\s*${sectionOrder[s]}\\s*:`);
      for (let i = 0; i < lines.length; i++) {
        if (re.test(lines[i])) {
          insertLineIdx = i;
          break;
        }
      }
      if (insertLineIdx >= 0) break;
    }

    if (insertLineIdx < 0) {
      for (let i = lines.length - 1; i >= 0; i--) {
        if (lines[i].trim() === '}') {
          insertLineIdx = i;
          break;
        }
      }
    }

    if (insertLineIdx >= 0) {
      const hasBlankBefore = insertLineIdx > 0 && lines[insertLineIdx - 1].trim() === '';
      const scaffold = scaffolds[sectionName] ?? `    ${sectionName}:\n`;
      const insertText = (hasBlankBefore ? '' : '\n') + scaffold + '\n';
      const pos = Math.min(lineOffset(lines, insertLineIdx), docText.length);
      return { text: insertText, pos };
    }
    const insertText = '\n' + (scaffolds[sectionName] ?? `    ${sectionName}:\n`);
    return { text: insertText, pos: docText.length };
  }

  if (field === 'meta' || field.startsWith('meta.')) {
    const metaKey = field === 'meta' ? value : field.split('.').slice(1).join('.');
    const metaVal = field === 'meta' ? `"<value>"` : `"${value}"`;
    let metaIdx = -1;
    let insertAfterIdx = -1;
    for (let i = 0; i < lines.length; i++) {
      if (/^\s*meta\s*:/.test(lines[i])) {
        metaIdx = i;
        insertAfterIdx = i;
        for (let j = i + 1; j < lines.length; j++) {
          if (/^\s+\S/.test(lines[j]) && !/^\s*(strings|condition)\s*:/.test(lines[j])) {
            insertAfterIdx = j;
          } else {
            break;
          }
        }
        break;
      }
    }
    if (metaIdx >= 0) {
      const indent = lines[insertAfterIdx].match(/^(\s*)/)?.[1] ?? '        ';
      const lineEnd = Math.min(lineOffset(lines, insertAfterIdx) + lines[insertAfterIdx].length, docText.length);
      return { text: `\n${indent}${metaKey} = ${metaVal}`, pos: lineEnd };
    }
    return { text: `\n        ${metaKey} = ${metaVal}`, pos: docText.length };
  }

  if (field === 'import') {
    return { text: `import "${value}"\n`, pos: 0 };
  }

  if (field === 'strings.modifiers' && cursorLine != null) {
    const lineIdx = Math.min(cursorLine - 1, lines.length - 1);
    const lineEnd = Math.min(lineOffset(lines, lineIdx + 1) - 1, docText.length);
    return { text: ` ${value}`, pos: lineEnd, inline: true };
  }

  return { text: `${value}\n`, pos: docText.length };
}

function defaultValueForType(schema: FieldSchema): string {
  switch (schema.type) {
    case FieldValueType.Number:
      return schema.placeholder ?? '';
    case FieldValueType.Boolean:
      return 'false';
    case FieldValueType.Enum:
      return schema.enumValues?.[0] ?? '';
    case FieldValueType.String:
      return '';
    default:
      return '';
  }
}

function stripDisplayCommas(text: string): string {
  return text.replace(/^,\s*\n?/, '').replace(/,\s*$/, '');
}

function formatValueForYaml(value: string, schema?: FieldSchema): string {
  if (!schema || !value) return `'${value}'`;
  switch (schema.type) {
    case FieldValueType.Number:
      return value;
    case FieldValueType.Boolean:
      return value;
    case FieldValueType.Enum:
      return value;
    default:
      return `'${value}'`;
  }
}

function buildVariantYamlText(
  field: string,
  variantName: string,
  variantSchema: FieldSchema | null,
  value: string,
  indent: string,
): string {
  if (!variantSchema) {
    return `${indent}${field}: ${variantName}\n`;
  }
  if (variantSchema.type === FieldValueType.Object && variantSchema.fields) {
    const defaults: Record<string, string> = {};
    for (const [k, s] of Object.entries(variantSchema.fields)) {
      defaults[k] = defaultValueForType(s);
    }
    return (
      `${indent}${field}:\n${indent}${INDENT}${variantName}:\n` +
      Object.entries(variantSchema.fields)
        .map(([k, s]) => {
          const v = defaults[k] || defaultValueForType(s);
          return `${indent}${INDENT}${INDENT}${k}: ${formatValueForYaml(v, s)}`;
        })
        .join('\n') +
      '\n'
    );
  }
  const formatted = formatValueForYaml(value || defaultValueForType(variantSchema), variantSchema);
  return `${indent}${field}:\n${indent}${INDENT}${variantName}: ${formatted}\n`;
}

function buildVariantJsonText(field: string, variantName: string, variantSchema: FieldSchema | null, value: string): string {
  if (!variantSchema) {
    return `"${field}": "${variantName}"`;
  }
  if (variantSchema.type === FieldValueType.Object && variantSchema.fields) {
    const entries: string[] = [];
    for (const [k, s] of Object.entries(variantSchema.fields)) {
      const v = defaultValueForType(s);
      entries.push(`"${k}": ${formatJsonPrimitive(v, s)}`);
    }
    return `"${field}": { "${variantName}": { ${entries.join(', ')} } }`;
  }
  const formatted = formatJsonPrimitive(value || defaultValueForType(variantSchema), variantSchema);
  return `"${field}": { "${variantName}": ${formatted} }`;
}

// Variant carrying a list payload (e.g. AutoTagLogic In/NotIn). `field:\n  Variant:\n    - 'v'`
export function buildVariantListYaml(field: string, variantName: string, values: string[], indent: string): string {
  const i1 = `${indent}${INDENT}`;
  const i2 = `${i1}${INDENT}`;
  if (values.length === 0) return `${indent}${field}:\n${i1}${variantName}: []\n`;
  return `${indent}${field}:\n${i1}${variantName}:\n` + values.map((v) => `${i2}- '${v}'`).join('\n') + '\n';
}

export function buildVariantListJson(field: string, variantName: string, values: string[]): string {
  return `"${field}": { "${variantName}": [${values.map((v) => `"${v}"`).join(', ')}] }`;
}

// --- Nested-list (stage) helpers (e.g. pipeline `order`) ---
// A list of stages, each a parallel group of images, serialized as Vec<Vec<String>> (always
// grouped — a single-image stage is still a one-element array). Empty images and empty stages
// are dropped. Each stage is emitted on one line (a flow array) so the structure stays readable.

function cleanStages(stages: string[][]): string[][] {
  return stages.map((s) => s.map((i) => i.trim()).filter(Boolean)).filter((s) => s.length > 0);
}

export function buildStageListYaml(field: string, stages: string[][], baseIndent: string): string {
  const cleaned = cleanStages(stages);
  if (cleaned.length === 0) return `${baseIndent}${field}: []\n`;
  const lines = cleaned.map((s) => `${baseIndent}${INDENT}- [${s.join(', ')}]`);
  return `${baseIndent}${field}:\n${lines.join('\n')}\n`;
}

export function buildStageListJson(stages: string[][], baseIndent: string): string {
  const cleaned = cleanStages(stages);
  if (cleaned.length === 0) return '[]';
  const inner = baseIndent + INDENT;
  const lines = cleaned.map((s) => `${inner}[${s.map((i) => `"${i}"`).join(', ')}]`);
  return `[\n${lines.join(',\n')}\n${baseIndent}]`;
}

// Variant carrying a key/value map payload (e.g. KwargDependency Map: <image> -> <kwarg>).
export function buildVariantMapYaml(
  field: string,
  variantName: string,
  entries: Array<{ key: string; value: string }>,
  indent: string,
): string {
  const i1 = `${indent}${INDENT}`;
  const i2 = `${i1}${INDENT}`;
  if (entries.length === 0) return `${indent}${field}:\n${i1}${variantName}: {}\n`;
  return (
    `${indent}${field}:\n${i1}${variantName}:\n` + entries.map((e) => `${i2}${e.key}: ${e.value ? `'${e.value}'` : "''"}`).join('\n') + '\n'
  );
}

export function buildVariantMapJson(field: string, variantName: string, entries: Array<{ key: string; value: string }>): string {
  return `"${field}": { "${variantName}": { ${entries.map((e) => `"${e.key}": "${e.value}"`).join(', ')} } }`;
}

// Emits a `field:` entry for a variant whose selected variant is `selected`, using an empty payload of
// the correct shape (unit -> bare name; scalar -> ''; list -> []; map -> {}; object -> default fields).
// Used when a variant field is nested inside an object form (the variant is picked in the form; its
// payload is then filled in the editor, or fully via the standalone variant suggestion).
export function buildVariantScaffoldYaml(
  field: string,
  variants: Record<string, FieldSchema | null>,
  selected: string,
  indent: string,
): string {
  const vs = variants[selected];
  if (!vs) return `${indent}${field}: ${selected}\n`;
  if (vs.type === FieldValueType.StringArray) return buildVariantListYaml(field, selected, [], indent);
  if (vs.type === FieldValueType.Object && !vs.fields) return buildVariantMapYaml(field, selected, [], indent);
  return buildVariantYamlText(field, selected, vs, '', indent);
}

export function buildVariantScaffoldJson(field: string, variants: Record<string, FieldSchema | null>, selected: string): string {
  const vs = variants[selected];
  if (!vs) return `"${field}": "${selected}"`;
  if (vs.type === FieldValueType.StringArray) return buildVariantListJson(field, selected, []);
  if (vs.type === FieldValueType.Object && !vs.fields) return buildVariantMapJson(field, selected, []);
  return buildVariantJsonText(field, selected, vs, '');
}

function buildObjectYamlText(field: string, schema: FieldSchema, values: Record<string, string>, indent: string = ''): string {
  if (!schema.fields) return `${indent}${field}: {}\n`;

  let text = `${indent}${field}:\n`;
  let addedAny = false;

  for (const [subKey, subSchema] of Object.entries(schema.fields)) {
    if (subSchema.type === FieldValueType.Object) {
      // A `placeholder` on an object schema marks a map (keyed by that placeholder, e.g. auto_tag),
      // whose `fields` describe the value shape — emit an empty map rather than its value fields.
      if (subSchema.fields && !subSchema.placeholder) {
        const subDefaults: Record<string, string> = {};
        for (const [k, s] of Object.entries(subSchema.fields)) {
          subDefaults[k] = defaultValueForType(s);
        }
        text += buildObjectYamlText(subKey, subSchema, subDefaults, indent + INDENT);
      } else {
        text += `${indent}${INDENT}${subKey}: {}\n`;
      }
      addedAny = true;
      continue;
    }
    if (subSchema.variants) {
      // a variant field nested in an object (e.g. results.kwarg, auto_tag.logic)
      const sel = values[subKey] || defaultValueForType(subSchema);
      text += buildVariantScaffoldYaml(subKey, subSchema.variants, sel, indent + INDENT);
      addedAny = true;
      continue;
    }
    if (subSchema.type === FieldValueType.StringArray) {
      const item = (values[subKey] ?? '').trim();
      text += item ? `${indent}${INDENT}${subKey}:\n${indent}${INDENT}${INDENT}- '${item}'\n` : `${indent}${INDENT}${subKey}: []\n`;
      addedAny = true;
      continue;
    }

    const val = values[subKey] ?? '';
    text += `${indent}${INDENT}${subKey}: ${formatValueForYaml(val || defaultValueForType(subSchema), subSchema)}\n`;
    addedAny = true;
  }

  if (!addedAny) return `${indent}${field}: {}\n`;
  return text;
}

function buildObjectJsonText(field: string, schema: FieldSchema, values: Record<string, string>): string {
  if (!schema.fields) return `"${field}": {}`;

  const entries: string[] = [];
  for (const [subKey, subSchema] of Object.entries(schema.fields)) {
    if (subSchema.type === FieldValueType.Object) {
      // see buildObjectYamlText: a placeholder marks a map; emit {} rather than its value fields
      if (subSchema.fields && !subSchema.placeholder) {
        const subDefaults: Record<string, string> = {};
        for (const [k, s] of Object.entries(subSchema.fields)) {
          subDefaults[k] = defaultValueForType(s);
        }
        entries.push(buildObjectJsonText(subKey, subSchema, subDefaults));
      } else {
        entries.push(`"${subKey}": {}`);
      }
    } else if (subSchema.variants) {
      const sel = values[subKey] || defaultValueForType(subSchema);
      entries.push(buildVariantScaffoldJson(subKey, subSchema.variants, sel));
    } else if (subSchema.type === FieldValueType.StringArray) {
      const item = (values[subKey] ?? '').trim();
      entries.push(`"${subKey}": ${item ? `["${item}"]` : '[]'}`);
    } else {
      const val = values[subKey] ?? '';
      entries.push(`"${subKey}": ${formatJsonPrimitive(val || defaultValueForType(subSchema), subSchema)}`);
    }
  }
  if (entries.length === 0) return `"${field}": {}`;
  return `"${field}": { ${entries.join(', ')} }`;
}

// --- Object-list (e.g. volumes) helpers ---

// Default values for an object entry, plus the active variant config key derived from
// the schema's discriminator field (variantField). Used to seed initial insert text.
function defaultObjectEntryValues(schema: FieldSchema): { values: Record<string, string>; activeVariantKey?: string } {
  const values: Record<string, string> = {};
  const fields = schema.fields ?? {};
  for (const [k, s] of Object.entries(fields)) {
    if (s.type === FieldValueType.Object && s.fields) continue;
    values[k] = defaultValueForType(s);
  }
  let activeVariantKey: string | undefined;
  if (schema.variantField) {
    const disc = values[schema.variantField.field] || '';
    activeVariantKey = schema.variantField.fieldMap[disc];
    const variantSchema = activeVariantKey ? fields[activeVariantKey] : undefined;
    if (variantSchema?.fields) {
      for (const [k2, s2] of Object.entries(variantSchema.fields)) {
        values[`${activeVariantKey}.${k2}`] = defaultValueForType(s2);
      }
    }
  }
  return { values, activeVariantKey };
}

// Builds a YAML `field:` header followed by one list-item object entry. Only the nested
// object named by activeVariantKey is rendered (other variant configs are skipped).
function buildObjectListItemYamlText(
  field: string,
  schema: FieldSchema,
  values: Record<string, string>,
  baseIndent: string,
  activeVariantKey?: string,
): string {
  const fields = schema.fields;
  const markerIndent = baseIndent + INDENT;
  const contentIndent = markerIndent + '  ';
  if (!fields) return `${baseIndent}${field}:\n${markerIndent}- {}\n`;

  let body = '';
  let first = true;
  for (const [key, sub] of Object.entries(fields)) {
    const prefix = first ? `${markerIndent}- ` : contentIndent;
    if (sub.type === FieldValueType.Object && sub.fields) {
      if (!activeVariantKey || key !== activeVariantKey) continue;
      body += `${prefix}${key}:\n`;
      for (const [k2, s2] of Object.entries(sub.fields)) {
        const v = values[`${key}.${k2}`] ?? '';
        body += `${contentIndent}${INDENT}${k2}: ${formatValueForYaml(v || s2.placeholder || '', s2)}\n`;
      }
      first = false;
      continue;
    }
    if (sub.type === FieldValueType.StringArray) {
      body += `${prefix}${key}: []\n`;
    } else {
      const v = values[key] ?? '';
      body += `${prefix}${key}: ${formatValueForYaml(v || sub.placeholder || '', sub)}\n`;
    }
    first = false;
  }
  if (!body) body = `${markerIndent}- {}\n`;
  return `${baseIndent}${field}:\n${body}`;
}

// Builds a single JSON object entry `{ ... }` for an object schema (active variant only).
function buildObjectEntryJsonText(schema: FieldSchema, values: Record<string, string>, activeVariantKey?: string): string {
  const fields = schema.fields;
  if (!fields) return '{}';
  const entries: string[] = [];
  for (const [key, sub] of Object.entries(fields)) {
    if (sub.type === FieldValueType.Object && sub.fields) {
      if (!activeVariantKey || key !== activeVariantKey) continue;
      const subEntries: string[] = [];
      for (const [k2, s2] of Object.entries(sub.fields)) {
        const v = values[`${key}.${k2}`] ?? '';
        subEntries.push(`"${k2}": ${formatJsonPrimitive(v || s2.placeholder || '', s2)}`);
      }
      entries.push(`"${key}": { ${subEntries.join(', ')} }`);
      continue;
    }
    if (sub.type === FieldValueType.StringArray) {
      entries.push(`"${key}": []`);
    } else {
      const v = values[key] ?? '';
      entries.push(`"${key}": ${formatJsonPrimitive(v || sub.placeholder || '', sub)}`);
    }
  }
  if (entries.length === 0) return '{}';
  return `{ ${entries.join(', ')} }`;
}

// Finds the alphabetical insertion position among sibling keys at a given indent
function findAlphabeticalInsertPos(
  field: string,
  lines: string[],
  scopeStart: number,
  scopeEnd: number,
  targetIndent: number,
): { insertBeforeLineIdx: number } | null {
  const siblings: { name: string; lineIdx: number }[] = [];

  for (let i = scopeStart; i < scopeEnd; i++) {
    const trimmed = lines[i].trimStart();
    if (!trimmed) continue;
    const lineIndent = lines[i].length - trimmed.length;
    if (lineIndent < targetIndent && trimmed.length > 0 && i > scopeStart) break;
    if (lineIndent === targetIndent) {
      const m = trimmed.match(/^([^:\s]+)\s*:/);
      if (m) siblings.push({ name: m[1], lineIdx: i });
    }
  }

  for (const sib of siblings) {
    if (sib.name.localeCompare(field) > 0) {
      return { insertBeforeLineIdx: sib.lineIdx };
    }
  }
  return null;
}

function buildYamlInsertText(
  field: string,
  value: string,
  docText: string,
  lines: string[],
  isList?: boolean,
  schema?: FieldSchema,
): InsertResult {
  const parts = field.split('.');
  if (parts.length === 1) {
    let entryText: string;
    if (isList && schema?.type === FieldValueType.Object && schema.fields) {
      const { values, activeVariantKey } = defaultObjectEntryValues(schema);
      entryText = buildObjectListItemYamlText(field, schema, values, '', activeVariantKey);
    } else if (schema?.type === FieldValueType.Object && schema.fields) {
      const defaults: Record<string, string> = {};
      for (const [k, s] of Object.entries(schema.fields)) {
        defaults[k] = defaultValueForType(s);
      }
      entryText = buildObjectYamlText(field, schema, defaults);
    } else if (schema?.nestedList) {
      // Minimal valid scaffold; the stage editor widget fills in the real grouped value on mount.
      entryText = buildStageListYaml(field, value ? [[value]] : [], '');
    } else if (schema?.type === FieldValueType.StringArray || isList) {
      entryText = `${field}:\n${INDENT}- '${value}'\n`;
    } else {
      entryText = `${field}: ${formatValueForYaml(value, schema)}\n`;
    }

    const alphaPos = findAlphabeticalInsertPos(field, lines, 0, lines.length, 0);
    if (alphaPos) {
      const pos = lineOffset(lines, alphaPos.insertBeforeLineIdx);
      return { text: entryText, pos };
    }

    const trailing = docText.endsWith('\n') ? '' : '\n';
    return { text: `${trailing}${entryText}`, pos: docText.length };
  }

  const leafKey = parts[parts.length - 1];
  let deepestFoundDepth = 0;
  let insertAfterIdx = -1;
  let nextChildIndent = 0;
  let searchStart = 0;
  let parentIndent = -1;
  let deepestScopeStart = 0;
  let deepestScopeEnd = 0;
  let inlineEmptyObjLineIdx = -1;

  for (let depth = 0; depth < parts.length - 1; depth++) {
    const key = parts[depth];
    let found = false;

    for (let i = searchStart; i < lines.length; i++) {
      const trimmed = lines[i].trimStart();
      if (!trimmed) continue;
      const lineIndent = lines[i].length - trimmed.length;

      if (depth > 0 && lineIndent <= parentIndent) break;

      if (trimmed.startsWith(`${key}:`) || trimmed.startsWith(`${key} :`)) {
        parentIndent = lineIndent;
        insertAfterIdx = i;
        let childIndent = lineIndent + INDENT_SIZE;
        let childDetected = false;
        deepestScopeStart = i + 1;

        const colonPos = trimmed.indexOf(':');
        const afterColon = trimmed.slice(colonPos + 1).trim();
        inlineEmptyObjLineIdx = /^\{\s*\}$/.test(afterColon) ? i : -1;

        for (let j = i + 1; j < lines.length; j++) {
          const jTrimmed = lines[j].trim();
          const jIndent = lines[j].length - lines[j].trimStart().length;
          if (!jTrimmed) {
            insertAfterIdx = j;
            continue;
          }
          if (jIndent > lineIndent) {
            insertAfterIdx = j;
            if (!childDetected) {
              childIndent = jIndent;
              childDetected = true;
            }
          } else break;
        }
        deepestScopeEnd = insertAfterIdx + 1;

        deepestFoundDepth = depth + 1;
        nextChildIndent = childIndent;
        searchStart = i + 1;
        found = true;
        break;
      }
    }

    if (!found) break;
  }

  if (deepestFoundDepth === 0) {
    const trailing = docText.endsWith('\n') ? '' : '\n';
    let text = trailing;
    for (let d = 0; d < parts.length - 1; d++) {
      text += INDENT.repeat(d) + parts[d] + ':\n';
    }
    if (schema?.type === FieldValueType.Object && schema.fields) {
      const leafIndent = INDENT.repeat(parts.length - 1);
      const defaults: Record<string, string> = {};
      for (const [k, s] of Object.entries(schema.fields)) {
        defaults[k] = defaultValueForType(s);
      }
      text += buildObjectYamlText(leafKey, schema, defaults, leafIndent);
    } else {
      text += INDENT.repeat(parts.length - 1) + `${leafKey}: ${formatValueForYaml(value, schema)}\n`;
    }

    const alphaPos = findAlphabeticalInsertPos(parts[0], lines, 0, lines.length, 0);
    if (alphaPos) {
      return { text: text.replace(/^\n/, ''), pos: lineOffset(lines, alphaPos.insertBeforeLineIdx) };
    }
    return { text, pos: docText.length };
  }

  const missingAncestors = parts.slice(deepestFoundDepth, parts.length - 1);
  let text = '';
  for (let d = 0; d < missingAncestors.length; d++) {
    text += ' '.repeat(nextChildIndent + d * INDENT_SIZE) + missingAncestors[d] + ':\n';
  }
  const leafIndent = nextChildIndent + missingAncestors.length * INDENT_SIZE;
  if (schema?.type === FieldValueType.StringArray || isList) {
    text += ' '.repeat(leafIndent) + `${leafKey}:\n` + ' '.repeat(leafIndent + INDENT_SIZE) + `- '${value}'`;
  } else if (schema?.type === FieldValueType.Object && schema.fields) {
    const defaults: Record<string, string> = {};
    for (const [k, s] of Object.entries(schema.fields)) {
      defaults[k] = defaultValueForType(s);
    }
    text += buildObjectYamlText(leafKey, schema, defaults, ' '.repeat(leafIndent)).replace(/\n$/, '');
  } else {
    text += ' '.repeat(leafIndent) + `${leafKey}: ${formatValueForYaml(value, schema)}`;
  }

  if (inlineEmptyObjLineIdx >= 0) {
    const lineStr = lines[inlineEmptyObjLineIdx];
    const objLineStart = lineOffset(lines, inlineEmptyObjLineIdx);
    const colonInLine = lineStr.indexOf(':');
    const replaceStart = objLineStart + colonInLine + 1;
    const replaceEnd = objLineStart + lineStr.length;
    return { text: '\n' + text, pos: replaceStart, replaceEnd };
  }

  // Try alphabetical placement within the deepest found scope
  const alphaKey = missingAncestors.length > 0 ? missingAncestors[0] : leafKey;
  const alphaPos = findAlphabeticalInsertPos(alphaKey, lines, deepestScopeStart, deepestScopeEnd, nextChildIndent);
  if (alphaPos) {
    return { text: text + '\n', pos: lineOffset(lines, alphaPos.insertBeforeLineIdx) };
  }

  const pos = Math.min(lineOffset(lines, insertAfterIdx + 1), docText.length);
  const needsNewline = insertAfterIdx < lines.length - 1 || !docText.endsWith('\n');
  text += needsNewline ? '\n' : '';

  return { text, pos };
}

// --- JSON helpers ---

// Extracts the formatting envelope from an insertText so rebuilt content
// preserves the leading comma/newline/indent and trailing comma/newline.
function jsonEnvelope(insertText: string): { prefix: string; trailing: string } {
  const quoteIdx = insertText.indexOf('"');
  const prefix = quoteIdx >= 0 ? insertText.slice(0, quoteIdx) : '';
  const trailing = insertText.match(/(,?\s*)$/)?.[1] ?? '';
  return { prefix, trailing };
}

// Re-formats a compact JSON value string as pretty-printed (2-space) JSON, re-indented so nested
// lines sit under a key already at `baseIndent`. Primitives, empty `{}`/`[]`, and anything that
// doesn't parse are returned unchanged (they are already single-line).
export function prettyJsonValue(value: string, baseIndent: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    return value;
  }
  if (parsed === null || typeof parsed !== 'object') return value;
  return JSON.stringify(parsed, null, INDENT)
    .split('\n')
    .map((line, i) => (i === 0 ? line : baseIndent + line))
    .join('\n');
}

// Pretty-prints a compact JSON object-entry fragment (`"key": <value>`), re-indenting object/array
// values under baseIndent. The map-entry analog of prettyJsonValue (used by the generic key path):
// the key and value start stay on one line; only object/array values expand. Fragments that don't
// match `"key": value`, or whose value is a primitive/empty `{}`/`[]`, are returned unchanged.
export function prettyJsonEntry(entry: string, baseIndent: string): string {
  const match = entry.match(/^("(?:[^"\\]|\\.)*")\s*:\s*([\s\S]+)$/);
  if (!match) return entry;
  const [, key, value] = match;
  return `${key}: ${prettyJsonValue(value, baseIndent)}`;
}

export interface MapEntryWrap {
  format: FormatType;
  parentKey: string;
  // The parent map key (e.g. `triggers`) is entirely absent from the document.
  parentMissing: boolean;
  // The parent exists but is empty (null / `{}`) and is being replaced (populate).
  isPopulate: boolean;
  // The initial buildInsertText output, used to recover the JSON envelope and YAML leading newline.
  insertText: string;
}

/**
 * Produces the final replacement text for a variant map entry (e.g. one `triggers` entry).
 * `child` is the already-built inner entry:
 *   - YAML: an indented `<name>: ...` block ending in a newline
 *   - JSON: a `"<name>": ...` fragment
 * The parent key (`triggers:` / `"triggers": { ... }`) is prepended when the parent is being
 * populated from empty (YAML) or is absent entirely (YAML and JSON), so the entry is never left
 * orphaned at the wrong nesting level. When the parent already exists, the child is emitted as-is
 * inside the format's envelope (appended to the existing map).
 */
export function buildMapEntryText(child: string, ctx: MapEntryWrap): string {
  const { format, parentKey, parentMissing, isPopulate, insertText } = ctx;
  if (format === FormatType.JSON) {
    const { prefix, trailing } = jsonEnvelope(insertText);
    // When the key is absent, emit a multi-line object (one entry per line) so later entries can be
    // appended with correct comma handling — an inline `{ ... }` would hide the existing key from
    // the inserter, which scans lines between the braces. JSON populate (`{}`) keeps the existing
    // key and fills between its braces instead, so it never reaches here.
    if (parentKey && parentMissing) {
      const keyIndent = prefix.slice(prefix.lastIndexOf('\n') + 1);
      const childIndent = keyIndent + INDENT;
      return `${prefix}"${parentKey}": {\n${childIndent}${prettyJsonEntry(child, childIndent)}\n${keyIndent}}${trailing}`;
    }
    // Pretty-print object/array values when the entry is on its own line (so it matches the rest of
    // the document); inline appends (`, "key": value` into a single-line object) must stay compact.
    if (prefix.includes('\n')) {
      const childIndent = prefix.slice(prefix.lastIndexOf('\n') + 1);
      return prefix + prettyJsonEntry(child, childIndent) + trailing;
    }
    return prefix + child + trailing;
  }
  const yamlPrefix = insertText.startsWith('\n') ? '\n' : '';
  // Populate replaces the existing empty `triggers:` line at its start, so no leading newline.
  if (parentKey && isPopulate) return `${parentKey}:\n${child}`;
  // Absent: insert the parent at the document position chosen by buildInsertText, preserving any
  // leading newline it computed.
  if (parentKey && parentMissing) return `${yamlPrefix}${parentKey}:\n${child}`;
  return yamlPrefix + child;
}

// --- JSON insertion helpers ---

function formatValueForJson(value: string, schema?: FieldSchema, isList?: boolean): string {
  if (isList && schema?.type === FieldValueType.Object && schema.fields) {
    const { values, activeVariantKey } = defaultObjectEntryValues(schema);
    return `[ ${buildObjectEntryJsonText(schema, values, activeVariantKey)} ]`;
  }
  if (schema?.type === FieldValueType.Object && schema.fields) {
    const entries: string[] = [];
    for (const [k, s] of Object.entries(schema.fields)) {
      if (s.type === FieldValueType.Object) {
        entries.push(`"${k}": ${formatValueForJson('', s)}`);
      } else if (s.type === FieldValueType.StringArray) {
        entries.push(`"${k}": []`);
      } else {
        const v = defaultValueForType(s);
        entries.push(`"${k}": ${formatJsonPrimitive(v, s)}`);
      }
    }
    if (entries.length === 0) return '{}';
    return `{ ${entries.join(', ')} }`;
  }
  if (schema?.nestedList) {
    // Minimal valid scaffold; the stage editor widget fills in the real grouped value on mount.
    return value ? `[["${value}"]]` : '[]';
  }
  if (schema?.type === FieldValueType.StringArray || isList) {
    return value ? `["${value}"]` : '[]';
  }
  return formatJsonPrimitive(value, schema);
}

function formatJsonPrimitive(value: string, schema?: FieldSchema): string {
  if (!schema || !value) return `"${value}"`;
  switch (schema.type) {
    case FieldValueType.Number:
      return value;
    case FieldValueType.Boolean:
      return value;
    default:
      return `"${value}"`;
  }
}

// Find the last line of a JSON value starting at lineIdx (handles multi-line arrays/objects)
function jsonValueEndLine(lines: string[], lineIdx: number, scopeClose: number): number {
  let depth = 0;
  for (let i = lineIdx; i < scopeClose; i++) {
    for (const ch of lines[i]) {
      if (ch === '{' || ch === '[') depth++;
      if (ch === '}' || ch === ']') depth--;
    }
    if (depth <= 0 || (i > lineIdx && depth === 0)) return i;
  }
  return lineIdx;
}

type InsertResult = { text: string; pos: number; inline?: boolean; replaceEnd?: number };

function buildJsonInsertText(
  field: string,
  value: string,
  docText: string,
  lines: string[],
  isList?: boolean,
  schema?: FieldSchema,
): InsertResult {
  const parts = field.split('.');
  const formattedValue = formatValueForJson(value, schema, isList);

  // Find root object boundaries
  let rootOpen = -1;
  let rootClose = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === '{' || lines[i].trim().startsWith('{')) {
      rootOpen = i;
      break;
    }
  }
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].trim().startsWith('}')) {
      rootClose = i;
      break;
    }
  }

  if (rootOpen < 0 || rootClose < 0) {
    return { text: `"${field}": ${formattedValue}`, pos: docText.length };
  }

  // Detect indent from first key
  let keyIndent = INDENT;
  for (let i = rootOpen + 1; i < rootClose; i++) {
    const m = lines[i].match(/^(\s+)"/);
    if (m) {
      keyIndent = m[1];
      break;
    }
  }

  if (parts.length === 1) {
    return insertJsonKeyAlphabetical(parts[0], formattedValue, lines, docText, rootOpen, rootClose, keyIndent);
  }

  // Multi-part: walk down JSON hierarchy to find parent object
  let scopeOpen = rootOpen;
  let scopeClose = rootClose;
  let indent = keyIndent;
  let foundDepth = 0;
  let nonObjectKeyLine = -1;
  let nonObjectScopeClose = rootClose;

  for (let depth = 0; depth < parts.length - 1; depth++) {
    const key = parts[depth];
    let found = false;

    for (let i = scopeOpen + 1; i < scopeClose; i++) {
      const trimmed = lines[i].trimStart();
      const m = trimmed.match(/^"([^"]+)"\s*:/);
      if (m && m[1] === key) {
        // Find the { for this key's value
        const afterColon = trimmed.slice(m[0].length).trim();
        if (afterColon.startsWith('{')) {
          const newOpen = i;
          // Find matching }
          let d = 0;
          for (let j = i; j <= scopeClose; j++) {
            for (const ch of lines[j]) {
              if (ch === '{') d++;
              if (ch === '}') d--;
            }
            if (d <= 0) {
              scopeOpen = newOpen;
              scopeClose = j;
              indent = indent + INDENT;
              foundDepth = depth + 1;
              found = true;
              break;
            }
          }
        } else {
          nonObjectKeyLine = i;
          nonObjectScopeClose = scopeClose;
        }
        break;
      }
    }

    if (!found) break;
  }

  if (foundDepth === parts.length - 1) {
    return insertJsonKeyAlphabetical(parts[parts.length - 1], formattedValue, lines, docText, scopeOpen, scopeClose, indent);
  }

  // Parent key exists but has a non-object value (null, primitive, etc.)
  // Replace the value with a nested object containing the remaining path
  if (nonObjectKeyLine >= 0) {
    const lineStr = lines[nonObjectKeyLine];
    const prefixMatch = lineStr.match(/^(\s*"[^"]+"\s*:\s*)/);
    if (prefixMatch) {
      const valueStart = lineOffset(lines, nonObjectKeyLine) + prefixMatch[0].length;
      const valueEndLine = jsonValueEndLine(lines, nonObjectKeyLine, nonObjectScopeClose);
      const valueEndStr = lines[valueEndLine];
      const trimmedEnd = valueEndStr.trimEnd();
      let valueEnd = lineOffset(lines, valueEndLine) + trimmedEnd.length;
      if (trimmedEnd.endsWith(',')) {
        valueEnd = lineOffset(lines, valueEndLine) + valueEndStr.lastIndexOf(',');
      }

      const remaining = parts.slice(foundDepth + 1);
      let newValue = formattedValue;
      for (let d = remaining.length - 1; d >= 0; d--) {
        newValue = `{ "${remaining[d]}": ${newValue} }`;
      }
      const keyLineIndent = lineStr.match(/^(\s*)/)?.[1] ?? '';
      return { text: prettyJsonValue(newValue, keyLineIndent), pos: valueStart, replaceEnd: valueEnd };
    }
  }

  // Parent object doesn't exist — insert the full nested structure
  const leafKey = parts[parts.length - 1];
  let nestedValue = formattedValue;
  for (let d = parts.length - 2; d >= foundDepth; d--) {
    const k = parts[d + 1] === leafKey ? leafKey : parts[d + 1];
    nestedValue = `{ "${k}": ${nestedValue} }`;
  }
  const insertKey = foundDepth > 0 ? parts[foundDepth] : parts[0];
  const targetScopeOpen = foundDepth > 0 ? scopeOpen : rootOpen;
  const targetScopeClose = foundDepth > 0 ? scopeClose : rootClose;
  const targetIndent = foundDepth > 0 ? indent : keyIndent;
  return insertJsonKeyAlphabetical(insertKey, nestedValue, lines, docText, targetScopeOpen, targetScopeClose, targetIndent);
}

function insertJsonKeyAlphabetical(
  key: string,
  formattedValue: string,
  lines: string[],
  _docText: string,
  scopeOpen: number,
  scopeClose: number,
  indent: string,
): { text: string; pos: number } {
  // Collect existing keys with their line ranges
  const existingKeys: { name: string; lineIdx: number; lastLineIdx: number }[] = [];
  for (let i = scopeOpen + 1; i < scopeClose; i++) {
    const trimmed = lines[i].trimStart();
    const m = trimmed.match(/^"([^"]+)"\s*:/);
    if (m) {
      const endLine = jsonValueEndLine(lines, i, scopeClose);
      existingKeys.push({ name: m[1], lineIdx: i, lastLineIdx: endLine });
      i = endLine;
    }
  }

  // Pretty-print object/array values multi-line, reindented under the key. The inline non-empty
  // append branch below keeps the raw (single-line) value so it stays on the existing line.
  const newEntry = `${indent}"${key}": ${prettyJsonValue(formattedValue, indent)}`;

  if (existingKeys.length === 0) {
    if (scopeOpen === scopeClose) {
      // Inline object on one line (`"key": {}` or `"key": { "a": 1 }`). The existing-key scan
      // above only inspects lines *between* the braces, so an inline object always lands here with
      // existingKeys empty — inspect the line directly to tell empty from non-empty.
      const lineStr = lines[scopeOpen];
      const colonIdx = lineStr.indexOf('":');
      const braceIdx = colonIdx >= 0 ? lineStr.indexOf('{', colonIdx) : -1;
      const closeIdx = lineStr.lastIndexOf('}');
      if (braceIdx >= 0 && closeIdx > braceIdx) {
        const inner = lineStr.slice(braceIdx + 1, closeIdx).trim();
        if (inner) {
          // Non-empty inline object: append `, "key": value` just after the last existing value so
          // the result stays inline and valid (e.g. `{ "a": 1, "key": value }`).
          let insertAt = closeIdx;
          while (insertAt > braceIdx + 1 && /\s/.test(lineStr[insertAt - 1])) insertAt--;
          const pos = lineOffset(lines, scopeOpen) + insertAt;
          return { text: `, "${key}": ${formattedValue}`, pos };
        }
        // Empty inline object: open it up and insert the first entry between the braces.
        const pos = lineOffset(lines, scopeOpen) + braceIdx + 1;
        const parentIndent = lineStr.match(/^(\s*)/)?.[1] ?? '';
        return { text: `\n${newEntry}\n${parentIndent}`, pos };
      }
    }
    const pos = lineOffset(lines, scopeClose);
    return { text: `${newEntry}\n`, pos };
  }

  // Find alphabetical position
  let insertBeforeIdx = -1;
  for (let i = 0; i < existingKeys.length; i++) {
    if (existingKeys[i].name.localeCompare(key) > 0) {
      insertBeforeIdx = i;
      break;
    }
  }

  if (insertBeforeIdx === 0) {
    // Insert before all existing keys
    const pos = lineOffset(lines, existingKeys[0].lineIdx);
    return { text: `${newEntry},\n`, pos };
  }

  if (insertBeforeIdx > 0) {
    // Insert between two keys — previous already has trailing comma
    const pos = lineOffset(lines, existingKeys[insertBeforeIdx].lineIdx);
    return { text: `${newEntry},\n`, pos };
  }

  // Insert at end — need to add comma to previous last entry
  const lastEntry = existingKeys[existingKeys.length - 1];
  const lastLine = lines[lastEntry.lastLineIdx];
  const endOfLastLine = lineOffset(lines, lastEntry.lastLineIdx) + lastLine.length;
  const lastTrimmed = lastLine.trimEnd();
  const alreadyHasComma = lastTrimmed.endsWith(',');

  if (alreadyHasComma) {
    const pos = lineOffset(lines, lastEntry.lastLineIdx + 1);
    return { text: `${newEntry}\n`, pos };
  }

  // Append comma to previous line, then add new entry
  return { text: `,\n${newEntry}`, pos: endOfLastLine };
}

export function buildInsertText(
  field: string,
  value: string,
  docText: string,
  format: FormatType,
  cursorLine?: number,
  isList?: boolean,
  schema?: FieldSchema,
): InsertResult {
  const lines = docText.split('\n');

  switch (format) {
    case FormatType.JSON:
      return buildJsonInsertText(field, value, docText, lines, isList, schema);
    case FormatType.YARA:
      return buildYaraInsertText(field, value, docText, lines, cursorLine);
    default:
      return buildYamlInsertText(field, value, docText, lines, isList, schema);
  }
}

function buildYamlRemoveRange(field: string, docText: string): { from: number; to: number; content: string } | null {
  const lines = docText.split('\n');
  const parts = field.split('.');

  let scopeStart = 0;
  let scopeEnd = lines.length;
  let minIndent = -1;

  for (let p = 0; p < parts.length - 1; p++) {
    const parentKey = parts[p];
    let found = false;
    for (let i = scopeStart; i < scopeEnd; i++) {
      const trimmed = lines[i].trimStart();
      const lineIndent = lines[i].length - trimmed.length;
      if (lineIndent <= minIndent) continue;
      if (trimmed.startsWith(`${parentKey}:`) || trimmed.startsWith(`${parentKey} :`)) {
        minIndent = lineIndent;
        scopeStart = i + 1;
        scopeEnd = lines.length;
        for (let j = i + 1; j < lines.length; j++) {
          const t = lines[j].trim();
          if (!t) continue;
          const ind = lines[j].length - lines[j].trimStart().length;
          if (ind <= minIndent) {
            scopeEnd = j;
            break;
          }
        }
        found = true;
        break;
      }
    }
    if (!found) return null;
  }

  const leafKey = parts[parts.length - 1];
  let startLineIdx = -1;
  let keyIndent = -1;

  for (let i = scopeStart; i < scopeEnd; i++) {
    const trimmed = lines[i].trimStart();
    const lineIndent = lines[i].length - trimmed.length;
    if (lineIndent <= minIndent) continue;
    if (trimmed.startsWith(`${leafKey}:`) || trimmed.startsWith(`${leafKey} :`)) {
      startLineIdx = i;
      keyIndent = lineIndent;
      break;
    }
  }

  if (startLineIdx < 0) return null;

  let endLineIdx = startLineIdx;
  for (let i = startLineIdx + 1; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    if (!trimmed) {
      endLineIdx = i;
      continue;
    }
    const lineIndent = lines[i].length - lines[i].trimStart().length;
    if (lineIndent > keyIndent) {
      endLineIdx = i;
    } else {
      break;
    }
  }

  while (endLineIdx > startLineIdx && lines[endLineIdx].trim() === '') {
    endLineIdx--;
  }

  let from = 0;
  for (let i = 0; i < startLineIdx; i++) from += lines[i].length + 1;
  let to = from;
  for (let i = startLineIdx; i <= endLineIdx; i++) to += lines[i].length + 1;
  to = Math.min(to, docText.length);

  const content = lines.slice(startLineIdx, endLineIdx + 1).join('\n');
  return { from, to, content };
}

function buildJsonRemoveRange(field: string, docText: string): { from: number; to: number; content: string } | null {
  const lines = docText.split('\n');
  const parts = field.split('.');

  let scopeOpen = -1;
  let scopeClose = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim().startsWith('{')) {
      scopeOpen = i;
      break;
    }
  }
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].trim().startsWith('}')) {
      scopeClose = i;
      break;
    }
  }
  if (scopeOpen < 0 || scopeClose < 0) return null;

  for (let depth = 0; depth < parts.length - 1; depth++) {
    const key = parts[depth];
    let found = false;
    for (let i = scopeOpen + 1; i < scopeClose; i++) {
      const m = lines[i].trimStart().match(/^"([^"]+)"\s*:/);
      if (m && m[1] === key) {
        const afterColon = lines[i].trimStart().slice(m[0].length).trim();
        if (afterColon.startsWith('{')) {
          let d = 0;
          for (let j = i; j <= scopeClose; j++) {
            for (const ch of lines[j]) {
              if (ch === '{') d++;
              if (ch === '}') d--;
            }
            if (d <= 0) {
              scopeOpen = i;
              scopeClose = j;
              found = true;
              break;
            }
          }
        }
        break;
      }
    }
    if (!found) return null;
  }

  const leafKey = parts[parts.length - 1];
  let keyLineIdx = -1;
  for (let i = scopeOpen + 1; i < scopeClose; i++) {
    const m = lines[i].trimStart().match(/^"([^"]+)"\s*:/);
    if (m && m[1] === leafKey) {
      keyLineIdx = i;
      break;
    }
  }
  if (keyLineIdx < 0) return null;

  const endLine = jsonValueEndLine(lines, keyLineIdx, scopeClose);

  const from = lineOffset(lines, keyLineIdx);
  let to = lineOffset(lines, endLine) + lines[endLine].length;

  // Handle trailing comma: if this line ends with comma, include it in the range
  const endText = lines[endLine].trimEnd();
  if (!endText.endsWith(',')) {
    // No trailing comma — check if previous sibling has a comma that should be removed
    for (let i = keyLineIdx - 1; i > scopeOpen; i--) {
      const prev = lines[i].trimEnd();
      if (!prev) continue;
      if (prev.endsWith(',')) {
        // Remove the comma from the previous line by extending from to include the newline before us
        const commaPos = lineOffset(lines, i) + lines[i].lastIndexOf(',');
        to = lineOffset(lines, endLine) + lines[endLine].length;
        // Include trailing newline
        if (to < docText.length) to++;
        const content = lines.slice(keyLineIdx, endLine + 1).join('\n');
        return { from: commaPos, to, content };
      }
      break;
    }
  }

  // Include trailing newline
  if (to < docText.length && docText[to] === '\n') to++;

  const content = lines.slice(keyLineIdx, endLine + 1).join('\n');
  return { from, to, content };
}

type RemoveRange = { from: number; to: number; content: string };

export function buildRemoveRange(field: string, docText: string, format: FormatType = FormatType.YAML): RemoveRange | null {
  if (format === FormatType.JSON) return buildJsonRemoveRange(field, docText);
  return buildYamlRemoveRange(field, docText);
}

const setViewRef = StateEffect.define<EditorView>();

const EDITABLE_RE = /(=\s*"|:\s*["']|-\s*')([^"']*)("|')/;
const EDITABLE_UNQUOTED_RE = /(:\s+)(\S+)\s*$/;

function findEditableRange(text: string): { before: string; value: string; after: string; quote: string } | null {
  const trimmed = text.trim();
  const m = trimmed.match(EDITABLE_RE);
  if (m) {
    const matchStart = m.index!;
    const before = trimmed.slice(0, matchStart + m[1].length);
    const value = m[2];
    const after = trimmed.slice(matchStart + m[1].length + m[2].length);
    return { before, value, after, quote: m[3] };
  }
  const mu = trimmed.match(EDITABLE_UNQUOTED_RE);
  if (mu) {
    const matchStart = mu.index!;
    const before = trimmed.slice(0, matchStart + mu[1].length);
    const value = mu[2];
    return { before, value, after: '', quote: '' };
  }
  return null;
}

function makeBtn(label: string, title: string, bgVar: string, onClick: () => void): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.textContent = label;
  btn.title = title;
  btn.style.cssText = [
    `background-color: var(${bgVar})`,
    'color: var(--thorium-button-text)',
    'border: none',
    'border-radius: 4px',
    'padding: 2px 10px',
    'cursor: pointer',
    'font-size: 11px',
    'font-weight: 600',
    'transition: filter 0.15s',
  ].join(';');
  btn.addEventListener('mouseenter', () => {
    btn.style.filter = 'brightness(1.15)';
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.filter = '';
  });
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    onClick();
  });
  return btn;
}

function validateSchemaValue(value: string, schema: FieldSchema): boolean {
  if (schema.required && !value) return false;
  if (!value) return true;
  switch (schema.type) {
    case FieldValueType.Number:
      return /^-?\d+(\.\d+)?$/.test(value.trim());
    case FieldValueType.Boolean:
      return value === 'true' || value === 'false';
    case FieldValueType.Enum:
      return schema.enumValues ? schema.enumValues.includes(value) : true;
    default:
      return true;
  }
}

function validateObjectSchema(values: Record<string, string>, schema: FieldSchema): { valid: boolean; errors: string[] } {
  if (!schema.fields) return { valid: true, errors: [] };
  const errors: string[] = [];
  for (const [key, subSchema] of Object.entries(schema.fields)) {
    const val = values[key] ?? '';
    if (subSchema.required && !val) {
      errors.push(`${key} is required`);
    } else if (val && !validateSchemaValue(val, subSchema)) {
      errors.push(`${key}: invalid ${subSchema.type} value`);
    }
  }
  return { valid: errors.length === 0, errors };
}

const inputBaseStyle = [
  'background: var(--thorium-highlight-panel-bg)',
  'color: var(--thorium-text)',
  'border: 1px solid var(--thorium-panel-border)',
  'border-radius: 3px',
  'padding: 0 4px',
  'font-family: monospace',
  'font-size: 13px',
  'line-height: 18px',
  'height: 20px',
  'box-sizing: border-box',
  'outline: none',
].join(';');

function makeFormInput(schema: FieldSchema, initialValue: string): HTMLInputElement {
  const input = document.createElement('input');
  input.type = 'text';
  if (schema.type === FieldValueType.Number) input.inputMode = 'numeric';
  input.value = initialValue;
  input.placeholder = schema.placeholder ?? '';
  input.style.cssText = `${inputBaseStyle};min-width:80px;max-width:200px;`;
  return input;
}

function makeFormSelect(options: readonly string[], initialValue: string): HTMLSelectElement {
  const select = document.createElement('select');
  select.style.cssText = `${inputBaseStyle};cursor:pointer;min-width:80px;`;
  const emptyOpt = document.createElement('option');
  emptyOpt.value = '';
  emptyOpt.textContent = 'select';
  select.appendChild(emptyOpt);
  for (const opt of options) {
    const option = document.createElement('option');
    option.value = opt;
    option.textContent = opt;
    if (opt === initialValue) option.selected = true;
    select.appendChild(option);
  }
  if (initialValue) select.value = initialValue;
  return select;
}

function makeBoolSelect(initialValue: string): HTMLSelectElement {
  return makeFormSelect(['true', 'false'] as const, initialValue);
}

interface TagMapRowsResult {
  container: HTMLElement;
  addRow: () => void;
  getEntries: () => Array<{ key: string; values: string[] }>;
}

function makeTagMapRows(
  placeholder: string,
  onChange: () => void,
  onKeydown: (e: KeyboardEvent) => void,
  startEmpty: boolean = false,
): TagMapRowsResult {
  const grid = document.createElement('div');
  grid.style.cssText = 'display:grid;grid-template-columns:1fr 1fr auto;gap:4px 6px;align-items:baseline;';

  const headerStyle = 'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--thorium-secondary-text);';
  const keyHeader = document.createElement('span');
  keyHeader.style.cssText = headerStyle;
  keyHeader.textContent = 'Key';
  const valHeader = document.createElement('span');
  valHeader.style.cssText = headerStyle;
  valHeader.textContent = 'Value';
  grid.appendChild(keyHeader);
  grid.appendChild(valHeader);
  grid.appendChild(document.createElement('span'));

  const rows: { keyEl: HTMLInputElement; valEl: HTMLInputElement; rowEls: HTMLElement[] }[] = [];

  const addRow = (): void => {
    const keyInput = document.createElement('input');
    keyInput.type = 'text';
    keyInput.placeholder = placeholder;
    keyInput.style.cssText = `${inputBaseStyle};min-width:60px;`;

    const valInput = document.createElement('input');
    valInput.type = 'text';
    valInput.placeholder = 'value';
    valInput.style.cssText = `${inputBaseStyle};min-width:60px;`;

    const removeBtn = document.createElement('button');
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove row';
    removeBtn.style.cssText =
      'background:none;border:none;color:var(--thorium-secondary-text);cursor:pointer;font-size:16px;padding:0 2px;line-height:20px;';

    const entry = { keyEl: keyInput, valEl: valInput, rowEls: [keyInput, valInput, removeBtn] as HTMLElement[] };
    rows.push(entry);

    removeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (rows.length === 1) {
        keyInput.value = '';
        valInput.value = '';
        onChange();
        return;
      }
      const idx = rows.indexOf(entry);
      if (idx >= 0) {
        entry.rowEls.forEach((el) => el.remove());
        rows.splice(idx, 1);
        onChange();
      }
    });

    const autoAddIfLast = () => {
      const idx = rows.indexOf(entry);
      if (idx === rows.length - 1 && (keyInput.value.trim() || valInput.value.trim())) {
        addRow();
      }
      onChange();
    };
    keyInput.addEventListener('input', autoAddIfLast);
    valInput.addEventListener('input', autoAddIfLast);
    keyInput.addEventListener('keydown', onKeydown as EventListener);
    valInput.addEventListener('keydown', onKeydown as EventListener);

    for (const el of entry.rowEls) grid.appendChild(el);
  };

  if (!startEmpty) addRow();

  const mergeEntries = (): Array<{ key: string; values: string[] }> => {
    const map = new Map<string, string[]>();
    for (const r of rows) {
      const k = r.keyEl.value.trim();
      const v = r.valEl.value.trim();
      if (!k) continue;
      const existing = map.get(k) ?? [];
      if (v) existing.push(v);
      map.set(k, existing);
    }
    return Array.from(map.entries()).map(([key, values]) => ({ key, values }));
  };

  return {
    container: grid,
    addRow,
    getEntries: mergeEntries,
  };
}

interface ListRowsResult {
  container: HTMLElement;
  getValues: () => string[];
  focusFirst: () => void;
}

// A single-column dynamic list editor (bulleted rows), used for StringArray fields and for
// variant list payloads (e.g. AutoTagLogic In/NotIn). Auto-adds a row as the last one is filled.
function makeListRows(placeholder: string, onChange: () => void, onKeydown: (e: KeyboardEvent) => void): ListRowsResult {
  const container = document.createElement('div');
  container.style.cssText = 'display:flex;flex-direction:column;gap:4px;';
  const inputs: HTMLInputElement[] = [];

  const addRow = (): HTMLInputElement => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:baseline;gap:4px;';
    const bullet = document.createElement('span');
    bullet.style.cssText = 'color:var(--thorium-text);font-style:italic;white-space:pre;line-height:20px;';
    bullet.textContent = '-';
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = placeholder;
    input.style.cssText = `${inputBaseStyle};min-width:80px;max-width:240px;flex:1;`;

    const removeBtn = document.createElement('button');
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove row';
    removeBtn.style.cssText =
      'background:none;border:none;color:var(--thorium-secondary-text);cursor:pointer;font-size:16px;padding:0 2px;line-height:20px;';
    removeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (inputs.length === 1) {
        input.value = '';
        onChange();
        return;
      }
      const idx = inputs.indexOf(input);
      if (idx >= 0) {
        inputs.splice(idx, 1);
        row.remove();
        onChange();
      }
    });

    input.addEventListener('input', () => {
      const idx = inputs.indexOf(input);
      if (idx === inputs.length - 1 && input.value.trim()) addRow();
      onChange();
    });
    input.addEventListener('keydown', onKeydown as EventListener);

    inputs.push(input);
    row.appendChild(bullet);
    row.appendChild(input);
    row.appendChild(removeBtn);
    container.appendChild(row);
    return input;
  };

  addRow();

  return {
    container,
    getValues: () => inputs.map((i) => i.value.trim()).filter((v) => v !== ''),
    focusFirst: () => inputs[0]?.focus(),
  };
}

// Like makeListRows, but each row is a <select> of `options` (e.g. the images in a pipeline group).
// Auto-adds a new empty select once the last one is chosen (tag-style). Same ListRowsResult shape.
function makeSelectRows(options: readonly string[], onChange: () => void, onKeydown: (e: KeyboardEvent) => void): ListRowsResult {
  const container = document.createElement('div');
  container.style.cssText = 'display:flex;flex-direction:column;gap:4px;';
  const selects: HTMLSelectElement[] = [];

  const addRow = (): HTMLSelectElement => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:baseline;gap:4px;';
    const bullet = document.createElement('span');
    bullet.style.cssText = 'color:var(--thorium-text);font-style:italic;white-space:pre;line-height:20px;';
    bullet.textContent = '-';

    const select = makeFormSelect(options, '');
    select.style.cssText += ';min-width:120px;max-width:240px;flex:1;';

    const removeBtn = document.createElement('button');
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove image';
    removeBtn.style.cssText =
      'background:none;border:none;color:var(--thorium-secondary-text);cursor:pointer;font-size:16px;padding:0 2px;line-height:20px;';
    removeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (selects.length === 1) {
        select.value = '';
        onChange();
        return;
      }
      const idx = selects.indexOf(select);
      if (idx >= 0) {
        selects.splice(idx, 1);
        row.remove();
        onChange();
      }
    });

    select.addEventListener('change', () => {
      const idx = selects.indexOf(select);
      if (idx === selects.length - 1 && select.value.trim()) addRow();
      onChange();
    });
    select.addEventListener('keydown', onKeydown as EventListener);

    selects.push(select);
    row.appendChild(bullet);
    row.appendChild(select);
    row.appendChild(removeBtn);
    container.appendChild(row);
    return select;
  };

  addRow();

  return {
    container,
    getValues: () => selects.map((s) => s.value.trim()).filter((v) => v !== ''),
    focusFirst: () => selects[0]?.focus(),
  };
}

class PreviewWidget extends WidgetType {
  constructor(
    readonly insertText: string,
    readonly viewRef: { current: EditorView | null },
    readonly inline: boolean = false,
    readonly schema?: FieldSchema,
    readonly proposal?: PreviewProposal | null,
    readonly oldContent?: string,
  ) {
    super();
  }

  eq(other: PreviewWidget): boolean {
    return this.proposal === other.proposal && this.inline === other.inline;
  }

  private get isList(): boolean {
    return this.schema?.type === FieldValueType.StringArray || this.proposal?.isList === true;
  }

  private setupWrapper(wrapper: HTMLElement): void {
    wrapper.style.contain = 'inline-size';
    requestAnimationFrame(() => {
      const editor = wrapper.closest('.cm-editor');
      if (editor) editor.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
  }

  toDOM(): HTMLElement {
    if (this.proposal?.isRemoval) return this.toRemovalDOM();
    if (this.proposal?.isMapEntry) return this.toMapEntryDOM();
    if (this.inline) return this.toInlineDOM();
    if (this.schema?.variants) return this.toVariantDOM();
    if (this.schema?.nestedList) return this.toStageListDOM();
    if (this.proposal?.isList && this.schema?.type === FieldValueType.Object && this.schema.fields) return this.toObjectListDOM();
    if (this.schema?.type === FieldValueType.Object && this.schema.fields) return this.toObjectDOM();
    if (this.isList) return this.toListDOM();

    const firstLine = this.insertText.split('\n').find((l) => l.trim().length > 0) ?? this.insertText;
    const leadingSpaces = firstLine.match(/^(\s*)/)?.[1].length ?? 0;
    const marginLeft = `${leadingSpaces}ch`;

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-wrap: wrap',
      'align-items: flex-end',
      'gap: 8px',
      'padding: 4px 10px',
      `margin: 0 4px 0 ${marginLeft}`,
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const editable = findEditableRange(this.insertText);
    const contentContainer = document.createElement('span');
    contentContainer.style.cssText = ['flex: 1 1 auto', 'min-width: 0'].join(';');

    const acceptBtn = makeBtn('Accept', 'Insert this field (Enter)', '--thorium-ok-bg', () => {
      this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    if (editable) {
      const labelStyle = ['color: var(--thorium-text)', 'font-style: italic', 'white-space: pre', 'line-height: 20px'].join(';');

      const trimmedText = this.insertText.trim();
      const editMatch = trimmedText.match(EDITABLE_RE);
      const editLineIdx = editMatch ? trimmedText.slice(0, editMatch.index).split('\n').length - 1 : 0;
      const allLines = trimmedText.split('\n');

      const headerLines = allLines.slice(0, editLineIdx);
      if (headerLines.length > 0) {
        const headerSpan = document.createElement('div');
        headerSpan.style.cssText = ['color: var(--thorium-text)', 'font-style: italic', 'white-space: pre', 'line-height: 20px'].join(';');
        headerSpan.textContent = stripDisplayCommas(headerLines.join('\n'));
        contentContainer.appendChild(headerSpan);
      }

      const editableRow = document.createElement('div');
      editableRow.style.cssText = ['display: flex', 'align-items: baseline', 'gap: 0'].join(';');

      const isNonStringType =
        this.schema?.type === FieldValueType.Number ||
        this.schema?.type === FieldValueType.Boolean ||
        this.schema?.type === FieldValueType.Enum;
      const format = this.proposal?.format ?? FormatType.YAML;

      const prefixSpan = document.createElement('span');
      prefixSpan.style.cssText = labelStyle;
      const editableLine = allLines[editLineIdx] ?? '';
      const lineMatch = editableLine.match(EDITABLE_RE);
      let prefixText = lineMatch ? editableLine.slice(0, lineMatch.index! + lineMatch[1].length) : editable.before;
      if (isNonStringType && format !== FormatType.JSON) prefixText = prefixText.replace(/['"]$/, '');
      prefixSpan.textContent = prefixText;

      let inputEl: HTMLInputElement | HTMLSelectElement;

      if (this.schema?.type === FieldValueType.Enum && this.schema.enumValues) {
        inputEl = makeFormSelect(this.schema.enumValues, editable.value === '<value>' ? '' : editable.value);
        inputEl.style.cssText += ';vertical-align:baseline;';
      } else if (this.schema?.type === FieldValueType.Boolean) {
        inputEl = makeBoolSelect(editable.value === '<value>' ? '' : editable.value);
        inputEl.style.cssText += ';vertical-align:baseline;';
      } else {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = editable.value === '<value>' ? '' : editable.value;
        input.placeholder = this.schema?.placeholder ?? 'value';
        input.style.cssText = [
          'background: var(--thorium-highlight-panel-bg)',
          'color: var(--thorium-text)',
          'border: 1px solid var(--thorium-panel-border)',
          'border-radius: 3px',
          'padding: 0 4px',
          'font-family: monospace',
          'font-size: 13px',
          'line-height: 18px',
          'height: 20px',
          'box-sizing: border-box',
          'min-width: 80px',
          'max-width: 300px',
          'outline: none',
          'vertical-align: baseline',
        ].join(';');
        if (this.schema?.type === FieldValueType.Number) input.inputMode = 'numeric';
        inputEl = input;
      }

      let unitSelect: HTMLSelectElement | null = null;
      if (this.schema?.unit && this.schema.unit.options.length > 1) {
        unitSelect = document.createElement('select');
        unitSelect.style.cssText = `${inputBaseStyle};cursor:pointer;min-width:50px;max-width:80px;margin-left:4px;vertical-align:baseline;`;
        for (const opt of this.schema.unit.options) {
          const option = document.createElement('option');
          option.value = opt.suffix;
          option.textContent = opt.label;
          if (opt.label === this.schema.unit.defaultUnit) option.selected = true;
          unitSelect.appendChild(option);
        }
      }

      const suffixSpan = document.createElement('span');
      suffixSpan.style.cssText = labelStyle;
      let suffixText = editable.after;
      if (isNonStringType && format !== FormatType.JSON) suffixText = suffixText.replace(/^['"]/, '');
      suffixText = stripDisplayCommas(suffixText);
      suffixSpan.textContent = suffixText;

      const rebuildText = () => {
        if (this.schema?.transform) {
          const fieldName = this.insertText.trim().split(':')[0].trim();
          const result = this.schema.transform(inputEl.value);
          if (format === FormatType.JSON) {
            const { prefix, trailing } = jsonEnvelope(this.insertText);
            return `${prefix}${fieldName}: ${result.json}${trailing}`;
          }
          const indent = this.insertText.match(/^(\s*)/)?.[1] ?? '';
          return `${indent}${fieldName}: ${result.yaml}\n`;
        }
        const indent = this.insertText.match(/^(\s*)/)?.[1] ?? '';
        const rest = this.insertText.trimStart();
        const m = rest.match(EDITABLE_RE);
        if (m) {
          let val = inputEl.value;
          const unitSuffix = unitSelect ? unitSelect.value : '';
          if (unitSuffix) val = val + unitSuffix;
          const colonPrefix = rest.slice(0, (m.index ?? 0) + m[1].length).replace(/['"]$/, '');
          const afterSuffix = rest.slice((m.index ?? 0) + m[0].length).replace(/^['"]/, '');
          const formatted =
            format === FormatType.JSON
              ? formatJsonPrimitive(val, this.schema)
              : this.schema
                ? formatValueForYaml(val, this.schema)
                : `'${val}'`;
          return indent + colonPrefix + formatted + afterSuffix;
        }
        return this.insertText;
      };

      const updateValidation = () => {
        if (!this.schema) return;
        if (this.schema.transform) {
          const result = this.schema.transform(inputEl.value);
          inputEl.style.borderColor = result.valid ? 'var(--thorium-panel-border)' : 'var(--thorium-danger-bg)';
          acceptBtn.style.opacity = result.valid ? '1' : '0.4';
          acceptBtn.style.pointerEvents = result.valid ? 'auto' : 'none';
          return;
        }
        const isValid = validateSchemaValue(inputEl.value, this.schema);
        inputEl.style.borderColor = isValid ? 'var(--thorium-panel-border)' : 'var(--thorium-danger-bg)';
        acceptBtn.style.opacity = isValid ? '1' : '0.4';
        acceptBtn.style.pointerEvents = isValid ? 'auto' : 'none';
      };

      const onInputChange = () => {
        updateValidation();
        setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(rebuildText()) }), 0);
      };

      inputEl.addEventListener('input', onInputChange);
      inputEl.addEventListener('change', onInputChange);
      if (unitSelect) {
        unitSelect.addEventListener('change', onInputChange);
      }

      inputEl.addEventListener('keydown', ((e: KeyboardEvent) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          e.stopPropagation();
          if (this.schema?.transform) {
            if (!this.schema.transform(inputEl.value).valid) return;
          } else if (this.schema && !validateSchemaValue(inputEl.value, this.schema)) return;
          this.viewRef.current?.dispatch({ effects: updateInsertText.of(rebuildText()) });
          setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          this.viewRef.current?.dispatch({ effects: clearPreview.of() });
        }
      }) as EventListener);

      editableRow.appendChild(prefixSpan);
      editableRow.appendChild(inputEl);
      if (unitSelect) editableRow.appendChild(unitSelect);
      editableRow.appendChild(suffixSpan);
      contentContainer.appendChild(editableRow);

      const footerLines = allLines.slice(editLineIdx + 1).filter((l) => l.trim().length > 0);
      if (footerLines.length > 0) {
        const footerSpan = document.createElement('div');
        footerSpan.style.cssText = [
          'color: var(--thorium-text)',
          'font-style: italic',
          'white-space: pre',
          'line-height: 20px',
          'opacity: 0.7',
        ].join(';');
        footerSpan.textContent = footerLines.join('\n');
        contentContainer.appendChild(footerSpan);
      }

      setTimeout(() => {
        inputEl.focus();
        updateValidation();
      }, 0);
    } else {
      const textSpan = document.createElement('span');
      textSpan.style.cssText = ['color: var(--thorium-text)', 'font-style: italic', 'white-space: pre-wrap', 'word-break: break-word'].join(
        ';',
      );
      textSpan.textContent = stripDisplayCommas(this.insertText.trim());
      contentContainer.appendChild(textSpan);
    }

    const btnContainer = document.createElement('span');
    btnContainer.style.cssText = ['display: flex', 'gap: 6px', 'flex-shrink: 0'].join(';');

    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(contentContainer);
    wrapper.appendChild(btnContainer);
    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toObjectDOM(): HTMLElement {
    const schema = this.schema!;
    const fields = schema.fields!;
    const fieldName = this.insertText.trim().split(':')[0].trim();

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 2px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const headerSpan = document.createElement('div');
    headerSpan.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;line-height:20px;';
    headerSpan.textContent = `${fieldName}:`;
    wrapper.appendChild(headerSpan);

    const formContainer = document.createElement('div');
    formContainer.style.cssText = `display:grid;grid-template-columns:auto 1fr;gap:4px 6px;align-items:baseline;padding-left:16px;`;

    const inputs = new Map<string, HTMLInputElement | HTMLSelectElement>();
    const unitSelects = new Map<string, HTMLSelectElement>();

    const errorArea = document.createElement('div');
    errorArea.style.cssText = 'font-size:11px;color:var(--thorium-danger-bg);min-height:14px;padding-left:16px;';

    const acceptBtn = makeBtn('Accept', 'Insert this field (Enter)', '--thorium-ok-bg', () => {
      const values = collectValues();
      const { valid } = validateObjectSchema(values, schema);
      if (valid) {
        this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
      }
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    const collectValues = (): Record<string, string> => {
      const values: Record<string, string> = {};
      for (const [key, el] of inputs) {
        let val = el.value;
        const unitSel = unitSelects.get(key);
        if (unitSel && val) val = val + unitSel.value;
        values[key] = val;
      }
      return values;
    };

    const rebuildAndValidate = () => {
      const values = collectValues();
      const format = this.proposal?.format ?? FormatType.YAML;
      let text: string;
      if (format === FormatType.JSON) {
        const cleanName = fieldName.replace(/^"|"$/g, '');
        const entry = buildObjectJsonText(cleanName, schema, values);
        const { prefix, trailing } = jsonEnvelope(this.insertText);
        text = prefix + entry + (trailing || '\n');
      } else {
        const indent = this.insertText.match(/^(\s*)/)?.[1] ?? '';
        text = indent + buildObjectYamlText(fieldName, schema, values);
      }
      setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(text) }), 0);

      const { valid, errors } = validateObjectSchema(values, schema);
      errorArea.textContent = errors.length > 0 ? errors[0] : '';
      acceptBtn.style.opacity = valid ? '1' : '0.4';
      acceptBtn.style.pointerEvents = valid ? 'auto' : 'none';

      for (const [key, el] of inputs) {
        const subSchema = fields[key];
        if (!subSchema) continue;
        const isFieldValid = validateSchemaValue(el.value, subSchema);
        el.style.borderColor = isFieldValid ? 'var(--thorium-panel-border)' : 'var(--thorium-danger-bg)';
      }
    };

    let firstInput: HTMLInputElement | HTMLSelectElement | null = null;

    for (const [subKey, subSchema] of Object.entries(fields)) {
      if (subSchema.type === FieldValueType.Object && subSchema.fields) continue;

      const label = document.createElement('span');
      label.style.cssText = 'color:var(--thorium-text);font-style:italic;white-space:nowrap;text-align:right;';
      label.textContent = `${subKey}:`;
      if (subSchema.required) {
        const req = document.createElement('span');
        req.style.cssText = 'color:var(--thorium-danger-bg);margin-left:2px;';
        req.textContent = '*';
        label.appendChild(req);
      }

      let inputEl: HTMLInputElement | HTMLSelectElement;
      const initialValue = defaultValueForType(subSchema);

      if (subSchema.type === FieldValueType.Enum && subSchema.enumValues) {
        inputEl = makeFormSelect(subSchema.enumValues, initialValue);
      } else if (subSchema.type === FieldValueType.Boolean) {
        inputEl = makeBoolSelect(initialValue);
      } else {
        inputEl = makeFormInput(subSchema, initialValue === subSchema.placeholder ? '' : initialValue);
      }

      inputEl.addEventListener('input', rebuildAndValidate);
      inputEl.addEventListener('change', rebuildAndValidate);
      inputEl.addEventListener('keydown', ((e: KeyboardEvent) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          e.stopPropagation();
          const values = collectValues();
          const { valid } = validateObjectSchema(values, schema);
          if (valid) {
            setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
          }
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          this.viewRef.current?.dispatch({ effects: clearPreview.of() });
        }
      }) as EventListener);

      inputs.set(subKey, inputEl);
      if (!firstInput) firstInput = inputEl;

      formContainer.appendChild(label);

      if (subSchema.unit && subSchema.unit.options.length > 1) {
        const inputRow = document.createElement('div');
        inputRow.style.cssText = 'display:flex;align-items:baseline;gap:4px;';
        inputRow.appendChild(inputEl);
        const unitSel = document.createElement('select');
        unitSel.style.cssText = `${inputBaseStyle};cursor:pointer;min-width:50px;max-width:80px;`;
        for (const opt of subSchema.unit.options) {
          const option = document.createElement('option');
          option.value = opt.suffix;
          option.textContent = opt.label;
          if (opt.label === subSchema.unit.defaultUnit) option.selected = true;
          unitSel.appendChild(option);
        }
        unitSel.addEventListener('change', rebuildAndValidate);
        unitSelects.set(subKey, unitSel);
        inputRow.appendChild(unitSel);
        formContainer.appendChild(inputRow);
      } else {
        formContainer.appendChild(inputEl);
      }
    }

    wrapper.appendChild(formContainer);
    wrapper.appendChild(errorArea);

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    setTimeout(() => {
      if (firstInput) firstInput.focus();
      rebuildAndValidate();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  // Renders a structured form for one object entry in a list (e.g. a Volume under `volumes:`).
  // Scalar fields become inputs; a `variantField` discriminator (archetype) reveals the
  // matching nested config object, re-rendered when the discriminator changes.
  private toObjectListDOM(): HTMLElement {
    const schema = this.schema!;
    const fields = schema.fields!;
    const proposal = this.proposal!;
    const format = proposal.format;
    const fieldName = proposal.field.split('.').pop()!;
    const variantField = schema.variantField;

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 2px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const headerSpan = document.createElement('div');
    headerSpan.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;line-height:20px;';
    headerSpan.textContent = `${fieldName}:`;
    wrapper.appendChild(headerSpan);

    const formContainer = document.createElement('div');
    formContainer.style.cssText = 'display:grid;grid-template-columns:auto 1fr;gap:4px 6px;align-items:baseline;padding-left:16px;';

    const variantArea = document.createElement('div');
    variantArea.style.cssText = 'display:grid;grid-template-columns:auto 1fr;gap:4px 6px;align-items:baseline;padding-left:32px;';

    const errorArea = document.createElement('div');
    errorArea.style.cssText = 'font-size:11px;color:var(--thorium-danger-bg);min-height:14px;padding-left:16px;';

    const inputs = new Map<string, HTMLInputElement | HTMLSelectElement>();
    const variantInputs = new Map<string, HTMLInputElement | HTMLSelectElement>();

    const acceptBtn = makeBtn('Accept', 'Insert this entry (Enter)', '--thorium-ok-bg', () => {
      if (validateAll().valid) this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    const activeVariantKey = (): string | undefined => {
      if (!variantField) return undefined;
      const disc = inputs.get(variantField.field)?.value ?? '';
      return variantField.fieldMap[disc];
    };

    const collectValues = (): Record<string, string> => {
      const values: Record<string, string> = {};
      for (const [k, el] of inputs) values[k] = el.value;
      const vk = activeVariantKey();
      if (vk) for (const [k, el] of variantInputs) values[`${vk}.${k}`] = el.value;
      return values;
    };

    const validateAll = (): { valid: boolean; errors: string[] } => {
      const values = collectValues();
      const { errors } = validateObjectSchema(values, schema);
      const all = [...errors];
      const vk = activeVariantKey();
      const variantSchema = vk ? fields[vk] : undefined;
      if (variantSchema?.fields) {
        for (const [k2, s2] of Object.entries(variantSchema.fields)) {
          const v = values[`${vk}.${k2}`] ?? '';
          if (s2.required && !v) all.push(`${k2} is required`);
          else if (v && !validateSchemaValue(v, s2)) all.push(`${k2}: invalid ${s2.type} value`);
        }
      }
      return { valid: all.length === 0, errors: all };
    };

    const onKeydown = ((e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        if (validateAll().valid) {
          rebuildText();
          setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
        }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        this.viewRef.current?.dispatch({ effects: clearPreview.of() });
      }
    }) as EventListener;

    const rebuildText = () => {
      const values = collectValues();
      const vk = activeVariantKey();
      let text: string;
      if (format === FormatType.JSON) {
        const { prefix, trailing } = jsonEnvelope(this.insertText);
        text = `${prefix}"${fieldName}": [ ${buildObjectEntryJsonText(schema, values, vk)} ]${trailing || '\n'}`;
      } else {
        const yamlPrefix = this.insertText.startsWith('\n') ? '\n' : '';
        text = yamlPrefix + buildObjectListItemYamlText(fieldName, schema, values, '', vk);
      }
      setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(text) }), 0);

      const { valid, errors } = validateAll();
      errorArea.textContent = errors.length > 0 ? errors[0] : '';
      acceptBtn.style.opacity = valid ? '1' : '0.4';
      acceptBtn.style.pointerEvents = valid ? 'auto' : 'none';
    };

    // Builds one labeled input row inside a grid container and returns the input element
    const makeFieldRow = (container: HTMLElement, sub: FieldSchema, label: string): HTMLInputElement | HTMLSelectElement => {
      const labelEl = document.createElement('span');
      labelEl.style.cssText = 'color:var(--thorium-text);font-style:italic;white-space:nowrap;text-align:right;';
      labelEl.textContent = `${label}:`;
      if (sub.required) {
        const req = document.createElement('span');
        req.style.cssText = 'color:var(--thorium-danger-bg);margin-left:2px;';
        req.textContent = '*';
        labelEl.appendChild(req);
      }

      let inputEl: HTMLInputElement | HTMLSelectElement;
      const initial = defaultValueForType(sub);
      if (sub.type === FieldValueType.Enum && sub.enumValues) inputEl = makeFormSelect(sub.enumValues, initial);
      else if (sub.type === FieldValueType.Boolean) inputEl = makeBoolSelect(initial);
      else inputEl = makeFormInput(sub, initial === sub.placeholder ? '' : initial);

      inputEl.addEventListener('input', rebuildText);
      inputEl.addEventListener('change', rebuildText);
      inputEl.addEventListener('keydown', onKeydown);
      container.appendChild(labelEl);
      container.appendChild(inputEl);
      return inputEl;
    };

    // Re-renders the nested config inputs for the currently selected discriminator value
    const renderVariantArea = () => {
      variantArea.innerHTML = '';
      variantInputs.clear();
      const vk = activeVariantKey();
      const variantSchema = vk ? fields[vk] : undefined;
      if (!vk || !variantSchema?.fields) return;
      const head = document.createElement('div');
      head.style.cssText = 'grid-column:1 / -1;color:var(--thorium-text);font-style:italic;font-weight:600;line-height:18px;';
      head.textContent = `${vk}:`;
      variantArea.appendChild(head);
      for (const [k2, s2] of Object.entries(variantSchema.fields)) {
        variantInputs.set(k2, makeFieldRow(variantArea, s2, k2));
      }
    };

    let firstInput: HTMLInputElement | HTMLSelectElement | null = null;
    for (const [key, sub] of Object.entries(fields)) {
      if (sub.type === FieldValueType.Object && sub.fields) continue; // nested configs handled by variantArea
      const el = makeFieldRow(formContainer, sub, key);
      inputs.set(key, el);
      if (!firstInput) firstInput = el;
      if (variantField && key === variantField.field) {
        el.addEventListener('change', () => {
          renderVariantArea();
          rebuildText();
        });
      }
    }

    wrapper.appendChild(formContainer);
    wrapper.appendChild(variantArea);
    wrapper.appendChild(errorArea);

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    renderVariantArea();
    setTimeout(() => {
      if (firstInput) firstInput.focus();
      rebuildText();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toVariantDOM(): HTMLElement {
    const schema = this.schema!;
    const variants = schema.variants!;
    const variantNames = Object.keys(variants);
    const proposal = this.proposal!;
    const format = proposal.format;
    const fieldName = this.insertText.trim().split(':')[0].trim().replace(/^"|"$/g, '');

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 4px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const headerRow = document.createElement('div');
    headerRow.style.cssText = 'display:flex;align-items:baseline;gap:6px;';

    const fieldLabel = document.createElement('span');
    fieldLabel.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;line-height:20px;';
    fieldLabel.textContent = `${fieldName}:`;

    const variantSelect = document.createElement('select');
    variantSelect.style.cssText = `${inputBaseStyle};cursor:pointer;min-width:100px;`;
    for (const name of variantNames) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      variantSelect.appendChild(opt);
    }

    headerRow.appendChild(fieldLabel);
    headerRow.appendChild(variantSelect);
    wrapper.appendChild(headerRow);

    const inputArea = document.createElement('div');
    inputArea.style.cssText = 'padding-left:16px;min-height:0;';
    wrapper.appendChild(inputArea);

    const acceptBtn = makeBtn('Accept', 'Insert (Enter)', '--thorium-ok-bg', () => {
      rebuildText();
      setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    let activeInput: HTMLInputElement | null = null;
    let listRows: ListRowsResult | null = null;
    let mapRows: TagMapRowsResult | null = null;

    const renderInputArea = () => {
      inputArea.innerHTML = '';
      activeInput = null;
      listRows = null;
      mapRows = null;
      const selected = variantSelect.value;
      const variantSchema = variants[selected];
      if (!variantSchema) return;

      if (variantSchema.type === FieldValueType.Object && variantSchema.fields) {
        const grid = document.createElement('div');
        grid.style.cssText = 'display:grid;grid-template-columns:auto 1fr;gap:4px 6px;align-items:baseline;';
        for (const [subKey, subSchema] of Object.entries(variantSchema.fields)) {
          const label = document.createElement('span');
          label.style.cssText = 'color:var(--thorium-text);font-style:italic;white-space:nowrap;';
          label.textContent = `${subKey}:`;
          grid.appendChild(label);

          const input = makeFormInput(subSchema, '');
          input.addEventListener('input', () => rebuildText());
          input.addEventListener('keydown', onKeydown as EventListener);
          grid.appendChild(input);
          if (!activeInput) activeInput = input;
        }
        inputArea.appendChild(grid);
      } else if (variantSchema.type === FieldValueType.StringArray) {
        // list payload (e.g. AutoTagLogic In/NotIn)
        listRows = makeListRows(variantSchema.placeholder ?? 'value', rebuildText, onKeydown);
        inputArea.appendChild(listRows.container);
      } else if (variantSchema.type === FieldValueType.Object) {
        // object without fields => free key/value map (e.g. KwargDependency Map)
        mapRows = makeTagMapRows(variantSchema.placeholder ?? 'key', rebuildText, onKeydown, false);
        inputArea.appendChild(mapRows.container);
      } else {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:baseline;gap:6px;';
        const varLabel = document.createElement('span');
        varLabel.style.cssText = 'color:var(--thorium-text);font-style:italic;';
        varLabel.textContent = `${selected}:`;
        row.appendChild(varLabel);

        const input = makeFormInput(variantSchema, '');
        input.addEventListener('input', () => rebuildText());
        input.addEventListener('keydown', onKeydown as EventListener);
        row.appendChild(input);
        inputArea.appendChild(row);
        activeInput = input;
      }
    };

    const rebuildText = () => {
      const selected = variantSelect.value;
      const variantSchema = variants[selected];
      const indent = this.insertText.match(/^(\s*)/)?.[1]?.replace(/\n/g, '') ?? '';
      const yamlPrefix = this.insertText.startsWith('\n') ? '\n' : '';
      const { prefix: envPrefix, trailing: envTrailing } = jsonEnvelope(this.insertText);
      let text: string;

      if (variantSchema && variantSchema.type === FieldValueType.StringArray) {
        const vals = listRows?.getValues() ?? [];
        text =
          format === FormatType.JSON
            ? envPrefix + buildVariantListJson(fieldName, selected, vals) + envTrailing
            : yamlPrefix + buildVariantListYaml(fieldName, selected, vals, indent);
      } else if (variantSchema && variantSchema.type === FieldValueType.Object && !variantSchema.fields) {
        const entries = (mapRows?.getEntries() ?? []).map((e) => ({ key: e.key, value: e.values[0] ?? '' }));
        text =
          format === FormatType.JSON
            ? envPrefix + buildVariantMapJson(fieldName, selected, entries) + envTrailing
            : yamlPrefix + buildVariantMapYaml(fieldName, selected, entries, indent);
      } else if (variantSchema && variantSchema.type === FieldValueType.Object && variantSchema.fields) {
        const inputs = inputArea.querySelectorAll('input');
        if (format === FormatType.JSON) {
          const entries = Object.entries(variantSchema.fields).map(([k, s], i) => {
            const v = inputs[i]?.value ?? '';
            return `"${k}": ${formatJsonPrimitive(v || defaultValueForType(s), s)}`;
          });
          text = `${envPrefix}"${fieldName}": { "${selected}": { ${entries.join(', ')} } }${envTrailing}`;
        } else {
          const baseText = buildVariantYamlText(fieldName, selected, variantSchema, '', indent);
          const newLines = baseText.split('\n').map((line) => {
            for (const [k] of Object.entries(variantSchema.fields!)) {
              if (line.trim().startsWith(`${k}:`)) {
                const idx = Object.keys(variantSchema.fields!).indexOf(k);
                const inp = inputs[idx];
                if (inp) {
                  const lineIndent = line.match(/^(\s*)/)?.[1] ?? '';
                  const s = variantSchema.fields![k];
                  return `${lineIndent}${k}: ${formatValueForYaml(inp.value || defaultValueForType(s), s)}`;
                }
              }
            }
            return line;
          });
          text = yamlPrefix + newLines.join('\n');
        }
      } else {
        const value = activeInput ? activeInput.value : '';
        text =
          format === FormatType.JSON
            ? envPrefix + buildVariantJsonText(fieldName, selected, variantSchema, value) + envTrailing
            : yamlPrefix + buildVariantYamlText(fieldName, selected, variantSchema, value, indent);
      }
      setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(text) }), 0);
    };

    const onKeydown = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        rebuildText();
        setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        this.viewRef.current?.dispatch({ effects: clearPreview.of() });
      }
    };

    variantSelect.addEventListener('change', () => {
      renderInputArea();
      rebuildText();
      if (activeInput) setTimeout(() => activeInput!.focus(), 0);
    });
    variantSelect.addEventListener('keydown', onKeydown as EventListener);

    renderInputArea();

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    setTimeout(() => {
      if (activeInput) activeInput.focus();
      else variantSelect.focus();
      rebuildText();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toListDOM(): HTMLElement {
    const proposal = this.proposal!;
    const format = proposal.format;
    const schema = this.schema;
    const templateText = this.insertText;
    const leafKey = proposal.field.split('.').pop()!;

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 2px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const headerSpan = document.createElement('div');
    headerSpan.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;line-height:20px;';
    headerSpan.textContent = `${leafKey}:`;
    wrapper.appendChild(headerSpan);

    const listContainer = document.createElement('div');
    listContainer.style.cssText = 'display:flex;flex-direction:column;gap:4px;padding-left:16px;';

    const inputs: HTMLInputElement[] = [];

    const acceptBtn = makeBtn('Accept', 'Insert this field (Enter)', '--thorium-ok-bg', () => {
      rebuildText();
      setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    const rebuildText = () => {
      const values = inputs.map((inp) => inp.value).filter((v) => v.trim() !== '');
      let newText: string;

      if (format === FormatType.JSON) {
        const bracketStart = templateText.indexOf('[');
        const bracketEnd = templateText.lastIndexOf(']');
        if (bracketStart >= 0 && bracketEnd > bracketStart) {
          const jsonVals = values.map((v) => `"${v}"`).join(', ');
          newText = templateText.slice(0, bracketStart + 1) + jsonVals + templateText.slice(bracketEnd);
        } else {
          newText = templateText;
        }
      } else {
        const colonIdx = templateText.indexOf(':');
        const prefix = templateText.slice(0, colonIdx + 1);
        const baseIndent = templateText.match(/^(\s*)/)?.[1] ?? '';
        const itemIndent = baseIndent + INDENT;

        if (values.length === 0) {
          newText = `${prefix} []\n`;
        } else {
          newText = `${prefix}\n`;
          for (const v of values) {
            newText += `${itemIndent}- '${v}'\n`;
          }
        }
      }

      setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(newText) }), 0);
    };

    const addInputRow = (value: string = ''): HTMLInputElement => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:baseline;gap:4px;';

      const bullet = document.createElement('span');
      bullet.style.cssText = 'color:var(--thorium-text);font-style:italic;white-space:pre;line-height:20px;';
      bullet.textContent = '-';

      const input = document.createElement('input');
      input.type = 'text';
      input.value = value;
      input.placeholder = schema?.placeholder ?? 'value';
      input.style.cssText = `${inputBaseStyle};min-width:80px;max-width:300px;flex:1;`;

      input.addEventListener('input', () => {
        const idx = inputs.indexOf(input);
        if (idx === inputs.length - 1 && input.value.trim() !== '') {
          const newInput = addInputRow();
          setTimeout(() => newInput.scrollIntoView({ block: 'nearest' }), 0);
        }
        rebuildText();
      });

      input.addEventListener('keydown', ((e: KeyboardEvent) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          e.stopPropagation();
          const idx = inputs.indexOf(input);
          if (input.value.trim() !== '' && idx === inputs.length - 1) {
            const newInput = addInputRow();
            setTimeout(() => newInput.focus(), 0);
          } else if (idx < inputs.length - 1) {
            inputs[idx + 1].focus();
          } else {
            rebuildText();
            setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
          }
        }
        if (e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          this.viewRef.current?.dispatch({ effects: clearPreview.of() });
        }
        if (e.key === 'Backspace' && input.value === '' && inputs.length > 1) {
          e.preventDefault();
          const idx = inputs.indexOf(input);
          inputs.splice(idx, 1);
          row.remove();
          if (idx > 0) inputs[idx - 1].focus();
          else if (inputs.length > 0) inputs[0].focus();
          rebuildText();
        }
      }) as EventListener);

      row.appendChild(bullet);
      row.appendChild(input);
      listContainer.appendChild(row);
      inputs.push(input);
      return input;
    };

    const firstInput = addInputRow();
    wrapper.appendChild(listContainer);

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    setTimeout(() => {
      firstInput.focus();
      rebuildText();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  // Editor for a `nestedList` field (pipeline `order`): a list of stages, each holding one or more
  // images that run in parallel. Serializes to Vec<Vec<String>> (always grouped). Reuses
  // makeListRows for each stage's parallel-image group.
  private toStageListDOM(): HTMLElement {
    const proposal = this.proposal!;
    const format = proposal.format;
    const schema = this.schema;
    const templateText = this.insertText;
    const leafKey = proposal.field.split('.').pop()!;

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 2px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const headerSpan = document.createElement('div');
    headerSpan.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;line-height:20px;';
    headerSpan.textContent = `${leafKey}:`;
    wrapper.appendChild(headerSpan);

    const hint = document.createElement('div');
    hint.style.cssText = 'color:var(--thorium-secondary-text);font-size:11px;margin:0 0 4px 16px;';
    hint.textContent = 'Stages run in order; images within a stage run in parallel.';
    wrapper.appendChild(hint);

    const stagesContainer = document.createElement('div');
    stagesContainer.style.cssText = 'display:flex;flex-direction:column;gap:8px;padding-left:16px;';

    const stages: ListRowsResult[] = [];

    const collectStages = (): string[][] => stages.map((s) => s.getValues());

    const rebuildText = () => {
      const collected = collectStages();
      let newText: string;
      if (format === FormatType.JSON) {
        const open = templateText.indexOf('[');
        const close = templateText.lastIndexOf(']');
        const beforeKey = templateText.slice(0, Math.max(templateText.indexOf('"'), 0));
        const keyIndent = beforeKey.slice(beforeKey.lastIndexOf('\n') + 1);
        const value = buildStageListJson(collected, keyIndent);
        newText = open >= 0 && close > open ? templateText.slice(0, open) + value + templateText.slice(close + 1) : templateText;
      } else {
        const leadingNl = templateText.startsWith('\n') ? '\n' : '';
        const baseIndent = templateText.slice(leadingNl.length).match(/^(\s*)/)?.[1] ?? '';
        newText = leadingNl + buildStageListYaml(leafKey, collected, baseIndent);
      }
      setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(newText) }), 0);
    };

    const acceptBtn = makeBtn('Accept', 'Insert this field (Enter)', '--thorium-ok-bg', () => {
      rebuildText();
      setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    const onKeydown = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        rebuildText();
        setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        this.viewRef.current?.dispatch({ effects: clearPreview.of() });
      }
    };

    // When the group's images are known, each stage's images are picked from a select dropdown;
    // otherwise fall back to free-text rows (e.g. group not set or images not yet loaded).
    const imageOptions = schema?.enumValues;

    const renumber = () => {
      Array.from(stagesContainer.children).forEach((block, i) => {
        const label = block.querySelector('[data-stage-label]');
        if (label) label.textContent = `Stage ${i + 1}`;
      });
    };

    // Hide the remove-stage link while only one stage remains (nothing to remove).
    const syncRemoveLinks = () => {
      stagesContainer.querySelectorAll<HTMLElement>('[data-remove-stage]').forEach((el) => {
        el.style.display = stages.length > 1 ? '' : 'none';
      });
    };

    const addStage = (): ListRowsResult => {
      const block = document.createElement('div');
      block.style.cssText =
        'display:flex;flex-direction:column;gap:2px;border-left:2px solid var(--thorium-panel-border);padding-left:8px;';

      // Stage title with the remove link directly after it (left-aligned).
      const head = document.createElement('div');
      head.style.cssText = 'display:flex;align-items:baseline;gap:8px;';
      const label = document.createElement('span');
      label.dataset.stageLabel = 'true';
      label.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;font-size:11px;';
      label.textContent = `Stage ${stages.length + 1}`;
      const removeStageBtn = document.createElement('button');
      removeStageBtn.dataset.removeStage = 'true';
      removeStageBtn.textContent = 'remove stage';
      removeStageBtn.style.cssText =
        'background:none;border:none;color:var(--thorium-secondary-text);cursor:pointer;font-size:11px;text-decoration:underline;';
      head.appendChild(label);
      head.appendChild(removeStageBtn);

      const rows =
        imageOptions && imageOptions.length
          ? makeSelectRows(imageOptions, rebuildText, onKeydown)
          : makeListRows(schema?.placeholder ?? 'image-name', rebuildText, onKeydown);

      removeStageBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (stages.length === 1) return; // keep at least one stage
        const idx = stages.indexOf(rows);
        if (idx >= 0) {
          stages.splice(idx, 1);
          block.remove();
          renumber();
          syncRemoveLinks();
          rebuildText();
        }
      });

      block.appendChild(head);
      block.appendChild(rows.container);
      stagesContainer.appendChild(block);
      stages.push(rows);
      syncRemoveLinks();
      return rows;
    };

    const firstStage = addStage();
    wrapper.appendChild(stagesContainer);

    // '+ Stage' sits right after the last stage's images (not in the Accept/Dismiss row).
    const addStageBtn = makeBtn('+ Stage', 'Add a sequential stage', '--thorium-info-bg', () => {
      const s = addStage();
      rebuildText();
      setTimeout(() => s.focusFirst(), 0);
    });
    const addStageRow = document.createElement('div');
    addStageRow.style.cssText = 'display:flex;padding-left:16px;margin-top:4px;';
    addStageRow.appendChild(addStageBtn);
    wrapper.appendChild(addStageRow);

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;margin-top:8px;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    setTimeout(() => {
      firstStage.focusFirst();
      rebuildText();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toMapEntryDOM(): HTMLElement {
    const proposal = this.proposal!;
    const schema = this.schema;
    const parts = proposal.field.split('.');
    const placeholderKey = parts[parts.length - 1];

    const isSimpleValue =
      !schema?.fields &&
      (schema?.type === FieldValueType.String ||
        schema?.type === FieldValueType.Number ||
        schema?.type === FieldValueType.Boolean ||
        !schema?.type);

    if (isSimpleValue) return this.toKeyValueMapDOM();
    if (schema?.variants) return this.toVariantMapEntryDOM();

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 2px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    // Name input row
    const nameRow = document.createElement('div');
    nameRow.style.cssText = 'display:flex;align-items:baseline;gap:6px;';

    const nameLabel = document.createElement('span');
    nameLabel.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;white-space:nowrap;';
    nameLabel.textContent = 'name:';

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.value = '';
    nameInput.placeholder = placeholderKey;
    nameInput.style.cssText = `${inputBaseStyle};min-width:120px;max-width:250px;flex:1;`;

    nameRow.appendChild(nameLabel);
    nameRow.appendChild(nameInput);
    wrapper.appendChild(nameRow);

    const isSimpleEnum = schema?.type === FieldValueType.Enum;
    const isObjectEntry = schema?.type === FieldValueType.Object && schema.fields;

    const subInputs = new Map<string, HTMLInputElement | HTMLSelectElement>();

    const errorArea = document.createElement('div');
    errorArea.style.cssText = 'font-size:11px;color:var(--thorium-danger-bg);min-height:0;';

    const acceptBtn = makeBtn('Accept', 'Insert this entry (Enter)', '--thorium-ok-bg', () => {
      if (!nameInput.value.trim()) {
        errorArea.textContent = 'Name is required';
        return;
      }
      if (isObjectEntry) {
        const values: Record<string, string> = {};
        for (const [key, el] of subInputs) values[key] = el.value;
        const { valid } = validateObjectSchema(values, schema);
        if (!valid) return;
      }
      this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    const rebuildText = () => {
      const entryName = nameInput.value.trim() || placeholderKey;
      const newField = [...parts.slice(0, -1), entryName].join('.');
      const docText = this.viewRef.current?.state.doc.toString() ?? '';

      if (isObjectEntry && subInputs.size > 0) {
        const baseResult = buildInsertText(newField, '', docText, proposal.format, undefined, false, schema);
        const values: Record<string, string> = {};
        for (const [key, el] of subInputs) values[key] = el.value;
        let finalText: string;
        if (proposal.format === FormatType.JSON) {
          const entry = buildObjectJsonText(entryName, schema, values);
          const colonPos = baseResult.text.indexOf('": ');
          if (colonPos >= 0) {
            const prefix = baseResult.text.slice(0, baseResult.text.indexOf('"'));
            const trailing = baseResult.text.match(/(,?\s*)$/)?.[1] ?? '\n';
            finalText = prefix + entry + trailing;
          } else {
            finalText = entry + '\n';
          }
        } else {
          const indent = baseResult.text.match(/^(\s*)/)?.[1] ?? '';
          const objectText = buildObjectYamlText(entryName, schema, values, indent);
          finalText = baseResult.text.endsWith('\n') ? objectText : objectText.replace(/\n$/, '');
        }
        setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(finalText) }), 0);

        const { valid, errors } = validateObjectSchema(values, schema);
        const nameValid = entryName.length > 0;
        const allValid = nameValid && valid;
        errorArea.textContent = !nameValid ? 'Name is required' : errors.length > 0 ? errors[0] : '';
        acceptBtn.style.opacity = allValid ? '1' : '0.4';
        acceptBtn.style.pointerEvents = allValid ? 'auto' : 'none';
      } else {
        const result = buildInsertText(newField, proposal.value, docText, proposal.format, undefined, proposal.isList, schema);
        setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(result.text) }), 0);

        const isValid = entryName.length > 0;
        errorArea.textContent = isValid ? '' : 'Name is required';
        acceptBtn.style.opacity = isValid ? '1' : '0.4';
        acceptBtn.style.pointerEvents = isValid ? 'auto' : 'none';
      }
    };

    const onKeydown = ((e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        if (nameInput.value.trim()) {
          rebuildText();
          setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
        }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        this.viewRef.current?.dispatch({ effects: clearPreview.of() });
      }
    }) as EventListener;

    nameInput.addEventListener('input', rebuildText);
    nameInput.addEventListener('keydown', onKeydown);

    if (isSimpleEnum && proposal.value) {
      const valueRow = document.createElement('div');
      valueRow.style.cssText = 'display:flex;align-items:baseline;gap:6px;padding-left:16px;';
      const valueLabel = document.createElement('span');
      valueLabel.style.cssText = 'color:var(--thorium-text);font-style:italic;';
      valueLabel.textContent = `value: ${proposal.value}`;
      valueRow.appendChild(valueLabel);
      wrapper.appendChild(valueRow);
    } else if (isObjectEntry) {
      const formContainer = document.createElement('div');
      formContainer.style.cssText = 'display:grid;grid-template-columns:auto 1fr;gap:4px 6px;align-items:baseline;padding-left:16px;';

      for (const [subKey, subSchema] of Object.entries(schema?.fields ?? {})) {
        if (subSchema.type === FieldValueType.Object && subSchema.fields) continue;

        const label = document.createElement('span');
        label.style.cssText = 'color:var(--thorium-text);font-style:italic;white-space:nowrap;text-align:right;';
        label.textContent = `${subKey}:`;
        if (subSchema.required) {
          const req = document.createElement('span');
          req.style.cssText = 'color:var(--thorium-danger-bg);margin-left:2px;';
          req.textContent = '*';
          label.appendChild(req);
        }

        let inputEl: HTMLInputElement | HTMLSelectElement;
        const initialValue = defaultValueForType(subSchema);

        if (subSchema.type === FieldValueType.Enum && subSchema.enumValues) {
          inputEl = makeFormSelect(subSchema.enumValues, initialValue);
        } else if (subSchema.type === FieldValueType.Boolean) {
          inputEl = makeBoolSelect(initialValue);
        } else {
          inputEl = makeFormInput(subSchema, initialValue === subSchema.placeholder ? '' : initialValue);
        }

        inputEl.addEventListener('input', rebuildText);
        inputEl.addEventListener('change', rebuildText);
        inputEl.addEventListener('keydown', onKeydown);

        subInputs.set(subKey, inputEl);
        formContainer.appendChild(label);
        formContainer.appendChild(inputEl);
      }

      wrapper.appendChild(formContainer);
    }

    wrapper.appendChild(errorArea);

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    setTimeout(() => {
      nameInput.focus();
      rebuildText();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toVariantMapEntryDOM(): HTMLElement {
    const proposal = this.proposal!;
    const schema = this.schema!;
    const variants = schema.variants!;
    const variantNames = Object.keys(variants);
    const parts = proposal.field.split('.');
    const placeholderKey = parts[parts.length - 1];
    const parentKey = parts.length > 1 ? parts[0] : '';
    const format = proposal.format;

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 0',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const labelStyle = 'color:var(--thorium-text);font-style:italic;white-space:pre;line-height:22px;';

    // Root key label (e.g., "triggers:") — shown for YAML populate only
    if (parentKey && !(format === FormatType.JSON && proposal.isReplace)) {
      const rootLabel = document.createElement('div');
      rootLabel.style.cssText = labelStyle + 'font-weight:600;';
      rootLabel.textContent = `${parentKey}:`;
      wrapper.appendChild(rootLabel);
    }

    // Name input row — indent 2sp
    const nameRow = document.createElement('div');
    nameRow.style.cssText = 'display:flex;align-items:baseline;gap:4px;padding-left:2ch;';
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.value = '';
    nameInput.placeholder = placeholderKey;
    nameInput.style.cssText = `${inputBaseStyle};min-width:100px;max-width:200px;`;
    const nameColon = document.createElement('span');
    nameColon.style.cssText = labelStyle;
    nameColon.textContent = ':';
    nameRow.appendChild(nameInput);
    nameRow.appendChild(nameColon);
    wrapper.appendChild(nameRow);

    // Variant type row — indent 4sp
    const typeRow = document.createElement('div');
    typeRow.style.cssText = 'display:flex;align-items:baseline;gap:4px;padding-left:4ch;margin-top:2px;';
    const variantSelect = document.createElement('select');
    variantSelect.style.cssText = `${inputBaseStyle};cursor:pointer;min-width:100px;`;
    for (const name of variantNames) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      variantSelect.appendChild(opt);
    }
    typeRow.appendChild(variantSelect);
    wrapper.appendChild(typeRow);

    // Dynamic input area — indent 6sp for sub-fields
    const inputArea = document.createElement('div');
    inputArea.style.cssText = 'padding-left:6ch;min-height:0;';
    wrapper.appendChild(inputArea);

    const errorArea = document.createElement('div');
    errorArea.style.cssText = 'font-size:11px;color:var(--thorium-danger-bg);min-height:0;padding-left:6ch;';
    wrapper.appendChild(errorArea);

    const validate = (): boolean => {
      const selected = variantSelect.value;
      const variantSchema = variants[selected];
      // Only validate key/value pairs when Tag variant is selected
      if (variantSchema && variantSchema.type === FieldValueType.Object && variantSchema.fields) {
        for (const tmg of tagMapGroups.values()) {
          const rowInputs = tmg.container.querySelectorAll<HTMLInputElement>('input[type="text"]');
          for (let i = 0; i < rowInputs.length; i += 2) {
            const keyEl = rowInputs[i];
            const valEl = rowInputs[i + 1];
            if (valEl && valEl.value.trim() && !keyEl?.value.trim()) {
              errorArea.textContent = 'Key is required when a value is specified';
              return false;
            }
          }
        }
      }
      errorArea.textContent = '';
      return true;
    };

    const acceptBtn = makeBtn('Accept', 'Insert this entry (Enter)', '--thorium-ok-bg', () => {
      if (!nameInput.value.trim()) {
        errorArea.textContent = 'Name is required';
        return;
      }
      if (!validate()) return;
      rebuildText();
      setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    let selectInputs = new Map<string, HTMLSelectElement>();
    let tagMapGroups = new Map<string, TagMapRowsResult>();

    const renderInputArea = () => {
      inputArea.innerHTML = '';
      selectInputs = new Map();
      tagMapGroups = new Map();
      const selected = variantSelect.value;
      const variantSchema = variants[selected];
      if (!variantSchema) return;

      if (variantSchema.type === FieldValueType.Object && variantSchema.fields) {
        const tagHints: Record<string, string> = { required: 'must have', not: 'must not have' };
        for (const [subKey, subSchema] of Object.entries(variantSchema.fields)) {
          const label = document.createElement('div');
          label.style.cssText =
            'color:var(--thorium-text);font-style:italic;white-space:pre;line-height:22px;font-weight:600;margin-top:2px;';
          const hint = tagHints[subKey];
          label.textContent = hint ? `${subKey} (${hint}):` : `${subKey}:`;
          inputArea.appendChild(label);

          if (subSchema.type === FieldValueType.StringArray && subSchema.enumValues) {
            const sel = makeFormSelect(subSchema.enumValues, '');
            sel.addEventListener('change', rebuildText);
            sel.addEventListener('keydown', onKeydown as EventListener);
            selectInputs.set(subKey, sel);
            inputArea.appendChild(sel);
          } else if (subSchema.type === FieldValueType.Object && !subSchema.fields) {
            const tmRows = makeTagMapRows(subSchema.placeholder ?? 'key', rebuildText, onKeydown, false);
            tmRows.container.style.cssText += ';padding-left:2ch;';
            tagMapGroups.set(subKey, tmRows);
            inputArea.appendChild(tmRows.container);
          } else {
            const input = makeFormInput(subSchema, '');
            input.addEventListener('input', rebuildText);
            input.addEventListener('keydown', onKeydown as EventListener);
            input.dataset.fieldKey = subKey;
            inputArea.appendChild(input);
          }
        }
      } else {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:baseline;gap:6px;';
        const valLabel = document.createElement('span');
        valLabel.style.cssText = labelStyle;
        valLabel.textContent = `${selected}:`;
        row.appendChild(valLabel);
        const input = makeFormInput(variantSchema, '');
        input.addEventListener('input', rebuildText);
        input.addEventListener('keydown', onKeydown as EventListener);
        row.appendChild(input);
        inputArea.appendChild(row);
      }
    };

    const isPopulate = proposal.isReplace === true;

    const rebuildText = () => {
      const entryName = nameInput.value.trim() || placeholderKey;
      const selected = variantSelect.value;
      const variantSchema = variants[selected];

      // For populate (replacing triggers: {}), build text from scratch with parent key
      // For add (appending to existing triggers:), use buildInsertText to find position
      const buildYamlContent = (indent: string): string => {
        if (!variantSchema) return `${indent}${INDENT}${entryName}: ${selected}\n`;
        if (!(variantSchema.type === FieldValueType.Object && variantSchema.fields)) {
          const input = inputArea.querySelector('input');
          const value = input?.value ?? '';
          return `${indent}${INDENT}${entryName}:\n${indent}${INDENT}${INDENT}${selected}: ${formatValueForYaml(value || defaultValueForType(variantSchema), variantSchema)}\n`;
        }
        let content = `${indent}${INDENT}${entryName}:\n${indent}${INDENT}${INDENT}${selected}:\n`;
        for (const [k] of Object.entries(variantSchema.fields)) {
          const sel = selectInputs.get(k);
          const tmg = tagMapGroups.get(k);
          const i3 = `${indent}${INDENT}${INDENT}${INDENT}`;
          const i4 = `${indent}${INDENT}${INDENT}${INDENT}${INDENT}`;
          if (sel) {
            const val = sel.value;
            if (!val) content += `${i3}${k}: []\n`;
            else content += `${i3}${k}:\n${i4}- ${val}\n`;
          } else if (tmg) {
            const entries = tmg.getEntries();
            if (entries.length === 0) content += `${i3}${k}: {}\n`;
            else {
              content += `${i3}${k}:\n`;
              for (const e of entries) {
                if (e.values.length === 0) content += `${i4}${e.key}: []\n`;
                else {
                  content += `${i4}${e.key}:\n`;
                  for (const v of e.values) content += `${i4}${INDENT}- '${v}'\n`;
                }
              }
            }
          } else {
            const s = variantSchema.fields[k];
            const inp = inputArea.querySelector<HTMLInputElement>(`input[data-field-key="${k}"]`);
            content += `${i3}${k}: ${formatValueForYaml(inp?.value || s.placeholder || '', s)}\n`;
          }
        }
        return content;
      };

      const buildJsonContent = (): string => {
        if (!variantSchema) return `"${entryName}": "${selected}"`;
        if (!(variantSchema.type === FieldValueType.Object && variantSchema.fields)) {
          const input = inputArea.querySelector('input');
          const value = input?.value ?? '';
          return `"${entryName}": { "${selected}": ${formatJsonPrimitive(value || defaultValueForType(variantSchema), variantSchema)} }`;
        }
        const subEntries: string[] = [];
        for (const [k, s] of Object.entries(variantSchema.fields)) {
          const sel = selectInputs.get(k);
          const tmg = tagMapGroups.get(k);
          if (sel) {
            const val = sel.value;
            subEntries.push(`"${k}": ${val ? `["${val}"]` : '[]'}`);
          } else if (tmg) {
            const entries = tmg.getEntries();
            const mapEntries = entries.map((e) => `"${e.key}": [${e.values.map((v) => `"${v}"`).join(', ')}]`);
            subEntries.push(`"${k}": { ${mapEntries.join(', ')} }`);
          } else {
            const inp = inputArea.querySelector<HTMLInputElement>(`input[data-field-key="${k}"]`);
            subEntries.push(`"${k}": ${formatJsonPrimitive(inp?.value || s.placeholder || '', s)}`);
          }
        }
        return `"${entryName}": { "${selected}": { ${subEntries.join(', ')} } }`;
      };

      // The parent map key (`triggers`) may be absent (add to a doc with no `triggers:`), empty
      // (populate), or already populated (append). Detect absence from the live doc so the parent
      // is emitted when missing rather than orphaning the entry at the wrong nesting level.
      const docText = this.viewRef.current?.state.doc.toString() ?? '';
      const parentMissing =
        !!parentKey &&
        (format === FormatType.JSON
          ? !new RegExp(`"${parentKey}"\\s*:`).test(docText)
          : !docText.split('\n').some((l) => {
              const t = l.trimStart();
              return t.startsWith(`${parentKey}:`) || t.startsWith(`${parentKey} :`);
            }));

      const child = format === FormatType.JSON ? buildJsonContent() : buildYamlContent('');
      const text = buildMapEntryText(child, { format, parentKey, parentMissing, isPopulate, insertText: this.insertText });

      setTimeout(() => this.viewRef.current?.dispatch({ effects: updateInsertText.of(text) }), 0);

      validate();
      const isValid = nameInput.value.trim().length > 0;
      acceptBtn.style.opacity = isValid ? '1' : '0.4';
      acceptBtn.style.pointerEvents = isValid ? 'auto' : 'none';
    };

    const onKeydown = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        if (validate()) {
          rebuildText();
          setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
        }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        this.viewRef.current?.dispatch({ effects: clearPreview.of() });
      }
    };

    nameInput.addEventListener('input', rebuildText);
    nameInput.addEventListener('keydown', onKeydown as EventListener);
    variantSelect.addEventListener('change', () => {
      renderInputArea();
      rebuildText();
    });
    variantSelect.addEventListener('keydown', onKeydown as EventListener);

    renderInputArea();

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;margin-top:4px;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    setTimeout(() => {
      nameInput.focus();
      rebuildText();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toKeyValueMapDOM(): HTMLElement {
    const proposal = this.proposal!;
    const schema = this.schema;
    const parts = proposal.field.split('.');
    const parentField = parts.slice(0, -1).join('.');
    const placeholderKey = parts[parts.length - 1];
    const format = proposal.format;

    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 2px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const headerSpan = document.createElement('div');
    headerSpan.style.cssText = 'color:var(--thorium-text);font-style:italic;font-weight:600;line-height:20px;';
    headerSpan.textContent = `${parentField || placeholderKey}:`;
    wrapper.appendChild(headerSpan);

    const gridContainer = document.createElement('div');
    gridContainer.style.cssText = 'display:grid;grid-template-columns:1fr 1fr auto;gap:4px 6px;align-items:baseline;padding-left:16px;';

    const rows: { keyEl: HTMLInputElement; valEl: HTMLInputElement; row: HTMLElement[] }[] = [];

    const errorArea = document.createElement('div');
    errorArea.style.cssText = 'font-size:11px;color:var(--thorium-danger-bg);min-height:0;padding-left:16px;';

    const acceptBtn = makeBtn('Accept', 'Insert entries (Enter)', '--thorium-ok-bg', () => {
      if (!validate()) return;
      rebuildText();
      setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    const validate = (): boolean => {
      for (const r of rows) {
        if (r.valEl.value.trim() && !r.keyEl.value.trim()) {
          errorArea.textContent = 'Key is required when a value is specified';
          r.keyEl.style.borderColor = 'var(--thorium-danger-bg)';
          acceptBtn.style.opacity = '0.4';
          acceptBtn.style.pointerEvents = 'none';
          return false;
        }
        r.keyEl.style.borderColor = 'var(--thorium-panel-border)';
      }
      const hasAny = rows.some((r) => r.keyEl.value.trim());
      if (!hasAny) {
        errorArea.textContent = 'At least one key is required';
        acceptBtn.style.opacity = '0.4';
        acceptBtn.style.pointerEvents = 'none';
        return false;
      }
      errorArea.textContent = '';
      acceptBtn.style.opacity = '1';
      acceptBtn.style.pointerEvents = 'auto';
      return true;
    };

    const rebuildText = () => {
      const entries = rows.filter((r) => r.keyEl.value.trim()).map((r) => ({ key: r.keyEl.value.trim(), val: r.valEl.value }));

      if (entries.length === 0) {
        this.viewRef.current?.dispatch({ effects: updateInsertText.of('') });
        return;
      }

      const docText = this.viewRef.current?.state.doc.toString() ?? '';
      const lines = docText.split('\n');

      if (entries.length === 1) {
        const field = parentField ? `${parentField}.${entries[0].key}` : entries[0].key;
        const result = buildInsertText(field, entries[0].val, docText, format, undefined, false, schema);
        this.viewRef.current?.dispatch({ effects: updateInsertText.of(result.text) });
        return;
      }

      let parentIndent = '';
      let parentFound = false;
      if (parentField) {
        for (const l of lines) {
          const trimmed = l.trimStart();
          if (trimmed.startsWith(`${parentField}:`) || trimmed.startsWith(`${parentField} :`)) {
            parentIndent = l.slice(0, l.length - trimmed.length);
            parentFound = true;
            break;
          }
        }
      }
      const childIndent = parentIndent + INDENT;
      let text = '';
      if (parentField && !parentFound) {
        text += `${parentField}:\n`;
      }
      for (const e of entries) {
        text += `${childIndent}${e.key}: ${formatValueForYaml(e.val, schema)}\n`;
      }

      if (this.insertText.startsWith('\n') && !text.startsWith('\n')) {
        text = '\n' + text;
      }

      this.viewRef.current?.dispatch({ effects: updateInsertText.of(text) });
    };

    const addRow = (): { keyEl: HTMLInputElement; valEl: HTMLInputElement } => {
      const keyInput = document.createElement('input');
      keyInput.type = 'text';
      keyInput.placeholder = placeholderKey;
      keyInput.style.cssText = `${inputBaseStyle};min-width:80px;`;

      const valInput = document.createElement('input');
      valInput.type = 'text';
      valInput.placeholder = schema?.placeholder ?? 'value';
      valInput.style.cssText = `${inputBaseStyle};min-width:80px;`;

      const removeBtn = document.createElement('button');
      removeBtn.textContent = '×';
      removeBtn.title = 'Remove row';
      removeBtn.style.cssText =
        'background:none;border:none;color:var(--thorium-secondary-text);cursor:pointer;font-size:16px;padding:0 2px;line-height:20px;';
      removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const idx = rows.findIndex((r) => r.keyEl === keyInput);
        if (idx >= 0 && rows.length > 1) {
          rows[idx].row.forEach((el) => el.remove());
          rows.splice(idx, 1);
          validate();
          rebuildText();
        }
      });

      const rowEls = [keyInput, valInput, removeBtn];
      const entry = { keyEl: keyInput, valEl: valInput, row: rowEls };
      rows.push(entry);
      for (const el of rowEls) gridContainer.appendChild(el);

      const onInput = () => {
        const idx = rows.indexOf(entry);
        if (idx === rows.length - 1 && keyInput.value.trim()) {
          addRow();
        }
        validate();
        rebuildText();
      };

      keyInput.addEventListener('input', onInput);
      valInput.addEventListener('input', onInput);

      const onKeydown = ((e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          this.viewRef.current?.dispatch({ effects: clearPreview.of() });
        }
        if (e.key === 'Enter') {
          e.preventDefault();
          e.stopPropagation();
          if (validate()) {
            rebuildText();
            setTimeout(() => this.viewRef.current?.dispatch({ effects: acceptPreview.of() }), 0);
          }
        }
      }) as EventListener;

      keyInput.addEventListener('keydown', onKeydown);
      valInput.addEventListener('keydown', onKeydown);

      return { keyEl: keyInput, valEl: valInput };
    };

    // Column headers
    const keyHeader = document.createElement('span');
    keyHeader.style.cssText =
      'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--thorium-secondary-text);';
    keyHeader.textContent = 'Key';
    const valHeader = document.createElement('span');
    valHeader.style.cssText =
      'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--thorium-secondary-text);';
    valHeader.textContent = 'Value';
    gridContainer.appendChild(keyHeader);
    gridContainer.appendChild(valHeader);
    gridContainer.appendChild(document.createElement('span'));

    const firstRow = addRow();
    wrapper.appendChild(gridContainer);
    wrapper.appendChild(errorArea);

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:center;';
    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);

    setTimeout(() => {
      firstRow.keyEl.focus();
      validate();
    }, 0);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toRemovalDOM(): HTMLElement {
    const wrapper = document.createElement('div');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: flex',
      'flex-direction: column',
      'gap: 2px',
      'padding: 8px 12px',
      'margin: 0 4px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-danger-bg, #e74c3c)',
      'border-radius: 6px',
      'font-family: monospace',
      'font-size: 13px',
      'box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)',
    ].join(';');

    const headerSpan = document.createElement('div');
    headerSpan.style.cssText = 'color:var(--thorium-danger-bg, #e74c3c);font-weight:600;line-height:20px;';
    headerSpan.textContent = `Remove '${this.proposal?.field ?? ''}'?`;
    wrapper.appendChild(headerSpan);

    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = 'display:flex;gap:6px;justify-content:flex-end;';

    const removeBtn = makeBtn('Remove', 'Remove this field', '--thorium-danger-bg', () => {
      this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
    });
    const cancelBtn = makeBtn('Cancel', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    btnContainer.appendChild(removeBtn);
    btnContainer.appendChild(cancelBtn);
    wrapper.appendChild(btnContainer);

    this.setupWrapper(wrapper);
    return wrapper;
  }

  private toInlineDOM(): HTMLElement {
    const wrapper = document.createElement('span');
    wrapper.className = 'cm-suggestion-preview';
    wrapper.style.cssText = [
      'display: inline-flex',
      'align-items: center',
      'gap: 6px',
      'padding: 1px 8px',
      'margin-left: 8px',
      'background-color: var(--thorium-panel-bg)',
      'border: 1px solid var(--thorium-panel-border)',
      'border-left: 3px solid var(--thorium-info-secondary-bg)',
      'border-radius: 4px',
      'font-family: monospace',
      'font-size: 13px',
      'vertical-align: baseline',
    ].join(';');

    const textSpan = document.createElement('span');
    textSpan.style.cssText = ['color: var(--thorium-text)', 'font-style: italic', 'white-space: pre'].join(';');
    textSpan.textContent = this.insertText.trim();
    wrapper.appendChild(textSpan);

    const btnContainer = document.createElement('span');
    btnContainer.style.cssText = ['display: inline-flex', 'gap: 4px', 'flex-shrink: 0'].join(';');

    const acceptBtn = makeBtn('Accept', 'Insert (Enter)', '--thorium-ok-bg', () => {
      this.viewRef.current?.dispatch({ effects: acceptPreview.of() });
    });
    const dismissBtn = makeBtn('Dismiss', 'Cancel (Escape)', '--thorium-info-bg', () => {
      this.viewRef.current?.dispatch({ effects: clearPreview.of() });
    });

    btnContainer.appendChild(acceptBtn);
    btnContainer.appendChild(dismissBtn);
    wrapper.appendChild(btnContainer);
    this.setupWrapper(wrapper);
    return wrapper;
  }

  ignoreEvent(): boolean {
    return true;
  }
}

interface ProposalState {
  proposal: PreviewProposal | null;
  insertText: string;
  insertPos: number;
  inline: boolean;
  removeFrom?: number;
  removeTo?: number;
  oldContent?: string;
}

export const previewState = StateField.define<ProposalState>({
  create() {
    return { proposal: null, insertText: '', insertPos: 0, inline: false };
  },

  update(state, tr) {
    const empty: ProposalState = { proposal: null, insertText: '', insertPos: 0, inline: false };
    for (const effect of tr.effects) {
      if (effect.is(addPreview)) {
        const docText = tr.state.doc.toString();

        if (effect.value.isRemoval) {
          const range = buildRemoveRange(effect.value.field, docText, effect.value.format);
          if (range) {
            return {
              proposal: effect.value,
              insertText: range.content,
              insertPos: range.from,
              inline: false,
              removeFrom: range.from,
              removeTo: range.to,
            };
          }
          return empty;
        }

        if (effect.value.isReplace && !(effect.value.isMapEntry && effect.value.format === FormatType.JSON)) {
          const removeField = effect.value.isMapEntry
            ? effect.value.field.split('.').slice(0, -1).join('.') || effect.value.field
            : effect.value.field;
          const range = buildRemoveRange(removeField, docText, effect.value.format);
          if (range) {
            const stripped = docText.slice(0, range.from) + docText.slice(range.to);
            const result = buildInsertText(
              effect.value.field,
              effect.value.value,
              stripped,
              effect.value.format,
              undefined,
              effect.value.isList,
              effect.value.schema,
            );
            return {
              proposal: effect.value,
              insertText: result.text,
              insertPos: range.from,
              inline: false,
              removeFrom: range.from,
              removeTo: range.to,
              oldContent: range.content,
            };
          }
          return empty;
        }

        const result = buildInsertText(
          effect.value.field,
          effect.value.value,
          docText,
          effect.value.format,
          effect.value.cursorLine,
          effect.value.isList,
          effect.value.schema,
        );
        return {
          proposal: effect.value,
          insertText: result.text,
          insertPos: result.pos,
          inline: result.inline ?? false,
          ...(result.replaceEnd != null ? { removeFrom: result.pos, removeTo: result.replaceEnd } : {}),
        };
      }
      if (effect.is(clearPreview)) return empty;
      if (effect.is(acceptPreview)) return empty;
      if (effect.is(updateInsertText)) {
        return { ...state, insertText: effect.value };
      }
    }
    if (tr.docChanged && state.proposal) return empty;
    return state;
  },
});

const previewDecoField = StateField.define<{ decos: DecorationSet; viewRef: { current: EditorView | null } }>({
  create() {
    return { decos: Decoration.none, viewRef: { current: null } };
  },

  update(state, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setViewRef)) {
        return { ...state, viewRef: { current: effect.value } };
      }
    }

    const prev = tr.startState.field(previewState);
    const curr = tr.state.field(previewState);
    if (prev === curr) return state;

    if (!curr.proposal) {
      return { ...state, decos: Decoration.none };
    }

    const iPos = Math.min(curr.insertPos, tr.state.doc.length);
    const widget = new PreviewWidget(curr.insertText, state.viewRef, curr.inline, curr.proposal?.schema, curr.proposal, curr.oldContent);

    if (curr.inline) {
      const widgetDeco = Decoration.widget({ widget, side: 1, block: false }).range(iPos);
      return { ...state, decos: Decoration.set([widgetDeco]) };
    }

    const doc = tr.state.doc;
    let decoPos = iPos;

    // For replace operations, position widget after the content being replaced
    if (curr.removeFrom != null && curr.removeTo != null && !curr.proposal?.isRemoval) {
      decoPos = Math.min(curr.removeTo, doc.length);
    }

    // Place widget at end of last content line to avoid visual blank line gap
    if (decoPos === doc.length && doc.length > 0) {
      const line = doc.lineAt(decoPos);
      if (line.from === decoPos && line.length === 0 && decoPos > 0) {
        decoPos = doc.lineAt(decoPos - 1).to;
      }
    }
    const line = doc.lineAt(decoPos);
    const side = decoPos === doc.length || decoPos !== line.from ? 1 : -1;
    const widgetDeco = Decoration.widget({ widget, side, block: true }).range(decoPos);

    const rangeList = [widgetDeco];

    if (curr.removeFrom != null && curr.removeTo != null) {
      const rfrom = Math.min(curr.removeFrom, doc.length);
      const rto = Math.min(curr.removeTo, doc.length);
      const strikethrough = Decoration.line({ class: 'cm-line-strikethrough' });
      for (let pos = rfrom; pos < rto; ) {
        const l = doc.lineAt(pos);
        if (l.length > 0) rangeList.push(strikethrough.range(l.from));
        pos = l.to + 1;
      }
    }

    rangeList.sort((a, b) => a.from - b.from || a.value.startSide - b.value.startSide);

    return {
      ...state,
      decos: Decoration.set(rangeList),
    };
  },

  provide(field) {
    return EditorView.decorations.from(field, (val) => val.decos);
  },
});

export function handleAcceptEffect(view: EditorView): boolean {
  const state = view.state.field(previewState);
  if (!state.proposal) return false;

  view.dispatch({ effects: acceptPreview.of() });
  return true;
}

export function createPreviewExtensions() {
  const acceptFilter = EditorState.transactionFilter.of((tr) => {
    for (const effect of tr.effects) {
      if (effect.is(acceptPreview)) {
        const state = tr.startState.field(previewState);
        if (state.proposal) {
          if (state.removeFrom != null && state.removeTo != null) {
            const insert = state.proposal?.isRemoval ? '' : state.insertText;
            return [
              tr,
              {
                changes: { from: state.removeFrom, to: state.removeTo, insert },
              },
            ];
          }
          return [
            tr,
            {
              changes: { from: state.insertPos, insert: state.insertText },
            },
          ];
        }
      }
    }
    return tr;
  });

  const keyHandler = EditorView.domEventHandlers({
    keydown(event, view) {
      const state = view.state.field(previewState);
      if (!state.proposal) return false;

      const target = event.target as HTMLElement;
      if (target.closest('.cm-suggestion-preview')) return false;

      if (event.key === 'Enter') {
        event.preventDefault();
        view.dispatch({ effects: acceptPreview.of() });
        return true;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        view.dispatch({ effects: clearPreview.of() });
        return true;
      }
      return false;
    },
  });

  const viewRefPlugin = EditorView.updateListener.of((update) => {
    const field = update.state.field(previewDecoField);
    if (!field.viewRef.current) {
      setTimeout(() => update.view.dispatch({ effects: setViewRef.of(update.view) }), 0);
    }

    const prev = update.startState.field(previewState);
    const curr = update.state.field(previewState);
    if (curr.proposal && prev.proposal !== curr.proposal) {
      const pos = curr.removeTo != null && !curr.proposal.isRemoval ? curr.removeTo : curr.insertPos;
      setTimeout(() => {
        update.view.dispatch({
          effects: EditorView.scrollIntoView(Math.min(pos, update.state.doc.length), { y: 'nearest' }),
        });
      }, 0);
    }
  });

  return [previewState, previewDecoField, acceptFilter, keyHandler, viewRefPlugin];
}
