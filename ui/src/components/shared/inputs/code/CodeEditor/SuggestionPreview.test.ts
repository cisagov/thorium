import { describe, test, expect } from 'vitest';
import { parseDocument } from 'yaml';

// project imports
import { FormatType } from '@utilities/rules/types';
import {
  buildInsertText,
  buildMapEntryText,
  buildRemoveRange,
  buildStageListJson,
  buildStageListYaml,
  buildVariantScaffoldYaml,
  buildVariantScaffoldJson,
  buildVariantListYaml,
  buildVariantMapYaml,
  prettyJsonEntry,
  prettyJsonValue,
} from './SuggestionPreview';
import {
  VOLUME_ENTRY_SCHEMA,
  IMAGE_FIELD_SCHEMAS,
  KWARG_DEPENDENCY_SCHEMA,
  AUTO_TAG_LOGIC_SCHEMA,
} from '@utilities/rules/tools/image/schema';
import { PIPELINE_FIELD_SCHEMAS, TAG_TRIGGER_SCHEMA } from '@utilities/rules/tools/pipeline/schema';

const YAML_BASE = 'group: test\nname: p\norder:\n  - img1';
const JSON_BASE = '{\n  "group": "test",\n  "name": "p",\n  "order": ["img1"]\n}';

describe('buildInsertText — YAML triggers', () => {
  test('insert into empty triggers: {} replaces braces', () => {
    const doc = YAML_BASE + '\ntriggers: {}';
    const result = buildInsertText('triggers.my-trig', 'NewSample', doc, FormatType.YAML);
    expect(result.text).toContain('my-trig');
    expect(result.replaceEnd).toBeDefined();
  });

  test('insert into populated triggers appends entry', () => {
    const doc = YAML_BASE + '\ntriggers:\n  t1: NewSample';
    const result = buildInsertText('triggers.t2', 'NewSample', doc, FormatType.YAML);
    expect(result.text).toContain('t2');
    const full = doc.slice(0, result.pos) + result.text + doc.slice(result.pos);
    expect(full).toContain('t1: NewSample');
    expect(full).toContain('t2');
  });

  test('insert when triggers key absent creates full structure', () => {
    const result = buildInsertText('triggers.my-trig', 'NewSample', YAML_BASE, FormatType.YAML);
    expect(result.text).toContain('triggers:');
    expect(result.text).toMatch(/triggers:\n\s+my-trig/);
  });

  test('preserves 2-space indentation', () => {
    const doc = YAML_BASE + '\ntriggers:\n  t1: NewSample';
    const result = buildInsertText('triggers.t2', 'NewSample', doc, FormatType.YAML);
    const lines = result.text.split('\n').filter((l) => l.trim());
    for (const line of lines) {
      const indent = line.length - line.trimStart().length;
      expect(indent % 2).toBe(0);
    }
  });
});

// --- JSON trigger tests ---

describe('buildInsertText — JSON triggers', () => {
  test('inserts into empty "triggers": {}', () => {
    const doc = '{\n  "group": "test",\n  "name": "p",\n  "order": ["img1"],\n  "triggers": {}\n}';
    const result = buildInsertText('triggers.my-trig', 'NewSample', doc, FormatType.JSON);

    expect(result.text).toContain('"my-trig"');
  });

  test('inserts into populated triggers', () => {
    const doc = '{\n  "group": "test",\n  "name": "p",\n  "triggers": {\n    "t1": "NewSample"\n  }\n}';
    const result = buildInsertText('triggers.t2', 'NewSample', doc, FormatType.JSON);

    expect(result.text).toContain('"t2"');
    // Verify comma handling -- the result text or its placement should handle commas
    const constructed = doc.slice(0, result.pos) + result.text + doc.slice(result.replaceEnd ?? result.pos);
    // Should contain both t1 and t2
    expect(constructed).toContain('"t1"');
    expect(constructed).toContain('"t2"');
  });

  test('inserts when triggers key is absent', () => {
    const result = buildInsertText('triggers.my-trig', 'NewSample', JSON_BASE, FormatType.JSON);

    expect(result.text).toContain('"triggers"');
    expect(result.text).toContain('"my-trig"');
  });

  test('each JSON insert produces valid JSON when applied', () => {
    // Absent triggers
    const r1 = buildInsertText('triggers.my-trig', 'NewSample', JSON_BASE, FormatType.JSON);
    const full1 = JSON_BASE.slice(0, r1.pos) + r1.text + JSON_BASE.slice(r1.replaceEnd ?? r1.pos);
    expect(() => JSON.parse(full1) as unknown).not.toThrow();

    // Populated triggers
    const docPop = '{\n  "group": "test",\n  "triggers": {\n    "t1": "val"\n  }\n}';
    const r2 = buildInsertText('triggers.t2', 'val2', docPop, FormatType.JSON);
    const full2 = docPop.slice(0, r2.pos) + r2.text + docPop.slice(r2.replaceEnd ?? r2.pos);
    expect(() => JSON.parse(full2) as unknown).not.toThrow();

    // Empty triggers
    const docEmpty = '{\n  "group": "test",\n  "triggers": {}\n}';
    const r3 = buildInsertText('triggers.t3', 'val3', docEmpty, FormatType.JSON);
    const full3 = docEmpty.slice(0, r3.pos) + r3.text + docEmpty.slice(r3.replaceEnd ?? r3.pos);
    expect(() => JSON.parse(full3) as unknown).not.toThrow();
  });
});

// --- JSON comma handling ---

describe('buildInsertText — JSON comma handling', () => {
  const IMAGE_DOC = '{\n  "group": "test",\n  "image": "registry/img:latest",\n  "type": "docker"\n}';

  test('mid-object insertion gets trailing comma', () => {
    // "spawn_limit" sorts between "image" and "type"
    const result = buildInsertText('spawn_limit', 'Unlimited', IMAGE_DOC, FormatType.JSON);
    expect(result.text).toMatch(/,\n$/);
    const full = IMAGE_DOC.slice(0, result.pos) + result.text + IMAGE_DOC.slice(result.pos);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
  });

  test('end-of-object insertion adds comma to previous line', () => {
    // "volumes" sorts after "type" (last key)
    const result = buildInsertText('volumes', '[]', IMAGE_DOC, FormatType.JSON);
    expect(result.text).toMatch(/^,/);
    const full = IMAGE_DOC.slice(0, result.pos) + result.text + IMAGE_DOC.slice(result.pos);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
  });

  test('end-of-object insertion places new entry on a new line', () => {
    const result = buildInsertText('volumes', '[]', IMAGE_DOC, FormatType.JSON);
    expect(result.text).toContain('\n');
    const full = IMAGE_DOC.slice(0, result.pos) + result.text + IMAGE_DOC.slice(result.pos);
    const lines = full.split('\n');
    const volumesLine = lines.find((l) => l.includes('"volumes"'));
    const typeLine = lines.find((l) => l.includes('"type"'));
    expect(volumesLine).toBeDefined();
    expect(typeLine).toBeDefined();
    expect(lines.indexOf(volumesLine!)).toBeGreaterThan(lines.indexOf(typeLine!));
  });

  test('first-position insertion gets trailing comma', () => {
    // "a_field" sorts before all existing keys
    const result = buildInsertText('a_field', 'val', IMAGE_DOC, FormatType.JSON);
    expect(result.text).toMatch(/,\n$/);
    const full = IMAGE_DOC.slice(0, result.pos) + result.text + IMAGE_DOC.slice(result.pos);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
  });

  test('multiple sequential insertions produce valid JSON', () => {
    let doc = IMAGE_DOC;

    // Insert in middle
    const r1 = buildInsertText('spawn_limit', 'Unlimited', doc, FormatType.JSON);
    doc = doc.slice(0, r1.pos) + r1.text + doc.slice(r1.replaceEnd ?? r1.pos);
    expect(() => JSON.parse(doc) as unknown).not.toThrow();

    // Insert at end
    const r2 = buildInsertText('volumes', 'data', doc, FormatType.JSON);
    doc = doc.slice(0, r2.pos) + r2.text + doc.slice(r2.replaceEnd ?? r2.pos);
    expect(() => JSON.parse(doc) as unknown).not.toThrow();

    // Insert at beginning
    const r3 = buildInsertText('a_first', 'val', doc, FormatType.JSON);
    doc = doc.slice(0, r3.pos) + r3.text + doc.slice(r3.replaceEnd ?? r3.pos);
    expect(() => JSON.parse(doc) as unknown).not.toThrow();
  });
});

// --- YAML removal tests ---

describe('buildRemoveRange — YAML', () => {
  test('removes triggers from doc with triggers: {}', () => {
    const doc = YAML_BASE + '\ntriggers: {}';
    const result = buildRemoveRange('triggers', doc, FormatType.YAML);

    expect(result).not.toBeNull();
    expect(result!.content).toContain('triggers');
    // from/to should cover the triggers line
    const removed = doc.slice(result!.from, result!.to);
    expect(removed).toContain('triggers');
  });

  test('removes triggers from doc with multi-line triggers block', () => {
    const doc = YAML_BASE + '\ntriggers:\n  t1: val1\n  t2: val2';
    const result = buildRemoveRange('triggers', doc, FormatType.YAML);

    expect(result).not.toBeNull();
    expect(result!.content).toContain('triggers');
    expect(result!.content).toContain('t1');
    expect(result!.content).toContain('t2');
    // The range should cover all trigger lines
    expect(result!.to).toBeGreaterThan(result!.from);
    const removed = doc.slice(result!.from, result!.to);
    expect(removed).toContain('triggers:');
    expect(removed).toContain('t1: val1');
    expect(removed).toContain('t2: val2');
  });

  test('returns null for non-existent field', () => {
    const result = buildRemoveRange('nonexistent', YAML_BASE, FormatType.YAML);
    expect(result).toBeNull();
  });
});

// --- JSON removal tests ---

describe('buildRemoveRange — JSON', () => {
  test('removes triggers from doc with "triggers": {}', () => {
    const doc = '{\n  "group": "test",\n  "triggers": {}\n}';
    const result = buildRemoveRange('triggers', doc, FormatType.JSON);

    expect(result).not.toBeNull();
    expect(result!.content).toContain('"triggers"');
    const remaining = doc.slice(0, result!.from) + doc.slice(result!.to);
    // remaining should still have the group key but not triggers
    expect(remaining).toContain('"group"');
    expect(remaining).not.toContain('"triggers"');
  });

  test('removes triggers from nested JSON', () => {
    const doc = '{\n  "group": "test",\n  "triggers": {\n    "t1": "v1",\n    "t2": "v2"\n  }\n}';
    const result = buildRemoveRange('triggers', doc, FormatType.JSON);

    expect(result).not.toBeNull();
    expect(result!.content).toContain('"triggers"');
    // The range should cover the entire triggers block
    const removed = doc.slice(result!.from, result!.to);
    expect(removed).toContain('"triggers"');
  });

  test('returns null for non-existent field', () => {
    const result = buildRemoveRange('nonexistent', JSON_BASE, FormatType.JSON);
    expect(result).toBeNull();
  });
});

// --- Tag trigger sub-fields must nest under the `.Tag` variant key ---

describe('buildInsertText — Tag trigger sub-fields nest under .Tag', () => {
  const TAG_DOC = `${YAML_BASE}\ntriggers:\n  t1:\n    Tag:\n      tag_types:\n        - Files`;

  test('required is inserted inside t1.Tag, not as a sibling of Tag (YAML)', () => {
    const schema = TAG_TRIGGER_SCHEMA.fields!.required;
    const result = buildInsertText('triggers.t1.Tag.required', '', TAG_DOC, FormatType.YAML, undefined, false, schema);
    const full = TAG_DOC.slice(0, result.pos) + result.text + TAG_DOC.slice(result.replaceEnd ?? result.pos);
    const parsed = parseDocument(full).toJS() as { triggers: { t1: { Tag?: Record<string, unknown>; required?: unknown } } };
    expect(parsed.triggers.t1.Tag).toHaveProperty('required');
    expect(parsed.triggers.t1.required).toBeUndefined();
  });

  test('tag_types is inserted inside an empty t1.Tag (YAML)', () => {
    const doc = `${YAML_BASE}\ntriggers:\n  t1:\n    Tag:`;
    const schema = TAG_TRIGGER_SCHEMA.fields!.tag_types;
    const result = buildInsertText('triggers.t1.Tag.tag_types', '', doc, FormatType.YAML, undefined, false, schema);
    const full = doc.slice(0, result.pos) + result.text + doc.slice(result.replaceEnd ?? result.pos);
    const parsed = parseDocument(full).toJS() as { triggers: { t1: { Tag?: Record<string, unknown> } } };
    expect(parsed.triggers.t1.Tag).toHaveProperty('tag_types');
  });
});

// --- volumes renders as a structured list of Volume objects (not a single string) ---

describe('buildInsertText — volumes object list', () => {
  test('YAML: produces a structured list item under volumes:', () => {
    const result = buildInsertText('volumes', '', YAML_BASE, FormatType.YAML, undefined, true, VOLUME_ENTRY_SCHEMA);
    const full = YAML_BASE.slice(0, result.pos) + result.text + YAML_BASE.slice(result.replaceEnd ?? result.pos);
    const parsed = parseDocument(full).toJS() as { volumes: Array<Record<string, unknown>> };
    expect(Array.isArray(parsed.volumes)).toBe(true);
    expect(parsed.volumes).toHaveLength(1);
    expect(parsed.volumes[0]).toHaveProperty('name');
    expect(parsed.volumes[0]).toHaveProperty('archetype');
    expect(parsed.volumes[0]).toHaveProperty('mount_path');
    // the default archetype (HostPath) reveals only its matching nested config
    expect(parsed.volumes[0]).toHaveProperty('host_path');
    expect(parsed.volumes[0]).not.toHaveProperty('nfs');
  });

  test('JSON: produces an array containing one Volume object', () => {
    const result = buildInsertText('volumes', '', JSON_BASE, FormatType.JSON, undefined, true, VOLUME_ENTRY_SCHEMA);
    const full = JSON_BASE.slice(0, result.pos) + result.text + JSON_BASE.slice(result.replaceEnd ?? result.pos);
    const parsed = JSON.parse(full) as { volumes: Array<Record<string, unknown>> };
    expect(Array.isArray(parsed.volumes)).toBe(true);
    expect(parsed.volumes[0]).toHaveProperty('archetype');
    expect(parsed.volumes[0]).toHaveProperty('host_path');
  });
});

// --- A3/B2: object populate emits valid enum/bool defaults and never leaks placeholders as values ---

describe('buildInsertText — object populate value correctness', () => {
  const apply = (doc: string, r: { pos: number; text: string; replaceEnd?: number }) =>
    doc.slice(0, r.pos) + r.text + doc.slice(r.replaceEnd ?? r.pos);

  test('YAML: populate dependencies has valid enum/bool defaults, no empty enums/bools, no placeholder leak', () => {
    const r = buildInsertText('dependencies', '', YAML_BASE, FormatType.YAML, undefined, false, IMAGE_FIELD_SCHEMAS.dependencies);
    const text = r.text;
    expect(text).not.toMatch(/strategy: ''/);
    expect(text).not.toMatch(/naming: ''/);
    expect(text).not.toMatch(/enabled: ''/);
    expect(text).not.toMatch(/use_parent_cache: ''/);
    expect(text).toContain('strategy: Paths');
    expect(text).toContain('enabled: false');
    // placeholders are hints, not values
    expect(text).not.toContain("kwarg: 'samples'");
    expect(() => {
      parseDocument(apply(YAML_BASE, r)).toJS();
    }).not.toThrow();
  });

  test('JSON: populate dependencies has valid enum/bool defaults and parses', () => {
    const r = buildInsertText('dependencies', '', JSON_BASE, FormatType.JSON, undefined, false, IMAGE_FIELD_SCHEMAS.dependencies);
    expect(r.text).not.toContain('"strategy": ""');
    expect(r.text).not.toContain('"enabled": ""');
    expect(r.text).toContain('"strategy": "Paths"');
    expect(() => JSON.parse(apply(JSON_BASE, r)) as unknown).not.toThrow();
  });

  test('output_collection populate emits auto_tag as an empty map, not logic/key fields', () => {
    const y = buildInsertText('output_collection', '', YAML_BASE, FormatType.YAML, undefined, false, IMAGE_FIELD_SCHEMAS.output_collection);
    const parsed = parseDocument(apply(YAML_BASE, y)).toJS() as { output_collection: { auto_tag: Record<string, unknown> } };
    expect(parsed.output_collection.auto_tag).toEqual({});
  });
});

// --- A4/A6: data-carrying variant fields serialize to the correct shape per variant ---

describe('variant scaffold serialization (KwargDependency / AutoTagLogic)', () => {
  const kwarg = KWARG_DEPENDENCY_SCHEMA.variants!;
  const logic = AUTO_TAG_LOGIC_SCHEMA.variants!;

  test('unit variant is bare; payload variants are single-key objects (YAML)', () => {
    expect(buildVariantScaffoldYaml('kwarg', kwarg, 'None', '  ')).toBe('  kwarg: None\n');
    expect(buildVariantScaffoldYaml('kwarg', kwarg, 'List', '  ')).toContain('List:');
    expect(buildVariantScaffoldYaml('kwarg', kwarg, 'Map', '  ')).toContain('Map: {}');
    expect(buildVariantScaffoldYaml('logic', logic, 'Exists', '  ')).toBe('  logic: Exists\n');
    expect(buildVariantScaffoldYaml('logic', logic, 'In', '  ')).toContain('In: []');
  });

  test('JSON scaffolds', () => {
    expect(buildVariantScaffoldJson('kwarg', kwarg, 'None')).toBe('"kwarg": "None"');
    expect(buildVariantScaffoldJson('kwarg', kwarg, 'Map')).toContain('"Map"');
    expect(buildVariantScaffoldJson('logic', logic, 'In')).toContain('"In": []');
  });

  test('list and map payloads produce valid YAML', () => {
    const listYaml = buildVariantListYaml('logic', 'In', ['a', 'b'], '');
    const p = parseDocument(listYaml).toJS() as { logic: { In: string[] } };
    expect(p.logic.In).toEqual(['a', 'b']);

    const mapYaml = buildVariantMapYaml('kwarg', 'Map', [{ key: 'img', value: '--r' }], '');
    const pm = parseDocument(mapYaml).toJS() as { kwarg: { Map: Record<string, string> } };
    expect(pm.kwarg.Map).toEqual({ img: '--r' });
  });

  test('populate results: kwarg defaults to the valid bare None variant', () => {
    const r = buildInsertText(
      'results',
      '',
      YAML_BASE,
      FormatType.YAML,
      undefined,
      false,
      IMAGE_FIELD_SCHEMAS.dependencies.fields!.results,
    );
    expect(r.text).toContain('kwarg: None');
    expect(r.text).not.toContain('kwarg: List');
  });
});

// --- buildMapEntryText: parent-key emission for variant map entries (e.g. triggers) ---

describe('buildMapEntryText — emits the parent map key only when needed', () => {
  test('YAML absent: wraps the entry under triggers: (preserving the leading newline)', () => {
    const text = buildMapEntryText('  my-trig: NewSample\n', {
      format: FormatType.YAML,
      parentKey: 'triggers',
      parentMissing: true,
      isPopulate: false,
      insertText: '\ntriggers:\n  trigger-name: \n',
    });
    expect(text).toBe('\ntriggers:\n  my-trig: NewSample\n');
  });

  test('YAML populate (empty triggers:): emits triggers: with no leading newline', () => {
    const text = buildMapEntryText('  my-trig: NewSample\n', {
      format: FormatType.YAML,
      parentKey: 'triggers',
      parentMissing: false,
      isPopulate: true,
      insertText: 'triggers:\n  trigger-name: \n',
    });
    expect(text).toBe('triggers:\n  my-trig: NewSample\n');
  });

  test('YAML append (triggers already populated): no parent key re-emitted', () => {
    const text = buildMapEntryText('  my-trig: NewSample\n', {
      format: FormatType.YAML,
      parentKey: 'triggers',
      parentMissing: false,
      isPopulate: false,
      insertText: '\n  trigger-name: \n',
    });
    expect(text).toBe('\n  my-trig: NewSample\n');
    expect(text).not.toContain('triggers:');
  });

  test('JSON absent: wraps the entry in a multi-line "triggers": { ... } (so later adds get commas)', () => {
    const text = buildMapEntryText('"my-trig": "NewSample"', {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: true,
      isPopulate: false,
      insertText: ',\n  "triggers": { "trigger-name": "" }',
    });
    expect(text).toBe(',\n  "triggers": {\n    "my-trig": "NewSample"\n  }');
  });

  test('JSON populate (empty {}): fills between braces, no extra wrapper', () => {
    const text = buildMapEntryText('"my-trig": "NewSample"', {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: false,
      isPopulate: true,
      insertText: '\n    "trigger-name": ""\n  ',
    });
    expect(text).toBe('\n    "my-trig": "NewSample"\n  ');
    expect(text).not.toContain('"triggers"');
  });

  test('JSON append (triggers already populated): no wrapper, keeps the comma', () => {
    const text = buildMapEntryText('"my-trig": "NewSample"', {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: false,
      isPopulate: false,
      insertText: ',\n    "trigger-name": ""',
    });
    expect(text).toBe(',\n    "my-trig": "NewSample"');
  });

  test('applied to a doc with no triggers, the result introduces a valid triggers section', () => {
    // YAML — appended at end of a doc that has no triggers key
    const yamlChild = '  my-trig: NewSample\n';
    const yamlText = buildMapEntryText(yamlChild, {
      format: FormatType.YAML,
      parentKey: 'triggers',
      parentMissing: true,
      isPopulate: false,
      insertText: '\ntriggers:\n' + yamlChild,
    });
    const yamlParsed = parseDocument(YAML_BASE + yamlText).toJS() as { triggers: Record<string, unknown> };
    expect(yamlParsed.triggers).toHaveProperty('my-trig', 'NewSample');

    // JSON — spliced after the last key of a doc that has no triggers key
    const jsonText = buildMapEntryText('"my-trig": "NewSample"', {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: true,
      isPopulate: false,
      insertText: ',\n  "triggers": {}',
    });
    const insertPos = JSON_BASE.lastIndexOf(']') + 1;
    const jsonFull = JSON_BASE.slice(0, insertPos) + jsonText + JSON_BASE.slice(insertPos);
    const jsonParsed = JSON.parse(jsonFull) as { triggers: Record<string, unknown> };
    expect(jsonParsed.triggers).toHaveProperty('my-trig', 'NewSample');
  });

  test('JSON append: an object (Tag) variant value is pretty-printed multi-line and stays valid', () => {
    const child = '"webhook": { "Tag": { "tag_types": {}, "required": [], "not": [] } }';
    const text = buildMapEntryText(child, {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: false,
      isPopulate: false,
      insertText: ',\n    "trigger-name": ""',
    });
    expect(text.split('\n').length).toBeGreaterThan(3); // not a single compact blob
    expect(text).toContain('\n      "Tag": {'); // entry(4sp) -> Tag(6sp)
    const doc = '{\n  "group": "test",\n  "triggers": {\n    "t1": "NewSample"\n  }\n}';
    const at = doc.lastIndexOf('"NewSample"') + '"NewSample"'.length;
    const full = doc.slice(0, at) + text + doc.slice(at);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
    expect((JSON.parse(full) as { triggers: Record<string, unknown> }).triggers).toHaveProperty('webhook');
  });

  test('JSON absent: an object (Tag) variant value is pretty-printed inside the new triggers wrapper', () => {
    const child = '"webhook": { "Tag": { "tag_types": {}, "required": [], "not": [] } }';
    const text = buildMapEntryText(child, {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: true,
      isPopulate: false,
      insertText: ',\n  "triggers": {}',
    });
    expect(text).toContain('"triggers": {');
    expect(text).toContain('\n      "Tag": {'); // triggers(2sp) -> webhook(4sp) -> Tag(6sp)
    const at = JSON_BASE.lastIndexOf(']') + 1;
    const full = JSON_BASE.slice(0, at) + text + JSON_BASE.slice(at);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
    expect((JSON.parse(full) as { triggers: Record<string, { Tag: unknown }> }).triggers.webhook).toHaveProperty('Tag');
  });
});

// --- JSON: adding multiple triggers keeps commas valid (regression for inline-object insertion) ---

describe('JSON multi-trigger comma handling', () => {
  test('adding a second trigger to a multi-line triggers object inserts a comma', () => {
    const doc = '{\n  "group": "test",\n  "triggers": {\n    "t1": "NewSample"\n  }\n}';
    const r = buildInsertText('triggers.t2', 'NewSample', doc, FormatType.JSON);
    const full = doc.slice(0, r.pos) + r.text + doc.slice(r.replaceEnd ?? r.pos);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
    const parsed = JSON.parse(full) as { triggers: Record<string, unknown> };
    expect(parsed.triggers).toHaveProperty('t1', 'NewSample');
    expect(parsed.triggers).toHaveProperty('t2', 'NewSample');
  });

  test('inserting into an inline non-empty object appends with a comma (no mangling)', () => {
    const doc = '{\n  "group": "test",\n  "triggers": { "t1": "NewSample" }\n}';
    const r = buildInsertText('triggers.t2', 'NewSample', doc, FormatType.JSON);
    const full = doc.slice(0, r.pos) + r.text + doc.slice(r.replaceEnd ?? r.pos);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
    const parsed = JSON.parse(full) as { triggers: Record<string, unknown> };
    expect(parsed.triggers).toHaveProperty('t1', 'NewSample');
    expect(parsed.triggers).toHaveProperty('t2', 'NewSample');
  });

  test('full flow: add two triggers to a doc with no triggers — both valid, comma present', () => {
    let doc = '{\n  "group": "test",\n  "name": "p",\n  "order": ["img1"]\n}';

    // First add (triggers absent): the widget computes the envelope from the placeholder field,
    // then buildMapEntryText emits the real entry wrapped in a fresh multi-line parent.
    const env1 = buildInsertText('triggers.trigger-name', 'NewSample', doc, FormatType.JSON);
    const text1 = buildMapEntryText('"t1": "NewSample"', {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: true,
      isPopulate: false,
      insertText: env1.text,
    });
    doc = doc.slice(0, env1.pos) + text1 + doc.slice(env1.replaceEnd ?? env1.pos);
    expect(() => JSON.parse(doc) as unknown).not.toThrow();

    // Second add (triggers now exists, multi-line): parentMissing is false, entry appended.
    const env2 = buildInsertText('triggers.trigger-name', 'NewSample', doc, FormatType.JSON);
    const text2 = buildMapEntryText('"t2": "NewSample"', {
      format: FormatType.JSON,
      parentKey: 'triggers',
      parentMissing: false,
      isPopulate: false,
      insertText: env2.text,
    });
    doc = doc.slice(0, env2.pos) + text2 + doc.slice(env2.replaceEnd ?? env2.pos);
    expect(() => JSON.parse(doc) as unknown).not.toThrow();

    const parsed = JSON.parse(doc) as { triggers: Record<string, unknown> };
    expect(parsed.triggers).toHaveProperty('t1', 'NewSample');
    expect(parsed.triggers).toHaveProperty('t2', 'NewSample');
  });
});

// --- prettyJsonValue: object/array values become multi-line, reindented under the key ---

describe('prettyJsonValue', () => {
  test('object value becomes multi-line, reindented to baseIndent', () => {
    expect(prettyJsonValue('{ "a": 1, "b": "x" }', '  ')).toBe('{\n    "a": 1,\n    "b": "x"\n  }');
  });

  test('array value becomes multi-line', () => {
    expect(prettyJsonValue('["a", "b"]', '')).toBe('[\n  "a",\n  "b"\n]');
  });

  test('primitives and empty containers pass through unchanged', () => {
    expect(prettyJsonValue('"hello"', '  ')).toBe('"hello"');
    expect(prettyJsonValue('42', '  ')).toBe('42');
    expect(prettyJsonValue('{}', '  ')).toBe('{}');
    expect(prettyJsonValue('[]', '  ')).toBe('[]');
  });

  test('unparseable input is returned unchanged', () => {
    expect(prettyJsonValue('not json', '  ')).toBe('not json');
  });
});

// --- prettyJsonEntry: pretty-prints a `"key": value` fragment (used for variant map entries) ---

describe('prettyJsonEntry', () => {
  test('object value expands multi-line; key stays on the first line', () => {
    expect(prettyJsonEntry('"k": { "a": 1, "b": "x" }', '  ')).toBe('"k": {\n    "a": 1,\n    "b": "x"\n  }');
  });

  test('primitive values and empty containers are left inline', () => {
    expect(prettyJsonEntry('"k": "v"', '  ')).toBe('"k": "v"');
    expect(prettyJsonEntry('"k": {}', '  ')).toBe('"k": {}');
  });

  test('keys containing escaped quotes are handled', () => {
    expect(prettyJsonEntry('"a\\"b": ["x"]', '')).toBe('"a\\"b": [\n  "x"\n]');
  });

  test('non-entry input is returned unchanged', () => {
    expect(prettyJsonEntry('not an entry', '  ')).toBe('not an entry');
  });
});

// --- JSON object suggestions insert pretty-printed (multi-line) ---

describe('buildInsertText — JSON pretty output', () => {
  test('object populate (dependencies) is multi-line, indented, and parses', () => {
    const r = buildInsertText('dependencies', '', JSON_BASE, FormatType.JSON, undefined, false, IMAGE_FIELD_SCHEMAS.dependencies);
    expect(r.text.split('\n').length).toBeGreaterThan(3); // not a single inline blob
    expect(r.text).toContain('"strategy": "Paths"'); // value preserved verbatim
    expect(r.text).toMatch(/\n {4}"/); // nested keys indented under the parent key
    const full = JSON_BASE.slice(0, r.pos) + r.text + JSON_BASE.slice(r.replaceEnd ?? r.pos);
    expect(() => JSON.parse(full) as unknown).not.toThrow();
    expect((JSON.parse(full) as { dependencies: Record<string, unknown> }).dependencies).toHaveProperty('results');
  });
});

// --- order stage list (nestedList): parallel stages, always grouped Vec<Vec<String>> ---

describe('order stage list (nestedList)', () => {
  test('buildStageListYaml emits grouped flow-array stages', () => {
    expect(buildStageListYaml('order', [['a']], '')).toBe('order:\n  - [a]\n');
    expect(buildStageListYaml('order', [['a'], ['b', 'c']], '')).toBe('order:\n  - [a]\n  - [b, c]\n');
  });

  test('buildStageListYaml drops empty images/stages; all-empty -> []', () => {
    expect(buildStageListYaml('order', [[' a ', ''], [], ['  ']], '')).toBe('order:\n  - [a]\n');
    expect(buildStageListYaml('order', [], '')).toBe('order: []\n');
  });

  test('buildStageListJson emits compact one-line-per-stage arrays, reindented', () => {
    expect(buildStageListJson([['a'], ['b', 'c']], '')).toBe('[\n  ["a"],\n  ["b", "c"]\n]');
    expect(buildStageListJson([['a'], ['b', 'c']], '  ')).toBe('[\n    ["a"],\n    ["b", "c"]\n  ]');
    expect(buildStageListJson([], '')).toBe('[]');
  });

  test('YAML and JSON outputs round-trip to Vec<Vec<String>>', () => {
    const yaml = buildStageListYaml('order', [['a'], ['b', 'c']], '');
    expect((parseDocument(yaml).toJS() as { order: string[][] }).order).toEqual([['a'], ['b', 'c']]);

    const json = JSON.parse(`{ "order": ${buildStageListJson([['a'], ['b', 'c']], '')} }`) as { order: string[][] };
    expect(json.order).toEqual([['a'], ['b', 'c']]);
  });

  test('buildInsertText scaffolds a valid order for the nestedList schema (YAML + JSON)', () => {
    const order = PIPELINE_FIELD_SCHEMAS.order;
    const y = buildInsertText('order', '', 'group: g\nname: n', FormatType.YAML, undefined, false, order);
    expect(y.text).toContain('order:');

    const jdoc = '{\n  "group": "g",\n  "name": "n"\n}';
    const j = buildInsertText('order', '', jdoc, FormatType.JSON, undefined, false, order);
    const full = jdoc.slice(0, j.pos) + j.text + jdoc.slice(j.replaceEnd ?? j.pos);
    expect(full).toContain('"order"');
    expect(() => JSON.parse(full) as unknown).not.toThrow();
  });
});
