import { afterEach, describe, test, expect } from 'vitest';
import { PipelineChecker } from './index';
import { removeLine, removeBlock, replaceLine } from '../../test-helpers';
import { FieldValueType, FormatType, Severity } from '../../types';

const VALID_PIPELINE = `group: analysis
name: triage
order:
    - file-info
    - - yara-scanner
      - clamav
    - report-generator
sla: 604800
description: Standard triage pipeline`;

const pipelineChecker = new PipelineChecker();

function pipelineErrors(text: string) {
  return pipelineChecker.check(text).diagnostics.filter((d) => d.severity === Severity.Error);
}

function pipelineWarnings(text: string) {
  return pipelineChecker.check(text).diagnostics.filter((d) => d.severity === Severity.Warning);
}

function pipelineSuggestions(text: string) {
  return pipelineChecker.check(text).suggestions;
}

describe('PipelineChecker', () => {
  describe('valid pipeline', () => {
    test('produces no errors for valid pipeline request', () => {
      const errs = pipelineErrors(VALID_PIPELINE);
      expect(errs).toHaveLength(0);
    });

    test('produces no warnings for valid pipeline request', () => {
      const warns = pipelineWarnings(VALID_PIPELINE);
      expect(warns).toHaveLength(0);
    });
  });

  describe('required field errors', () => {
    test('missing group', () => {
      const text = removeLine(VALID_PIPELINE, 'group:');
      const errs = pipelineErrors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'group'"))).toBe(true);
    });

    test('missing name', () => {
      const text = removeLine(VALID_PIPELINE, 'name:');
      const errs = pipelineErrors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'name'"))).toBe(true);
    });

    test('missing order', () => {
      const text = removeBlock(VALID_PIPELINE, 'order');
      const errs = pipelineErrors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'order'"))).toBe(true);
    });
  });

  describe('type validation', () => {
    test('sla must be a number', () => {
      const text = replaceLine(VALID_PIPELINE, 'sla:', 'sla: fast');
      const errs = pipelineErrors(text);
      expect(errs.some((e) => e.message.includes("'sla' must be a number"))).toBe(true);
    });

    test('order must be an array', () => {
      const text = replaceLine(VALID_PIPELINE, 'order:', 'order: not-an-array');
      const errs = pipelineErrors(text);
      expect(errs.some((e) => e.message.includes("'order' must be an array"))).toBe(true);
    });
  });

  describe('unknown field warnings', () => {
    test('unknown top-level field', () => {
      const text = VALID_PIPELINE + '\nfoobar: baz';
      const warns = pipelineWarnings(text);
      expect(warns.some((w) => w.message.includes("Unknown pipeline field: 'foobar'"))).toBe(true);
    });
  });

  describe('suggestions', () => {
    test('missing optional fields are suggested', () => {
      const minimal = `group: test\nname: minimal\norder:\n    - tool1`;
      const s = pipelineSuggestions(minimal);
      const fields = s.map((sg) => sg.field);
      expect(fields).toContain('description');
      expect(fields).toContain('sla');
      // triggers is a map, suggested via its map-entry (not a `triggers` struct field)
      expect(fields).toContain('triggers.trigger-name');
      expect(fields).not.toContain('triggers');
    });

    test('missing required pipeline fields are suggested', () => {
      const noGroup = `name: minimal\norder:\n    - tool1`;
      const s = pipelineSuggestions(noGroup);
      const groupSugg = s.find((sg) => sg.field === 'group');
      expect(groupSugg).toBeDefined();
      expect(groupSugg!.message).toContain('Required');
    });

    test('pipeline suggestions are sorted alphabetically', () => {
      const minimal = `group: test\nname: minimal\norder:\n    - tool1`;
      const s = pipelineSuggestions(minimal);
      const fields = s.map((sg) => sg.field);
      const sorted = [...fields].sort((a, b) => a.localeCompare(b));
      expect(fields).toEqual(sorted);
    });

    test('pipeline suggestions carry schemas', () => {
      const minimal = `group: test\nname: minimal\norder:\n    - tool1`;
      const s = pipelineSuggestions(minimal);
      const slaSugg = s.find((sg) => sg.field === 'sla');
      expect(slaSugg?.schema).toBeDefined();
      expect(slaSugg!.schema!.type).toBe(FieldValueType.Number);

      const descSugg = s.find((sg) => sg.field === 'description');
      expect(descSugg?.schema).toBeDefined();
      expect(descSugg!.schema!.type).toBe(FieldValueType.String);
    });
  });
});

describe('pipeline order image validation', () => {
  afterEach(() => {
    pipelineChecker.clearValidImageNames();
  });

  test('no errors when validImageNames is not set', () => {
    const errs = pipelineErrors(VALID_PIPELINE);
    expect(errs).toHaveLength(0);
  });

  test('valid images produce no errors', () => {
    pipelineChecker.setValidImageNames('analysis', ['file-info', 'yara-scanner', 'clamav', 'report-generator']);
    const errs = pipelineErrors(VALID_PIPELINE);
    expect(errs).toHaveLength(0);
  });

  test('unknown image in flat order produces error', () => {
    pipelineChecker.setValidImageNames('analysis', ['yara-scanner', 'clamav', 'report-generator']);
    const errs = pipelineErrors(VALID_PIPELINE);
    expect(errs.some((e) => e.message.includes("Image 'file-info' not found in group 'analysis'"))).toBe(true);
  });

  test('unknown image in parallel step produces error', () => {
    pipelineChecker.setValidImageNames('analysis', ['file-info', 'report-generator']);
    const errs = pipelineErrors(VALID_PIPELINE);
    expect(errs.some((e) => e.message.includes("Image 'yara-scanner' not found"))).toBe(true);
    expect(errs.some((e) => e.message.includes("Image 'clamav' not found"))).toBe(true);
  });

  test('group mismatch between checker and YAML skips validation', () => {
    pipelineChecker.setValidImageNames('other-group', ['some-image']);
    const errs = pipelineErrors(VALID_PIPELINE);
    expect(errs).toHaveLength(0);
  });

  test('clearValidImageNames removes validation', () => {
    pipelineChecker.setValidImageNames('analysis', []);
    let errs = pipelineErrors(VALID_PIPELINE);
    expect(errs.length).toBeGreaterThan(0);

    pipelineChecker.clearValidImageNames();
    errs = pipelineErrors(VALID_PIPELINE);
    expect(errs).toHaveLength(0);
  });
});

describe('pipeline null triggers', () => {
  test('null triggers offers a Populate map-entry (like empty {}), not a struct populate', () => {
    const text = `group: test\nname: minimal\norder:\n    - tool1\ntriggers:`;
    const s = pipelineSuggestions(text);
    const trigName = s.find((sg) => sg.field === 'triggers.trigger-name');
    expect(trigName).toBeDefined();
    expect(trigName!.isMapEntry).toBe(true);
    expect(trigName!.isReplace).toBe(true);
    expect(trigName!.message).toBe('Populate triggers');
    // no whole-object struct populate that would create a literal `trigger-name` key
    expect(s.some((sg) => sg.field === 'triggers' && sg.isReplace)).toBe(false);
  });
});

const BASE = `group: test\nname: p\norder:\n  - img1`;

describe('trigger suggestion states', () => {
  test('S1: triggers absent — offers Add trigger map-entry, not a struct suggestion', () => {
    const s = pipelineSuggestions(BASE);
    const addTrig = s.find((sg) => sg.field === 'triggers.trigger-name');
    expect(addTrig).toBeDefined();
    expect(addTrig!.message).toBe('Add trigger');
    expect(addTrig!.isMapEntry).toBe(true);
    expect(addTrig!.isReplace).toBeFalsy();

    // The whole-object `triggers` struct suggestion must NOT be offered: it would route to a struct
    // form that renders `trigger-name` as a literal field (no name/Tag editing). The map entry above
    // is the only path for creating the section.
    expect(s.some((sg) => sg.field === 'triggers')).toBe(false);
  });

  test('S2: triggers null — behaves like empty {} (Populate trigger-name, no struct populate)', () => {
    const text = `${BASE}\ntriggers:`;
    const s = pipelineSuggestions(text);

    // null triggers must NOT offer a whole-object populate (that would create a literal `trigger-name` key)
    const trigField = s.find((sg) => sg.field === 'triggers' && sg.isReplace && !sg.isMapEntry);
    expect(trigField).toBeUndefined();

    const trigName = s.find((sg) => sg.field === 'triggers.trigger-name');
    expect(trigName).toBeDefined();
    expect(trigName!.message).toBe('Populate triggers');
    expect(trigName!.isMapEntry).toBe(true);
    expect(trigName!.isReplace).toBe(true);
  });

  test('S3: triggers empty {} — offers Populate triggers with isReplace', () => {
    const text = `${BASE}\ntriggers: {}`;
    const s = pipelineSuggestions(text);

    const trigName = s.find((sg) => sg.field === 'triggers.trigger-name');
    expect(trigName).toBeDefined();
    expect(trigName!.message).toBe('Populate triggers');
    expect(trigName!.isMapEntry).toBe(true);
    expect(trigName!.isReplace).toBe(true);
  });

  test('S4: one NewSample trigger — offers Add trigger without isReplace', () => {
    const text = `${BASE}\ntriggers:\n  t1: NewSample`;
    const s = pipelineSuggestions(text);

    const trigName = s.find((sg) => sg.field === 'triggers.trigger-name');
    expect(trigName).toBeDefined();
    expect(trigName!.message).toBe('Add trigger');
    expect(trigName!.isMapEntry).toBe(true);
    expect(trigName!.isReplace).toBeFalsy();
  });

  test('S5: one Tag trigger — offers Add trigger and sub-field suggestions', () => {
    const text = `${BASE}\ntriggers:\n  t1:\n    Tag:\n      tag_types:\n        - Files`;
    const s = pipelineSuggestions(text);

    const addTrig = s.find((sg) => sg.field === 'triggers.trigger-name');
    expect(addTrig).toBeDefined();
    expect(addTrig!.message).toBe('Add trigger');

    // sub-fields nest under the Tag variant key, not directly under the trigger name
    const reqSugg = s.find((sg) => sg.field === 'triggers.t1.Tag.required');
    expect(reqSugg).toBeDefined();
    expect(reqSugg!.message).toBe('Tags that must be present to trigger');

    const notSugg = s.find((sg) => sg.field === 'triggers.t1.Tag.not');
    expect(notSugg).toBeDefined();
    expect(notSugg!.message).toBe('Tags that must not be present to trigger');

    // tag_types already present under Tag → not re-suggested; and nothing at the wrong level
    expect(s.some((sg) => sg.field.endsWith('tag_types'))).toBe(false);
    expect(s.some((sg) => sg.field === 'triggers.t1.required')).toBe(false);
    expect(s.some((sg) => sg.field === 'triggers.t1.not')).toBe(false);
  });

  test('S6: two triggers — offers Add trigger without isReplace', () => {
    const text = `${BASE}\ntriggers:\n  t1: NewSample\n  t2: NewSample`;
    const s = pipelineSuggestions(text);

    const trigName = s.find((sg) => sg.field === 'triggers.trigger-name');
    expect(trigName).toBeDefined();
    expect(trigName!.message).toBe('Add trigger');
    expect(trigName!.isReplace).toBeFalsy();
  });

  test('S7: invalid trigger value — offers Set trigger type', () => {
    const text = `${BASE}\ntriggers:\n  t1: BadValue`;
    const s = pipelineSuggestions(text);

    const setType = s.find((sg) => sg.field === 'triggers.t1');
    expect(setType).toBeDefined();
    expect(setType!.message).toBe('Set trigger type');
  });

  test('S8: empty string trigger value — offers Set trigger type', () => {
    const text = `${BASE}\ntriggers:\n  t1: ""`;
    const s = pipelineSuggestions(text);

    const setType = s.find((sg) => sg.field === 'triggers.t1');
    expect(setType).toBeDefined();
    expect(setType!.message).toBe('Set trigger type');
  });
});

describe('order suggestions', () => {
  afterEach(() => {
    pipelineChecker.clearValidImageNames();
  });

  test('empty order [] offers "Populate order" (isReplace)', () => {
    const order = pipelineSuggestions('group: g\nname: n\norder: []').find((sg) => sg.field === 'order');
    expect(order).toBeDefined();
    expect(order!.message).toBe('Populate order');
    expect(order!.isReplace).toBe(true);
  });

  test('JSON empty order [] also offers "Populate order"', () => {
    const json = new PipelineChecker();
    json.format = FormatType.JSON;
    const order = json.check('{\n  "group": "g",\n  "name": "n",\n  "order": []\n}').suggestions.find((sg) => sg.field === 'order');
    expect(order?.message).toBe('Populate order');
    expect(order?.isReplace).toBe(true);
  });

  test('order suggestion carries the matching group images as sorted enumValues (not value chips)', () => {
    pipelineChecker.setValidImageNames('g', ['b-img', 'a-img']);
    const order = pipelineSuggestions('group: g\nname: n\norder: []').find((sg) => sg.field === 'order');
    expect(order?.schema?.enumValues).toEqual(['a-img', 'b-img']);
    expect(order?.values).toBeUndefined();
  });

  test('no enumValues when the checker group does not match the document group', () => {
    pipelineChecker.setValidImageNames('other', ['x']);
    const order = pipelineSuggestions('group: g\nname: n\norder: []').find((sg) => sg.field === 'order');
    expect(order?.schema?.enumValues).toBeUndefined();
  });

  test('absent order (Add) also carries group images when set', () => {
    pipelineChecker.setValidImageNames('g', ['img1']);
    const order = pipelineSuggestions('group: g\nname: n').find((sg) => sg.field === 'order');
    expect(order).toBeDefined();
    expect(order!.schema?.enumValues).toEqual(['img1']);
  });
});
