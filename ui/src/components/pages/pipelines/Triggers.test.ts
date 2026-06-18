import { describe, it, expect } from 'vitest';

// project imports
import { TriggerKind, formToTriggers, triggerNameErrors, triggersToForm, validateTriggers } from './Triggers';
import type { FormTrigger } from './Triggers';
import type { EventTrigger } from '@models/pipelines';
import { TagTypes } from '@models/tags';

/// Build a Tag-kind form trigger, overriding any fields supplied
function tagForm(patch: Partial<FormTrigger> = {}): FormTrigger {
  return {
    name: 'trig',
    kind: TriggerKind.Tag,
    tagTypes: [TagTypes.Files],
    required: [],
    not: [],
    ...patch,
  };
}

describe('triggersToForm', () => {
  it('maps a NewSample trigger to a NewSample form entry', () => {
    const forms = triggersToForm({ onUpload: 'NewSample' });
    expect(forms).toEqual([{ name: 'onUpload', kind: TriggerKind.NewSample, tagTypes: [], required: [], not: [] }]);
  });

  it('flattens a Tag trigger map into key/value rows', () => {
    const triggers: Record<string, EventTrigger> = {
      onTag: {
        Tag: {
          tag_types: [TagTypes.Files, TagTypes.Repos],
          required: { family: ['emotet', 'trickbot'] },
          not: { tlp: ['red'] },
        },
      },
    };
    const forms = triggersToForm(triggers);
    expect(forms).toEqual([
      {
        name: 'onTag',
        kind: TriggerKind.Tag,
        tagTypes: [TagTypes.Files, TagTypes.Repos],
        required: [
          { key: 'family', value: 'emotet' },
          { key: 'family', value: 'trickbot' },
        ],
        not: [{ key: 'tlp', value: 'red' }],
      },
    ]);
  });
});

describe('formToTriggers', () => {
  it('serializes a NewSample form back to the string literal', () => {
    const forms: FormTrigger[] = [{ name: 'onUpload', kind: TriggerKind.NewSample, tagTypes: [], required: [], not: [] }];
    expect(formToTriggers(forms)).toEqual({ onUpload: 'NewSample' });
  });

  it('aggregates key/value rows into a tag filter map and drops empty rows', () => {
    const forms = [
      tagForm({
        name: 'onTag',
        tagTypes: [TagTypes.Files],
        required: [
          { key: 'family', value: 'emotet' },
          { key: 'family', value: 'trickbot' },
          { key: '', value: '' },
          { key: 'os', value: '' },
        ],
        not: [{ key: 'tlp', value: 'red' }],
      }),
    ];
    expect(formToTriggers(forms)).toEqual({
      onTag: {
        Tag: {
          tag_types: [TagTypes.Files],
          required: { family: ['emotet', 'trickbot'] },
          not: { tlp: ['red'] },
        },
      },
    });
  });

  it('de-duplicates repeated values for the same key', () => {
    const forms = [
      tagForm({
        required: [
          { key: 'family', value: 'emotet' },
          { key: 'family', value: 'emotet' },
        ],
      }),
    ];
    const trigger = formToTriggers(forms).trig;
    // narrow to the Tag variant to read the aggregated required map
    expect(typeof trigger === 'object' && 'Tag' in trigger ? trigger.Tag.required.family : null).toEqual(['emotet']);
  });

  it('skips triggers with blank names', () => {
    const forms = [tagForm({ name: '   ' })];
    expect(formToTriggers(forms)).toEqual({});
  });

  it('round-trips a Tag trigger through form and back', () => {
    const triggers: Record<string, EventTrigger> = {
      onTag: { Tag: { tag_types: [TagTypes.Files], required: { family: ['emotet'] }, not: {} } },
    };
    expect(formToTriggers(triggersToForm(triggers))).toEqual(triggers);
  });
});

describe('validateTriggers', () => {
  it('passes for valid unique triggers', () => {
    expect(validateTriggers([tagForm({ name: 'a' }), tagForm({ name: 'b' })])).toBe(false);
  });

  it('fails when a name is blank', () => {
    expect(validateTriggers([tagForm({ name: '' })])).toBe(true);
  });

  it('fails on duplicate names', () => {
    expect(validateTriggers([tagForm({ name: 'dup' }), tagForm({ name: 'dup' })])).toBe(true);
  });

  it('fails when a Tag trigger has no tag types', () => {
    expect(validateTriggers([tagForm({ tagTypes: [] })])).toBe(true);
  });

  it('passes a NewSample trigger with no tag types', () => {
    expect(validateTriggers([{ name: 'x', kind: TriggerKind.NewSample, tagTypes: [], required: [], not: [] }])).toBe(false);
  });
});

describe('triggerNameErrors', () => {
  it('returns Required for a blank name', () => {
    expect(triggerNameErrors([tagForm({ name: '' })])).toEqual(['Required']);
  });

  it('returns null for a valid unique name', () => {
    expect(triggerNameErrors([tagForm({ name: 'scan' })])).toEqual([null]);
  });

  it('flags both occurrences of a duplicate name', () => {
    expect(triggerNameErrors([tagForm({ name: 'dup' }), tagForm({ name: 'dup' })])).toEqual([
      'Duplicate trigger name',
      'Duplicate trigger name',
    ]);
  });

  it('reports errors per index independently', () => {
    expect(triggerNameErrors([tagForm({ name: 'ok' }), tagForm({ name: '' })])).toEqual([null, 'Required']);
  });
});
