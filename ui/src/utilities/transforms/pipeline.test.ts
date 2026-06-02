import { describe, test, expect } from 'vitest';

// project imports
import { pipelineToEditorObject, editorObjectToPipelineCreate, editorObjectToPipelineUpdate } from './pipeline';

describe('pipelineToEditorObject', () => {
  test('strips creator and bans', () => {
    const pipeline = {
      group: 'analysis',
      name: 'triage',
      creator: 'admin',
      bans: { 'uuid-1': { id: 'uuid-1', ban_kind: { Generic: { msg: 'test' } } } },
      order: [['step-a'], 'step-b'],
      sla: 604800,
    };
    const result = pipelineToEditorObject(pipeline);
    expect(result).not.toHaveProperty('creator');
    expect(result).not.toHaveProperty('bans');
  });

  test('preserves editable fields', () => {
    const pipeline = {
      group: 'analysis',
      name: 'triage',
      order: ['step-a'],
      sla: 86400,
      description: 'A pipeline',
      triggers: { 'new-upload': 'NewSample' },
    };
    const result = pipelineToEditorObject(pipeline);
    expect(result.group).toBe('analysis');
    expect(result.name).toBe('triage');
    expect(result.order).toEqual(['step-a']);
    expect(result.sla).toBe(86400);
    expect(result.description).toBe('A pipeline');
    expect(result.triggers).toEqual({ 'new-upload': 'NewSample' });
  });

  test('converts null description to empty string', () => {
    const pipeline = { group: 'test', name: 'p', order: ['a'], description: null };
    const result = pipelineToEditorObject(pipeline);
    expect(result.description).toBe('');
  });

  test('preserves non-null description', () => {
    const pipeline = { group: 'test', name: 'p', order: ['a'], description: 'Hello' };
    const result = pipelineToEditorObject(pipeline);
    expect(result.description).toBe('Hello');
  });

  test('returns empty object when pipeline has only read-only fields', () => {
    const pipeline = { creator: 'admin', bans: {} };
    const result = pipelineToEditorObject(pipeline);
    expect(Object.keys(result)).toHaveLength(0);
  });
});

describe('editorObjectToPipelineCreate', () => {
  test('returns null when group is missing', () => {
    expect(editorObjectToPipelineCreate({ name: 'test', order: ['a'] })).toBeNull();
  });

  test('returns null when name is missing', () => {
    expect(editorObjectToPipelineCreate({ group: 'analysis', order: ['a'] })).toBeNull();
  });

  test('returns null when order is missing', () => {
    expect(editorObjectToPipelineCreate({ group: 'analysis', name: 'test' })).toBeNull();
  });

  test('returns null when all required fields missing', () => {
    expect(editorObjectToPipelineCreate({})).toBeNull();
  });

  test('returns copy of object when all required fields present', () => {
    const obj = { group: 'analysis', name: 'test', order: ['step-a'], sla: 86400 };
    const result = editorObjectToPipelineCreate(obj);
    expect(result).toEqual(obj);
    expect(result).not.toBe(obj);
  });

  test('preserves optional fields', () => {
    const obj = {
      group: 'analysis',
      name: 'test',
      order: ['step-a'],
      sla: 3600,
      description: 'A pipeline',
      triggers: { auto: 'NewSample' },
    };
    const result = editorObjectToPipelineCreate(obj)!;
    expect(result.sla).toBe(3600);
    expect(result.description).toBe('A pipeline');
    expect(result.triggers).toEqual({ auto: 'NewSample' });
  });
});

describe('editorObjectToPipelineUpdate', () => {
  const ORIGINAL = {
    group: 'analysis',
    name: 'triage',
    order: ['step-a', 'step-b'],
    sla: 604800,
    description: 'Original description',
    triggers: { 'new-upload': 'NewSample' },
  };

  test('returns null when original has no group', () => {
    expect(editorObjectToPipelineUpdate({}, { name: 'test' })).toBeNull();
  });

  test('returns null when original has no name', () => {
    expect(editorObjectToPipelineUpdate({}, { group: 'analysis' })).toBeNull();
  });

  test('returns group and name from original pipeline', () => {
    const result = editorObjectToPipelineUpdate({ order: ['a'] }, ORIGINAL)!;
    expect(result.group).toBe('analysis');
    expect(result.name).toBe('triage');
  });

  describe('order diffs', () => {
    test('includes order when changed', () => {
      const obj = { ...ORIGINAL, order: ['step-c'] };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.order).toEqual(['step-c']);
    });

    test('omits order when unchanged', () => {
      const obj = { ...ORIGINAL };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data).not.toHaveProperty('order');
    });

    test('detects parallel order changes', () => {
      const obj = { ...ORIGINAL, order: [['step-a', 'step-b'], 'step-c'] };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.order).toEqual([['step-a', 'step-b'], 'step-c']);
    });
  });

  describe('sla diffs', () => {
    test('includes sla when changed', () => {
      const obj = { ...ORIGINAL, sla: 3600 };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.sla).toBe(3600);
    });

    test('omits sla when unchanged', () => {
      const result = editorObjectToPipelineUpdate({ ...ORIGINAL }, ORIGINAL)!;
      expect(result.data).not.toHaveProperty('sla');
    });
  });

  describe('description handling', () => {
    test('includes description when changed', () => {
      const obj = { ...ORIGINAL, description: 'Updated' };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.description).toBe('Updated');
    });

    test('omits description when unchanged', () => {
      const result = editorObjectToPipelineUpdate({ ...ORIGINAL }, ORIGINAL)!;
      expect(result.data).not.toHaveProperty('description');
      expect(result.data).not.toHaveProperty('clear_description');
    });

    test('sets clear_description when description removed and original had one', () => {
      const obj = { ...ORIGINAL, description: '' };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.clear_description).toBe(true);
      expect(result.data).not.toHaveProperty('description');
    });

    test('sets clear_description for whitespace-only description', () => {
      const obj = { ...ORIGINAL, description: '   ' };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.clear_description).toBe(true);
    });

    test('does not set clear_description when original had no description', () => {
      const original = { ...ORIGINAL, description: undefined };
      const obj = { ...original, description: '' };
      const result = editorObjectToPipelineUpdate(obj, original)!;
      expect(result.data).not.toHaveProperty('clear_description');
    });
  });

  describe('trigger diffs', () => {
    test('detects added triggers', () => {
      const obj = {
        ...ORIGINAL,
        triggers: { 'new-upload': 'NewSample', 'new-trigger': 'NewSample' },
      };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.triggers).toEqual({ 'new-trigger': 'NewSample' });
      expect(result.data).not.toHaveProperty('remove_triggers');
    });

    test('detects removed triggers', () => {
      const obj = { ...ORIGINAL, triggers: {} };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.remove_triggers).toEqual(['new-upload']);
      expect(result.data).not.toHaveProperty('triggers');
    });

    test('detects changed trigger values', () => {
      const obj = {
        ...ORIGINAL,
        triggers: {
          'new-upload': {
            Tag: { tag_types: ['Files'], required: {}, not: {} },
          },
        },
      };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.triggers).toHaveProperty('new-upload');
    });

    test('no trigger diffs when triggers unchanged', () => {
      const result = editorObjectToPipelineUpdate({ ...ORIGINAL }, ORIGINAL)!;
      expect(result.data).not.toHaveProperty('triggers');
      expect(result.data).not.toHaveProperty('remove_triggers');
    });

    test('handles absent triggers on both sides', () => {
      const original = { group: 'g', name: 'n', order: ['a'] };
      const obj = { order: ['a'] };
      const result = editorObjectToPipelineUpdate(obj, original)!;
      expect(result.data).not.toHaveProperty('triggers');
      expect(result.data).not.toHaveProperty('remove_triggers');
    });
  });

  describe('combined update', () => {
    test('handles all diff types in single update', () => {
      const obj = {
        order: ['new-step'],
        sla: 1800,
        description: 'New desc',
        triggers: { added: 'NewSample' },
      };
      const result = editorObjectToPipelineUpdate(obj, ORIGINAL)!;
      expect(result.data.order).toEqual(['new-step']);
      expect(result.data.sla).toBe(1800);
      expect(result.data.description).toBe('New desc');
      expect(result.data.triggers).toEqual({ added: 'NewSample' });
      expect(result.data.remove_triggers).toEqual(['new-upload']);
    });

    test('empty data when nothing changed', () => {
      const result = editorObjectToPipelineUpdate({ ...ORIGINAL }, ORIGINAL)!;
      expect(Object.keys(result.data)).toHaveLength(0);
    });
  });
});
