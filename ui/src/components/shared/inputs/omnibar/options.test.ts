import { describe, it, expect } from 'vitest';

// project imports
import { ClauseCondition } from './ClauseTypes';
import { addDepthOptions, addEntityLayerOptions, addTagOptions, ENTITY_LAYER_CATEGORIES, type OmnibarOptionMap } from './options';

describe('addEntityLayerOptions', () => {
  it('returns the map unchanged when there are no kinds', () => {
    const map: OmnibarOptionMap = {};
    expect(addEntityLayerOptions(map, [])).toBe(map);
  });

  it('adds all four layer categories with Is/IsOneOf conditions and creatable=false', () => {
    const kinds = ['Folder', 'Device'];
    const next = addEntityLayerOptions({}, kinds);
    expect(Object.keys(next).sort()).toEqual([...ENTITY_LAYER_CATEGORIES].sort());
    for (const category of ENTITY_LAYER_CATEGORIES) {
      const field = next[category].fields[category];
      expect(field.values).toEqual(kinds);
      expect(field.conditions).toEqual([ClauseCondition.Is, ClauseCondition.IsOneOf]);
      expect(field.creatable).toBe(false);
      expect(field.category).toBe(category);
    }
  });

  it('does not mutate the input map', () => {
    const map: OmnibarOptionMap = {};
    addEntityLayerOptions(map, ['Folder']);
    expect(map).toEqual({});
  });
});

describe('addDepthOptions', () => {
  it('adds a creatable depth field of "1".."maxDepth" (default 10)', () => {
    const field = addDepthOptions({}).depth.fields.depth;
    expect(field.values).toEqual(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']);
    expect(field.conditions).toEqual([ClauseCondition.Is]);
    expect(field.creatable).toBe(true);
    expect(field.category).toBe('depth');
  });

  it('honors a custom maxDepth', () => {
    expect(addDepthOptions({}, 3).depth.fields.depth.values).toEqual(['1', '2', '3']);
  });

  it('does not mutate the input map', () => {
    const map: OmnibarOptionMap = {};
    addDepthOptions(map);
    expect(map).toEqual({});
  });
});

describe('addTagOptions', () => {
  it('offers IsOneOf alongside Is so merged is-one-of tag clauses render/edit in the omnibar', () => {
    const field = addTagOptions({}, { FileType: ['PE32', 'ELF'] }).tag.fields.FileType;
    expect(field.conditions).toEqual([ClauseCondition.Is, ClauseCondition.IsOneOf]);
    expect(field.values).toEqual(['PE32', 'ELF']);
    expect(field.creatable).toBe(true);
  });
});
