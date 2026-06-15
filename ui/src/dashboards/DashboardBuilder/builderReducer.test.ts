import { describe, expect, it } from 'vitest';

// project imports
import {
  DEFAULT_BUILDER_STATE,
  LABEL_KEY_PARAM,
  LABEL_VALUE_PARAM,
  builderReducer,
  labelMapFromParams,
  seedLabelParams,
  selectionKey,
  selectionsFromParams,
  selectionsToLabelParams,
} from './builderReducer';
import { selectionsToSeedParams } from './builderDescriptors';
import { BuilderAction, BuilderActionKind, BuilderSelection, BuilderState, SelectionKind } from './types';

/// A sample file selection used across tests.
const FILE_A: BuilderSelection = { kind: SelectionKind.File, sha256: 'a'.repeat(64), label: 'malware.exe' };
/// A second file selection sharing the identity of FILE_A but with a different label (same resource).
const FILE_A_RELABELED: BuilderSelection = { kind: SelectionKind.File, sha256: 'a'.repeat(64), label: 'renamed.exe' };
/// A device entity selection.
const ENTITY_A: BuilderSelection = { kind: SelectionKind.Entity, id: 'uuid-1', label: 'laptop-1' };
/// A repo selection.
const REPO_A: BuilderSelection = { kind: SelectionKind.Repo, url: 'https://github.com/x/y', label: 'y' };
/// A tag selection.
const TAG_A: BuilderSelection = { kind: SelectionKind.Tag, key: 'FileType', value: 'PE32', label: 'FileType: PE32' };

/**
 * Build a state variant without repetition.
 *
 * @param patch - The partial state to overlay onto the default.
 * @returns A fresh state with the patch applied.
 */
function stateWith(patch: Partial<BuilderState>): BuilderState {
  return { ...structuredClone(DEFAULT_BUILDER_STATE), ...patch };
}

describe('selectionKey', () => {
  it('keys by identity, not label', () => {
    expect(selectionKey(FILE_A)).toBe(selectionKey(FILE_A_RELABELED));
  });
  it('distinguishes kinds and identities', () => {
    expect(selectionKey(FILE_A)).not.toBe(selectionKey(ENTITY_A));
    expect(selectionKey(REPO_A)).not.toBe(selectionKey(TAG_A));
  });
  it('distinguishes tags by key + value', () => {
    const other: BuilderSelection = { kind: SelectionKind.Tag, key: 'FileType', value: 'ELF', label: 'FileType: ELF' };
    expect(selectionKey(TAG_A)).not.toBe(selectionKey(other));
  });
  it('keeps key/value splits unambiguous when either half contains spaces', () => {
    // a space delimiter would collapse these two distinct tags to the same key ('a b c'); the NUL delimiter
    // keeps them distinct
    const a: BuilderSelection = { kind: SelectionKind.Tag, key: 'a', value: 'b c', label: 'a: b c' };
    const b: BuilderSelection = { kind: SelectionKind.Tag, key: 'a b', value: 'c', label: 'a b: c' };
    expect(selectionKey(a)).not.toBe(selectionKey(b));
  });
});

describe('builderReducer — Add', () => {
  it('appends a new selection', () => {
    const next = builderReducer(DEFAULT_BUILDER_STATE, { type: BuilderActionKind.Add, selection: FILE_A });
    expect(next.selected).toEqual([FILE_A]);
    expect(next.removed).toEqual([]);
  });
  it('is a no-op (same reference) when the item is already selected', () => {
    const start = stateWith({ selected: [FILE_A] });
    const next = builderReducer(start, { type: BuilderActionKind.Add, selection: FILE_A_RELABELED });
    expect(next).toBe(start);
    expect(next.selected).toHaveLength(1);
  });
  it('resurrects an item that was in removed', () => {
    const start = stateWith({ selected: [ENTITY_A], removed: [FILE_A] });
    const next = builderReducer(start, { type: BuilderActionKind.Add, selection: FILE_A });
    expect(next.removed).toEqual([]);
    expect(next.selected).toEqual([ENTITY_A, FILE_A]);
  });
});

describe('builderReducer — Remove', () => {
  it('moves a selected item into removed', () => {
    const start = stateWith({ selected: [FILE_A, ENTITY_A] });
    const next = builderReducer(start, { type: BuilderActionKind.Remove, selection: FILE_A });
    expect(next.selected).toEqual([ENTITY_A]);
    expect(next.removed).toEqual([FILE_A]);
  });
  it('is a no-op when the item is not selected', () => {
    const start = stateWith({ selected: [ENTITY_A] });
    const next = builderReducer(start, { type: BuilderActionKind.Remove, selection: FILE_A });
    expect(next).toBe(start);
  });
  it('does not duplicate an item already in removed', () => {
    const start = stateWith({ selected: [FILE_A], removed: [FILE_A_RELABELED] });
    const next = builderReducer(start, { type: BuilderActionKind.Remove, selection: FILE_A });
    expect(next.removed).toHaveLength(1);
    expect(next.selected).toEqual([]);
  });
});

describe('builderReducer — Readd', () => {
  it('restores a removed item to selected', () => {
    const start = stateWith({ selected: [ENTITY_A], removed: [FILE_A] });
    const next = builderReducer(start, { type: BuilderActionKind.Readd, selection: FILE_A });
    expect(next.selected).toEqual([ENTITY_A, FILE_A]);
    expect(next.removed).toEqual([]);
  });
  it('is a no-op when the item is not in removed', () => {
    const start = stateWith({ selected: [ENTITY_A] });
    const next = builderReducer(start, { type: BuilderActionKind.Readd, selection: FILE_A });
    expect(next).toBe(start);
  });
});

describe('builderReducer — remove then re-add round-trip', () => {
  it('restores the exact selection after a remove', () => {
    let state: BuilderState = stateWith({ selected: [FILE_A, ENTITY_A] });
    state = builderReducer(state, { type: BuilderActionKind.Remove, selection: FILE_A });
    expect(state.removed).toEqual([FILE_A]);
    state = builderReducer(state, { type: BuilderActionKind.Readd, selection: FILE_A });
    expect(state.selected).toEqual([ENTITY_A, FILE_A]);
    expect(state.removed).toEqual([]);
  });
});

describe('builderReducer — HydrateFromParams', () => {
  it('populates selected from seed params and clears removed', () => {
    const params = new URLSearchParams();
    params.append('sample', 'a'.repeat(64));
    params.append('entity', 'uuid-1');
    params.append('repo', 'https://github.com/x/y');
    params.append('tag', `${encodeURIComponent('FileType')}:${encodeURIComponent('PE32')}`);
    const start = stateWith({ selected: [ENTITY_A], removed: [REPO_A] });
    const next = builderReducer(start, { type: BuilderActionKind.HydrateFromParams, params });
    expect(next.removed).toEqual([]);
    const keys = next.selected.map(selectionKey);
    expect(keys).toContain(selectionKey(FILE_A));
    expect(keys).toContain(selectionKey(ENTITY_A));
    expect(keys).toContain(selectionKey(REPO_A));
    expect(keys).toContain(selectionKey(TAG_A));
    expect(next.selected).toHaveLength(4);
  });
  it('hydrates to an empty selection when params carry no resources', () => {
    const next = builderReducer(stateWith({ selected: [FILE_A] }), {
      type: BuilderActionKind.HydrateFromParams,
      params: new URLSearchParams('depth=3'),
    });
    expect(next.selected).toEqual([]);
    expect(next.removed).toEqual([]);
  });
});

describe('display-label URL params', () => {
  it('round-trips chip labels through hydration, keyed by identity not order', () => {
    // encode the seed ids AND the display labels, then merge into one param set as the builder does
    const selections: BuilderSelection[] = [FILE_A, ENTITY_A, REPO_A, TAG_A];
    const params = selectionsToSeedParams(selections, 2);
    selectionsToLabelParams(selections).forEach((value, key) => params.append(key, value));
    const next = builderReducer(DEFAULT_BUILDER_STATE, { type: BuilderActionKind.HydrateFromParams, params });
    // each hydrated chip carries its rich label rather than the raw id
    const byKey = new Map(next.selected.map((s) => [selectionKey(s), s.label]));
    expect(byKey.get(selectionKey(FILE_A))).toBe('malware.exe');
    expect(byKey.get(selectionKey(ENTITY_A))).toBe('laptop-1');
    expect(byKey.get(selectionKey(REPO_A))).toBe('y');
    expect(byKey.get(selectionKey(TAG_A))).toBe('FileType: PE32');
  });
  it('falls back to the seed id when no label param is present', () => {
    const params = new URLSearchParams();
    params.append('entity', 'uuid-1');
    const next = builderReducer(DEFAULT_BUILDER_STATE, { type: BuilderActionKind.HydrateFromParams, params });
    expect(next.selected).toEqual([{ kind: SelectionKind.Entity, id: 'uuid-1', label: 'uuid-1' }]);
  });
});

describe('seedLabelParams', () => {
  it('builds an identity-keyed "name (id)" label for an entity link', () => {
    const params = seedLabelParams(SelectionKind.Entity, 'uuid-1', 'laptop-1');
    // hydrating from the entity seed + these label params yields the "name (id)" chip
    params.append('entity', 'uuid-1');
    const next = builderReducer(DEFAULT_BUILDER_STATE, { type: BuilderActionKind.HydrateFromParams, params });
    expect(next.selected).toEqual([{ kind: SelectionKind.Entity, id: 'uuid-1', label: 'laptop-1 (uuid-1)' }]);
  });
  it('falls back to the identity when the name is empty', () => {
    const params = seedLabelParams(SelectionKind.Entity, 'uuid-1', '');
    expect(params.get('lv')).toBe('uuid-1');
  });
});

describe('labelMapFromParams', () => {
  it('zips equal-length key/value params into a map', () => {
    const params = new URLSearchParams();
    params.append(LABEL_KEY_PARAM, selectionKey(FILE_A));
    params.append(LABEL_VALUE_PARAM, 'malware.exe');
    params.append(LABEL_KEY_PARAM, selectionKey(ENTITY_A));
    params.append(LABEL_VALUE_PARAM, 'laptop-1');
    const map = labelMapFromParams(params);
    expect(map.get(selectionKey(FILE_A))).toBe('malware.exe');
    expect(map.get(selectionKey(ENTITY_A))).toBe('laptop-1');
    expect(map.size).toBe(2);
  });
  it('drops a trailing key that has no paired value (mismatched lengths)', () => {
    // an extra lk without a matching lv must not produce a map entry for the unpaired key, nor throw
    const params = new URLSearchParams();
    params.append(LABEL_KEY_PARAM, selectionKey(FILE_A));
    params.append(LABEL_VALUE_PARAM, 'malware.exe');
    params.append(LABEL_KEY_PARAM, selectionKey(ENTITY_A));
    const map = labelMapFromParams(params);
    expect(map.get(selectionKey(FILE_A))).toBe('malware.exe');
    expect(map.has(selectionKey(ENTITY_A))).toBe(false);
    expect(map.size).toBe(1);
  });
  it('drops a trailing value that has no paired key (mismatched lengths)', () => {
    const params = new URLSearchParams();
    params.append(LABEL_KEY_PARAM, selectionKey(FILE_A));
    params.append(LABEL_VALUE_PARAM, 'malware.exe');
    params.append(LABEL_VALUE_PARAM, 'orphan-label');
    const map = labelMapFromParams(params);
    expect(map.get(selectionKey(FILE_A))).toBe('malware.exe');
    expect(map.size).toBe(1);
  });
  it('returns an empty map when no label params are present', () => {
    expect(labelMapFromParams(new URLSearchParams('depth=3')).size).toBe(0);
  });
});

describe('selectionsFromParams', () => {
  it('builds selections across all kinds with seed ids as fallback labels', () => {
    const params = new URLSearchParams();
    params.append('sample', 'a'.repeat(64));
    params.append('entity', 'uuid-1');
    params.append('repo', 'https://github.com/x/y');
    params.append('tag', `${encodeURIComponent('FileType')}:${encodeURIComponent('PE32')}`);
    const selections = selectionsFromParams(params);
    const byKey = new Map(selections.map((s) => [selectionKey(s), s.label]));
    // no label params, so each chip falls back to its raw seed identity
    expect(byKey.get(selectionKey(FILE_A))).toBe('a'.repeat(64));
    expect(byKey.get(selectionKey(ENTITY_A))).toBe('uuid-1');
    expect(byKey.get(selectionKey(REPO_A))).toBe('https://github.com/x/y');
    expect(byKey.get(selectionKey(TAG_A))).toBe('FileType: PE32');
    expect(selections).toHaveLength(4);
  });
  it('adopts the richer label from paired label params when present', () => {
    const params = new URLSearchParams();
    params.append('entity', 'uuid-1');
    params.append(LABEL_KEY_PARAM, selectionKey(ENTITY_A));
    params.append(LABEL_VALUE_PARAM, 'laptop-1');
    const selections = selectionsFromParams(params);
    expect(selections).toEqual([ENTITY_A]);
  });
  it('returns an empty list when params carry no resources', () => {
    expect(selectionsFromParams(new URLSearchParams('depth=3'))).toEqual([]);
  });
});

describe('builderReducer — Clear', () => {
  it('resets to the default state', () => {
    const start = stateWith({ selected: [FILE_A], removed: [ENTITY_A] });
    const next = builderReducer(start, { type: BuilderActionKind.Clear });
    expect(next).toEqual(DEFAULT_BUILDER_STATE);
  });
});

describe('builderReducer — unknown action', () => {
  it('returns state unchanged', () => {
    const start = stateWith({ selected: [FILE_A] });
    // cast through unknown: an unrecognized action can't be constructed from the typed union
    const next = builderReducer(start, { type: 'Bogus' } as unknown as BuilderAction);
    expect(next).toBe(start);
  });
});
