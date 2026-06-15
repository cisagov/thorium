// spec: ./SPEC.md

// project imports
import { labelFor } from './builderDescriptors';
import { decodeSeedParams } from '../Dashboard/seedParams';
import { BuilderAction, BuilderActionKind, BuilderSelection, BuilderState, SelectionKind } from './types';

/// The URL param carrying a selection's identity key for its display label (paired with {@link LABEL_VALUE_PARAM}).
export const LABEL_KEY_PARAM = 'lk';
/// The URL param carrying a selection's display label (paired with {@link LABEL_KEY_PARAM}).
export const LABEL_VALUE_PARAM = 'lv';

/// Delimiter separating a tag selection's key from its value in {@link selectionKey}. A NUL byte
/// (`String.fromCharCode(0)`, kept as an expression so no literal NUL sits in the source and the file
/// stays text/diffable) can never appear in user-entered tag input, so it keeps distinct key/value splits
/// unambiguous — `{key:'a', value:'b c'}` and `{key:'a b', value:'c'}` produce different identity keys.
const TAG_KEY_VALUE_DELIMITER = String.fromCharCode(0);

/**
 * The empty starting state for the builder: no selections and no removed items.
 */
export const DEFAULT_BUILDER_STATE: BuilderState = {
  selected: [],
  removed: [],
};

/**
 * Compute the identity key for a selection.
 *
 * Identity is by kind + seed identifier (sha256 / uuid / url / key+value), NEVER by label, so two
 * selections that point at the same resource with different display labels are treated as equal for
 * dedup, remove, and re-add. The `kind` prefix prevents cross-kind collisions (e.g. a repo url that
 * happens to equal an entity name).
 *
 * @param selection - The selection to key.
 * @returns A stable string uniquely identifying the selection's resource.
 */
export function selectionKey(selection: BuilderSelection): string {
  switch (selection.kind) {
    // file identity is its sha256
    case SelectionKind.File:
      return `${SelectionKind.File}:${selection.sha256}`;
    // entity identity is its uuid
    case SelectionKind.Entity:
      return `${SelectionKind.Entity}:${selection.id}`;
    // repo identity is its full url
    case SelectionKind.Repo:
      return `${SelectionKind.Repo}:${selection.url}`;
    // tag identity is key + value, joined by a NUL delimiter that cannot appear in tag input so distinct
    // key/value splits never collide (see TAG_KEY_VALUE_DELIMITER)
    case SelectionKind.Tag:
      return `${SelectionKind.Tag}:${selection.key}${TAG_KEY_VALUE_DELIMITER}${selection.value}`;
  }
}

/**
 * Encode the display labels of a set of selections as paired URL params.
 *
 * Emits one `LABEL_KEY_PARAM`/`LABEL_VALUE_PARAM` pair per selection, in lockstep order, so the
 * builder's chips stay human-readable across refresh and sharing (the seed params alone carry only
 * raw ids). The keys are selection identity keys ({@link selectionKey}) so a label re-attaches to the
 * right chip on hydration regardless of order.
 *
 * @param selections - The selections whose labels to encode.
 * @returns Params carrying the label pairs (empty when there are no selections).
 */
export function selectionsToLabelParams(selections: BuilderSelection[]): URLSearchParams {
  const params = new URLSearchParams();
  for (const selection of selections) {
    params.append(LABEL_KEY_PARAM, selectionKey(selection));
    params.append(LABEL_VALUE_PARAM, selection.label);
  }
  return params;
}

/**
 * Build the display-label URL params for a single seed resource linked in from a details page.
 *
 * Lets the dashboard entry-point button pre-seed the builder with a human-readable chip (e.g.
 * `laptop-1 (uuid)`) instead of a bare id: it emits the same `LABEL_KEY_PARAM`/`LABEL_VALUE_PARAM`
 * pair the builder writes for its own chips, keyed by the resource's selection identity and labelled
 * via the shared {@link labelFor}. Restricted to the seedable single-object kinds (files, entities,
 * repos); tags are entered in-builder, never linked in.
 *
 * # Arguments
 *
 * * `kind` - The selection kind of the linked-in resource.
 * * `identity` - The resource's seed identity (sha256 / uuid / url).
 * * `name` - The resource's human-readable name, when known.
 *
 * @returns Params carrying the single label pair.
 */
export function seedLabelParams(
  kind: SelectionKind.File | SelectionKind.Entity | SelectionKind.Repo,
  identity: string,
  name?: string | null,
): URLSearchParams {
  // build the matching selection so its identity key is derived the same way the builder derives it
  let selection: BuilderSelection;
  switch (kind) {
    case SelectionKind.File:
      selection = { kind, sha256: identity, label: labelFor(kind, identity, name) };
      break;
    case SelectionKind.Entity:
      selection = { kind, id: identity, label: labelFor(kind, identity, name) };
      break;
    case SelectionKind.Repo:
      selection = { kind, url: identity, label: labelFor(kind, identity, name) };
      break;
  }
  const params = new URLSearchParams();
  params.append(LABEL_KEY_PARAM, selectionKey(selection));
  params.append(LABEL_VALUE_PARAM, selection.label);
  return params;
}

/**
 * Build a selection-key → display-label map from the paired label params on a URL.
 *
 * The inverse of {@link selectionsToLabelParams}: `LABEL_KEY_PARAM` and `LABEL_VALUE_PARAM` are read in
 * insertion order (`getAll` preserves it) and zipped by index, so each label re-attaches to its chip.
 *
 * Exported for direct unit testing of the zip guard (per ui/CLAUDE.md testability rule).
 *
 * @param params - The URL search params to read the label pairs from.
 * @returns A map from selection identity key to display label.
 */
export function labelMapFromParams(params: URLSearchParams): Map<string, string> {
  const keys = params.getAll(LABEL_KEY_PARAM);
  const values = params.getAll(LABEL_VALUE_PARAM);
  const map = new Map<string, string>();
  // lk/lv are appended in lockstep, so equal-index entries pair up
  for (let i = 0; i < keys.length && i < values.length; i += 1) {
    map.set(keys[i], values[i]);
  }
  return map;
}

/**
 * Build the list of selections from a decoded {@link import('../Dashboard/seedParams').decodeSeedParams}
 * result.
 *
 * Each selection starts with its seed identity as a placeholder label, then adopts the richer label
 * carried by the URL's paired label params ({@link labelMapFromParams}) when one is present — so a chip
 * linked in from an entity page reads e.g. `laptop-1 (uuid)` instead of a bare uuid. Hydration needs no
 * async lookups.
 *
 * Exported for direct unit testing (per ui/CLAUDE.md testability rule).
 *
 * @param params - The URL search params to decode via the dashboard codec.
 * @returns The selections represented by the params, in a stable kind order.
 */
export function selectionsFromParams(params: URLSearchParams): BuilderSelection[] {
  const { seed } = decodeSeedParams(params);
  const labels = labelMapFromParams(params);
  const selections: BuilderSelection[] = [];
  // apply the URL-carried display label (if any) to a freshly built selection
  const push = (selection: BuilderSelection) => {
    const label = labels.get(selectionKey(selection));
    selections.push(label ? { ...selection, label } : selection);
  };
  // files
  for (const sha256 of seed.samples ?? []) {
    push({ kind: SelectionKind.File, sha256, label: sha256 });
  }
  // entities (any kind incl. devices)
  for (const id of seed.entities ?? []) {
    push({ kind: SelectionKind.Entity, id, label: id });
  }
  // repos
  for (const url of seed.repos ?? []) {
    push({ kind: SelectionKind.Repo, url, label: url });
  }
  // tags: one selection per (key, value)
  if (seed.tags) {
    for (const key of Object.keys(seed.tags)) {
      for (const value of seed.tags[key]) {
        push({ kind: SelectionKind.Tag, key, value, label: `${key}: ${value}` });
      }
    }
  }
  return selections;
}

/**
 * The pure reducer driving the builder's selection state.
 *
 * Exported so it can be unit-tested directly (per ui/CLAUDE.md testability rule). See the SPEC for the
 * full behavior contract; in brief: Add is idempotent and resurrects removed items, Remove parks an
 * item in `removed`, Readd restores it, HydrateFromParams replaces `selected` from URL params, Clear
 * resets to the empty state, and an unknown action returns the state unchanged.
 *
 * @param state - The current builder state.
 * @param action - The action to apply.
 * @returns The next builder state (a new object when anything changed, else the same reference).
 */
export function builderReducer(state: BuilderState, action: BuilderAction): BuilderState {
  switch (action.type) {
    case BuilderActionKind.Add: {
      const key = selectionKey(action.selection);
      // dup-add is a no-op so the same row can be "added" repeatedly without duplicating
      if (state.selected.some((s) => selectionKey(s) === key)) {
        return state;
      }
      // adding an item that was removed resurrects it: drop it from `removed` and append to `selected`
      const removed = state.removed.filter((s) => selectionKey(s) !== key);
      return { selected: [...state.selected, action.selection], removed };
    }
    case BuilderActionKind.Remove: {
      const key = selectionKey(action.selection);
      // no-op if the item is not currently selected
      if (!state.selected.some((s) => selectionKey(s) === key)) {
        return state;
      }
      // move the selected item into `removed`, keeping the removed copy for re-add
      const selected = state.selected.filter((s) => selectionKey(s) !== key);
      const target = state.selected.find((s) => selectionKey(s) === key) as BuilderSelection;
      // guard against a stale duplicate already sitting in `removed`
      const removed = state.removed.some((s) => selectionKey(s) === key) ? state.removed : [...state.removed, target];
      return { selected, removed };
    }
    case BuilderActionKind.Readd: {
      const key = selectionKey(action.selection);
      // no-op if the item is not in the removed list
      if (!state.removed.some((s) => selectionKey(s) === key)) {
        return state;
      }
      // move the removed item back into `selected`
      const target = state.removed.find((s) => selectionKey(s) === key) as BuilderSelection;
      const removed = state.removed.filter((s) => selectionKey(s) !== key);
      // avoid a duplicate if the item is somehow already selected
      const selected = state.selected.some((s) => selectionKey(s) === key) ? state.selected : [...state.selected, target];
      return { selected, removed };
    }
    case BuilderActionKind.HydrateFromParams:
      // deep-link / entry-point hydration: replace selections from the URL and clear removed history
      return { selected: selectionsFromParams(action.params), removed: [] };
    case BuilderActionKind.Clear:
      return DEFAULT_BUILDER_STATE;
    default:
      // unknown action: leave state untouched (exhaustiveness guard for future action kinds)
      return state;
  }
}
