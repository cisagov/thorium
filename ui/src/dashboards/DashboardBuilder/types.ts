// spec: ./SPEC.md

// project imports
import { Entities } from '@models/entities/entities';

/**
 * The kind of a dashboard-builder selection.
 *
 * Discriminant for {@link BuilderSelection}. Each variant carries a different seed identifier
 * (sha256 / entity uuid / repo url / tag key+value) that maps into a distinct field of `Seed`.
 * A string enum (not a union) so `descriptorFor`/`selectionsToSeedParams` get exhaustiveness
 * checking and autocomplete.
 */
export enum SelectionKind {
  /// A file seed, identified by its sha256 (maps to `Seed.samples`).
  File = 'File',
  /// An entity seed of any kind incl. devices, identified by its uuid (maps to `Seed.entities`).
  Entity = 'Entity',
  /// A repo seed, identified by its full url (maps to `Seed.repos`).
  Repo = 'Repo',
  /// A tag seed, identified by its key + value (maps to `Seed.tags`).
  Tag = 'Tag',
}

/**
 * A single file selection in the builder.
 *
 * `sha256` is the seed identity used for dedup and encoding; `label` is the human-readable display
 * (built via `getUniqueFileNames`) shown as a chip in the selection panel.
 */
export interface FileSelection {
  /// The discriminant marking this as a file selection.
  kind: SelectionKind.File;
  /// The file's sha256 — its seed identity.
  sha256: string;
  /// The human-readable display label (derived filename, falling back to the sha256).
  label: string;
}

/**
 * A single entity selection in the builder (any entity kind, including devices).
 *
 * `id` is the entity uuid used for dedup and encoding; `label` is the entity `name`.
 */
export interface EntitySelection {
  /// The discriminant marking this as an entity selection.
  kind: SelectionKind.Entity;
  /// The entity's uuid — its seed identity.
  id: string;
  /// The human-readable display label (the entity name, falling back to the uuid).
  label: string;
}

/**
 * A single repo selection in the builder.
 *
 * `url` is the full repo url used for dedup and encoding (matches `Seed.repos` and `/repo/*` routes);
 * `label` is the repo `name`.
 */
export interface RepoSelection {
  /// The discriminant marking this as a repo selection.
  kind: SelectionKind.Repo;
  /// The repo's full url — its seed identity.
  url: string;
  /// The human-readable display label (the repo name, falling back to the url).
  label: string;
}

/**
 * A single tag selection in the builder.
 *
 * `key`/`value` together are the seed identity and map into `Seed.tags` (`key -> [value]`);
 * `label` is `key: value` for display.
 */
export interface TagSelection {
  /// The discriminant marking this as a tag selection.
  kind: SelectionKind.Tag;
  /// The tag key.
  key: string;
  /// The tag value.
  value: string;
  /// The human-readable display label (`key: value`).
  label: string;
}

/**
 * A resource chosen in the builder to seed the dashboard graph.
 *
 * A discriminated union over {@link SelectionKind}; each variant carries the seed identifier its
 * kind needs plus a stable display `label`. Selection identity (for dedup / remove / re-add) is by
 * kind + seed identifier, never by label — see {@link selectionKey}.
 */
export type BuilderSelection = FileSelection | EntitySelection | RepoSelection | TagSelection;

/**
 * The pure state of the builder's selection machine.
 *
 * `selected` is the ordered list of chosen resources. `removed` holds items the user removed and is
 * the ONLY source of re-add options offered by the selection panel (accidental-removal undo), so the
 * re-add dropdown never lists the whole catalog.
 */
export interface BuilderState {
  /// The ordered list of currently selected resources.
  selected: BuilderSelection[];
  /// Previously removed resources, offered as the sole re-add options.
  removed: BuilderSelection[];
}

/**
 * The kinds of actions the {@link builderReducer} handles.
 *
 * A string enum so the reducer's `switch` is exhaustiveness-checked.
 */
export enum BuilderActionKind {
  /// Add a selection (no-op if already selected; resurrects it from `removed` if present).
  Add = 'Add',
  /// Move a selection from `selected` into `removed`.
  Remove = 'Remove',
  /// Move a selection from `removed` back into `selected` (undo a removal).
  Readd = 'Readd',
  /// Replace `selected` with selections decoded from URL seed params; clears `removed`.
  HydrateFromParams = 'HydrateFromParams',
  /// Reset to {@link DEFAULT_BUILDER_STATE}.
  Clear = 'Clear',
}

/**
 * Add a selection to the builder.
 *
 * Idempotent for an already-selected item; if the item is in `removed`, it is moved back to `selected`.
 */
export interface AddAction {
  /// The action discriminant.
  type: BuilderActionKind.Add;
  /// The selection to add.
  selection: BuilderSelection;
}

/**
 * Remove a selection from the builder, moving it into `removed`.
 */
export interface RemoveAction {
  /// The action discriminant.
  type: BuilderActionKind.Remove;
  /// The selection to remove.
  selection: BuilderSelection;
}

/**
 * Re-add a previously removed selection (undo a removal).
 */
export interface ReaddAction {
  /// The action discriminant.
  type: BuilderActionKind.Readd;
  /// The removed selection to restore.
  selection: BuilderSelection;
}

/**
 * Hydrate the builder's `selected` list from URL seed params.
 *
 * Used by deep links and details-page entry points; the params are decoded via the dashboard codec.
 */
export interface HydrateFromParamsAction {
  /// The action discriminant.
  type: BuilderActionKind.HydrateFromParams;
  /// The URL search params to decode selections from.
  params: URLSearchParams;
}

/**
 * Reset the builder to its default (empty) state.
 */
export interface ClearAction {
  /// The action discriminant.
  type: BuilderActionKind.Clear;
}

/**
 * The union of all actions accepted by {@link builderReducer}.
 */
export type BuilderAction = AddAction | RemoveAction | ReaddAction | HydrateFromParamsAction | ClearAction;

/**
 * A special value in the resource-type picker denoting Tag mode (rather than an entity kind).
 *
 * Tag mode has no browse list; it presents a key/value entry via `TagSelect`. Modeled as its own
 * string constant (distinct from every {@link Entities} value) so the picker's value is a simple
 * discriminated string.
 */
export const TAG_MODE = 'Tag' as const;

/**
 * The value the resource-type picker resolves to: either an entity kind to browse or Tag mode.
 *
 * `Entities.File` / `Entities.Repo` / the other entity kinds drive the config-driven browse list;
 * {@link TAG_MODE} switches to the tag key/value entry instead.
 */
export type BrowseMode = Entities | typeof TAG_MODE;

/**
 * The entity kinds the builder exposes for browsing in the resource-type picker, in display order.
 *
 * A curated subset of {@link Entities}: File and Repo first (the common seeds), then the entity
 * kinds that have browsing configs and make sense as dashboard seeds. Kept as an explicit list so
 * the picker order is stable and intentional rather than derived from object-key order.
 */
export const BROWSABLE_KINDS: Entities[] = [
  Entities.File,
  Entities.Repo,
  Entities.Device,
  Entities.Vendor,
  Entities.NetworkConnection,
  Entities.WindowsProcess,
  Entities.WindowsProcessTree,
  Entities.SigmaRule,
  Entities.Flag,
  Entities.Incident,
  Entities.Other,
];
