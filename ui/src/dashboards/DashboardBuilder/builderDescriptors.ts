// spec: ./SPEC.md

// project imports
import { encodeSeedParams } from '../Dashboard/seedParams';
import { BuilderSelection, SelectionKind } from './types';
import { getUniqueFileNames } from '@utilities/files';
import { Entity, EntityMetaMap } from '@models/entities/entities';
import { Repo } from '@models/entities/repos';
import { Sample } from '@models/files';
import type { Seed } from '@models/trees';

/**
 * A tag key/value pair, the descriptor source for a {@link SelectionKind.Tag} selection.
 *
 * Tags have no single-object browse row (they come from `TagSelect` in the UI), so their descriptor
 * source is this explicit pair rather than an entity object.
 */
export interface TagSource {
  /// The tag key.
  key: string;
  /// The tag value.
  value: string;
}

/**
 * The source object `descriptorFor` accepts for each {@link SelectionKind}.
 *
 * Files come from a {@link Sample}, repos from a {@link Repo}, entities from an {@link Entity} of any
 * kind, and tags from a {@link TagSource} pair. The union keeps `descriptorFor` a single exhaustive
 * entry point over all selectable kinds.
 */
export type DescriptorSource = Sample | Repo | Entity<keyof EntityMetaMap> | TagSource;

/**
 * Build the human-readable display label for a seed resource from its identity and optional name.
 *
 * The single source of truth for how a selection chip reads, shared by {@link descriptorFor} (browsed
 * rows) and the dashboard entry-point button (linked-in seeds) so both render identically. Labels
 * always fall back to the raw identity so a chip is never blank:
 *
 * - `File` / `Repo` → the name when present, else the identity (sha256 / url).
 * - `Entity`        → `name (id)` so the id stays visible, else just the id when unnamed.
 * - `Tag`           → the identity, which is already the `key: value` display.
 *
 * # Arguments
 *
 * * `kind` - The selection kind the label is for.
 * * `identity` - The kind's seed identity (sha256 / uuid / url, or `key: value` for tags).
 * * `name` - The human-readable name to prefer, when available.
 *
 * @returns The display label.
 */
export function labelFor(kind: SelectionKind, identity: string, name?: string | null): string {
  switch (kind) {
    // files/repos: the human-readable name if we have one, else the raw identity
    case SelectionKind.File:
    case SelectionKind.Repo:
      return name && name.length > 0 ? name : identity;
    // entities: "name (id)" so the id stays visible, else just the id when unnamed
    case SelectionKind.Entity:
      return name && name.length > 0 ? `${name} (${identity})` : identity;
    // tags: the identity IS the display ("key: value"); the name arg is unused
    case SelectionKind.Tag:
      return identity;
  }
}

/**
 * Build a {@link BuilderSelection} from a browsed source object for a given kind.
 *
 * Extracts the kind's seed identifier and a human-readable label (via {@link labelFor}) with an
 * exhaustive `switch` over {@link SelectionKind} (adding a new kind without handling it is a compile
 * error). Labels always fall back to the identity so a chip is never blank:
 *
 * - `File`   → `sha256` + `getUniqueFileNames(sample.submissions)` (falls back to the sha256).
 * - `Repo`   → `url` + `repo.name` (falls back to the url).
 * - `Entity` → `id` + `name (id)` (falls back to just the id when unnamed).
 * - `Tag`    → `key`/`value` + `key: value`.
 *
 * @param kind - The kind of selection to build.
 * @param source - The source object matching `kind` (`Sample` / `Repo` / `Entity` / `TagSource`).
 * @returns The built selection.
 */
export function descriptorFor(kind: SelectionKind, source: DescriptorSource): BuilderSelection {
  switch (kind) {
    case SelectionKind.File: {
      // files: sha256 identity + derived filename label, falling back to the sha256 when empty
      const sample = source as Sample;
      const derived = getUniqueFileNames(sample.submissions ?? []);
      return { kind: SelectionKind.File, sha256: sample.sha256, label: labelFor(SelectionKind.File, sample.sha256, derived) };
    }
    case SelectionKind.Repo: {
      // repos: url identity + name label, falling back to the url
      const repo = source as Repo;
      return { kind: SelectionKind.Repo, url: repo.url, label: labelFor(SelectionKind.Repo, repo.url, repo.name) };
    }
    case SelectionKind.Entity: {
      // entities: uuid identity + a "name (id)" label so the id stays visible, falling back to just
      // the id when the entity is unnamed
      const entity = source as Entity<keyof EntityMetaMap>;
      return { kind: SelectionKind.Entity, id: entity.id, label: labelFor(SelectionKind.Entity, entity.id, entity.name) };
    }
    case SelectionKind.Tag: {
      // tags: key/value identity + "key: value" label
      const tag = source as TagSource;
      return { kind: SelectionKind.Tag, key: tag.key, value: tag.value, label: `${tag.key}: ${tag.value}` };
    }
  }
}

/**
 * Build a {@link Seed} from a set of builder selections and encode it (plus depth) into URL params.
 *
 * Groups the selections by kind into the `Seed` arrays (`samples`/`entities`/`repos`) and the
 * `Seed.tags` `key -> value[]` map, then delegates to `encodeSeedParams` so the URL contract lives in
 * exactly one place (the dashboard codec) and stays round-trippable with `decodeSeedParams`. An empty
 * selection list yields a valid, decodable params object carrying only `depth`.
 *
 * @param selections - The selected resources to encode.
 * @param depth - The crawl depth to encode (validated/clamped by `encodeSeedParams`).
 * @returns The encoded `URLSearchParams` for `/dashboard/view`.
 */
export function selectionsToSeedParams(selections: BuilderSelection[], depth: number): URLSearchParams {
  const seed: Seed = {};
  const samples: string[] = [];
  const entities: string[] = [];
  const repos: string[] = [];
  const tags: { [key: string]: string[] } = {};
  // fan each selection out into its seed field by kind
  for (const selection of selections) {
    switch (selection.kind) {
      case SelectionKind.File:
        samples.push(selection.sha256);
        break;
      case SelectionKind.Entity:
        entities.push(selection.id);
        break;
      case SelectionKind.Repo:
        repos.push(selection.url);
        break;
      case SelectionKind.Tag: {
        // accumulate values under their key so multiple values for one key coexist
        const existing = tags[selection.key] ?? [];
        existing.push(selection.value);
        tags[selection.key] = existing;
        break;
      }
    }
  }
  // only attach non-empty fields so the encoded params carry no empty keys
  if (samples.length > 0) {
    seed.samples = samples;
  }
  if (entities.length > 0) {
    seed.entities = entities;
  }
  if (repos.length > 0) {
    seed.repos = repos;
  }
  if (Object.keys(tags).length > 0) {
    seed.tags = tags;
  }
  return encodeSeedParams(seed, depth);
}
