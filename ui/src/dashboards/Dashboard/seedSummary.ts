// spec: ./SPEC.md

// project imports
import { getUniqueFileNames } from '@utilities/files';
import { TreeNode, TreeNodeKey, type Seed } from '@models/trees';

/// The number of leading characters shown for a sha256/id when no human-readable name resolves.
const SHORT_ID_LENGTH = 12;

/**
 * The kind of a resolved seed-summary item, used to label/group the human-readable list.
 */
export enum SeedSummaryKind {
  /// A seeded file (resolved to its display name, or a short sha256 fallback).
  File = 'file',
  /// A seeded entity (resolved to its name, or a short id fallback).
  Entity = 'entity',
  /// A seeded repo (resolved to its name, or the url fallback).
  Repo = 'repo',
  /// A seeded tag (rendered as `key: value`).
  Tag = 'tag',
}

/**
 * One human-readable entry in the dashboard's "seeded by" summary.
 */
export interface SeedSummaryItem {
  /// Which kind of seed resource this item came from.
  kind: SeedSummaryKind;
  /// The display label (file/entity/repo name, or `key: value` for a tag; with fallbacks).
  label: string;
}

/**
 * Shorten a long identifier (sha256 / uuid) to its leading characters for a compact fallback label.
 *
 * @param id - The identifier to shorten.
 * @returns The first {@link SHORT_ID_LENGTH} characters (unchanged when already shorter).
 */
function shortId(id: string): string {
  return id.length > SHORT_ID_LENGTH ? id.slice(0, SHORT_ID_LENGTH) : id;
}

/**
 * Build sha256 → Sample, id → Entity, and url → Repo lookups by scanning `data_map` once.
 *
 * The graph's `data_map` is keyed by opaque node ids, so the seed's sha256s/ids/urls can't index it
 * directly; this walk re-keys each node by its own identity so the summary can resolve display names in
 * O(1). Later nodes with a duplicate identity are ignored (first-seen wins).
 *
 * @param dataMap - The graph's node map (opaque node id → {@link TreeNode}).
 * @returns The `{ samples, entities, repos }` identity lookups.
 */
function buildLookups(dataMap: { [nodeId: string]: TreeNode }): {
  samples: Map<string, TreeNode[TreeNodeKey.Sample]>;
  entities: Map<string, TreeNode[TreeNodeKey.Entity]>;
  repos: Map<string, TreeNode[TreeNodeKey.Repo]>;
} {
  const samples = new Map<string, TreeNode[TreeNodeKey.Sample]>();
  const entities = new Map<string, TreeNode[TreeNodeKey.Entity]>();
  const repos = new Map<string, TreeNode[TreeNodeKey.Repo]>();
  for (const node of Object.values(dataMap)) {
    const sample = node[TreeNodeKey.Sample];
    if (sample && !samples.has(sample.sha256)) {
      samples.set(sample.sha256, sample);
    }
    const entity = node[TreeNodeKey.Entity];
    if (entity && !entities.has(entity.id)) {
      entities.set(entity.id, entity);
    }
    const repo = node[TreeNodeKey.Repo];
    if (repo && !repos.has(repo.url)) {
      repos.set(repo.url, repo);
    }
  }
  return { samples, entities, repos };
}

/**
 * Resolve the resources a dashboard was seeded from into a human-readable summary list.
 *
 * Each seed resource is turned into a labeled item, resolving display names against the graph's
 * `data_map` (scanned once into identity lookups) with graceful fallbacks when a node hasn't loaded:
 *
 * - **File** (`seed.samples[].sha256`) → the resolved `Sample`'s display name via
 *   {@link getUniqueFileNames}; falls back to a short sha256 when the node is missing or has no usable
 *   submission names.
 * - **Entity** (`seed.entities[].id`) → the resolved `Entity`'s `name`; falls back to a short id.
 * - **Repo** (`seed.repos[].url`) → the resolved `Repo`'s `name`; falls back to the url.
 * - **Tag** (`seed.tags[key] = value[]`) → `key: value`, one item per value.
 *
 * Items are emitted in a stable order (files, entities, repos, then tags) so the summary line is
 * deterministic. An empty seed yields an empty list.
 *
 * @param seed - The decoded dashboard seed.
 * @param dataMap - The graph's node map used to resolve names (may be empty before the graph loads).
 * @returns The ordered, human-readable seed-summary items.
 */
export function collectSeedSummary(seed: Seed, dataMap: { [nodeId: string]: TreeNode }): SeedSummaryItem[] {
  const { samples, entities, repos } = buildLookups(dataMap);
  const items: SeedSummaryItem[] = [];
  // files: resolve the sample's display name, falling back to a short sha256
  for (const sha256 of seed.samples ?? []) {
    const sample = samples.get(sha256);
    const derived = sample ? getUniqueFileNames(sample.submissions) : '';
    items.push({ kind: SeedSummaryKind.File, label: derived !== '' ? derived : shortId(sha256) });
  }
  // entities: resolve the entity name, falling back to a short id
  for (const id of seed.entities ?? []) {
    const entity = entities.get(id);
    const name = entity?.name ?? '';
    items.push({ kind: SeedSummaryKind.Entity, label: name !== '' ? name : shortId(id) });
  }
  // repos: resolve the repo name, falling back to the url
  for (const url of seed.repos ?? []) {
    const repo = repos.get(url);
    const name = repo?.name ?? '';
    items.push({ kind: SeedSummaryKind.Repo, label: name !== '' ? name : url });
  }
  // tags: one `key: value` item per value under each key
  if (seed.tags) {
    for (const key of Object.keys(seed.tags)) {
      for (const value of seed.tags[key]) {
        items.push({ kind: SeedSummaryKind.Tag, label: `${key}: ${value}` });
      }
    }
  }
  return items;
}
