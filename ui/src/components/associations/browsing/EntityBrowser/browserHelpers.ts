// spec: ./EntityBrowser.spec.md

// project imports
import {
  ascendFirstParent,
  buildTreeIndex,
  childIdsOf,
  contextualDisplayEdges,
  displayChildren,
  parentIdsOf,
  toDisplayCfg,
  TreeIndex,
} from '../treeHelpers';
import { classifyNode } from '../../graph/data';
import { getNodeName, getOrCreate } from '../../utilities';
import {
  EffectiveChild,
  EntityLayerConfig,
  FilterCriteria,
  FlagStat,
  KindGroup,
  LayerPolicy,
  RootDescriptor,
  RootSpec,
  SORT_PRIORITY,
  SortMode,
  TraversalConfig,
} from './types';
import { flatTagsToTags } from '@components/shared/info/info';
import { Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { getStringFieldListFromClauses } from '@components/shared/inputs/omnibar/utils';
import { DangerTagKeys, FileInfoTagKeys, MitreTagKeys } from '@components/tags/tag_groups';
import { filterIncludedTags } from '@components/tags/utilities';
import { Entities } from '@models/entities';
import { Confidence } from '@models/entities/flag';
import { RequestTags, TagOptions, Tags } from '@models/tags';
import { Graph, NodeType, TreeNode, TreeNodeKey } from '@models/trees';

/** Deduplicate a string list preserving order. */
function uniq(values: string[]): string[] {
  return Array.from(new Set(values));
}

/**
 * Find the graph node id for the file with the given sha256, if present.
 *
 * @param graph - The shared association graph.
 * @param sha256 - The file sha256 to locate.
 * @returns The node id, or `undefined` when the file isn't in the graph.
 */
export function findFileNodeHash(graph: Graph, sha256: string): string | undefined {
  for (const [nodeId, node] of Object.entries(graph.data_map ?? {})) {
    if (node[TreeNodeKey.Sample]?.sha256 === sha256) {
      return nodeId;
    }
  }
  return undefined;
}

/** The display label for a node id (its name), falling back to the raw id. */
function labelFor(graph: Graph, nodeId: string): string {
  const node = graph.data_map[nodeId];
  return node ? getNodeName(node, 100) || nodeId : nodeId;
}

/** The classified {@link NodeType} of a node id (guards nodes missing from `data_map`). */
export function nodeTypeOf(nodeId: string, graph: Graph): NodeType {
  if (!(nodeId in graph.data_map)) return NodeType.Other;
  return classifyNode(nodeId, graph).nodeType;
}

/**
 * Resolve a node type's effective {@link LayerPolicy}. Precedence: explicit `Show`/`Hide`/`Exclude` clause →
 * `Include` whitelist membership (Show) → component default → `Include` present but not listed (PassThrough,
 * so a whitelist still surfaces included types nested under non-included ones) → fallback.
 */
export function resolvePolicy(nodeType: NodeType, cfg: TraversalConfig): LayerPolicy {
  const explicit = cfg.clausePolicies[nodeType];
  if (explicit !== undefined) return explicit;
  if (cfg.includeSet?.has(nodeType)) return LayerPolicy.Show;
  const dflt = cfg.defaultPolicies[nodeType];
  if (dflt !== undefined) return dflt;
  if (cfg.includeSet) return LayerPolicy.PassThrough;
  return cfg.fallback;
}

/** The nested {@link Tags} carried by any node kind (flat Tag-node tags are normalized). */
export function getNodeTags(node: TreeNode): Tags {
  if (node[TreeNodeKey.Sample]) return node[TreeNodeKey.Sample].tags ?? {};
  if (node[TreeNodeKey.Repo]) return node[TreeNodeKey.Repo].tags ?? {};
  if (node[TreeNodeKey.Entity]) return node[TreeNodeKey.Entity].tags ?? {};
  if (node[TreeNodeKey.Tag]) return flatTagsToTags(node[TreeNodeKey.Tag].tags);
  return {};
}

/**
 * Tag keys never rendered as header chips: internal plumbing (`Results`/`Parent`/`submitter`) and
 * unbounded-cardinality folder hashes (a distinct hash per folder), which would flood a row with noise. The
 * danger classification is already surfaced by the row's danger dot, so danger tags still render as chips
 * (they carry descriptive value, e.g. `TLP: RED`) — only the keys here are suppressed. Kept as a `Set` for
 * O(1) membership while flattening a node's tags. Any key ending in `Sha256` is also treated as noise
 * (per-file/folder/filesystem hashes are unbounded-cardinality identifiers, not descriptive tags).
 */
export const HEADER_HIDDEN_TAG_KEYS = new Set<string>([
  'Results',
  'Parent',
  'submitter',
  'FolderAllSha256',
  'FolderDataSha256',
  'FolderNamesSha256',
  // free-text flag-scan tags: long values that wrap and blow out the header chip row — they belong in the
  // expandable details, not a header chip (the short flag tags like FlagConfidence/FlagSuspicion still show)
  'FlagContent',
  'FlagReasoning',
]);

/**
 * Header display label for a tag key. Backend flag-scan tags are keyed with a redundant `Flag` prefix
 * (`FlagConfidence`, `FlagSuspicion`, …); strip it so they read as plain `Confidence`/`Suspicion` chips
 * alongside the other header tags. A kind's policy relabel (when present) always wins over the generic strip.
 *
 * @param key - The raw tag key.
 * @param policyLabel - The kind policy's relabel for this key, if any (takes precedence).
 * @returns The label to render on the chip.
 */
function headerTagLabel(key: string, policyLabel?: string): string {
  if (policyLabel) return policyLabel;
  if (/^Flag[A-Z]/.test(key)) return key.slice(4);
  return key;
}

/** The default maximum number of tag chips rendered inline in a row header before the rest collapse to `+N`. */
export const HEADER_TAG_LIMIT = 6;

// Uppercased significance sets used to rank non-priority header tags (danger → ATT&CK/MBC → file-info → other).
const DANGER_KEYS_UPPER = new Set(DangerTagKeys.map((k) => k.toUpperCase()));
const MITRE_KEYS_UPPER = new Set(MitreTagKeys.map((k) => k.toUpperCase()));
const FILEINFO_KEYS_UPPER = new Set(FileInfoTagKeys.map((k) => k.toUpperCase()));

/**
 * Per-entity-kind header-tag curation: which intrinsic tags to hide, which to show first (in order), and how
 * to relabel their (backend-generated) keys for display. Keyed by {@link Entities} kind. E.g. a WindowsProcess
 * emits `PID`/`ProcessName`/`ProcessCommand`/`ProcessImagePath`/`ProcessIsWow64`/… intrinsic tags; the header
 * shows only the identifying few, ordered and de-prefixed, and drops the low-signal internals.
 */
interface HeaderTagPolicy {
  /** Extra tag keys (beyond the global noise set) to hide for this kind. */
  hidden?: Set<string>;
  /** Tag keys to show first, in this order (before the significance-ranked remainder). */
  priority?: string[];
  /** Display relabeling for a tag key (e.g. strip the `Process` prefix). */
  relabel?: Record<string, string>;
}

const HEADER_TAG_POLICIES: Partial<Record<Entities, HeaderTagPolicy>> = {
  [Entities.WindowsProcess]: {
    hidden: new Set(['ProcessIsWow64', 'ProcessOffset', 'ProcessThreads', 'ProcessHandles', 'ProcessSessionID', 'ParentPID']),
    priority: ['PID', 'ProcessName', 'ProcessCommand', 'ProcessImagePath'],
    relabel: { ProcessName: 'Name', ProcessCommand: 'Command', ProcessImagePath: 'ImagePath' },
  },
};

/** Significance rank for ordering a header tag key: 0 danger, 1 ATT&CK/MBC, 2 file-info, 3 everything else. */
function headerTagRank(key: string): number {
  const upper = key.toUpperCase();
  if (DANGER_KEYS_UPPER.has(upper)) return 0;
  if (MITRE_KEYS_UPPER.has(upper)) return 1;
  if (FILEINFO_KEYS_UPPER.has(upper)) return 2;
  return 3;
}

/** True when a tag key is header noise: the global hidden set, or any unbounded `…Sha256` hash key. */
function isHeaderNoiseKey(key: string): boolean {
  return HEADER_HIDDEN_TAG_KEYS.has(key) || /sha256$/i.test(key);
}

/** A single flattened tag key/value pair for header display. */
export interface DisplayTag {
  /** The raw tag key (e.g. `ProcessCommand`) — used for identity/dedup. */
  key: string;
  /** The display label (relabeled per the kind's {@link HeaderTagPolicy}, else the raw key). */
  label: string;
  /** One value under that key (e.g. `PE32`). */
  value: string;
}

/** The capped set of a node's display tags: the shown chips plus how many (and which) overflowed the cap. */
export interface DisplayTags {
  /** The tag chips to render, ordered by kind priority then significance, capped at the limit. */
  shown: DisplayTag[];
  /** How many tag pairs were dropped past the cap (0 when everything fit). */
  overflow: number;
  /** The `label: value` labels of the overflowed pairs, for the `+N` chip's tooltip. */
  overflowLabels: string[];
}

/**
 * Flatten a node's tags into a capped, curated list of `key/value` pairs for the row header.
 *
 * Drops the global noise keys ({@link HEADER_HIDDEN_TAG_KEYS} + any `…Sha256` hash) and the kind's
 * policy-hidden keys, then orders: the kind's **priority** keys first (in policy order), then the remainder by
 * **significance** (danger → ATT&CK/MBC → file-info → other), then key/value for stability. Keys are relabeled
 * per the kind's policy (e.g. a WindowsProcess's `ProcessCommand` → `Command`). Only the first `limit` pairs
 * are returned; the rest are reported as an overflow count + labels for a single `+N` chip. Cheap per rendered
 * row (O(tags) over one node's own tags; only paginated rows mount) and memoized on the node.
 *
 * @param node - The tree node whose tags to flatten.
 * @param limit - The maximum number of chips to return (defaults to {@link HEADER_TAG_LIMIT}).
 * @returns The shown chips plus the overflow count and labels.
 */
export function getDisplayTags(node: TreeNode, limit: number = HEADER_TAG_LIMIT): DisplayTags {
  const tags = getNodeTags(node);
  const kind = node[TreeNodeKey.Entity]?.kind;
  const policy = kind ? HEADER_TAG_POLICIES[kind] : undefined;
  // rank tuple per pair: (priority index or Infinity, significance rank) so a stable multi-key sort applies
  const pairs: { key: string; value: string; prio: number; rank: number }[] = [];
  for (const [key, values] of Object.entries(tags)) {
    if (isHeaderNoiseKey(key) || policy?.hidden?.has(key)) continue;
    const prioIndex = policy?.priority?.indexOf(key) ?? -1;
    const prio = prioIndex === -1 ? Number.POSITIVE_INFINITY : prioIndex;
    const rank = headerTagRank(key);
    for (const value of Object.keys(values ?? {})) {
      pairs.push({ key, value, prio, rank });
    }
  }
  // priority keys first (policy order), then significance, then key/value for a stable deterministic order
  pairs.sort((a, b) => a.prio - b.prio || a.rank - b.rank || a.key.localeCompare(b.key) || a.value.localeCompare(b.value));
  const all: DisplayTag[] = pairs.map((p) => ({ key: p.key, label: headerTagLabel(p.key, policy?.relabel?.[p.key]), value: p.value }));
  const shown = all.slice(0, limit);
  const rest = all.slice(limit);
  return { shown, overflow: rest.length, overflowLabels: rest.map((t) => `${t.label}: ${t.value}`) };
}

/** The groups a node belongs to (Entity `.groups`; Sample/Repo from their submissions). */
export function nodeGroups(node: TreeNode): string[] {
  if (node[TreeNodeKey.Entity]) return node[TreeNodeKey.Entity].groups ?? [];
  if (node[TreeNodeKey.Sample]) return uniq((node[TreeNodeKey.Sample].submissions ?? []).flatMap((s) => s.groups ?? []));
  if (node[TreeNodeKey.Repo]) return uniq((node[TreeNodeKey.Repo].submissions ?? []).flatMap((s) => s.groups ?? []));
  return [];
}

/** True when a node carries any danger-classified tag. */
export function hasDangerTags(tags: Tags): boolean {
  return Object.keys(filterIncludedTags(tags, DangerTagKeys)).length > 0;
}

/** The first parent of a node in the index, or null. */
function firstParent(index: TreeIndex, nodeId: string): string | null {
  return parentIdsOf(index, nodeId)[0] ?? null;
}

/**
 * Resolve a {@link RootSpec} into concrete root descriptors.
 *
 * `sha256` locates the file node; `nodes` is passed through; `initial` ascends each seed node to its tree
 * root (mirroring the association tree's root resolution) so a dashboard view starts from the top.
 *
 * @param graph - The shared graph.
 * @param spec - How to determine roots.
 * @param index - Optional prebuilt index (used for the `initial` ascent; built on demand otherwise).
 * @returns The resolved roots (id + label), possibly empty.
 */
export function resolveRoots(graph: Graph, spec: RootSpec, index?: TreeIndex): RootDescriptor[] {
  switch (spec.kind) {
    case 'sha256': {
      const id = findFileNodeHash(graph, spec.sha256);
      return id ? [{ id, label: labelFor(graph, id) }] : [];
    }
    case 'nodes':
      return spec.roots;
    case 'initial': {
      const idx = index ?? buildTreeIndex(graph);
      const roots: string[] = [];
      // Set alongside the array so the "already collected this root" check is O(1) rather than a linear scan
      // per seed (the ascent can converge many seeds onto the same tree root)
      const rootSet = new Set<string>();
      for (const initialId of graph.initial) {
        // the topmost first-parent ancestor is the last element of the ascent chain
        const chain = ascendFirstParent(idx, initialId.toString(), firstParent);
        const top = chain[chain.length - 1];
        if (!rootSet.has(top)) {
          rootSet.add(top);
          roots.push(top);
        }
      }
      return roots.map((id) => ({ id, label: labelFor(graph, id) }));
    }
  }
}

/**
 * Build the focus breadcrumb: the first-parent ancestor chain from the topmost ancestor **down to and
 * including** `focusRoot`, so a re-rooted (focused) tree can show a clickable trail back up to the natural
 * roots. Ascends via {@link firstParent} (the same first-parent rule {@link resolveRoots} uses for `initial`),
 * guarding cycles, then reverses to top→down order. The last entry is `focusRoot` itself (the current head).
 *
 * @param graph - The shared graph (for labels).
 * @param index - The tree index (for `parentsOf`).
 * @param focusRoot - The node the tree is currently re-rooted at.
 * @returns The ancestor chain top→down, each with a display label; `[focusRoot]` when it has no parent.
 */
export function focusBreadcrumb(graph: Graph, index: TreeIndex, focusRoot: string): RootDescriptor[] {
  // ascent yields [focus, parent, …, top]; present it top→down so the trail reads left-to-right into the subtree
  const chain = ascendFirstParent(index, focusRoot, firstParent);
  chain.reverse();
  return chain.map((id) => ({ id, label: labelFor(graph, id) }));
}

/** Whether a node is within the config's depth bound (null bound = unbounded). Shared by effectiveChildren/filterTree. */
function withinDepth(id: string, cfg: TraversalConfig): boolean {
  return cfg.maxDepth == null || (cfg.distances.get(id) ?? 0) <= cfg.maxDepth;
}

/**
 * Compute the children to render under a parent, applying each child's resolved {@link LayerPolicy} and the
 * depth bound.
 *
 * - Beyond-`maxDepth` and `Skip` children are pruned (not rendered, not explored).
 * - `PassThrough` children are elided: we traverse *through* them and graft their qualifying descendants onto
 *   this parent, recording a breadcrumb of the elided node names so the row still explains its origin.
 * - `Show` children are kept.
 *
 * A **per-path** visited set guards cycles while still allowing a node reachable via two distinct paths
 * (DAG re-convergence) to render under each. Grafted `Show` results are deduped by id within this call.
 *
 * @param parentId - The node whose children to resolve.
 * @param index - The edge-carrying tree index.
 * @param graph - The shared graph (for node types + labels).
 * @param cfg - Layer policies + depth bound.
 * @param path - Node ids already on the rendered path to `parentId` (inclusive) — for cycle guarding.
 * @param reverseDepth - How many reversed hops preceded `parentId`.
 * @param viaReversed - Whether `parentId` was itself reached via a reversed edge.
 * @returns The effective children, each with its edge and any pass-through breadcrumb.
 */
export function effectiveChildren(
  parentId: string,
  index: TreeIndex,
  graph: Graph,
  cfg: TraversalConfig,
  path: Set<string>,
  reverseDepth = 0,
  viaReversed = false,
): EffectiveChild[] {
  const out: EffectiveChild[] = [];
  const seen = new Set<string>();
  const displayCfg = toDisplayCfg(cfg);
  // re-root ("gear") mode: span the whole component from the root by treating every edge as bidirectional
  // and dropping the reverse-depth cap / reverse-arrival suppression — only the per-path cycle guard bounds it
  const spanningCfg = cfg.spanning ? { orientation: displayCfg.orientation, bidirectional: () => true } : null;

  // `nodeViaReversed`/`nodeReverseDepth` are the ARRIVAL context of `nodeId`; contextualDisplayEdges applies
  // the reverse-traversal rules (suppress forward fan-out of a reverse-reached node; bound reverse hops), and
  // each surviving edge yields the CHILD's context, carried on the EffectiveChild so the row expands correctly.
  // In spanning mode those rules are skipped so a re-rooted view reaches every connected node.
  const walk = (nodeId: string, currentPath: Set<string>, breadcrumb: string[], nodeViaReversed: boolean, nodeReverseDepth: number) => {
    const edges = spanningCfg
      ? displayChildren(index, nodeId, spanningCfg)
      : contextualDisplayEdges(index, nodeId, displayCfg, nodeViaReversed, nodeReverseDepth);
    for (const edge of edges) {
      const childId = edge.id;
      if (currentPath.has(childId)) continue; // cycle guard, scoped to this path
      // user-hidden nodes vanish with their whole subtree BEFORE the policy check, so hiding a PassThrough
      // node also suppresses the descendants it would otherwise graft up (not just the node itself)
      if (cfg.hiddenNodes?.has(childId)) continue;
      if (!withinDepth(childId, cfg)) continue; // depth bound (prunes deeper nodes, incl. through pass-through)
      const isReverse = !!edge.reversed;
      // Spanning (re-root) treats every edge uniformly and bypasses contextualDisplayEdges, so it carries NO
      // reverse-arrival context. Pinning both to neutral keeps the (nodeId, viaReversed, reverseDepth) keys used
      // by filterTree's / flaggedOnlyVisible's cycle guards FINITE — otherwise reverseDepth grows without bound
      // across the hub↔leaf bounces a bidirectional spanning walk creates, and those guards never fire
      // (RangeError: Maximum call stack size exceeded). It also lets the expand-affordance checks judge re-rooted
      // rows at neutral context instead of a bogus deep-reverse context. Non-spanning behavior is unchanged.
      const childViaReversed = spanningCfg ? false : isReverse ? true : nodeViaReversed;
      const childReverseDepth = spanningCfg ? 0 : isReverse ? nodeReverseDepth + 1 : nodeReverseDepth;
      const policy = resolvePolicy(nodeTypeOf(childId, graph), cfg);
      if (policy === LayerPolicy.Skip) continue;
      if (policy === LayerPolicy.PassThrough) {
        const nextPath = new Set(currentPath);
        nextPath.add(childId);
        const via = labelFor(graph, childId) || edge.label;
        walk(childId, nextPath, [...breadcrumb, via], childViaReversed, childReverseDepth);
        continue;
      }
      // Show
      if (seen.has(childId)) continue;
      seen.add(childId);
      out.push({ edge, viaReversed: childViaReversed, reverseDepth: childReverseDepth, ...(breadcrumb.length ? { breadcrumb } : {}) });
    }
  };

  walk(parentId, path, [], viaReversed, reverseDepth);
  return out;
}

/**
 * Group sibling children by their {@link NodeType}, preserving first-appearance order.
 *
 * @param children - Effective children of a single parent.
 * @param graph - The shared graph (for node classification).
 * @returns Kind groups in the order their kinds first appear.
 */
export function groupByKind(children: EffectiveChild[], graph: Graph): KindGroup[] {
  const groups: KindGroup[] = [];
  const byType = new Map<NodeType, KindGroup>();
  for (const child of children) {
    const nodeType = nodeTypeOf(child.edge.id, graph);
    let group = byType.get(nodeType);
    if (!group) {
      group = { nodeType, children: [] };
      byType.set(nodeType, group);
      groups.push(group);
    }
    group.children.push(child);
  }
  return groups;
}

/** True when a node's tags satisfy every tag filter (each key present with an any-of value; case-insensitive). */
function matchesTags(nodeTags: Tags, filter: RequestTags): boolean {
  return Object.entries(filter).every(([key, values]) => {
    const nodeKey = Object.keys(nodeTags).find((k) => k.toLowerCase() === key.toLowerCase());
    if (!nodeKey) return false;
    if (values.length === 0) return true;
    const nodeVals = Object.keys(nodeTags[nodeKey]).map((v) => v.toLowerCase());
    return values.some((v) => nodeVals.includes(v.toLowerCase()));
  });
}

/** True when a node satisfies every active filter category (text AND tags AND groups AND flagged). */
function nodeMatches(nodeId: string, graph: Graph, criteria: FilterCriteria): boolean {
  const node = graph.data_map[nodeId];
  if (!node) return false;
  if (criteria.text && !getNodeName(node, 1000).toLowerCase().includes(criteria.text.toLowerCase())) return false;
  if (Object.keys(criteria.tags).length > 0 && !matchesTags(getNodeTags(node), criteria.tags)) return false;
  if (criteria.groups.length > 0) {
    const groups = nodeGroups(node).map((g) => g.toLowerCase());
    if (!criteria.groups.some((g) => groups.includes(g.toLowerCase()))) return false;
  }
  if (criteria.flaggedOnly && !criteria.flaggedNodes.has(nodeId)) return false;
  return true;
}

/**
 * Visibility for a pure **Flagged Only** view (no text/tag/group filter): a forward walk that descends **only
 * through flagged nodes** — mirroring the render's arrival context and per-path cycle guard — and, at each
 * flagged node, surfaces its **immediate display associations** even when they are unflagged. So an item
 * directly attached to a flag/tag (e.g. the `SigmaRule` shown under a `Flag`, or an entity a flagged node
 * points at) stays visible, while a *deeper* unflagged branch collapses to just below that first unflagged
 * relation, and a branch with no flagged content is never entered. Relies on flag/danger counts already
 * propagating to the containing spine ({@link computeFlagStats}), so the path from a root down to each flagged
 * node is itself flagged and thus walked. Hidden nodes are already dropped by `effectiveChildren`.
 *
 * @param rootIds - The tree roots to walk from.
 * @param index - The tree index.
 * @param graph - The shared graph.
 * @param flaggedNodes - The precomputed flagged-node set.
 * @param cfg - The traversal/display config.
 * @returns The visible node ids for the Flagged-Only view.
 */
function flaggedOnlyVisible(
  rootIds: string[],
  index: TreeIndex,
  graph: Graph,
  flaggedNodes: Set<string>,
  cfg: TraversalConfig,
): Set<string> {
  const visible = new Set<string>();
  // guard against re-walking a flagged node reached again via the same arrival context (DAG re-convergence)
  const seen = new Set<string>();
  const walk = (nodeId: string, viaReversed: boolean, reverseDepth: number, path: Set<string>) => {
    visible.add(nodeId);
    const key = `${nodeId}:${viaReversed ? 1 : 0}:${reverseDepth}`;
    if (seen.has(key)) return;
    seen.add(key);
    for (const child of effectiveChildren(nodeId, index, graph, cfg, path, reverseDepth, viaReversed)) {
      // a flagged node's direct association stays visible even if unflagged; only flagged children are descended
      visible.add(child.edge.id);
      if (flaggedNodes.has(child.edge.id)) {
        const nextPath = new Set(path);
        nextPath.add(child.edge.id);
        walk(child.edge.id, child.viaReversed ?? false, child.reverseDepth ?? 0, nextPath);
      }
    }
  };
  for (const rootId of rootIds) {
    if (flaggedNodes.has(rootId)) walk(rootId, false, 0, new Set([rootId]));
  }
  return visible;
}

/**
 * Compute the set of node ids to render under an active filter: a node is visible if it matches the criteria,
 * or any of its (policy/depth-resolved) descendants does — so ancestors of matches stay reachable. Traverses
 * only the currently-loaded graph.
 *
 * When **Flagged Only** is the sole active filter, delegates to {@link flaggedOnlyVisible}, which additionally
 * surfaces each flagged node's direct associations and collapses deeper unflagged branches (decision 3).
 *
 * @returns The set of visible node ids. When no filter is active, callers should skip filtering entirely.
 */
export function filterTree(rootIds: string[], index: TreeIndex, graph: Graph, criteria: FilterCriteria, cfg: TraversalConfig): Set<string> {
  // Flagged Only, with nothing else active, uses the forward flagged-spine walk (direct associations + collapse)
  if (criteria.flaggedOnly && !criteria.text && Object.keys(criteria.tags).length === 0 && criteria.groups.length === 0) {
    return flaggedOnlyVisible(rootIds, index, graph, criteria.flaggedNodes, cfg);
  }
  const visible = new Set<string>();
  // Memoize each node's "does it or a descendant match" answer, keyed by (id, arrival context), because
  // reverse edges make a node's children depend on how it was reached. `inProgress` guards cycles. This keeps
  // the pass O(V × 2 × REVERSE_MAX) instead of exploding across the many paths bidirectional edges create —
  // the depth bound (`maxDepth`) is null on the file tab and cannot be relied on to bound cost.
  const memo = new Map<string, boolean>();
  const inProgress = new Set<string>();

  const dfs = (nodeId: string, viaReversed: boolean, reverseDepth: number): boolean => {
    const memoKey = `${nodeId}:${viaReversed ? 1 : 0}:${reverseDepth}`;
    const cached = memo.get(memoKey);
    if (cached !== undefined) {
      if (cached) visible.add(nodeId);
      return cached;
    }
    if (inProgress.has(memoKey)) return false; // cycle: contribute nothing on this arc
    inProgress.add(memoKey);
    let subtreeMatch = false;
    // pass the node's own id as the cycle-guard path; the memo (not a full ancestor path) bounds the DFS,
    // matching the same reverse-context the renderer uses
    for (const child of effectiveChildren(nodeId, index, graph, cfg, new Set([nodeId]), reverseDepth, viaReversed)) {
      if (dfs(child.edge.id, child.viaReversed ?? false, child.reverseDepth ?? 0)) subtreeMatch = true;
    }
    inProgress.delete(memoKey);
    const result = nodeMatches(nodeId, graph, criteria) || subtreeMatch;
    memo.set(memoKey, result);
    if (result) visible.add(nodeId);
    return result;
  };

  for (const rootId of rootIds) {
    dfs(rootId, false, 0);
  }
  return visible;
}

/**
 * Read the `Show`/`Hide`/`Exclude`/`Include` omnibar clauses into a layer config. Clause values are raw
 * {@link NodeType} enum values, so they map straight to policy keys. When a type appears under multiple
 * verbs, Exclude wins over Hide over Show.
 */
export function getEntityLayerConfigFromClauses(clauses: Clause[]): EntityLayerConfig {
  const policies: EntityLayerConfig['policies'] = {};
  for (const k of getStringFieldListFromClauses(clauses, 'Show')) policies[k as NodeType] = LayerPolicy.Show;
  for (const k of getStringFieldListFromClauses(clauses, 'Hide')) policies[k as NodeType] = LayerPolicy.PassThrough;
  for (const k of getStringFieldListFromClauses(clauses, 'Exclude')) policies[k as NodeType] = LayerPolicy.Skip;
  const include = getStringFieldListFromClauses(clauses, 'Include');
  return { policies, includeSet: include.length ? new Set(include as NodeType[]) : null };
}

/** Read the traversal `depth` from the omnibar clauses (last valid positive integer), else the default. */
export function getDepthFromClauses(clauses: Clause[], dflt = 1): number {
  const values = getStringFieldListFromClauses(clauses, 'depth')
    .map(Number)
    .filter((n) => Number.isInteger(n) && n > 0);
  return values.length ? values[values.length - 1] : dflt;
}

/** Collect the tag key→values present anywhere in the pulled graph, for the omnibar tag options. */
export function collectTagOptions(graph: Graph): TagOptions {
  // accumulate into per-key Sets so each value is deduped in O(1); materializing the arrays once at the end
  // avoids the O(n·tags) rebuild that copying `out[key]` into a fresh Set per node would incur
  const accum = new Map<string, Set<string>>();
  for (const node of Object.values(graph.data_map ?? {})) {
    for (const [key, values] of Object.entries(getNodeTags(node))) {
      const set = getOrCreate(accum, key, () => new Set<string>());
      for (const value of Object.keys(values ?? {})) set.add(value);
    }
  }
  const out: TagOptions = {};
  for (const [key, set] of accum) out[key] = Array.from(set);
  return out;
}

/** Collect the groups present on any node in the pulled graph, for the omnibar group options. */
export function collectGroupOptions(graph: Graph): string[] {
  const set = new Set<string>();
  for (const node of Object.values(graph.data_map ?? {})) {
    for (const group of nodeGroups(node)) set.add(group);
  }
  return Array.from(set).sort();
}

/** The result of {@link computeFlagStats}: the flagged-node set plus per-node subtree flag aggregates. */
export interface FlagAggregation {
  /**
   * Every node a flag or danger count reaches under the directed propagation model (see
   * {@link computeFlagStats}): each `Flag`/danger-tagged node plus every node its count flows to (the flagged
   * entity and its containing spine, a rule's flag when it carries tags). Drives the "Flagged Only" filter.
   */
  flagged: Set<string>;
  /** Per-node flag aggregate (count / max suspicion / max confidence / danger-tag count). See {@link FlagStat}. */
  stats: Map<string, FlagStat>;
}

/** Map a {@link Confidence} to an ordinal (higher = more confident) for aggregation/sorting. */
function confidenceRank(confidence?: Confidence): number {
  switch (confidence) {
    case Confidence.Fact:
      return 3;
    case Confidence.Likely:
      return 2;
    case Confidence.Unsure:
      return 1;
    default:
      return 0;
  }
}

/** Create a zeroed {@link FlagStat}. */
function blankFlagStat(): FlagStat {
  return { flags: 0, suspicion: 0, confidence: 0, dangerTags: 0 };
}

/**
 * Whether a node is a `Flag` or `SigmaRule` entity — the two kinds whose counts propagate **downward** (a Flag
 * to the entity it flags, a SigmaRule to the Flag it created) instead of up a containment spine, and which are
 * therefore never climbed *into* as a "whole".
 *
 * @param node - The tree node (may be undefined for ids missing from `data_map`).
 * @returns True when the node is a Flag or SigmaRule entity.
 */
function isFlagOrRule(node: TreeNode | undefined): boolean {
  const kind = node?.[TreeNodeKey.Entity]?.kind;
  return kind === Entities.Flag || kind === Entities.SigmaRule;
}

/**
 * Count the danger-classified tag pairs on a node — each key/value under a {@link DangerTagKeys} key (so two
 * `YARAHIT` values count as two). Drives the aggregated danger-tag badge.
 *
 * @param tags - The node's tags.
 * @returns The number of danger tag key/value pairs.
 */
function countDangerTags(tags: Tags): number {
  let count = 0;
  for (const values of Object.values(filterIncludedTags(tags, DangerTagKeys))) {
    count += Object.keys(values ?? {}).length;
  }
  return count;
}

/**
 * The nodes a count on `id` flows *to* next — the one directed step of the propagation walk. Purely a function
 * of **direction** (the tree index's parent/child, derived from the stored association direction) and the
 * node's **entity type**; the association *kind* is intentionally ignored (a tool may record a generic
 * `AssociatedWith` where a more specific kind would fit, so kind is unreliable):
 * - A `Flag`/`SigmaRule` propagates **down** to what it points at ({@link childIdsOf}) — a Flag to the entity
 *   it flags, a SigmaRule to the Flag it created. It does not climb to its own parents (so a flag count never
 *   reaches the rule that created it, and a rule shows no flag count).
 * - Every other node propagates **up** to the whole that contains it ({@link parentIdsOf}), excluding any
 *   parent that is itself a `Flag`/`SigmaRule` — those only appear as a "parent" because they point down at
 *   this node, so climbing into them would flow a count the wrong way (into a flag/rule, or across to a shared
 *   subject's other flags).
 *
 * Because this trusts the stored direction, an association authored with a reversed direction (e.g. a part
 * pointed at its whole) will aggregate the wrong way — that is a data-source bug, not compensated for here.
 *
 * @param graph - The shared graph (for node entity types).
 * @param index - The tree index.
 * @param id - The node whose propagation targets to resolve.
 * @returns The next node ids the count flows to.
 */
function propagationTargets(graph: Graph, index: TreeIndex, id: string): string[] {
  if (isFlagOrRule(graph.data_map[id])) {
    return childIdsOf(index, id);
  }
  return parentIdsOf(index, id).filter((pid) => !isFlagOrRule(graph.data_map[pid]));
}

/**
 * Propagate one seed's contribution across the directed propagation graph (see {@link propagationTargets}),
 * applying `apply` to every reached node once and marking each flagged. A per-seed visited guard tallies each
 * node a single time, so counts stay distinct even where the graph re-converges (DAG).
 *
 * @param graph - The shared graph.
 * @param index - The tree index.
 * @param stats - The per-node stats map to update.
 * @param flagged - The flagged-node set to extend.
 * @param seedId - The node to start from (a Flag for flag counts; any danger-tagged node for tag counts).
 * @param apply - Applies this seed's contribution to a reached node's stat.
 */
function propagateFromSeed(
  graph: Graph,
  index: TreeIndex,
  stats: Map<string, FlagStat>,
  flagged: Set<string>,
  seedId: string,
  apply: (stat: FlagStat) => void,
): void {
  const visited = new Set<string>();
  const queue = [seedId];
  let i = 0;
  while (i < queue.length) {
    const current = queue[i++];
    if (visited.has(current)) continue;
    visited.add(current);
    flagged.add(current);
    apply(getOrCreate(stats, current, blankFlagStat));
    for (const next of propagationTargets(graph, index, current)) {
      if (!visited.has(next)) queue.push(next);
    }
  }
}

/**
 * Compute flag significance for the whole graph in a **single pass**, so rows/levels never re-crawl the tree
 * per render. The result is memoized by callers on the graph version.
 *
 * Propagation is keyed on **direction** and **entity type**, not the association kind (see
 * {@link propagationTargets}). Counts flow from a *part* up to the *whole* that contains it, with two directed
 * exceptions for the flag chain: a `SigmaRule`'s counts flow **down** to the `Flag` it created, a `Flag`'s
 * counts flow **down** to the entity it flags, and from that entity the count resumes climbing the containment
 * spine (process → process tree → memory dump/file). A per-seed visited guard tallies each node once, so a
 * memory dump shows the total beneath it while a single process shows only its own.
 *
 * Consequences of the directed model: a flag count never reaches the `SigmaRule` that created it (a rule shows
 * **0** flags — implicitly interesting, so no badge is needed), and a flagged entity's own danger tags climb
 * to its wholes but never flow back down into the flag. A rule's or flag's *own* danger tags do propagate
 * forward (rule → flag → entity → up), matching the flag chain's direction.
 *
 * Danger-classified tags use the same walk (via {@link FlagStat.dangerTags}); a danger-tagged node contributes
 * its tag-pair count to itself and every whole above it.
 *
 * Returns both the {@link FlagAggregation.flagged} set (flags + danger-tagged nodes + every node a count
 * reaches, for the Flagged-Only filter) and the per-node {@link FlagAggregation.stats} map (for the
 * flag/danger-count badges and sorting).
 *
 * @param graph - The shared graph.
 * @param index - The tree index (for `parentsOf`/`childrenOf`).
 * @returns The flagged set and the per-node subtree flag stats.
 */
export function computeFlagStats(graph: Graph, index: TreeIndex): FlagAggregation {
  const flagged = new Set<string>();
  const stats = new Map<string, FlagStat>();
  // collect Flag entities (with suspicion/confidence) and danger-tagged nodes (with their danger-tag count)
  const flagSeeds: { id: string; suspicion: number; confidence: number }[] = [];
  const dangerSeeds: { id: string; count: number }[] = [];
  for (const [id, node] of Object.entries(graph.data_map ?? {})) {
    const entity = node[TreeNodeKey.Entity];
    if (entity?.kind === Entities.Flag) {
      const meta = (entity.metadata as { Flag?: { suspicion?: number; confidence?: Confidence } } | undefined)?.Flag;
      flagSeeds.push({ id, suspicion: meta?.suspicion ?? 0, confidence: confidenceRank(meta?.confidence) });
    }
    const dangerCount = countDangerTags(getNodeTags(node));
    if (dangerCount > 0) dangerSeeds.push({ id, count: dangerCount });
  }
  // each flag propagates to the entity it flags and up that entity's containing spine, folding suspicion/confidence
  for (const seed of flagSeeds) {
    propagateFromSeed(graph, index, stats, flagged, seed.id, (stat) => {
      stat.flags += 1;
      if (seed.suspicion > stat.suspicion) stat.suspicion = seed.suspicion;
      if (seed.confidence > stat.confidence) stat.confidence = seed.confidence;
    });
  }
  // danger tags propagate the same way, so a node shows the danger tags on it and everything it contains
  for (const seed of dangerSeeds) {
    propagateFromSeed(graph, index, stats, flagged, seed.id, (stat) => {
      stat.dangerTags += seed.count;
    });
  }
  return { flagged, stats };
}

/** Read a single {@link SortMode} field from a (possibly missing) {@link FlagStat}, defaulting to 0. */
function flagStatField(stat: FlagStat | undefined, mode: SortMode): number {
  if (!stat) return 0;
  switch (mode) {
    case SortMode.Flags:
      return stat.flags;
    case SortMode.Suspicion:
      return stat.suspicion;
    case SortMode.Confidence:
      return stat.confidence;
  }
}

/**
 * Compare two node ids by their subtree flag stats for the given primary {@link SortMode}, **descending**
 * (most significant first). The selected mode leads; the remaining {@link SORT_PRIORITY} modes tiebreak in
 * order. Returns 0 when all three are equal, so a stable sort preserves the pre-existing order.
 *
 * @param a - First node id.
 * @param b - Second node id.
 * @param stats - The per-node flag stats from {@link computeFlagStats}.
 * @param primary - The selected sort mode (leads the comparison).
 * @returns Negative if `a` sorts first, positive if `b` sorts first, 0 if tied.
 */
export function compareByFlagStats(a: string, b: string, stats: Map<string, FlagStat>, primary: SortMode): number {
  const statA = stats.get(a);
  const statB = stats.get(b);
  // primary mode first, then the remaining modes as tiebreakers in the fixed priority order
  for (const mode of [primary, ...SORT_PRIORITY.filter((m) => m !== primary)]) {
    const diff = flagStatField(statB, mode) - flagStatField(statA, mode);
    if (diff !== 0) return diff;
  }
  return 0;
}
