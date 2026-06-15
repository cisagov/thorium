// project imports
import { getNodeName } from '@components/associations/utilities';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';
import { buildTagBrowseHref, filterExcludedTags } from '@components/tags/utilities';
import { formatAddress } from '@utilities/disassembly';
import { humanize } from '@utilities/humanize';
import { Entities, entityLabel } from '@models/entities/entities';
import { EntityRequest, entityRequestKind } from '@models/entities/requests';
import { Origin } from '@models/files';
import { Tags } from '@models/tags';
import { TreeNode, TreeNodeKey } from '@models/trees';

// spec: ./SPEC.md

/**
 * Whether a field renders inline (`Label: value` on one line) or stacked (label above value). When a
 * field omits this, {@link resolveFieldLayout} derives it from the value kind / render type.
 */
export enum FieldLayout {
  /** Label and value on the same line — short scalars, booleans, times, and short word-lists. */
  Inline = 'inline',
  /** Label above value — long values (hashes, URLs, paths, ids, prose) and multi-value arrays. */
  Stacked = 'stacked',
}

/** How a field's value should be rendered by {@link InfoField}. */
export enum FieldRender {
  /** Default: a `FieldBadge` (scalars, arrays, booleans). */
  Badge = 'badge',
  /** Render the (ISO string) value via the shared `<Time>` component. */
  Time = 'time',
  /** Plain text — used for long values (e.g. descriptions) that shouldn't be badge-wrapped. */
  Text = 'text',
  /** A full path / long name: italicized block that breaks at any character (e.g. `image_path`). */
  Path = 'path',
  /** Render the value as collapsible markdown (long text, auto-formatted, expand/collapse). */
  Markdown = 'markdown',
  /** Render the value in a monospace code block (sigma YAML, disassembly, decompiled source), collapsible. */
  Code = 'code',
}

/** A named entity reference: display its `text` (the name) linked to its details page. */
export interface InfoLink {
  /** The visible text — an entity name we know the exact identity of. */
  text: string;
  /** The details-page path for that entity (omit to render the name as plain text). */
  href?: string;
}

/** A single labelled metadata row for {@link InfoSection}. */
export interface InfoField {
  label: string;
  /** Scalar/array/boolean value rendered per {@link InfoField.render}, or `undefined` to omit. */
  value?: string | number | boolean | string[] | null;
  /**
   * A list of named entity links rendered as chips (each the entity's name linked to its details
   * page). Use instead of `value` when the field references specific entities we can identify exactly
   * (e.g. a Device's vendors) — takes precedence over `value` when set.
   */
  links?: InfoLink[];
  /**
   * When set, the value is rendered as a link to this path. This is per-field so a field can link to
   * the resource it *describes* — e.g. a File's SHA256 links to that file, but a FileSystem's SHA256
   * (or a Folder's `filesystem_id`) links to the filesystem entity rather than a `/file/` page.
   */
  href?: string;
  /** Open {@link InfoField.href} in a new tab. Defaults to `true` (links are usually cross-resource). */
  external?: boolean;
  /** Render as a danger (critical) badge. */
  danger?: boolean;
  /** How to render the value; defaults to {@link FieldRender.Badge}. */
  render?: FieldRender;
  /**
   * Force inline/stacked layout. Omit to auto-derive from value kind (see {@link resolveFieldLayout}).
   * Set {@link FieldLayout.Inline} on short scalars and short word-lists that read cleanly on one line.
   */
  layout?: FieldLayout;
}

/** An ordered group of fields, with its own heading, delimiter, and optional tag scope. */
export interface InfoSection {
  /** Optional heading rendered as a `Subtitle` (omitted for the leading identifier block). */
  heading?: string;
  /** The rows in this section. */
  fields?: InfoField[];
  /** Draw an `<hr>` delimiter after this section (expanded variant only). */
  borderAfter?: boolean;
  /**
   * Tag keys to surface *in this section*. When set, this section renders only those tag keys and the
   * catch-all tag block excludes them. When unset on every section, all tags render in the catch-all
   * block (the default behavior).
   */
  tagKeys?: string[];
}

/** Severity of a model-level {@link InfoNote}. */
export enum InfoNoteLevel {
  Warning = 'warning',
  Info = 'info',
}

/** A short model-level note rendered below the summary (e.g. the tree's duplicate-node warning). */
export interface InfoNote {
  level: InfoNoteLevel;
  message: string;
}

/**
 * A normalized, render-agnostic representation of an entity / file / repo, consumed by the shared
 * {@link EntitySummary} renderer (graph panel + hover overlays), fed by the adapters below.
 */
export interface InfoModel {
  /** Display label for the kind (e.g. `File`, `Repo`, `Device`, `Windows Process`). */
  kind: string;
  /** Primary title (name / sha256 / url). */
  title: string;
  /** Optional link for the title (omitted when there's no detail page, e.g. an unsaved request). */
  titleHref?: string;
  description?: string | null;
  /** Ordered sections — intentionally customizable per (pseudo-)entity kind. */
  sections: InfoSection[];
  /**
   * The resource this summary describes (File / Repo / entity kind). When set, tags render as links to
   * that resource's browse page filtered by the tag. Omitted for pseudo-resources (e.g. a Tag node).
   */
  resource?: Entities;
  /** Nested tags for the shared tag renderer. */
  tags?: Tags;
  /** Optional notes rendered below the summary. */
  notes?: InfoNote[];
}

/**
 * The non-field parts of a summary that can be hidden via an exclude list. Field rows are excluded by
 * their `label`; these tokens cover the parts that aren't fields.
 */
export enum SummaryPart {
  /** The uppercase kind label (e.g. `DEVICE`). */
  Kind = 'kind',
  /** The primary title (name / sha256 / url). */
  Title = 'title',
  /** The description / markdown block. */
  Description = 'description',
  /** The tags block. */
  Tags = 'tags',
}

/** The resolved visibility of a summary's parts after applying an exclude list. */
export interface SummaryVisibility {
  kind: boolean;
  title: boolean;
  description: boolean;
  tags: boolean;
  /** Sections with excluded field rows removed and now-empty sections dropped. */
  sections: InfoSection[];
}

/**
 * Resolve which parts of a summary should render given an exclude list.
 *
 * Each exclude entry matches a field's `label` (case-insensitive) or a {@link SummaryPart} token. Use
 * this to omit redundant info — e.g. a list that already shows the entity name passes `[SummaryPart.Title]`
 * so the preview doesn't repeat it, while the graph node overlay passes nothing and shows everything.
 *
 * @param model - The model to filter.
 * @param exclude - Labels / part tokens to hide (case-insensitive).
 * @returns Which top-level parts to show and the surviving sections.
 */
export function applyExclusions(model: InfoModel, exclude?: string[]): SummaryVisibility {
  const hidden = new Set((exclude ?? []).map((e) => e.toLowerCase()));
  const isHidden = (key: string) => hidden.has(key.toLowerCase());
  const sections = model.sections
    .map((s) => ({ ...s, fields: s.fields?.filter((f) => !isHidden(f.label)) }))
    .filter((s) => (s.fields?.length ?? 0) > 0 || (s.tagKeys?.length ?? 0) > 0);
  return {
    kind: !isHidden(SummaryPart.Kind),
    title: !isHidden(SummaryPart.Title),
    description: !isHidden(SummaryPart.Description),
    tags: !isHidden(SummaryPart.Tags),
    sections,
  };
}

/** Stringify an unknown array element for display (objects -> JSON, scalars -> string). */
function scalarString(v: unknown): string {
  if (v === null || v === undefined) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'bigint') return v.toString();
  return JSON.stringify(v);
}

/**
 * Convert an entity request's flat tags (`{ key: [values] }`) into the nested {@link Tags} shape
 * (`{ key: { value: [] } }`) so the shared tag renderer can be reused.
 */
export function flatTagsToTags(flat: Record<string, string[]>): Tags {
  const out: Tags = {};
  for (const [key, values] of Object.entries(flat ?? {})) {
    out[key] = Object.fromEntries((values ?? []).map((v) => [v, []]));
  }
  return out;
}

/** True when a field carries a displayable value (drops `undefined`/empty so no blank rows render). */
function hasValue(f: InfoField): boolean {
  if (f.links && f.links.length > 0) return true;
  return f.value !== undefined && f.value !== null && f.value !== '' && !(Array.isArray(f.value) && f.value.length === 0);
}

/** Drop `undefined`/empty fields so the renderer doesn't show blank rows. */
function compact(fields: InfoField[]): InfoField[] {
  return fields.filter(hasValue);
}

/**
 * Drop empty fields from each section and then drop sections that end up with no visible content, so
 * the expanded variant never renders a lone heading + delimiter.
 */
function compactSections(sections: InfoSection[]): InfoSection[] {
  return sections
    .map((s) => ({ ...s, fields: s.fields ? compact(s.fields) : undefined }))
    .filter((s) => (s.fields?.length ?? 0) > 0 || (s.tagKeys?.length ?? 0) > 0);
}

/** Build the detail-page link for an entity kind + id (falls back to undefined for unknown kinds). */
function entityHref(kind: Entities, id: string): string | undefined {
  const base = getDetailsBasePathByEntity(kind);
  return base ? `${base}/${id}` : undefined;
}

/** Short label for a sample's origin (the origin variant name, e.g. `Downloaded`, `Carved`). */
function originLabel(origin?: Origin): string | undefined {
  if (!origin) return undefined;
  // a Rust unit variant (`None`) serializes as the bare string "None", not `{ None: ... }`. `Object.keys` on a
  // string returns character indices ("0","1",…), so guard it: a bare string carries no meaningful origin.
  if (typeof origin === 'string') return undefined;
  return Object.keys(origin)[0];
}

/** First 12 chars of a sha256, for a compact provenance reference. */
function shortSha(sha?: string): string {
  return sha ? sha.slice(0, 12) : '';
}

/**
 * A compact human-readable provenance line for a sample's {@link Origin} — the variant plus its key payload
 * (parent sha, download url, transform command, wire/incident endpoints, source repo/commit, carved parent),
 * so an analyst sees *where the file came from* without opening the file page. Returns `undefined` for the
 * `None` origin (nothing to say).
 *
 * @param origin - The submission's origin (tagged union).
 * @returns A short `Variant: detail` string, or `undefined`.
 */
export function originDetail(origin?: Origin): string | undefined {
  // guard the bare-string unit variant (`None`) — it has no payload to describe (see `originLabel`)
  if (!origin || typeof origin === 'string') return undefined;
  if (origin.Downloaded) return `Downloaded: ${origin.Downloaded.url}`;
  if (origin.Unpacked)
    return `Unpacked from ${shortSha(origin.Unpacked.parent)}${origin.Unpacked.tool ? ` (${origin.Unpacked.tool})` : ''}`;
  if (origin.Transformed) {
    const t = origin.Transformed;
    return `Transformed from ${shortSha(t.parent)}${t.tool ? ` via ${t.tool}` : ''}${t.cmd ? `: ${t.cmd}` : ''}`;
  }
  if (origin.Wire) {
    const w = origin.Wire;
    return `Wire (${w.sniffer})${w.source ? ` ${w.source}` : ''}${w.destination ? ` → ${w.destination}` : ''}`;
  }
  if (origin.Incident) return `Incident ${origin.Incident.incident}${origin.Incident.cover_term ? ` (${origin.Incident.cover_term})` : ''}`;
  if (origin.MemoryDump) return `Memory dump from ${shortSha(origin.MemoryDump.parent)}`;
  if (origin.Source) return `Built from ${origin.Source.repo}${origin.Source.commit ? `@${shortSha(origin.Source.commit)}` : ''}`;
  if (origin.Carved) return `Carved from ${shortSha(origin.Carved.parent)}${origin.Carved.tool ? ` (${origin.Carved.tool})` : ''}`;
  return undefined;
}

/** Deduplicate a list of maybe-empty strings, preserving first-seen order. */
export function uniqStrings(values: (string | null | undefined)[]): string[] {
  return [...new Set(values.filter((v): v is string => v != null && v !== ''))];
}

/** Format an ISO timestamp as `YYYY-MM-DD HH:MM:SS` (mirrors the shared `<Time>` display). */
export function formatTimestamp(iso: string): string {
  const [date, rest] = iso.split('T');
  if (!rest) return iso;
  return `${date} ${rest.split('.')[0]}`;
}

/**
 * Backend-generated intrinsic tag keys for a WindowsProcess. These duplicate the typed metadata fields shown
 * in the details body, so they're excluded from the catch-all Tags block to avoid showing the same process
 * data twice (the header curates/relabels a subset of these separately).
 */
const WINDOWS_PROCESS_INTRINSIC_TAG_KEYS = [
  'PID',
  'ParentPID',
  'ProcessName',
  'ProcessImagePath',
  'ProcessCommand',
  'ProcessOffset',
  'ProcessThreads',
  'ProcessHandles',
  'ProcessIsWow64',
  'ProcessSessionID',
];

/**
 * Per-kind metadata field builders for a full (typed) entity. Returns leading identifier rows and the
 * per-kind detail rows; {@link treeNodeToInfo} assembles these into sections.
 *
 * @param entity - The typed entity tree node (kind-tagged, with resolved metadata).
 * @returns The entity's leading identifier rows and its per-kind detail field rows.
 */
export function entityFields(entity: NonNullable<TreeNode['Entity']>): { identifiers: InfoField[]; fields: InfoField[] } {
  const m = entity.metadata as Record<string, Record<string, unknown>>;
  switch (entity.kind) {
    case Entities.Device: {
      const d = m.Device as {
        urls?: string[];
        vendors?: { name: string; id: string }[];
        critical_system?: boolean;
        sensitive_location?: boolean;
        critical_sectors?: string[];
      };
      return {
        identifiers: [],
        fields: compact([
          // each vendor is a known entity — link its name straight to the vendor's details page
          {
            label: 'Vendors',
            links: d.vendors?.map((v) => ({ text: v.name, href: entityHref(Entities.Vendor, v.id) })),
            layout: FieldLayout.Inline,
          },
          { label: 'Critical System', value: !!d.critical_system, danger: !!d.critical_system },
          { label: 'Critical Sectors', value: d.critical_sectors, danger: true, layout: FieldLayout.Inline },
          { label: 'Sensitive Location', value: !!d.sensitive_location, danger: !!d.sensitive_location },
          { label: 'URLs', value: d.urls },
        ]),
      };
    }
    case Entities.Vendor: {
      const v = m.Vendor as { countries?: { name: string }[]; critical_sectors?: string[] };
      return {
        identifiers: [],
        fields: compact([
          { label: 'Countries', value: v.countries?.map((c) => c.name), layout: FieldLayout.Inline },
          { label: 'Critical Sectors', value: v.critical_sectors, danger: true, layout: FieldLayout.Inline },
        ]),
      };
    }
    case Entities.Collection: {
      const c = m.Collection as {
        collection_kind?: string;
        collection_tags?: Record<string, string[]>;
        start?: string | null;
        end?: string | null;
        tags_case_insensitive?: boolean;
        ignore_groups?: boolean;
      };
      // the tag filter that defines the collection — flatten `{key: [values]}` into `key: v1, v2` rows
      const collectionTags = c.collection_tags
        ? Object.entries(c.collection_tags).map(([key, values]) => `${key}: ${(values ?? []).join(', ')}`)
        : undefined;
      return {
        identifiers: [],
        fields: compact([
          { label: 'Type', value: c.collection_kind, layout: FieldLayout.Inline },
          { label: 'Collection Tags', value: collectionTags },
          { label: 'Newest', value: c.start ?? undefined, render: FieldRender.Time },
          { label: 'Oldest', value: c.end ?? undefined, render: FieldRender.Time },
          { label: 'Case Insensitive Tags', value: !!c.tags_case_insensitive },
          { label: 'Ignore Groups', value: !!c.ignore_groups },
        ]),
      };
    }
    case Entities.FileSystem: {
      const fs = m.FileSystem as { sha256: string; tools?: string[] };
      // A filesystem's sha256 identifies the filesystem entity itself, not a file — link to its detail page.
      return {
        identifiers: compact([{ label: 'SHA256', value: fs.sha256, href: entityHref(entity.kind, entity.id) }]),
        fields: compact([{ label: 'Tools', value: fs.tools, layout: FieldLayout.Inline }]),
      };
    }
    case Entities.Folder: {
      const fld = m.Folder as { all_sha256: string; names_sha256?: string; data_sha256?: string; filesystem_id?: string };
      return {
        // The folder's aggregate sha256 (all files' names + data) is its complete identity → folder details.
        identifiers: compact([{ label: 'SHA256', value: fld.all_sha256, href: entityHref(entity.kind, entity.id) }]),
        fields: compact([
          {
            label: 'Filesystem',
            value: fld.filesystem_id,
            href: fld.filesystem_id ? entityHref(Entities.FileSystem, fld.filesystem_id) : undefined,
          },
          // names_sha256 / data_sha256 each describe only part of the folder — link to browse sibling
          // folders sharing that partial hash rather than to a detail page.
          {
            label: 'Names SHA256',
            value: fld.names_sha256,
            href: fld.names_sha256 ? buildTagBrowseHref(Entities.Folder, 'names_sha256', fld.names_sha256) : undefined,
          },
          {
            label: 'Data SHA256',
            value: fld.data_sha256,
            href: fld.data_sha256 ? buildTagBrowseHref(Entities.Folder, 'data_sha256', fld.data_sha256) : undefined,
          },
        ]),
      };
    }
    case Entities.WindowsProcess: {
      const p = m.WindowsProcess as {
        pid?: number;
        parent_pid?: number;
        name?: string;
        image_path?: string;
        command?: string;
        offset?: number;
        threads?: number;
        handles?: number;
        is_wow64?: boolean;
        session_id?: number;
        create_time?: string;
        exit_time?: string;
      };
      return {
        identifiers: [],
        // `name` is intentionally omitted — it duplicates the entity name shown as the title. Image Path and
        // Command are the long, high-value fields for triage, so they render STACKED (full-width, wrapping)
        // rather than inline where a long command/path would crowd the label.
        fields: compact([
          { label: 'PID', value: p.pid },
          { label: 'Parent PID', value: p.parent_pid },
          { label: 'Image Path', value: p.image_path, render: FieldRender.Path, layout: FieldLayout.Stacked },
          { label: 'Command', value: p.command, render: FieldRender.Path, layout: FieldLayout.Stacked },
          { label: 'Offset', value: p.offset != null ? formatAddress(p.offset) : undefined, layout: FieldLayout.Inline },
          { label: 'Threads', value: p.threads, render: FieldRender.Text, layout: FieldLayout.Inline },
          { label: 'Handles', value: p.handles },
          { label: 'WOW64', value: !!p.is_wow64 },
          { label: 'Session', value: p.session_id, render: FieldRender.Text, layout: FieldLayout.Inline },
          { label: 'Created', value: p.create_time, render: FieldRender.Time },
          { label: 'Exited', value: p.exit_time, render: FieldRender.Time },
        ]),
      };
    }
    case Entities.WindowsProcessTree: {
      const t = m.WindowsProcessTree as { tools?: string[] };
      return { identifiers: [], fields: compact([{ label: 'Tools', value: t.tools, layout: FieldLayout.Inline }]) };
    }
    case Entities.NetworkConnection: {
      const n = m.NetworkConnection as {
        protocol?: string;
        source?: string;
        source_port?: number;
        destination?: string;
        destination_port?: number;
        state?: string;
        pid?: number;
        process?: string;
        create_time?: string;
      };
      const endpoint = (addr?: string, port?: number) => (addr ? (port != null ? `${addr}:${port}` : addr) : undefined);
      return {
        identifiers: [],
        fields: compact([
          { label: 'Protocol', value: n.protocol, layout: FieldLayout.Inline },
          { label: 'Source', value: endpoint(n.source, n.source_port), layout: FieldLayout.Inline },
          { label: 'Destination', value: endpoint(n.destination, n.destination_port), layout: FieldLayout.Inline },
          { label: 'State', value: n.state, layout: FieldLayout.Inline },
          { label: 'PID', value: n.pid },
          { label: 'Process', value: n.process, layout: FieldLayout.Inline },
          { label: 'Created', value: n.create_time, render: FieldRender.Time },
        ]),
      };
    }
    case Entities.SigmaRule: {
      const s = m.SigmaRule as { rule?: string; score?: number; applies_to?: string[]; actions?: unknown[] };
      return {
        identifiers: [],
        fields: compact([
          { label: 'Score', value: s.score },
          { label: 'Applies To', value: s.applies_to, layout: FieldLayout.Inline },
          { label: 'Actions', value: s.actions ? s.actions.length : undefined },
          // the raw sigma YAML — the defining content of the rule, shown as a collapsible code block
          { label: 'Rule', value: s.rule, render: FieldRender.Code },
        ]),
      };
    }
    case Entities.Flag: {
      const fl = m.Flag as { suspicion?: number; confidence?: string; content?: string | null; reasoning?: string };
      return {
        identifiers: [],
        fields: compact([
          { label: 'Suspicion', value: fl.suspicion, danger: (fl.suspicion ?? 0) > 0 },
          { label: 'Confidence', value: fl.confidence, layout: FieldLayout.Inline },
          { label: 'Content', value: fl.content ?? undefined, render: FieldRender.Text },
          { label: 'Reasoning', value: fl.reasoning, render: FieldRender.Text, layout: FieldLayout.Inline },
        ]),
      };
    }
    case Entities.Incident: {
      const i = m.Incident as {
        cover_term?: string | null;
        mission_teams?: string[];
        networks?: string[];
        machines?: string[];
        locations?: string[];
      };
      return {
        identifiers: [],
        fields: compact([
          { label: 'Cover Term', value: i.cover_term ?? undefined, layout: FieldLayout.Inline },
          { label: 'Mission Teams', value: i.mission_teams, layout: FieldLayout.Inline },
          { label: 'Networks', value: i.networks, layout: FieldLayout.Inline },
          { label: 'Machines', value: i.machines, layout: FieldLayout.Inline },
          { label: 'Locations', value: i.locations, layout: FieldLayout.Inline },
        ]),
      };
    }
    case Entities.CompiledFunction: {
      const f = m.CompiledFunction as { address?: number; disassembly?: { address?: number; instruction?: string }[] };
      // format the instruction list into an aligned `address  mnemonic` listing for the code block
      const asm = f.disassembly
        ?.map((ins) => `${ins.address != null ? formatAddress(ins.address) : ''}  ${ins.instruction ?? ''}`.trim())
        .join('\n');
      return {
        identifiers: [],
        fields: compact([
          { label: 'Address', value: f.address != null ? formatAddress(f.address) : undefined, layout: FieldLayout.Inline },
          { label: 'Instructions', value: f.disassembly ? f.disassembly.length : undefined },
          { label: 'Disassembly', value: asm || undefined, render: FieldRender.Code },
        ]),
      };
    }
    case Entities.DecompiledFunction: {
      const f = m.DecompiledFunction as { address?: number; tools?: string[]; content?: string };
      return {
        identifiers: [],
        fields: compact([
          { label: 'Address', value: f.address != null ? formatAddress(f.address) : undefined, layout: FieldLayout.Inline },
          { label: 'Tools', value: f.tools, layout: FieldLayout.Inline },
          // the decompiled source — the defining content, shown as a collapsible code block
          { label: 'Decompilation', value: f.content, render: FieldRender.Code },
        ]),
      };
    }
    case Entities.PeSection: {
      const s = m.PeSection as { md5?: string; raw_size?: number; virtual_size?: number; entropy?: number };
      return {
        identifiers: compact([{ label: 'MD5', value: s.md5 }]),
        fields: compact([
          { label: 'Raw Size', value: s.raw_size, layout: FieldLayout.Inline },
          { label: 'Virtual Size', value: s.virtual_size, layout: FieldLayout.Inline },
          { label: 'Entropy', value: s.entropy, layout: FieldLayout.Inline },
        ]),
      };
    }
    case Entities.PeImport: {
      const p = m.PeImport as { functions?: string[] };
      return {
        identifiers: [],
        fields: compact([
          { label: 'Functions', value: p.functions?.length, layout: FieldLayout.Inline },
          { label: 'Imported', value: p.functions },
        ]),
      };
    }
    default:
      return { identifiers: [], fields: [] };
  }
}

/**
 * Adapt a graph {@link TreeNode} (Sample / Repo / Tag / Entity, full objects) into an {@link InfoModel}.
 *
 * @param node - The tree node to describe.
 * @returns A normalized info model, or `null` for an empty/unknown node.
 */
export function treeNodeToInfo(node: TreeNode): InfoModel | null {
  if (node[TreeNodeKey.Sample]) {
    const s = node[TreeNodeKey.Sample];
    const subs = s.submissions ?? [];
    // aggregate ("mash") provenance across every submission (deduped) rather than paging through one at a
    // time — a file can be submitted multiple times with different submitters/groups/origins/descriptions
    const submitters = uniqStrings(subs.map((x) => x.submitter));
    const groups = uniqStrings(subs.flatMap((x) => x.groups ?? []));
    const origins = uniqStrings(subs.map((x) => originLabel(x.origin)));
    // richer per-submission provenance lines (deduped) beyond the bare variant name
    const originDetails = uniqStrings(subs.map((x) => originDetail(x.origin)));
    const uploads = uniqStrings(subs.map((x) => x.uploaded)).sort();
    const descriptions = uniqStrings(subs.map((x) => x.description));
    return {
      kind: 'File',
      title: getNodeName(node, 1000) || s.sha256,
      titleHref: `/file/${s.sha256}`,
      resource: Entities.File,
      // combine every submission's (unique) description, rendered as collapsible markdown
      description: descriptions.length > 0 ? descriptions.join('\n\n') : null,
      sections: compactSections([
        {
          // sha1/md5 are alternate identifiers of this same file — link them to its details page too
          fields: [
            { label: 'SHA256', value: s.sha256, href: `/file/${s.sha256}` },
            { label: 'SHA1', value: s.sha1, href: `/file/${s.sha256}` },
            { label: 'MD5', value: s.md5, href: `/file/${s.sha256}` },
          ],
          borderAfter: true,
        },
        {
          // count lives in the heading (no badge); the filename is already the title, so it's not repeated.
          // only submitter/origin/groups carry value badges — uploaded is plain text, hashes are links.
          heading: `Submission(s) ${subs.length}`,
          fields: [
            { label: 'Uploaded', value: uploads.map(formatTimestamp).join(', ') || undefined, render: FieldRender.Text },
            { label: 'Submitter', value: submitters, layout: FieldLayout.Inline },
            { label: 'Groups', value: groups, layout: FieldLayout.Inline },
            // prefer the richer provenance lines when available; fall back to the bare variant name(s)
            originDetails.length > 0
              ? { label: 'Origin', value: originDetails, render: FieldRender.Text }
              : { label: 'Origin', value: origins, layout: FieldLayout.Inline },
          ],
        },
      ]),
      tags: s.tags,
    };
  }
  if (node[TreeNodeKey.Repo]) {
    const r = node[TreeNodeKey.Repo];
    // The default checkout is a tagged union (Commit / Branch / Tag) — surface whichever is set.
    const checkout = r.default_checkout ? Object.entries(r.default_checkout).find(([, v]) => v != null) : undefined;
    return {
      kind: 'Repo',
      title: r.url,
      titleHref: `/repo/${r.url}`,
      resource: Entities.Repo,
      sections: compactSections([
        {
          fields: [
            { label: 'Provider', value: r.provider, layout: FieldLayout.Inline },
            { label: 'User', value: r.user, layout: FieldLayout.Inline },
            { label: 'Name', value: r.name, layout: FieldLayout.Inline },
          ],
          borderAfter: true,
        },
        {
          fields: [
            // Default Checkout can be a full commit hash — leave it to auto-derive (stacked)
            { label: 'Default Checkout', value: checkout ? `${checkout[0]}: ${checkout[1]}` : undefined },
            { label: 'Submissions', value: r.submissions?.length },
            { label: 'Earliest Commit', value: r.earliest, render: FieldRender.Time },
          ],
        },
      ]),
      tags: r.tags,
    };
  }
  if (node[TreeNodeKey.Tag]) {
    // a Tag node carries flat `{ key: string[] }` tags — normalize to nested Tags for the renderer
    return { kind: 'Tag', title: 'Tags', sections: [], tags: flatTagsToTags(node[TreeNodeKey.Tag].tags) };
  }
  if (node[TreeNodeKey.Entity]) {
    const e = node[TreeNodeKey.Entity];
    const { identifiers, fields } = entityFields(e);
    // a WindowsProcess's intrinsic tags duplicate its typed fields (shown above), so drop them from the
    // catch-all Tags block; other kinds show all their tags
    const bodyTags = e.kind === Entities.WindowsProcess ? filterExcludedTags(e.tags, WINDOWS_PROCESS_INTRINSIC_TAG_KEYS) : e.tags;
    return {
      kind: entityLabel(e.kind),
      title: e.name,
      titleHref: entityHref(e.kind, e.id),
      resource: e.kind,
      description: e.description,
      sections: compactSections([
        // the ID always links to this entity's own details page (its type + id); inline so the label
        // and UUID share one line
        {
          fields: [{ label: 'ID', value: e.id, href: entityHref(e.kind, e.id), layout: FieldLayout.Inline }, ...identifiers],
          borderAfter: fields.length > 0,
        },
        { fields },
      ]),
      tags: bodyTags,
    };
  }
  return null;
}

/**
 * Adapt an unsaved {@link EntityRequest} (kind-tagged raw metadata, no id) into an {@link InfoModel}.
 * Metadata fields are flattened generically since a request carries ids/codes rather than resolved objects.
 *
 * @param req - The entity request to describe.
 * @returns A normalized info model (no title link — the entity isn't created yet).
 */
export function entityRequestToInfo(req: EntityRequest): InfoModel {
  const kind = entityRequestKind(req);
  const inner = typeof req.metadata === 'object' ? (req.metadata[kind] as Record<string, unknown> | undefined) : undefined;
  const fields: InfoField[] = inner
    ? compact(
        Object.entries(inner)
          // `name` duplicates the title (req.name) shown at the top — don't repeat it in the body
          .filter(([key]) => key.toLowerCase() !== 'name')
          .map(([key, value]) => ({
            label: humanize(key),
            value: Array.isArray(value)
              ? (value as unknown[]).map(scalarString)
              : typeof value === 'object' && value !== null
                ? JSON.stringify(value)
                : (value as string | number | boolean | null),
          })),
      )
    : [];
  return {
    kind: entityLabel(kind),
    title: req.name,
    // the request's kind string is an entity kind; if it's not browsable the tag link builder no-ops
    resource: kind as Entities,
    description: req.description,
    sections: compactSections([{ fields }]),
    tags: flatTagsToTags(req.tags),
  };
}
