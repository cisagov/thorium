import { describe, it, expect } from 'vitest';

// project imports
import {
  applyExclusions,
  entityFields,
  entityRequestToInfo,
  FieldLayout,
  FieldRender,
  flatTagsToTags,
  formatTimestamp,
  InfoField,
  InfoModel,
  originDetail,
  SummaryPart,
  treeNodeToInfo,
  uniqStrings,
} from './info';
import { paramsToClauses } from '@components/shared/inputs/omnibar/urlState';
import { Entities } from '@models/entities/entities';
import { EntityRequest } from '@models/entities/requests';
import { Origin } from '@models/files';
import { TreeNode } from '@models/trees';

/** Flatten every field across a model's sections (order preserved) for assertions. */
function allFields(model: InfoModel | null): InfoField[] {
  return (model?.sections ?? []).flatMap((s) => s.fields ?? []);
}
const labelsOf = (model: InfoModel | null) => allFields(model).map((f) => f.label);
const fieldNamed = (model: InfoModel | null, label: string) => allFields(model).find((f) => f.label === label);

describe('flatTagsToTags', () => {
  it('nests flat key->values[] into key->{value->[]}', () => {
    expect(flatTagsToTags({ family: ['emotet', 'x'], tlp: ['amber'] })).toEqual({
      family: { emotet: [], x: [] },
      tlp: { amber: [] },
    });
  });
  it('handles empty input', () => {
    expect(flatTagsToTags({})).toEqual({});
  });
});

describe('treeNodeToInfo', () => {
  it('returns null for an empty/unknown node', () => {
    expect(treeNodeToInfo({})).toBeNull();
  });

  it('describes a Sample (File) with sha/md5 identifiers and a file link', () => {
    const node = {
      Sample: { sha256: 'a'.repeat(64), sha1: 'b'.repeat(40), md5: 'c'.repeat(32), submissions: [{}], tags: {} },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(info?.kind).toBe('File');
    expect(info?.titleHref).toBe(`/file/${'a'.repeat(64)}`);
    expect(labelsOf(info)).toEqual(expect.arrayContaining(['SHA256', 'SHA1', 'MD5']));
    expect(fieldNamed(info, 'SHA256')?.href).toBe(`/file/${'a'.repeat(64)}`);
  });

  it('aggregates provenance across all File submissions (no filename, count in the heading)', () => {
    const node = {
      Sample: {
        sha256: 'a'.repeat(64),
        sha1: 'b'.repeat(40),
        md5: 'c'.repeat(32),
        tags: {},
        submissions: [
          { name: 'old.exe', uploaded: '2024-01-01T00:00:00', submitter: 'alice', groups: ['g1'], origin: { None: 'None' } },
          {
            name: 'new.exe',
            uploaded: '2025-06-01T00:00:00',
            submitter: 'bob',
            groups: ['g2'],
            origin: { Downloaded: { url: 'http://x' } },
          },
        ],
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    // submitter/groups mash into deduped arrays across submissions
    expect(fieldNamed(info, 'Submitter')?.value).toEqual(['alice', 'bob']);
    expect(fieldNamed(info, 'Groups')?.value).toEqual(['g1', 'g2']);
    // origin shows the richer provenance line(s); the `None` origin contributes nothing
    expect(fieldNamed(info, 'Origin')?.value).toEqual(['Downloaded: http://x']);
    // filename is dropped (duplicates the title) and the count moves to the section heading (no field)
    expect(fieldNamed(info, 'Filename')).toBeUndefined();
    expect(fieldNamed(info, 'Submissions')).toBeUndefined();
    expect(info?.sections.some((s) => s.heading === 'Submission(s) 2')).toBe(true);
  });

  it('describes a Repo with a repo link and enriched details', () => {
    const node = {
      Repo: {
        url: 'github.com/x/y',
        provider: 'GitHub',
        user: 'x',
        name: 'y',
        tags: {},
        default_checkout: { Branch: 'main' },
        submissions: [{}, {}],
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(info?.kind).toBe('Repo');
    expect(info?.titleHref).toBe('/repo/github.com/x/y');
    expect(fieldNamed(info, 'Default Checkout')?.value).toBe('Branch: main');
    expect(fieldNamed(info, 'Submissions')?.value).toBe(2);
  });

  it('describes a Device entity with typed metadata fields + detail link', () => {
    const node = {
      Entity: {
        id: 'dev-1',
        name: 'Router',
        kind: Entities.Device,
        description: 'edge router',
        tags: {},
        metadata: { Device: { vendors: [{ name: 'Acme', id: 'v1' }], critical_system: true, critical_sectors: ['Energy'] } },
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(info?.kind).toBe('Device');
    expect(info?.title).toBe('Router');
    expect(info?.titleHref).toBe('/device/dev-1');
    expect(labelsOf(info)).toEqual(expect.arrayContaining(['ID', 'Vendors', 'Critical System', 'Critical Sectors']));
    expect(fieldNamed(info, 'Critical System')?.danger).toBe(true);
  });

  it('links each Device vendor name directly to its vendor details page', () => {
    const node = {
      Entity: {
        id: 'dev-1',
        name: 'Router',
        kind: Entities.Device,
        tags: {},
        metadata: {
          Device: {
            vendors: [
              { name: 'Acme', id: 'v1' },
              { name: 'Globex', id: 'v2' },
            ],
          },
        },
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'Vendors')?.links).toEqual([
      { text: 'Acme', href: '/vendor/v1' },
      { text: 'Globex', href: '/vendor/v2' },
    ]);
  });

  it('links a FileSystem sha256 to the filesystem detail page, not /file/', () => {
    const node = {
      Entity: {
        id: 'fs-1',
        name: 'root',
        kind: Entities.FileSystem,
        tags: {},
        metadata: { FileSystem: { sha256: 'd'.repeat(64), tools: ['x'] } },
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'SHA256')?.href).toBe('/filesystem/fs-1');
    expect(fieldNamed(info, 'SHA256')?.href).not.toContain('/file/');
  });

  it('links a Folder filesystem_id to the filesystem it belongs to', () => {
    const node = {
      Entity: {
        id: 'fld-1',
        name: 'bin',
        kind: Entities.Folder,
        tags: {},
        metadata: { Folder: { all_sha256: 'e'.repeat(64), filesystem_id: 'fs-9' } },
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'SHA256')?.href).toBe('/folder/fld-1');
    expect(fieldNamed(info, 'Filesystem')?.href).toBe('/filesystem/fs-9');
  });

  it("links a File's sha1 and md5 to the same file details page as sha256", () => {
    const sha = 'a'.repeat(64);
    const node = {
      Sample: { sha256: sha, sha1: 'b'.repeat(40), md5: 'c'.repeat(32), submissions: [{}], tags: {} },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'SHA1')?.href).toBe(`/file/${sha}`);
    expect(fieldNamed(info, 'MD5')?.href).toBe(`/file/${sha}`);
  });

  it('marks short File submission fields inline and hashes stacked', () => {
    const node = {
      Sample: {
        sha256: 'a'.repeat(64),
        sha1: 'b'.repeat(40),
        md5: 'c'.repeat(32),
        tags: {},
        submissions: [{ submitter: 'bob', groups: ['g1'], origin: { None: 'None' } }],
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'Submitter')?.layout).toBe(FieldLayout.Inline);
    expect(fieldNamed(info, 'Groups')?.layout).toBe(FieldLayout.Inline);
    // long hashes carry no inline override — they stack by derivation
    expect(fieldNamed(info, 'SHA256')?.layout).toBeUndefined();
  });

  it('renders a Repo Earliest Commit as a Time and marks provider/user/name inline', () => {
    const node = {
      Repo: {
        url: 'github.com/x/y',
        provider: 'GitHub',
        user: 'x',
        name: 'y',
        tags: {},
        earliest: '2025-01-02T03:04:05',
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'Earliest Commit')?.render).toBe(FieldRender.Time);
    expect(fieldNamed(info, 'Provider')?.layout).toBe(FieldLayout.Inline);
    expect(fieldNamed(info, 'Name')?.layout).toBe(FieldLayout.Inline);
  });

  it('marks Device Vendors and Critical Sectors inline while long-token arrays (URLs) stay stacked', () => {
    const node = {
      Entity: {
        id: 'dev-1',
        name: 'Router',
        kind: Entities.Device,
        tags: {},
        metadata: { Device: { vendors: [{ name: 'Acme', id: 'v1' }], critical_sectors: ['Energy', 'Water'], urls: ['http://x'] } },
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'Critical Sectors')?.layout).toBe(FieldLayout.Inline);
    expect(fieldNamed(info, 'Vendors')?.layout).toBe(FieldLayout.Inline);
    expect(fieldNamed(info, 'URLs')?.layout).toBeUndefined();
  });

  it('links an entity ID to its own details page', () => {
    const node = {
      Entity: { id: 'dev-1', name: 'Router', kind: Entities.Device, tags: {}, metadata: { Device: {} } },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(fieldNamed(info, 'ID')?.href).toBe('/device/dev-1');
  });

  it('tags the model with its browse resource (File / Repo / entity kind)', () => {
    const file = treeNodeToInfo({ Sample: { sha256: 'a'.repeat(64), submissions: [], tags: {} } } as unknown as TreeNode);
    expect(file?.resource).toBe(Entities.File);
    const repo = treeNodeToInfo({ Repo: { url: 'github.com/x/y', provider: 'GitHub', tags: {} } } as unknown as TreeNode);
    expect(repo?.resource).toBe(Entities.Repo);
    const device = treeNodeToInfo({
      Entity: { id: 'd1', name: 'R', kind: Entities.Device, tags: {}, metadata: { Device: {} } },
    } as unknown as TreeNode);
    expect(device?.resource).toBe(Entities.Device);
  });

  it("combines all File submissions' unique descriptions into the model description (not a field row)", () => {
    const node = {
      Sample: {
        sha256: 'a'.repeat(64),
        sha1: 'b'.repeat(40),
        md5: 'c'.repeat(32),
        tags: {},
        submissions: [
          { uploaded: '2024-01-01T00:00:00', description: 'old' },
          { uploaded: '2025-06-01T00:00:00', description: 'newest desc' },
          { uploaded: '2025-07-01T00:00:00', description: 'old' },
        ],
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    // deduped, joined by a blank line (markdown), not repeated per submission
    expect(info?.description).toBe('old\n\nnewest desc');
    expect(fieldNamed(info, 'Description')).toBeUndefined();
  });

  it('links Folder partial hashes to a tag-filtered folder browse that decodes back to a tag clause', () => {
    const node = {
      Entity: {
        id: 'fld-1',
        name: 'bin',
        kind: Entities.Folder,
        tags: {},
        metadata: {
          Folder: { all_sha256: 'e'.repeat(64), names_sha256: 'f'.repeat(64), data_sha256: '1'.repeat(64), filesystem_id: 'fs-9' },
        },
      },
    } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    // complete identity → folder details; sub-hashes → folder browse filtered by that tag
    expect(fieldNamed(info, 'SHA256')?.href).toBe('/folder/fld-1');
    const namesHref = fieldNamed(info, 'Names SHA256')?.href;
    expect(namesHref).toBeTruthy();
    expect(namesHref).toContain('/folders?');
    // round-trip: the query string must decode to a `tag` clause on the field name (proves bracket encoding is correct)
    const params = new URLSearchParams(namesHref!.split('?')[1]);
    const clause = paramsToClauses(params).find((c) => c.category === 'tag');
    expect(clause?.field).toBe('names_sha256');
    expect(clause && !('values' in clause.value) ? clause.value.value : undefined).toBe('f'.repeat(64));
  });

  it('drops sections that have no visible fields (compactSections)', () => {
    // a bare repo with only a provider should not produce an empty "Details" section
    const node = { Repo: { url: 'github.com/x/y', provider: 'GitHub', tags: {} } } as unknown as TreeNode;
    const info = treeNodeToInfo(node);
    expect(info?.sections.some((s) => s.heading === 'Details')).toBe(false);
  });
});

describe('entityRequestToInfo', () => {
  const base: EntityRequest = { name: 'n', metadata: {}, groups: [], tags: {}, description: null };

  it('flattens object-variant metadata into fields and converts tags', () => {
    const req: EntityRequest = {
      ...base,
      name: 'evil.exe',
      metadata: { Device: { urls: ['http://x'], critical_system: true } },
      tags: { family: ['emotet'] },
      description: 'sus',
    };
    const info = entityRequestToInfo(req);
    expect(info.kind).toBe('Device');
    expect(info.title).toBe('evil.exe');
    expect(info.titleHref).toBeUndefined();
    expect(info.description).toBe('sus');
    expect(labelsOf(info)).toEqual(expect.arrayContaining(['Urls', 'Critical System']));
    expect(info.tags).toEqual({ family: { emotet: [] } });
  });

  it('handles unit-variant (string) metadata with no fields', () => {
    const info = entityRequestToInfo({ ...base, metadata: 'Other' });
    expect(info.kind).toBe('Other');
    expect(allFields(info)).toEqual([]);
  });
});

describe('applyExclusions', () => {
  const model: InfoModel = {
    kind: 'Device',
    title: 'Router',
    description: 'edge router',
    sections: [
      { fields: [{ label: 'ID', value: 'dev-1' }], borderAfter: true },
      {
        heading: 'Details',
        fields: [
          { label: 'Vendors', value: ['Acme'] },
          { label: 'PID', value: 5 },
        ],
      },
    ],
    tags: { family: { emotet: [] } },
  };

  it('shows every part with no exclude list', () => {
    const v = applyExclusions(model);
    expect(v).toMatchObject({ kind: true, title: true, description: true, tags: true });
    expect(v.sections).toHaveLength(2);
  });

  it('hides a part token (case-insensitive) without touching fields', () => {
    const v = applyExclusions(model, ['TITLE']);
    expect(v.title).toBe(false);
    expect(v.kind).toBe(true);
    expect(v.sections).toHaveLength(2);
  });

  it('removes matching field rows and drops sections left empty', () => {
    // hiding both Details fields should drop the whole Details section
    const v = applyExclusions(model, ['Vendors', 'pid']);
    const labels = v.sections.flatMap((s) => s.fields ?? []).map((f) => f.label);
    expect(labels).toEqual(['ID']);
    expect(v.sections.some((s) => s.heading === 'Details')).toBe(false);
  });

  it('hides the tags block via the tags token', () => {
    expect(applyExclusions(model, [SummaryPart.Tags]).tags).toBe(false);
  });
});

describe('originDetail', () => {
  it('returns undefined for a missing origin and the None variant', () => {
    expect(originDetail(undefined)).toBeUndefined();
    // the unit `None` origin serializes as the bare string "None", not an object
    expect(originDetail('None')).toBeUndefined();
  });

  it('describes a Downloaded origin with its url', () => {
    expect(originDetail({ Downloaded: { url: 'http://evil.test/x' } })).toBe('Downloaded: http://evil.test/x');
  });

  it('describes an Unpacked origin with and without a tool', () => {
    const parent = 'a'.repeat(64);
    expect(originDetail({ Unpacked: { parent, dangling: false } })).toBe(`Unpacked from ${'a'.repeat(12)}`);
    expect(originDetail({ Unpacked: { parent, tool: 'unzip', dangling: false } })).toBe(`Unpacked from ${'a'.repeat(12)} (unzip)`);
  });

  it('describes a Transformed origin with optional tool and cmd branches', () => {
    const parent = 'b'.repeat(64);
    expect(originDetail({ Transformed: { parent, dangling: false, flags: [] } })).toBe(`Transformed from ${'b'.repeat(12)}`);
    expect(originDetail({ Transformed: { parent, tool: 'strings', cmd: 'strings -a', dangling: false, flags: [] } })).toBe(
      `Transformed from ${'b'.repeat(12)} via strings: strings -a`,
    );
  });

  it('describes a Wire origin with source and destination endpoints', () => {
    expect(originDetail({ Wire: { sniffer: 'zeek' } })).toBe('Wire (zeek)');
    expect(originDetail({ Wire: { sniffer: 'zeek', source: '10.0.0.1', destination: '10.0.0.2' } })).toBe(
      'Wire (zeek) 10.0.0.1 → 10.0.0.2',
    );
  });

  it('describes an Incident origin with an optional cover term', () => {
    expect(originDetail({ Incident: { incident: 'INC-1' } })).toBe('Incident INC-1');
    expect(originDetail({ Incident: { incident: 'INC-1', cover_term: 'Nightfall' } })).toBe('Incident INC-1 (Nightfall)');
  });

  it('describes a MemoryDump origin from its parent sha', () => {
    const parent = 'c'.repeat(64);
    expect(originDetail({ MemoryDump: { parent, dangling: false, reconstructed: [] } })).toBe(`Memory dump from ${'c'.repeat(12)}`);
  });

  it('describes a Source origin with an optional commit', () => {
    expect(originDetail({ Source: { repo: 'github.com/x/y', commit: '', flags: [], system: 'make', supporting: false } })).toBe(
      'Built from github.com/x/y',
    );
    const commit = 'd'.repeat(40);
    expect(originDetail({ Source: { repo: 'github.com/x/y', commit, flags: [], system: 'make', supporting: false } })).toBe(
      `Built from github.com/x/y@${'d'.repeat(12)}`,
    );
  });

  it('describes a Carved origin with an optional tool', () => {
    const parent = 'e'.repeat(64);
    expect(originDetail({ Carved: { parent, dangling: false, carved_origin: 'Unknown' } } as unknown as Origin)).toBe(
      `Carved from ${'e'.repeat(12)}`,
    );
    expect(originDetail({ Carved: { parent, tool: 'binwalk', dangling: false, carved_origin: 'Unknown' } } as unknown as Origin)).toBe(
      `Carved from ${'e'.repeat(12)} (binwalk)`,
    );
  });
});

describe('uniqStrings', () => {
  it('drops null/undefined/empty and dedupes preserving first-seen order', () => {
    expect(uniqStrings(['b', null, 'a', '', 'b', undefined, 'a'])).toEqual(['b', 'a']);
  });

  it('returns an empty array when everything is filtered out', () => {
    expect(uniqStrings([null, undefined, ''])).toEqual([]);
  });
});

describe('formatTimestamp', () => {
  it('splits an ISO timestamp into date + seconds-truncated time', () => {
    expect(formatTimestamp('2026-07-08T13:45:07.123Z')).toBe('2026-07-08 13:45:07');
  });

  it('returns the input unchanged when there is no time component', () => {
    expect(formatTimestamp('2026-07-08')).toBe('2026-07-08');
  });
});

describe('entityFields', () => {
  /** Build a typed entity tree-node payload for a single kind + metadata block. */
  function entity(kind: Entities, metadata: Record<string, unknown>): NonNullable<TreeNode['Entity']> {
    return { kind, id: 'id-1', name: 'thing', metadata: { [kind]: metadata }, tags: {} } as unknown as NonNullable<TreeNode['Entity']>;
  }
  const labels = (kind: Entities, metadata: Record<string, unknown>) => {
    const { identifiers, fields } = entityFields(entity(kind, metadata));
    return [...identifiers, ...fields].map((f) => f.label);
  };
  const field = (kind: Entities, metadata: Record<string, unknown>, label: string): InfoField | undefined => {
    const { identifiers, fields } = entityFields(entity(kind, metadata));
    return [...identifiers, ...fields].find((f) => f.label === label);
  };

  it('returns empty rows for an unhandled kind', () => {
    expect(entityFields(entity(Entities.File, {}))).toEqual({ identifiers: [], fields: [] });
  });

  it('formats a WindowsProcess address via formatAddress and stacks image path / command', () => {
    const meta = { pid: 4, image_path: 'C:/x.exe', command: 'x.exe -run', offset: 0x1000 };
    expect(field(Entities.WindowsProcess, meta, 'Offset')?.value).toBe('0x1000');
    expect(field(Entities.WindowsProcess, meta, 'Image Path')?.layout).toBe(FieldLayout.Stacked);
    expect(field(Entities.WindowsProcess, meta, 'Command')?.render).toBe(FieldRender.Path);
    // `name` duplicates the entity title, so it must never appear as a field row
    expect(labels(Entities.WindowsProcess, { ...meta, name: 'x.exe' })).not.toContain('Name');
  });

  it('joins NetworkConnection endpoints as addr:port only when a port is present', () => {
    const withPort = field(Entities.NetworkConnection, { source: '10.0.0.1', source_port: 80 }, 'Source');
    expect(withPort?.value).toBe('10.0.0.1:80');
    const noPort = field(Entities.NetworkConnection, { destination: '10.0.0.2' }, 'Destination');
    expect(noPort?.value).toBe('10.0.0.2');
  });

  it('renders a SigmaRule rule body as a collapsible code block and counts actions', () => {
    const meta = { rule: 'title: x', score: 5, actions: [{}, {}] };
    expect(field(Entities.SigmaRule, meta, 'Rule')?.render).toBe(FieldRender.Code);
    expect(field(Entities.SigmaRule, meta, 'Actions')?.value).toBe(2);
  });

  it('marks a Flag as danger when suspicion is positive', () => {
    expect(field(Entities.Flag, { suspicion: 3 }, 'Suspicion')?.danger).toBe(true);
    expect(field(Entities.Flag, { suspicion: 0 }, 'Suspicion')?.danger).toBe(false);
  });

  it('lists Incident scalar arrays inline', () => {
    expect(field(Entities.Incident, { mission_teams: ['red'] }, 'Mission Teams')?.layout).toBe(FieldLayout.Inline);
  });

  it('formats a CompiledFunction disassembly into an aligned address/instruction code block', () => {
    const meta = {
      address: 0x400000,
      disassembly: [
        { address: 0x400000, instruction: 'push rbp' },
        { address: 0x400001, instruction: 'ret' },
      ],
    };
    expect(field(Entities.CompiledFunction, meta, 'Address')?.value).toBe('0x400000');
    expect(field(Entities.CompiledFunction, meta, 'Instructions')?.value).toBe(2);
    const asm = field(Entities.CompiledFunction, meta, 'Disassembly');
    expect(asm?.render).toBe(FieldRender.Code);
    expect(asm?.value).toBe('0x400000  push rbp\n0x400001  ret');
  });

  it('renders a DecompiledFunction body as a code block', () => {
    expect(field(Entities.DecompiledFunction, { content: 'int main(){}' }, 'Decompilation')?.render).toBe(FieldRender.Code);
  });

  it('surfaces a PeSection md5 identifier and sizing fields', () => {
    const { identifiers } = entityFields(entity(Entities.PeSection, { md5: 'f'.repeat(32), raw_size: 512 }));
    expect(identifiers.map((f) => f.label)).toContain('MD5');
    expect(field(Entities.PeSection, { md5: 'f'.repeat(32), raw_size: 512 }, 'Raw Size')?.value).toBe(512);
  });

  it('reports PeImport function count and the imported names list', () => {
    const meta = { functions: ['CreateFileW', 'ReadFile'] };
    expect(field(Entities.PeImport, meta, 'Functions')?.value).toBe(2);
    expect(field(Entities.PeImport, meta, 'Imported')?.value).toEqual(['CreateFileW', 'ReadFile']);
  });
});
