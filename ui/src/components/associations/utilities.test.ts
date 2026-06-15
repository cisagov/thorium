import { describe, it, expect } from 'vitest';

// project imports
import { formatSubmissionNames, formatTagNames, getEdgeLabel, getNodeName, getOrCreate, stripFilePath, truncateString } from './utilities';
import { Direction } from '@models/trees';
import type { BranchNode, Graph, TreeNode } from '@models/trees';
import type { SubmissionChunk, Sample } from '@models/files';
import type { Repo } from '@models/entities/repos';
import type { EntityTypes } from '@models/entities/entities';
import { AssociationKind } from '@models/associations';

describe('stripFilePath', () => {
  it('returns the filename from a full path', () => {
    expect(stripFilePath('/home/user/file.txt')).toBe('file.txt');
  });

  it('returns the string itself if no slashes', () => {
    expect(stripFilePath('file.txt')).toBe('file.txt');
  });

  it('returns full path when pop returns empty string', () => {
    expect(stripFilePath('/home/user/')).toBe('/home/user/');
  });
});

describe('formatSubmissionNames', () => {
  it('returns comma-separated unique names from submissions', () => {
    const subs = [{ name: '/path/a.exe' }, { name: '/other/b.dll' }] as SubmissionChunk[];
    expect(formatSubmissionNames(subs)).toBe('a.exe, b.dll');
  });

  it('deduplicates identical filenames', () => {
    const subs = [{ name: 'test.exe' }, { name: 'test.exe' }] as SubmissionChunk[];
    expect(formatSubmissionNames(subs)).toBe('test.exe');
  });

  it('returns "No Valid Name" for empty submissions', () => {
    expect(formatSubmissionNames([])).toBe('No Valid Name');
  });

  it('returns empty string for submissions with no name', () => {
    const subs = [{ name: '' }] as SubmissionChunk[];
    expect(formatSubmissionNames(subs)).toBe('');
  });
});

describe('formatTagNames', () => {
  it('formats tags as quoted key: value pairs', () => {
    const tags = { TLP: ['RED'], MBC: ['T1059'] };
    expect(formatTagNames(tags, false)).toBe('"TLP: RED", "MBC: T1059"');
  });

  it('truncates long tag strings when truncate is true', () => {
    const tags = { VeryLongTagKey: ['very_long_value_that_exceeds_limit'] };
    const result = formatTagNames(tags, true);
    expect(result.length).toBeLessThanOrEqual(33);
    expect(result).toContain('...');
  });

  it('does not truncate short tag strings even when truncate is true', () => {
    const tags = { TLP: ['RED'] };
    const result = formatTagNames(tags, true);
    expect(result).toBe('"TLP: RED"');
  });

  it('handles empty tags', () => {
    expect(formatTagNames({}, false)).toBe('');
  });
});

describe('truncateString', () => {
  it('returns the full string when under max length', () => {
    expect(truncateString('hello', 10)).toBe('hello');
  });

  it('truncates with ellipsis in the middle', () => {
    const result = truncateString('abcdefghijklmnopqrstuvwxyz', 20);
    expect(result).toContain('...');
    expect(result.length).toBeLessThanOrEqual(22);
  });

  it('truncates to exact length for short max lengths', () => {
    expect(truncateString('abcdefghij', 5)).toBe('abcde');
  });

  it('returns the string when exactly at max length', () => {
    expect(truncateString('12345', 5)).toBe('12345');
  });
});

describe('getNodeName', () => {
  it('returns submission name for Sample nodes', () => {
    const node: TreeNode = { Sample: { submissions: [{ name: 'malware.exe' }] } as unknown as Sample };
    expect(getNodeName(node, 50)).toBe('malware.exe');
  });

  it('returns url for Repo nodes', () => {
    const node: TreeNode = { Repo: { url: 'https://github.com/test' } as unknown as Repo };
    expect(getNodeName(node, 50)).toBe('https://github.com/test');
  });

  it('returns formatted tag for Tag nodes', () => {
    const node: TreeNode = { Tag: { tags: { TLP: ['RED'] } } };
    expect(getNodeName(node, 50)).toBe('"TLP: RED"');
  });

  it('returns entity name for Entity nodes', () => {
    const node: TreeNode = { Entity: { name: 'TestDevice', kind: 'Device' } as unknown as EntityTypes };
    expect(getNodeName(node, 50)).toBe('TestDevice');
  });

  it('truncates long names to maxLength', () => {
    const node: TreeNode = { Repo: { url: 'https://github.com/very/long/repository/path' } as unknown as Repo };
    const result = getNodeName(node, 20);
    expect(result.length).toBeLessThanOrEqual(22);
  });

  it('returns empty string for unknown node types', () => {
    expect(getNodeName({}, 50)).toBe('');
  });
});

describe('getOrCreate', () => {
  it('inserts and returns a freshly created value on a miss', () => {
    const map = new Map<string, number[]>();
    const value = getOrCreate(map, 'a', () => []);
    value.push(1);
    // the returned reference is the one stored in the map, so mutations persist
    expect(map.get('a')).toBe(value);
    expect(map.get('a')).toEqual([1]);
  });

  it('returns the existing value without invoking the factory on a hit', () => {
    const map = new Map<string, number[]>([['a', [1]]]);
    let factoryCalls = 0;
    const value = getOrCreate(map, 'a', () => {
      factoryCalls += 1;
      return [];
    });
    expect(value).toEqual([1]);
    expect(factoryCalls).toBe(0);
  });
});

describe('getEdgeLabel', () => {
  const emptyGraph = { data_map: {}, branches: {}, initial: [], growable: [], id: '' } as Graph;

  function branchWith(relationship: Record<string, unknown>): BranchNode {
    return {
      relationship,
      node: 'b',
      direction: Direction.To,
      relationship_hash: 'h1',
    };
  }

  it('returns truncated URL for Downloaded origins with long URLs', () => {
    const node = branchWith({ Origin: { Downloaded: { url: 'https://example.com/long/path' } } });
    const result = getEdgeLabel('b', 'a', node, emptyGraph);
    expect(result).toContain('...');
  });

  it('returns full label for Downloaded origins with short URLs', () => {
    const node = branchWith({ Origin: { Downloaded: { url: 'http://x.co' } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Downloaded: http://x.co');
  });

  it('returns tool name for Unpacked origins', () => {
    const node = branchWith({ Origin: { Unpacked: { tool: 'unzip', parent: '', dangling: false } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Unpacked: unzip');
  });

  it('returns "Unpacked" when no tool specified', () => {
    const node = branchWith({ Origin: { Unpacked: { parent: '', dangling: false } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Unpacked');
  });

  it('returns tool name for Transformed origins', () => {
    const node = branchWith({ Origin: { Transformed: { tool: 'deobfuscator', parent: '', dangling: false, flags: [] } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Transformed: deobfuscator');
  });

  it('returns "Transformed" when no tool specified', () => {
    const node = branchWith({ Origin: { Transformed: { parent: '', dangling: false, flags: [] } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Transformed');
  });

  it('returns incident name for Incident origins', () => {
    const node = branchWith({ Origin: { Incident: { incident: 'INC-42' } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Incident: INC-42');
  });

  it('returns "Memory Dump" for MemoryDump origins', () => {
    const node = branchWith({ Origin: { MemoryDump: { parent: '', dangling: false, reconstructed: [] } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Memory Dump');
  });

  it('returns sniffer name for Wire origins', () => {
    const node = branchWith({ Origin: { Wire: { sniffer: 'tcpdump' } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Wire: tcpdump');
  });

  it('returns truncated commit for Source origins', () => {
    const node = branchWith({ Origin: { Source: { repo: '', commit: 'abc123def456', flags: [], system: 'make', supporting: false } } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Commit: abc123de');
  });

  it('returns association kind for Association relationships', () => {
    const node = branchWith({ Association: { kind: AssociationKind.AssociatedWith } });
    expect(getEdgeLabel('b', 'a', node, emptyGraph)).toBe('Association: Associated With');
  });

  it('returns empty string for tag relationships with no tags', () => {
    const graphWithTarget = { ...emptyGraph, data_map: { b: {} } } as Graph;
    const node = branchWith({});
    expect(getEdgeLabel('b', 'a', node, graphWithTarget)).toBe('');
  });

  it('returns formatted tag pairs for tag-based relationships', () => {
    const graphWithTags = {
      ...emptyGraph,
      data_map: { b: { Tag: { tags: { TLP: ['RED'] } } } },
    } as unknown as Graph;
    const node = branchWith({});
    expect(getEdgeLabel('b', 'a', node, graphWithTags)).toBe('TLP: RED');
  });
});
