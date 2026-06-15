import { describe, it, expect } from 'vitest';

// project imports
import { getLinkEndpoints, classifyNode, processInitialGraphData } from './data';
import { VisualState } from './types';
import { Direction, BlankGraph, NodeType } from '@models/trees';
import type { Graph, BranchNode } from '@models/trees';
import { Entities } from '@models/entities/entities';
import type { EntityTypes } from '@models/entities/entities';
import type { Sample } from '@models/files';
import type { Repo } from '@models/entities/repos';

function graphWith(patch: Partial<Graph>): Graph {
  return { ...structuredClone(BlankGraph), ...patch };
}

function branch(node: string, hash: string, direction = Direction.To): BranchNode {
  return { node, direction, relationship_hash: hash, relationship: {} };
}

describe('getLinkEndpoints', () => {
  it('extracts source and target from string-based links', () => {
    const link = { source: 'a', target: 'b', label: '', bidirectional: false };
    expect(getLinkEndpoints(link)).toEqual({ source: 'a', target: 'b' });
  });

  it('extracts IDs from object-based links', () => {
    const link = {
      source: { id: 'src-id' } as unknown as string,
      target: { id: 'tgt-id' } as unknown as string,
      label: '',
      bidirectional: false,
    };
    expect(getLinkEndpoints(link)).toEqual({ source: 'src-id', target: 'tgt-id' });
  });
});

describe('classifyNode', () => {
  it('classifies a Sample node as File type', () => {
    const graph = graphWith({
      data_map: { n1: { Sample: { submissions: [{ name: 'test.exe' }] } as unknown as Sample } },
      initial: ['n1'],
      growable: [],
    });
    const result = classifyNode('n1', graph);
    expect(result.nodeType).toBe(NodeType.File);
    expect(result.visualState).toBe(VisualState.Initial);
  });

  it('classifies a Repo node', () => {
    const graph = graphWith({
      data_map: { n1: { Repo: { url: 'https://github.com/test' } as unknown as Repo } },
      initial: [],
      growable: [],
    });
    const result = classifyNode('n1', graph);
    expect(result.nodeType).toBe(NodeType.Repo);
    expect(result.label).toBe('https://github.com/test');
  });

  it('classifies a Tag node', () => {
    const graph = graphWith({
      data_map: { n1: { Tag: { tags: { TLP: ['RED'] } } } },
      initial: [],
      growable: [],
    });
    const result = classifyNode('n1', graph);
    expect(result.nodeType).toBe(NodeType.Tag);
  });

  it('classifies an Entity node by kind', () => {
    const graph = graphWith({
      data_map: { n1: { Entity: { kind: Entities.Device, name: 'MyDevice' } as unknown as EntityTypes } },
      initial: [],
      growable: [],
    });
    const result = classifyNode('n1', graph);
    expect(result.nodeType).toBe(Entities.Device);
    expect(result.label).toBe('MyDevice');
  });

  it('renames a Windows process tree label when the name is a sha256 hash', () => {
    const graph = graphWith({
      data_map: { n1: { Entity: { kind: Entities.WindowsProcessTree, name: 'a'.repeat(64) } as unknown as EntityTypes } },
      initial: [],
      growable: [],
    });
    expect(classifyNode('n1', graph).label).toBe('Windows Process Tree');
  });

  it('keeps a Windows process tree label when the name is not a sha256 hash', () => {
    const graph = graphWith({
      data_map: { n1: { Entity: { kind: Entities.WindowsProcessTree, name: 'explorer.exe tree' } as unknown as EntityTypes } },
      initial: [],
      growable: [],
    });
    expect(classifyNode('n1', graph).label).toBe('explorer.exe tree');
  });

  it('returns Growable visual state for growable nodes', () => {
    const graph = graphWith({
      data_map: { n1: { Sample: { submissions: [{ name: 'a.bin' }] } as unknown as Sample } },
      initial: [],
      growable: ['n1'],
    });
    expect(classifyNode('n1', graph).visualState).toBe(VisualState.Growable);
  });

  it('returns Basic visual state for non-initial, non-growable nodes', () => {
    const graph = graphWith({
      data_map: { n1: { Sample: { submissions: [{ name: 'a.bin' }] } as unknown as Sample } },
      initial: [],
      growable: [],
    });
    expect(classifyNode('n1', graph).visualState).toBe(VisualState.Basic);
  });

  it('uses precomputed sets when provided', () => {
    const graph = graphWith({
      data_map: { n1: { Sample: { submissions: [{ name: 'a.bin' }] } as unknown as Sample } },
      initial: [],
      growable: [],
    });
    const precomputed = { growableSet: new Set(['n1']), initialSet: new Set<string>() };
    expect(classifyNode('n1', graph, precomputed).visualState).toBe(VisualState.Growable);
  });

  it('truncates long sample labels', () => {
    const longName = 'a'.repeat(50);
    const graph = graphWith({
      data_map: { n1: { Sample: { submissions: [{ name: longName }] } as unknown as Sample } },
      initial: [],
      growable: [],
    });
    const result = classifyNode('n1', graph);
    expect(result.label.length).toBeLessThanOrEqual(33);
    expect(result.label).toContain('...');
  });
});

describe('processInitialGraphData', () => {
  it('creates nodes and links from a simple graph', () => {
    const graph = graphWith({
      initial: ['a'],
      growable: [],
      data_map: {
        a: { Sample: { submissions: [{ name: 'file.exe' }] } as unknown as Sample },
        b: { Sample: { submissions: [{ name: 'child.dll' }] } as unknown as Sample },
      },
      branches: {
        a: [branch('b', 'h1')],
      },
    });
    const result = processInitialGraphData(graph);
    expect(result.nodes).toHaveLength(2);
    expect(result.links).toHaveLength(1);
    expect(result.nodes.map((n) => n.id).sort()).toEqual(['a', 'b']);
  });

  it('deduplicates nodes that appear in multiple branches', () => {
    const graph = graphWith({
      initial: ['a'],
      growable: [],
      data_map: {
        a: { Sample: { submissions: [{ name: 'a.exe' }] } as unknown as Sample },
        b: { Sample: { submissions: [{ name: 'b.exe' }] } as unknown as Sample },
        c: { Sample: { submissions: [{ name: 'c.exe' }] } as unknown as Sample },
      },
      branches: {
        a: [branch('b', 'h1')],
        b: [branch('c', 'h2')],
        c: [branch('a', 'h3')],
      },
    });
    const result = processInitialGraphData(graph);
    expect(result.nodes).toHaveLength(3);
  });

  it('deduplicates duplicate edges by relationship hash', () => {
    const graph = graphWith({
      initial: ['a'],
      growable: [],
      data_map: {
        a: { Sample: { submissions: [{ name: 'a.exe' }] } as unknown as Sample },
        b: { Sample: { submissions: [{ name: 'b.exe' }] } as unknown as Sample },
      },
      branches: {
        a: [branch('b', 'same-hash'), branch('b', 'same-hash')],
      },
    });
    const result = processInitialGraphData(graph);
    expect(result.links).toHaveLength(1);
  });

  it('handles an empty graph', () => {
    const result = processInitialGraphData(graphWith({}));
    expect(result.nodes).toHaveLength(0);
    expect(result.links).toHaveLength(0);
  });

  it('computes node degrees based on branch connections', () => {
    const graph = graphWith({
      initial: ['a'],
      growable: [],
      data_map: {
        a: { Sample: { submissions: [{ name: 'a.exe' }] } as unknown as Sample },
        b: { Sample: { submissions: [{ name: 'b.exe' }] } as unknown as Sample },
        c: { Sample: { submissions: [{ name: 'c.exe' }] } as unknown as Sample },
      },
      branches: {
        a: [branch('b', 'h1'), branch('c', 'h2')],
      },
    });
    const result = processInitialGraphData(graph);
    const nodeA = result.nodes.find((n) => n.id === 'a');
    expect(nodeA!.degree).toBe(2);
  });

  it('handles bidirectional edges', () => {
    const graph = graphWith({
      initial: ['a'],
      growable: [],
      data_map: {
        a: { Sample: { submissions: [{ name: 'a.exe' }] } as unknown as Sample },
        b: { Sample: { submissions: [{ name: 'b.exe' }] } as unknown as Sample },
      },
      branches: {
        a: [{ node: 'b', direction: Direction.Bidirectional, relationship_hash: 'h1', relationship: {} }],
      },
    });
    const result = processInitialGraphData(graph);
    expect(result.links[0].bidirectional).toBe(true);
  });

  it('respects Direction.From for edge source/target', () => {
    const graph = graphWith({
      initial: ['a'],
      growable: [],
      data_map: {
        a: { Sample: { submissions: [{ name: 'a.exe' }] } as unknown as Sample },
        b: { Sample: { submissions: [{ name: 'b.exe' }] } as unknown as Sample },
      },
      branches: {
        a: [{ node: 'b', direction: Direction.From, relationship_hash: 'h1', relationship: {} }],
      },
    });
    const result = processInitialGraphData(graph);
    expect(result.links[0].source).toBe('b');
    expect(result.links[0].target).toBe('a');
  });
});
