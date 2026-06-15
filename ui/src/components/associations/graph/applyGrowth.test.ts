import { describe, it, expect, vi } from 'vitest';

// project imports
import { applyGrowthToInstance } from './applyGrowth';
import { VisualState } from './types';
import type { GraphData, GraphInstance, GraphNode, GraphLink } from './types';
import type { LabelEntry } from './controls/controlsReducer';
import { NodeType } from '@models/trees';

/**
 * Build a graph node fixture with optional layout coordinates and visual state.
 *
 * @param id - The node id.
 * @param opts - Optional coordinates and visual state overrides.
 * @returns A minimally populated {@link GraphNode}.
 */
function node(id: string, opts: { x?: number; y?: number; z?: number; visualState?: VisualState } = {}): GraphNode {
  return {
    id,
    label: id,
    nodeType: NodeType.Other,
    visualState: opts.visualState ?? VisualState.Basic,
    score: 0,
    diameter: 1,
    degree: 0,
    x: opts.x,
    y: opts.y,
    z: opts.z,
  };
}

/**
 * Build a link between two endpoints (bare ids or node objects).
 *
 * @param source - The source id or node.
 * @param target - The target id or node.
 * @returns A minimally populated {@link GraphLink}.
 */
function link(source: string | GraphNode, target: string | GraphNode): GraphLink {
  return { source, target, label: '', bidirectional: false } as unknown as GraphLink;
}

/**
 * Build a fake ForceGraph3D instance capturing the imperative calls the merge makes.
 *
 * @returns A stub graph instance plus captured state for assertions.
 */
function fakeInstance() {
  const captured: { graphData?: GraphData; refreshCount: number } = { refreshCount: 0 };
  const nodeThreeObjectImpl = vi.fn();
  const gi = {
    graphData: vi.fn((data?: GraphData) => {
      if (data) captured.graphData = data;
      return gi;
    }),
    nodeThreeObject: vi.fn((impl?: unknown) => {
      if (impl === undefined) return nodeThreeObjectImpl;
      return gi;
    }),
    refresh: vi.fn(() => {
      captured.refreshCount += 1;
      return gi;
    }),
  } as unknown as GraphInstance;
  return { gi, captured };
}

/**
 * Invoke {@link applyGrowthToInstance} with the given prev/new data against a fresh fake instance.
 *
 * @param prevData - The current graph data.
 * @param newData - The freshly fetched graph data to merge in.
 * @returns The captured instance state, the graphDataRef, and the setNodeCount spy.
 */
function runGrowth(prevData: GraphData, newData: GraphData) {
  const { gi, captured } = fakeInstance();
  const graphInstanceRef = { current: gi };
  const labelSpritesRef = { current: new Map<string, LabelEntry>() };
  const graphDataRef = { current: prevData };
  const setNodeCount = vi.fn();
  applyGrowthToInstance(prevData, newData, graphInstanceRef, labelSpritesRef, graphDataRef, setNodeCount);
  return { captured, graphDataRef, setNodeCount, labelSpritesRef };
}

describe('applyGrowthToInstance', () => {
  it('no-ops when the graph instance is not yet mounted', () => {
    const prevData: GraphData = { nodes: [node('a')], links: [] };
    const newData: GraphData = { nodes: [node('a'), node('b')], links: [link('a', 'b')] };
    const graphDataRef = { current: prevData };
    const setNodeCount = vi.fn();
    applyGrowthToInstance(prevData, newData, { current: null }, { current: new Map() }, graphDataRef, setNodeCount);
    expect(setNodeCount).not.toHaveBeenCalled();
    expect(graphDataRef.current).toBe(prevData);
  });

  it('early-returns without touching the instance when nothing changed', () => {
    const prevData: GraphData = { nodes: [node('a'), node('b')], links: [link('a', 'b')] };
    const newData: GraphData = { nodes: [node('a'), node('b')], links: [link('a', 'b')] };
    const { captured, graphDataRef, setNodeCount } = runGrowth(prevData, newData);
    expect(captured.graphData).toBeUndefined();
    expect(setNodeCount).not.toHaveBeenCalled();
    expect(graphDataRef.current).toBe(prevData);
  });

  it('merges added nodes and links and updates node count', () => {
    const prevData: GraphData = { nodes: [node('a')], links: [] };
    const newData: GraphData = { nodes: [node('a'), node('b')], links: [link('a', 'b')] };
    const { captured, graphDataRef, setNodeCount } = runGrowth(prevData, newData);
    expect(captured.graphData!.nodes.map((n) => n.id).sort()).toEqual(['a', 'b']);
    expect(captured.graphData!.links).toHaveLength(1);
    expect(setNodeCount).toHaveBeenCalledWith(2);
    expect(graphDataRef.current).toBe(captured.graphData);
  });

  it('dedups links already present via canonical endpoint keys', () => {
    const prevData: GraphData = { nodes: [node('a'), node('b')], links: [link('a', 'b')] };
    // same edge re-fetched (as an object-endpoint link) plus a genuinely new edge
    const newData: GraphData = {
      nodes: [node('a'), node('b'), node('c')],
      links: [link(node('a'), node('b')), link('b', 'c')],
    };
    const { captured } = runGrowth(prevData, newData);
    expect(captured.graphData!.links).toHaveLength(2);
  });

  it('positions a newly added node near its existing parent within the jitter radius', () => {
    const prevData: GraphData = { nodes: [node('a', { x: 100, y: 200, z: 300 })], links: [] };
    const newData: GraphData = {
      nodes: [node('a', { x: 100, y: 200, z: 300 }), node('b')],
      links: [link('a', 'b')],
    };
    const { captured } = runGrowth(prevData, newData);
    const added = captured.graphData!.nodes.find((n) => n.id === 'b')!;
    // parent-relative jitter is +/- 15 (POSITION_JITTER / 2) around the parent position
    expect(Math.abs(added.x! - 100)).toBeLessThanOrEqual(15);
    expect(Math.abs(added.y! - 200)).toBeLessThanOrEqual(15);
    expect(Math.abs(added.z! - 300)).toBeLessThanOrEqual(15);
  });

  it('detects a visual-state change on an existing node and refreshes the label objects', () => {
    const prevData: GraphData = { nodes: [node('a', { visualState: VisualState.Growable })], links: [] };
    const newData: GraphData = { nodes: [node('a', { visualState: VisualState.Basic })], links: [] };
    const { captured, labelSpritesRef } = runGrowth(prevData, newData);
    const updated = captured.graphData!.nodes.find((n) => n.id === 'a')!;
    expect(updated.visualState).toBe(VisualState.Basic);
    expect(labelSpritesRef.current.size).toBe(0);
    expect(captured.refreshCount).toBe(1);
  });

  it('normalizes existing link endpoints to bare ids when a state change occurs', () => {
    const prevData: GraphData = {
      nodes: [node('a', { visualState: VisualState.Growable }), node('b')],
      links: [link(node('a'), node('b'))],
    };
    const newData: GraphData = {
      nodes: [node('a', { visualState: VisualState.Basic }), node('b')],
      links: [link(node('a'), node('b'))],
    };
    const { captured } = runGrowth(prevData, newData);
    const [merged] = captured.graphData!.links;
    expect(merged.source).toBe('a');
    expect(merged.target).toBe('b');
  });
});
