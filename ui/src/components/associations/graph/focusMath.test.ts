import { describe, it, expect } from 'vitest';

// project imports
import {
  collectNeighborhoodIds,
  computeBoundingSphere,
  computeCentroidRadius,
  FOCUS_FIT_PADDING,
  LONE_NODE_RADIUS,
  MIN_FOCUS_DISTANCE,
  sphereFitDistance,
} from './focusMath';
import { NodeType } from '@models/trees';
import { VisualState } from './types';
import type { GraphLink, GraphNode } from './types';

/**
 * Build a graph node fixture with optional layout coordinates.
 *
 * @param id - The node id.
 * @param pos - Optional world-space coordinates.
 * @returns A minimally populated {@link GraphNode}.
 */
function node(id: string, pos?: { x?: number; y?: number; z?: number }): GraphNode {
  return {
    id,
    label: id,
    nodeType: NodeType.Other,
    visualState: VisualState.Basic,
    score: 0,
    diameter: 1,
    degree: 0,
    ...pos,
  };
}

/**
 * Build a link between two endpoints, which may be bare ids or node objects.
 *
 * @param source - The source id or node.
 * @param target - The target id or node.
 * @returns A minimally populated {@link GraphLink}.
 */
function link(source: string | GraphNode, target: string | GraphNode): GraphLink {
  return { source, target, label: '', bidirectional: false } as unknown as GraphLink;
}

describe('computeBoundingSphere', () => {
  it('returns null when no member has a finite position', () => {
    const nodes = [node('a'), node('b', { x: NaN })];
    expect(computeBoundingSphere(nodes, new Set(['a', 'b']))).toBeNull();
  });

  it('floors the radius at LONE_NODE_RADIUS for a single positioned member', () => {
    const nodes = [node('a', { x: 10, y: 20, z: 30 })];
    const sphere = computeBoundingSphere(nodes, new Set(['a']));
    expect(sphere).not.toBeNull();
    expect(sphere!.center).toEqual({ x: 10, y: 20, z: 30 });
    expect(sphere!.radius).toBe(LONE_NODE_RADIUS);
  });

  it('computes centroid and farthest-member radius across positioned members', () => {
    const nodes = [node('a', { x: 0, y: 0, z: 0 }), node('b', { x: 100, y: 0, z: 0 })];
    const sphere = computeBoundingSphere(nodes, new Set(['a', 'b']));
    expect(sphere!.center).toEqual({ x: 50, y: 0, z: 0 });
    expect(sphere!.radius).toBe(50);
  });

  it('ignores nodes outside the member set', () => {
    const nodes = [node('a', { x: 0, y: 0, z: 0 }), node('far', { x: 1000, y: 0, z: 0 })];
    const sphere = computeBoundingSphere(nodes, new Set(['a']));
    expect(sphere!.center).toEqual({ x: 0, y: 0, z: 0 });
    expect(sphere!.radius).toBe(LONE_NODE_RADIUS);
  });
});

describe('computeCentroidRadius', () => {
  it('returns null when no node qualifies', () => {
    const nodes = [node('a'), node('b')];
    expect(computeCentroidRadius(nodes, () => true)).toBeNull();
  });

  it('returns the un-floored radius (no LONE_NODE_RADIUS floor)', () => {
    const nodes = [node('a', { x: 0, y: 0, z: 0 })];
    const result = computeCentroidRadius(nodes, () => true);
    expect(result!.center).toEqual({ x: 0, y: 0, z: 0 });
    expect(result!.radius).toBe(0);
  });

  it('computes centroid and farthest-member radius across included nodes', () => {
    const nodes = [node('a', { x: 0, y: 0, z: 0 }), node('b', { x: 100, y: 0, z: 0 })];
    const result = computeCentroidRadius(nodes, () => true);
    expect(result!.center).toEqual({ x: 50, y: 0, z: 0 });
    expect(result!.radius).toBe(50);
  });

  it('honors the include predicate to exclude nodes', () => {
    const nodes = [node('a', { x: 0, y: 0, z: 0 }), node('far', { x: 1000, y: 0, z: 0 })];
    const result = computeCentroidRadius(nodes, (n) => n.id === 'a');
    expect(result!.center).toEqual({ x: 0, y: 0, z: 0 });
    expect(result!.radius).toBe(0);
  });
});

describe('collectNeighborhoodIds', () => {
  it('collects direct neighbors with bare-string endpoints', () => {
    const links = [link('a', 'b'), link('c', 'a'), link('x', 'y')];
    expect(collectNeighborhoodIds(links, 'a')).toEqual(new Set(['a', 'b', 'c']));
  });

  it('resolves object endpoints via their id', () => {
    const links = [link(node('a'), node('b'))];
    expect(collectNeighborhoodIds(links, 'a')).toEqual(new Set(['a', 'b']));
  });

  it('always includes the node itself even with no links', () => {
    expect(collectNeighborhoodIds([], 'solo')).toEqual(new Set(['solo']));
  });
});

describe('sphereFitDistance', () => {
  it('floors small spheres at MIN_FOCUS_DISTANCE', () => {
    expect(sphereFitDistance(50, 0)).toBe(MIN_FOCUS_DISTANCE);
  });

  it('scales with radius and adds padding for large spheres', () => {
    const fov = 50;
    const radius = 1000;
    const expected = radius / Math.tan((fov * Math.PI) / 360) + FOCUS_FIT_PADDING;
    expect(sphereFitDistance(fov, radius)).toBeCloseTo(expected, 5);
  });
});
