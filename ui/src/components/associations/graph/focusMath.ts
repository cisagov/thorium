// project imports
import { getLinkEndpoints } from './data';
import type { GraphLink, GraphNode } from './types';

// spec: ./AssociationGraph.spec.md

/** Camera-to-target distance floor so focusing never lands uselessly close */
export const MIN_FOCUS_DISTANCE = 120;
/** World-unit breathing room added beyond the exact neighborhood fit distance */
export const FOCUS_FIT_PADDING = 40;
/** Bounding-sphere radius floor so a node with no neighbors still frames sensibly */
export const LONE_NODE_RADIUS = 30;

/** A point in world space */
export interface Vec3 {
  /** World-space x coordinate */
  x: number;
  /** World-space y coordinate */
  y: number;
  /** World-space z coordinate */
  z: number;
}

/** A bounding sphere enclosing a set of nodes */
export interface BoundingSphere {
  /** The centroid of the enclosed members */
  center: Vec3;
  /** The farthest member distance from the centroid, floored at {@link LONE_NODE_RADIUS} */
  radius: number;
}

/**
 * Computes the centroid of a set of graph nodes and the farthest member distance from it,
 * from their live layout positions. The radius is returned un-floored so callers can apply
 * whatever minimum framing radius suits their use.
 *
 * @param nodes - The live graph nodes to consider (carrying layout positions).
 * @param include - Predicate selecting which nodes participate; only nodes it accepts and
 *   that have a defined `x` are counted.
 * @returns The centroid and un-floored max radius, or `null` when no node qualifies.
 */
export const computeCentroidRadius = (
  nodes: GraphNode[],
  include: (node: GraphNode) => boolean,
): { center: Vec3; radius: number } | null => {
  // average the qualifying positioned members into a centroid
  let cx = 0,
    cy = 0,
    cz = 0,
    count = 0;
  for (const n of nodes) {
    if (n.x === undefined || !include(n)) continue;
    cx += n.x;
    cy += n.y ?? 0;
    cz += n.z ?? 0;
    count++;
  }
  if (count === 0) return null;
  cx /= count;
  cy /= count;
  cz /= count;
  // radius = farthest qualifying member from the centroid, left un-floored for the caller
  let radius = 0;
  for (const n of nodes) {
    if (n.x === undefined || !include(n)) continue;
    radius = Math.max(radius, Math.hypot(n.x - cx, (n.y ?? 0) - cy, (n.z ?? 0) - cz));
  }
  return { center: { x: cx, y: cy, z: cz }, radius };
};

/**
 * Computes the bounding sphere of a set of graph nodes from their live layout positions:
 * the center is the centroid of the positioned members and the radius is the farthest
 * member distance, floored at {@link LONE_NODE_RADIUS}.
 *
 * @param nodes - The live graph nodes to search for members (carrying layout positions).
 * @param memberIds - Ids of the nodes the sphere must enclose.
 * @returns The sphere, or `null` when no member has a finite position yet.
 */
export const computeBoundingSphere = (nodes: GraphNode[], memberIds: Set<string>): BoundingSphere | null => {
  const result = computeCentroidRadius(nodes, (n) => memberIds.has(n.id) && n.x !== undefined && isFinite(n.x));
  if (!result) return null;
  return { center: result.center, radius: Math.max(LONE_NODE_RADIUS, result.radius) };
};

/**
 * Collects a node's 1-hop neighborhood (the node itself plus every direct neighbor) from
 * the live link list.
 *
 * @param links - The live graph links (endpoints may be node objects or bare ids).
 * @param nodeId - The node whose neighborhood to collect.
 * @returns The set of member node ids, always including `nodeId`.
 */
export const collectNeighborhoodIds = (links: GraphLink[], nodeId: string): Set<string> => {
  const memberIds = new Set([nodeId]);
  for (const link of links) {
    const { source, target } = getLinkEndpoints(link);
    if (source === nodeId) memberIds.add(target);
    if (target === nodeId) memberIds.add(source);
  }
  return memberIds;
};

/**
 * Computes the camera distance that frames a bounding sphere of the given radius:
 * `radius / tan(fov / 2)` puts the sphere edge at the viewport edge, plus padding,
 * floored at {@link MIN_FOCUS_DISTANCE}.
 *
 * @param fovDeg - The perspective camera's vertical field of view in degrees.
 * @param radius - The bounding-sphere radius in world units.
 * @returns The camera-to-center distance in world units.
 */
export const sphereFitDistance = (fovDeg: number, radius: number): number => {
  return Math.max(MIN_FOCUS_DISTANCE, radius / Math.tan((fovDeg * Math.PI) / 360) + FOCUS_FIT_PADDING);
};
