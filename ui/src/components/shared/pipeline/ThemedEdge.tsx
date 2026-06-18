import React from 'react';
import { BaseEdge, type EdgeProps } from '@xyflow/react';

// project imports
import { HANDLE_CENTER_Y } from './order';
import CrabSVG from '@assets/icons/crab.svg?raw';

const CRAB_DATA_URI = `data:image/svg+xml;base64,${btoa(CrabSVG)}`;
const CRAB_SIZE = { w: 14, h: 12 };

const CORNER_RADIUS = 5;
const FLAT_THRESHOLD = 5;
// Fixed horizontal stub leaving the source's East handle / entering the target's West handle. Using a
// constant (instead of a dx-dependent gap) keeps turns sane for any node position, including a node
// dragged to the left of / far off its source (the old dx/5 gap went negative there and looped).
const STUB = 18;
// Vertical clearance for the backward "C-shape" detour so the routing lane clears the node bodies.
const LANE_CLEAR = 33;

type Point = [number, number];

interface ThemedEdgeData {
  routeFlat?: 'source' | 'target';
}

/**
 * Build an SVG path from an axis-aligned polyline, rounding each corner. Consecutive segments must be
 * horizontal or vertical (right angles); the corner radius is clamped to half of the shorter adjacent
 * segment so it never overshoots. Duplicate/zero-length points are dropped.
 */
function roundedOrthPath(rawPoints: Point[], radius: number): string {
  const points: Point[] = [];
  for (const p of rawPoints) {
    const prev = points[points.length - 1];
    if (!prev || prev[0] !== p[0] || prev[1] !== p[1]) points.push(p);
  }
  if (points.length < 2) return '';
  if (points.length === 2) return `M ${points[0][0]} ${points[0][1]} L ${points[1][0]} ${points[1][1]}`;

  const cmds: string[] = [`M ${points[0][0]} ${points[0][1]}`];
  for (let i = 1; i < points.length - 1; i++) {
    const [px, py] = points[i - 1];
    const [cx, cy] = points[i];
    const [nx, ny] = points[i + 1];
    const inLen = Math.hypot(cx - px, cy - py);
    const outLen = Math.hypot(nx - cx, ny - cy);
    const r = Math.min(radius, inLen / 2, outLen / 2);
    // unit directions along the incoming and outgoing segments
    const inUx = inLen ? (cx - px) / inLen : 0;
    const inUy = inLen ? (cy - py) / inLen : 0;
    const outUx = outLen ? (nx - cx) / outLen : 0;
    const outUy = outLen ? (ny - cy) / outLen : 0;
    cmds.push(`L ${cx - inUx * r} ${cy - inUy * r}`);
    cmds.push(`Q ${cx} ${cy} ${cx + outUx * r} ${cy + outUy * r}`);
  }
  const last = points[points.length - 1];
  cmds.push(`L ${last[0]} ${last[1]}`);
  return cmds.join(' ');
}

/**
 * Orthogonal (Manhattan) edge router for fixed East→West ports. Always emits horizontal/vertical
 * segments with right-angle corners, for any relative position of source/target (handles a target that
 * is to the left of, above, or below the source — e.g. a node dragged anywhere on the canvas).
 */
export function getOrthogonalPath(sx: number, sy: number, tx: number, ty: number, routeFlat?: 'source' | 'target'): string {
  const sameRow = Math.abs(sy - ty) < FLAT_THRESHOLD;
  const dx = tx - sx;

  // Same row and target ahead: a single straight horizontal line.
  if (sameRow && dx >= 0) return `M ${sx} ${sy} L ${tx} ${ty}`;

  // Forward with room: one vertical turn. `turnX` keeps parallel siblings converging on a single
  // vertical (near the source for routeFlat 'source', near the target for 'target', else the midpoint).
  if (dx >= 2 * STUB) {
    const turnX = routeFlat === 'source' ? sx + STUB : routeFlat === 'target' ? tx - STUB : sx + dx / 2;
    return roundedOrthPath(
      [
        [sx, sy],
        [turnX, sy],
        [turnX, ty],
        [tx, ty],
      ],
      CORNER_RADIUS,
    );
  }

  // Forward but cramped: single mid turn (no room for stub-placed turns).
  if (dx > 0) {
    const turnX = sx + dx / 2;
    return roundedOrthPath(
      [
        [sx, sy],
        [turnX, sy],
        [turnX, ty],
        [tx, ty],
      ],
      CORNER_RADIUS,
    );
  }

  // Backward (target at/left of source) or same-row backward: C-shape detour to a clearance lane,
  // back across, then into the target — two extra corners, still all right angles. The lane is placed
  // toward the graph's center line so the vertical legs point back at the spine: route up when the edge
  // is below the center line, flip down when it's above it (instead of always going up).
  const aboveCenter = (sy + ty) / 2 < HANDLE_CENTER_Y;
  const laneY = aboveCenter ? Math.max(sy, ty) + LANE_CLEAR : Math.min(sy, ty) - LANE_CLEAR;
  return roundedOrthPath(
    [
      [sx, sy],
      [sx + STUB, sy],
      [sx + STUB, laneY],
      [tx - STUB, laneY],
      [tx - STUB, ty],
      [tx, ty],
    ],
    CORNER_RADIUS,
  );
}

const ThemedEdge: React.FC<EdgeProps> = ({ id, sourceX, sourceY, targetX, targetY, style, markerEnd, data }) => {
  const edgeData = data as ThemedEdgeData | undefined;
  const edgePath = getOrthogonalPath(sourceX, sourceY, targetX, targetY, edgeData?.routeFlat);

  const theme = document.getElementById('root')?.getAttribute('theme') ?? '';

  return (
    <>
      <BaseEdge id={id} path={edgePath} style={style} markerEnd={markerEnd} />
      {theme === 'Crab' && (
        <image href={CRAB_DATA_URI} width={CRAB_SIZE.w} height={CRAB_SIZE.h} x={-CRAB_SIZE.w / 2} y={-CRAB_SIZE.h / 2}>
          <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
        </image>
      )}
    </>
  );
};

export const edgeTypes = { themedStep: ThemedEdge };
