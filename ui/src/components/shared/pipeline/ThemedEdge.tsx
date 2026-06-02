import React from 'react';
import { BaseEdge, type EdgeProps } from '@xyflow/react';

// project imports
import CrabSVG from '@assets/icons/crab.svg?raw';

const CRAB_DATA_URI = `data:image/svg+xml;base64,${btoa(CrabSVG)}`;
const CRAB_SIZE = { w: 14, h: 12 };

const CORNER_RADIUS = 5;
const FLAT_THRESHOLD = 5;

interface ThemedEdgeData {
  routeFlat?: 'source' | 'target';
}

function getOrthogonalPath(sx: number, sy: number, tx: number, ty: number, routeFlat?: 'source' | 'target'): string {
  if (Math.abs(sy - ty) < FLAT_THRESHOLD) return `M ${sx} ${sy} L ${tx} ${ty}`;

  const dx = tx - sx;
  const gap = Math.min(40, dx / 5);
  const dirY = ty > sy ? 1 : -1;

  // 3-segment Z-path: vertical turn near source
  if (routeFlat === 'source') {
    const turnX = sx + gap;
    const r = Math.min(CORNER_RADIUS, gap, Math.abs(ty - sy) / 2);
    return [
      `M ${sx} ${sy}`,
      `L ${turnX - r} ${sy}`,
      `Q ${turnX} ${sy} ${turnX} ${sy + r * dirY}`,
      `L ${turnX} ${ty - r * dirY}`,
      `Q ${turnX} ${ty} ${turnX + r} ${ty}`,
      `L ${tx} ${ty}`,
    ].join(' ');
  }

  // 3-segment Z-path: vertical turn near target
  if (routeFlat === 'target') {
    const turnX = tx - gap;
    const r = Math.min(CORNER_RADIUS, gap, Math.abs(ty - sy) / 2);
    return [
      `M ${sx} ${sy}`,
      `L ${turnX - r} ${sy}`,
      `Q ${turnX} ${sy} ${turnX} ${sy + r * dirY}`,
      `L ${turnX} ${ty - r * dirY}`,
      `Q ${turnX} ${ty} ${turnX + r} ${ty}`,
      `L ${tx} ${ty}`,
    ].join(' ');
  }

  // Default 5-segment S-path with centered horizontal run
  const turnX1 = sx + gap;
  const turnX2 = tx - gap;
  const midY = (sy + ty) / 2;
  const r = Math.min(CORNER_RADIUS, gap, Math.abs(midY - sy) / 2);

  return [
    `M ${sx} ${sy}`,
    `L ${turnX1 - r} ${sy}`,
    `Q ${turnX1} ${sy} ${turnX1} ${sy + r * dirY}`,
    `L ${turnX1} ${midY - r * dirY}`,
    `Q ${turnX1} ${midY} ${turnX1 + r} ${midY}`,
    `L ${turnX2 - r} ${midY}`,
    `Q ${turnX2} ${midY} ${turnX2} ${midY + r * dirY}`,
    `L ${turnX2} ${ty - r * dirY}`,
    `Q ${turnX2} ${ty} ${turnX2 + r} ${ty}`,
    `L ${tx} ${ty}`,
  ].join(' ');
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
