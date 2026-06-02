import type { GraphControls } from './types';

// Keys managed by auto-scaling — user overrides are tracked per-key
export const SIZE_SCALED_KEYS: ReadonlyArray<keyof GraphControls> = [
  'chargeStrength',
  'edgeLength',
  'velocityDecay',
  'nodeRelSize',
  'edgeWidth',
  'edgeOpacity',
  'arrowLength',
  'directionalParticles',
  'warmupTicks',
  'cooldownTime',
  'nodeLabelDensity',
];

interface ScaleRange {
  small: number;
  large: number;
}

const SCALE_RANGES: Record<string, ScaleRange> = {
  chargeStrength: { small: -200, large: -30 },
  edgeLength: { small: 30, large: 80 },
  velocityDecay: { small: 0.4, large: 0.6 },
  nodeRelSize: { small: 4, large: 2 },
  edgeWidth: { small: 1.0, large: 0.3 },
  edgeOpacity: { small: 0.2, large: 0.1 },
  arrowLength: { small: 3.5, large: 1.5 },
  directionalParticles: { small: 1, large: 0 },
  warmupTicks: { small: 0, large: 200 },
  cooldownTime: { small: 15000, large: 5000 },
  nodeLabelDensity: { small: 0.7, large: 0.3 },
};

const LOG_SMALL = Math.log10(30);
const LOG_LARGE = Math.log10(2000);
const LOG_RANGE = LOG_LARGE - LOG_SMALL;

// Lerp between small-graph and large-graph values using log10(nodeCount)
const lerp = (range: ScaleRange, t: number): number => range.small + (range.large - range.small) * t;

// Compute optimal graph control defaults for a given node count
export const computeSizeDefaults = (nodeCount: number): Partial<GraphControls> => {
  if (nodeCount <= 30) {
    return Object.fromEntries(SIZE_SCALED_KEYS.map((k) => [k, SCALE_RANGES[k].small]));
  }

  const t = Math.min(1, Math.max(0, (Math.log10(nodeCount) - LOG_SMALL) / LOG_RANGE));

  const result: Record<string, number> = {};
  for (const key of SIZE_SCALED_KEYS) {
    const range = SCALE_RANGES[key];
    let value = lerp(range, t);

    if (key === 'directionalParticles' || key === 'warmupTicks') {
      value = Math.round(value);
    } else if (key === 'chargeStrength' || key === 'cooldownTime' || key === 'edgeLength') {
      value = Math.round(value);
    } else {
      value = Math.round(value * 100) / 100;
    }

    result[key] = value;
  }

  return result;
};
