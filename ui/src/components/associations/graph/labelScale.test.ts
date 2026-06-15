import { describe, it, expect } from 'vitest';

// project imports
import {
  computeLabelScale,
  EDGE_LABEL_MAX_PX,
  EDGE_LABEL_MIN_PX,
  EDGE_LABEL_TARGET_PX,
  LABEL_MAX_PX,
  LABEL_MIN_PX,
  LABEL_TARGET_PX,
} from './labelScale';
import type { LabelScaleInput } from './labelScale';

// Baseline node-label input; tests override individual fields
function inputWith(patch: Partial<LabelScaleInput>): LabelScaleInput {
  return {
    labelDist: 300,
    viewportHeightPx: 900,
    fovDeg: 50,
    baseScaleY: 1,
    labelScale: 1,
    tierBoost: 1,
    targetPx: LABEL_TARGET_PX,
    minPx: LABEL_MIN_PX,
    maxPx: LABEL_MAX_PX,
    ...patch,
  };
}

// Project the multiplier back to on-screen pixels to verify the constant-size invariant
function reconstructPx(multiplier: number, input: LabelScaleInput): number {
  const pxPerWorldAtUnitDist = input.viewportHeightPx / (2 * Math.tan((input.fovDeg * Math.PI) / 360));
  return (multiplier * input.baseScaleY * pxPerWorldAtUnitDist) / input.labelDist;
}

describe('computeLabelScale', () => {
  it('renders approximately the target px at defaults', () => {
    const input = inputWith({});
    const px = reconstructPx(computeLabelScale(input), input);
    expect(px).toBeCloseTo(LABEL_TARGET_PX, 5);
  });

  it('clamps to the min px floor', () => {
    const input = inputWith({ labelScale: 0.5 });
    const px = reconstructPx(computeLabelScale(input), input);
    expect(px).toBeCloseTo(LABEL_MIN_PX, 5);
  });

  it('clamps to the max px ceiling', () => {
    const input = inputWith({ labelScale: 2, tierBoost: 1.4 });
    const px = reconstructPx(computeLabelScale(input), input);
    expect(px).toBeCloseTo(LABEL_MAX_PX, 5);
  });

  it('multiplier scales linearly with distance', () => {
    const near = computeLabelScale(inputWith({ labelDist: 150 }));
    const far = computeLabelScale(inputWith({ labelDist: 300 }));
    expect(far).toBeCloseTo(near * 2, 10);
  });

  it('on-screen px is independent of viewport height', () => {
    const short = inputWith({ viewportHeightPx: 600 });
    const tall = inputWith({ viewportHeightPx: 1200 });
    const shortPx = reconstructPx(computeLabelScale(short), short);
    const tallPx = reconstructPx(computeLabelScale(tall), tall);
    expect(shortPx).toBeCloseTo(tallPx, 10);
  });

  it('tier boost increases px within the clamp', () => {
    const base = inputWith({ tierBoost: 1 });
    const mid = inputWith({ tierBoost: 1.2 });
    const high = inputWith({ tierBoost: 1.4 });
    const basePx = reconstructPx(computeLabelScale(base), base);
    const midPx = reconstructPx(computeLabelScale(mid), mid);
    const highPx = reconstructPx(computeLabelScale(high), high);
    expect(midPx).toBeCloseTo(basePx * 1.2, 5);
    expect(highPx).toBeCloseTo(basePx * 1.4, 5);
  });

  it('clamps edge labels to the edge constants', () => {
    const target = inputWith({ targetPx: EDGE_LABEL_TARGET_PX, minPx: EDGE_LABEL_MIN_PX, maxPx: EDGE_LABEL_MAX_PX });
    expect(reconstructPx(computeLabelScale(target), target)).toBeCloseTo(EDGE_LABEL_TARGET_PX, 5);
    const ceiling = inputWith({
      labelScale: 2,
      tierBoost: 1.4,
      targetPx: EDGE_LABEL_TARGET_PX,
      minPx: EDGE_LABEL_MIN_PX,
      maxPx: EDGE_LABEL_MAX_PX,
    });
    expect(reconstructPx(computeLabelScale(ceiling), ceiling)).toBeCloseTo(EDGE_LABEL_MAX_PX, 5);
  });
});
