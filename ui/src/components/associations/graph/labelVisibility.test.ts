import { describe, it, expect } from 'vitest';

// project imports
import { EDGE_LABEL_BUDGET, NODE_LABEL_BUDGET, ZOOM_EXPONENT, selectVisibleLabels } from './labelVisibility';
import type { VisibleLabelParams } from './labelVisibility';

// Ranked ids large enough that budgets never saturate unless a test wants them to
const RANKED = Array.from({ length: 2000 }, (_, i) => `node-${String(i).padStart(4, '0')}`);

// Baseline params (at fit distance, default density); tests override individual fields
function paramsWith(patch: Partial<VisibleLabelParams>): VisibleLabelParams {
  return {
    density: 0.5,
    camDist: 1000,
    fitDist: 1000,
    pinnedIds: new Set<string>(),
    baseBudget: NODE_LABEL_BUDGET,
    ...patch,
  };
}

// Expected budget K for finite inputs, mirroring the documented formula
function expectedK(baseBudget: number, density: number, z: number): number {
  return Math.round(baseBudget * (density / 0.5) * z ** ZOOM_EXPONENT);
}

describe('selectVisibleLabels', () => {
  it('always includes pinned ids even when the budget is zero', () => {
    // baseBudget 2 at density 0.1 rounds the budget down to 0
    const pinned = new Set(['pin-a', 'pin-b']);
    const visible = selectVisibleLabels(RANKED, paramsWith({ density: 0.1, baseBudget: 2, pinnedIds: pinned }));
    expect(visible).toEqual(pinned);
  });

  it('reveals more labels as the camera zooms in', () => {
    const far = selectVisibleLabels(RANKED, paramsWith({ camDist: 1000 }));
    const near = selectVisibleLabels(RANKED, paramsWith({ camDist: 250 }));
    expect(near.size).toBeGreaterThan(far.size);
    expect(far.size).toBe(expectedK(NODE_LABEL_BUDGET, 0.5, 1));
    expect(near.size).toBe(expectedK(NODE_LABEL_BUDGET, 0.5, 4));
  });

  it('scales the budget linearly with density', () => {
    const half = selectVisibleLabels(RANKED, paramsWith({ density: 0.5 }));
    const full = selectVisibleLabels(RANKED, paramsWith({ density: 1.0 }));
    expect(full.size).toBe(half.size * 2);
  });

  it('clamps the zoom factor at 8 for extreme close-ups', () => {
    const atClamp = selectVisibleLabels(RANKED, paramsWith({ baseBudget: 4, camDist: 1000 / 8 }));
    const pastClamp = selectVisibleLabels(RANKED, paramsWith({ baseBudget: 4, camDist: 1000 / 100 }));
    expect(pastClamp).toEqual(atClamp);
    expect(atClamp.size).toBe(expectedK(4, 0.5, 8));
  });

  it('clamps the zoom factor at 1 at and past the fit distance', () => {
    const atFit = selectVisibleLabels(RANKED, paramsWith({ camDist: 1000 }));
    const zoomedOut = selectVisibleLabels(RANKED, paramsWith({ camDist: 3000 }));
    expect(zoomedOut).toEqual(atFit);
    expect(atFit.size).toBe(expectedK(NODE_LABEL_BUDGET, 0.5, 1));
  });

  it('is deterministic for equal inputs', () => {
    const first = selectVisibleLabels(RANKED, paramsWith({ camDist: 400, density: 0.7, pinnedIds: new Set(['pin-a']) }));
    const second = selectVisibleLabels(RANKED, paramsWith({ camDist: 400, density: 0.7, pinnedIds: new Set(['pin-a']) }));
    expect(second).toEqual(first);
  });

  it('returns exactly the ranked prefix of length K when nothing is pinned', () => {
    const visible = selectVisibleLabels(RANKED, paramsWith({}));
    expect([...visible]).toEqual(RANKED.slice(0, expectedK(NODE_LABEL_BUDGET, 0.5, 1)));
  });

  it('applies node and edge budgets independently', () => {
    const nodes = selectVisibleLabels(RANKED, paramsWith({ baseBudget: NODE_LABEL_BUDGET }));
    const edges = selectVisibleLabels(RANKED, paramsWith({ baseBudget: EDGE_LABEL_BUDGET }));
    expect(nodes.size).toBe(NODE_LABEL_BUDGET);
    expect(edges.size).toBe(EDGE_LABEL_BUDGET);
    expect(nodes.size).not.toBe(edges.size);
  });

  it('falls back to the fit-distance budget for non-finite camera inputs', () => {
    const baseline = selectVisibleLabels(RANKED, paramsWith({}));
    expect(selectVisibleLabels(RANKED, paramsWith({ camDist: NaN }))).toEqual(baseline);
    expect(selectVisibleLabels(RANKED, paramsWith({ fitDist: Infinity }))).toEqual(baseline);
    expect(selectVisibleLabels(RANKED, paramsWith({ camDist: 0 }))).toEqual(baseline);
  });

  it('caps the budget at the number of ranked ids', () => {
    const visible = selectVisibleLabels(RANKED.slice(0, 10), paramsWith({ camDist: 10 }));
    expect(visible.size).toBe(10);
  });
});
