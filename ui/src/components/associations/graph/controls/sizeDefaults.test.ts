import { describe, it, expect } from 'vitest';

// project imports
import { computeSizeDefaults, SIZE_SCALED_KEYS } from './sizeDefaults';

describe('computeSizeDefaults', () => {
  it('returns small-graph defaults for <= 30 nodes', () => {
    const d1 = computeSizeDefaults(1);
    const d30 = computeSizeDefaults(30);

    expect(d1.chargeStrength).toBe(-200);
    expect(d1.edgeLength).toBe(30);
    expect(d1.nodeRelSize).toBe(4);
    expect(d1.directionalParticles).toBe(1);
    expect(d1.warmupTicks).toBe(0);

    expect(d30).toEqual(d1);
  });

  it('returns all size-scaled keys', () => {
    const defaults = computeSizeDefaults(100);
    for (const key of SIZE_SCALED_KEYS) {
      expect(defaults).toHaveProperty(key);
    }
  });

  it('interpolates between small and large values for medium graphs', () => {
    const d200 = computeSizeDefaults(200);

    expect(d200.chargeStrength!).toBeGreaterThan(-200);
    expect(d200.chargeStrength!).toBeLessThan(-30);

    expect(d200.edgeLength!).toBeGreaterThan(30);
    expect(d200.edgeLength!).toBeLessThan(80);

    expect(d200.nodeRelSize!).toBeGreaterThan(2);
    expect(d200.nodeRelSize!).toBeLessThan(4);
  });

  it('approaches large-graph values at 1000+ nodes', () => {
    const d1000 = computeSizeDefaults(1000);
    const d5000 = computeSizeDefaults(5000);

    expect(d1000.chargeStrength!).toBeGreaterThan(-80);
    expect(d1000.directionalParticles).toBe(0);

    expect(d5000.chargeStrength!).toBe(-30);
    expect(d5000.edgeLength!).toBe(80);
    expect(d5000.nodeRelSize!).toBe(2);
    expect(d5000.cooldownTime!).toBe(5000);
  });

  it('scales monotonically — larger graphs get smaller charge, wider edges', () => {
    const counts = [50, 100, 200, 500, 1000, 2000];
    const results = counts.map((n) => computeSizeDefaults(n));

    for (let i = 1; i < results.length; i++) {
      expect(results[i].chargeStrength!).toBeGreaterThan(results[i - 1].chargeStrength!);
      expect(results[i].edgeLength!).toBeGreaterThanOrEqual(results[i - 1].edgeLength!);
      expect(results[i].nodeRelSize!).toBeLessThanOrEqual(results[i - 1].nodeRelSize!);
    }
  });
});
