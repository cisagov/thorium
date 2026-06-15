import { describe, it, expect } from 'vitest';

// project imports
import { computeScaledDimensions } from './image';

describe('computeScaledDimensions', () => {
  it('leaves images already within the bound unchanged', () => {
    expect(computeScaledDimensions(100, 80, 256)).toEqual({ width: 100, height: 80 });
  });

  it('does not upscale images smaller than the bound', () => {
    expect(computeScaledDimensions(10, 10, 256)).toEqual({ width: 10, height: 10 });
  });

  it('scales a wide image so the longest edge equals maxPx', () => {
    expect(computeScaledDimensions(1000, 500, 256)).toEqual({ width: 256, height: 128 });
  });

  it('scales a tall image so the longest edge equals maxPx', () => {
    expect(computeScaledDimensions(500, 1000, 256)).toEqual({ width: 128, height: 256 });
  });

  it('keeps a square image square', () => {
    expect(computeScaledDimensions(1024, 1024, 256)).toEqual({ width: 256, height: 256 });
  });

  it('rounds to whole pixels and never goes below 1', () => {
    const result = computeScaledDimensions(1000, 3, 256);
    expect(result.width).toBe(256);
    // 3 * (256/1000) = 0.768 -> rounds to 1, clamped to minimum 1
    expect(result.height).toBe(1);
  });

  it('handles a zero-sized source without dividing by zero', () => {
    expect(computeScaledDimensions(0, 0, 256)).toEqual({ width: 0, height: 0 });
  });
});
