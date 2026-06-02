import { describe, test, expect } from 'vitest';
import { clamp, boundElementSize } from './utilities';
import { CanvasType } from './bounds';

describe('clamp', () => {
  test('returns value when within range', () => {
    expect(clamp(5, 0, 10)).toBe(5);
  });

  test('returns min when value is below range', () => {
    expect(clamp(-5, 0, 10)).toBe(0);
  });

  test('returns max when value is above range', () => {
    expect(clamp(15, 0, 10)).toBe(10);
  });

  test('returns min when value equals min', () => {
    expect(clamp(0, 0, 10)).toBe(0);
  });

  test('returns max when value equals max', () => {
    expect(clamp(10, 0, 10)).toBe(10);
  });

  test('handles negative ranges', () => {
    expect(clamp(-5, -10, -1)).toBe(-5);
    expect(clamp(-15, -10, -1)).toBe(-10);
    expect(clamp(0, -10, -1)).toBe(-1);
  });

  test('handles zero-width range', () => {
    expect(clamp(5, 3, 3)).toBe(3);
    expect(clamp(1, 3, 3)).toBe(3);
  });
});

describe('boundElementSize', () => {
  const defaultBounds = {
    top: 0,
    bottom: 0,
    start: 0,
    end: 0,
    type: CanvasType.Document,
    width: 1000,
    height: 800,
  };

  const defaultPadding = { top: 0, bottom: 0, start: 0, end: 0 };

  test('returns original size when within bounds', () => {
    const result = boundElementSize({ width: 400, height: 300 }, defaultBounds, defaultPadding);
    expect(result).toEqual({ width: 400, height: 300 });
  });

  test('constrains width to bounds', () => {
    const result = boundElementSize({ width: 1200, height: 300 }, defaultBounds, defaultPadding);
    expect(result).toEqual({ width: 1000, height: 300 });
  });

  test('constrains height to bounds', () => {
    const result = boundElementSize({ width: 400, height: 1000 }, defaultBounds, defaultPadding);
    expect(result).toEqual({ width: 400, height: 800 });
  });

  test('constrains both dimensions', () => {
    const result = boundElementSize({ width: 1200, height: 1000 }, defaultBounds, defaultPadding);
    expect(result).toEqual({ width: 1000, height: 800 });
  });

  test('accounts for padding in size constraint', () => {
    const padding = { top: 10, bottom: 10, start: 20, end: 20 };
    const result = boundElementSize({ width: 1000, height: 800 }, defaultBounds, padding);
    expect(result).toEqual({ width: 960, height: 780 });
  });

  test('handles negative padding values (absolute value used)', () => {
    const padding = { top: -10, bottom: -10, start: -20, end: -20 };
    const result = boundElementSize({ width: 1000, height: 800 }, defaultBounds, padding);
    expect(result).toEqual({ width: 960, height: 780 });
  });

  test('element smaller than padding-adjusted bounds is unchanged', () => {
    const padding = { top: 50, bottom: 50, start: 50, end: 50 };
    const result = boundElementSize({ width: 200, height: 200 }, defaultBounds, padding);
    expect(result).toEqual({ width: 200, height: 200 });
  });
});
