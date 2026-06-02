import { describe, test, expect } from 'vitest';
import { calculateWindowResizeState, calculateCanvasResizeWindowPosition } from './resize';
import { Placement } from './placement';
import { CanvasType } from './bounds';

// shared fixtures
const bounds = {
  top: 0,
  bottom: 0,
  start: 0,
  end: 0,
  type: CanvasType.Document,
  width: 1200,
  height: 800,
};

const startPosition = {
  width: 400,
  height: 300,
  top: 100,
  left: 200,
  pointerX: 500,
  pointerY: 400,
};

const minSize = { width: 200, height: 150 };

describe('calculateWindowResizeState', () => {
  test('no mouse movement produces no resize', () => {
    const result = calculateWindowResizeState(startPosition, minSize, 0, 0, bounds, Placement.BottomRight);
    expect(result.size).toEqual({ width: 400, height: 300 });
    expect(result.position).toEqual({ top: 100, left: 200 });
    expect(result.resized).toBe(false);
  });

  test('dragging bottom edge down increases height', () => {
    // dy negative = mouse moved down, increasing window height
    const result = calculateWindowResizeState(startPosition, minSize, 0, -50, bounds, Placement.Bottom);
    expect(result.size.height).toBe(350);
    expect(result.position.top).toBe(100);
    expect(result.resized).toBe(true);
  });

  test('dragging bottom edge up decreases height', () => {
    // dy positive = mouse moved up, decreasing window height
    const result = calculateWindowResizeState(startPosition, minSize, 0, 50, bounds, Placement.Bottom);
    expect(result.size.height).toBe(250);
    expect(result.resized).toBe(true);
  });

  test('height cannot shrink below min size', () => {
    // try to shrink by 200 but max decrease is 300 - 150 = 150
    const result = calculateWindowResizeState(startPosition, minSize, 0, 200, bounds, Placement.Bottom);
    expect(result.size.height).toBe(150);
    expect(result.resized).toBe(true);
  });

  test('dragging top edge up moves top and increases height', () => {
    const result = calculateWindowResizeState(startPosition, minSize, 0, 50, bounds, Placement.Top);
    expect(result.size.height).toBe(350);
    expect(result.position.top).toBe(50);
    expect(result.resized).toBe(true);
  });

  test('dragging right edge right increases width', () => {
    // dx negative = mouse moved right
    const result = calculateWindowResizeState(startPosition, minSize, -80, 0, bounds, Placement.Right);
    expect(result.size.width).toBe(480);
    expect(result.position.left).toBe(200);
    expect(result.resized).toBe(true);
  });

  test('dragging left edge left moves left and increases width', () => {
    // dx positive = mouse moved left
    const result = calculateWindowResizeState(startPosition, minSize, 60, 0, bounds, Placement.Left);
    expect(result.size.width).toBe(460);
    expect(result.position.left).toBe(140);
    expect(result.resized).toBe(true);
  });

  test('corner drag resizes both dimensions', () => {
    const result = calculateWindowResizeState(startPosition, minSize, -50, -50, bounds, Placement.BottomRight);
    expect(result.size.width).toBe(450);
    expect(result.size.height).toBe(350);
    expect(result.resized).toBe(true);
  });

  test('width cannot shrink below min size via right edge', () => {
    // max decrease = 400 - 200 = 200, try 300
    const result = calculateWindowResizeState(startPosition, minSize, 300, 0, bounds, Placement.Right);
    expect(result.size.width).toBe(200);
  });

  test('top edge cannot move above canvas top', () => {
    // try to grow up by 200 but top is only 100px from canvas top
    const result = calculateWindowResizeState(startPosition, minSize, 0, 150, bounds, Placement.Top);
    // clamped: can only move up by 100 (top is at 100, bounds.top is 0)
    expect(result.position.top).toBe(0);
    expect(result.size.height).toBe(400);
  });
});

describe('calculateCanvasResizeWindowPosition', () => {
  const startBounds = {
    top: 0,
    bottom: 0,
    start: 0,
    end: 0,
    type: CanvasType.Document,
    width: 1200,
    height: 800,
  };

  const padding = { top: 4, bottom: 4, start: 4, end: 4 };

  test('proportional repositioning on canvas width increase', () => {
    const updatedBounds = { ...startBounds, width: 1400 };
    const size = { width: 400, height: 300 };
    const marginRatio = { start: 0.25, end: 0.75, top: 0.5, bottom: 0.5 };
    const startPosition = { top: 100, left: 200 };

    const result = calculateCanvasResizeWindowPosition(startBounds, updatedBounds, size, padding, marginRatio, startPosition);
    // dx = 200, changeLeft = round(0.25 * 200) = 50
    expect(result.left).toBe(250);
  });

  test('proportional repositioning on canvas height increase', () => {
    const updatedBounds = { ...startBounds, height: 1000 };
    const size = { width: 400, height: 300 };
    const marginRatio = { start: 0.5, end: 0.5, top: 0.25, bottom: 0.75 };
    const startPosition = { top: 100, left: 200 };

    const result = calculateCanvasResizeWindowPosition(startBounds, updatedBounds, size, padding, marginRatio, startPosition);
    // dy = 200, changeTop = round(0.25 * 200) = 50
    expect(result.top).toBe(150);
  });

  test('window stays within bounds when canvas shrinks', () => {
    const updatedBounds = { ...startBounds, width: 500, height: 400 };
    const size = { width: 400, height: 300 };
    const marginRatio = { start: 0.5, end: 0.5, top: 0.5, bottom: 0.5 };
    const startPosition = { top: 400, left: 600 };

    const result = calculateCanvasResizeWindowPosition(startBounds, updatedBounds, size, padding, marginRatio, startPosition);
    // window should be clamped to fit
    expect(result.left).toBeLessThanOrEqual(updatedBounds.width - size.width);
    expect(result.top).toBeLessThanOrEqual(updatedBounds.height - size.height);
  });

  test('window at origin stays at padding', () => {
    const updatedBounds = { ...startBounds, width: 800, height: 600 };
    const size = { width: 400, height: 300 };
    const marginRatio = { start: 0, end: 1, top: 0, bottom: 1 };
    const startPosition = { top: 0, left: 0 };

    const result = calculateCanvasResizeWindowPosition(startBounds, updatedBounds, size, padding, marginRatio, startPosition);
    expect(result.top).toBeGreaterThanOrEqual(padding.top);
    expect(result.left).toBeGreaterThanOrEqual(padding.start);
  });

  test('no change when canvas stays same size', () => {
    const size = { width: 400, height: 300 };
    const marginRatio = { start: 0.5, end: 0.5, top: 0.5, bottom: 0.5 };
    const startPosition = { top: 200, left: 300 };

    const result = calculateCanvasResizeWindowPosition(startBounds, startBounds, size, padding, marginRatio, startPosition);
    expect(result.top).toBe(200);
    expect(result.left).toBe(300);
  });
});
