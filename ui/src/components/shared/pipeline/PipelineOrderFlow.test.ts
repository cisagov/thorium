import { describe, test, expect } from 'vitest';

// project imports
import {
  ordersEqual,
  insertImageAtPosition,
  removeImageFromOrder,
  removeImageAtPosition,
  getImagesInOrder,
  estimateNodeWidth,
  estimateStageWidth,
  clusterStagesByX,
  NODE_HEIGHT,
  HANDLE_CENTER_Y,
} from './order';

// Layout constants mirrored from PipelineOrderFlow.tsx
// TERMINAL_OFFSET = 70, STEP_WIDTH = 300, CLUSTER_THRESHOLD = 100
// Stage 0 left-edge: x=70, Stage 1: x=370, Stage 2: x=670

describe('ordersEqual', () => {
  test('equal single-image stages', () => {
    expect(ordersEqual(['a', 'b'], ['a', 'b'])).toBe(true);
  });

  test('equal parallel stages (order-independent)', () => {
    expect(ordersEqual([['b', 'a']], [['a', 'b']])).toBe(true);
  });

  test('different lengths', () => {
    expect(ordersEqual(['a'], ['a', 'b'])).toBe(false);
  });

  test('different images in same position', () => {
    expect(ordersEqual(['a'], ['b'])).toBe(false);
  });

  test('different parallel group sizes', () => {
    expect(ordersEqual([['a', 'b']], [['a']])).toBe(false);
  });

  test('empty arrays are equal', () => {
    expect(ordersEqual([], [])).toBe(true);
  });

  test('mixed sequential and parallel', () => {
    expect(ordersEqual([['a', 'b'], 'c'], [['b', 'a'], 'c'])).toBe(true);
    expect(ordersEqual([['a', 'b'], 'c'], ['a', ['b', 'c']])).toBe(false);
  });
});

describe('insertImageAtPosition', () => {
  test('into empty order', () => {
    expect(insertImageAtPosition([], 'img-a', 100)).toEqual(['img-a']);
  });

  test('append after all stages', () => {
    const result = insertImageAtPosition(['a'], 'b', 600);
    expect(result).toEqual(['a', 'b']);
  });

  test('insert before first stage (flowX left of stage 0 node)', () => {
    // Stage 0's node left edge is at x=TERMINAL_OFFSET (70). A click in the Start→first-node
    // gap (flowX < 70) prepends a new leading stage rather than clustering into stage 0.
    const result = insertImageAtPosition(['a', 'b'], 'new', 30);
    expect(result).toEqual(['new', 'a', 'b']);
  });

  test('click in the gap between stages inserts a new sequential stage', () => {
    // Stage 0 at x=70, stage 1 at x=370. flowX=210 is outside both stages' threshold (|210-70|=140,
    // |210-370|=160, both > 100), and is left of stage 1, so it inserts a new stage between them.
    const result = insertImageAtPosition(['a', 'b'], 'new', 210);
    expect(result).toEqual(['a', 'new', 'b']);
  });

  test('cluster with existing stage as parallel', () => {
    // Stage 0 at x=70, flowX=80 is within threshold (|80-70|=10 <= 100)
    const result = insertImageAtPosition(['a', 'b'], 'new', 80);
    expect(result).toEqual([['a', 'new'], 'b']);
  });

  test('click far left of the first stage prepends a new leading stage', () => {
    // flowX=-200 is left of stage 0's node (< TERMINAL_OFFSET=70), so it prepends a leading stage.
    const result = insertImageAtPosition(['a'], 'new', -200);
    expect(result).toEqual(['new', 'a']);
  });

  test('cluster into existing parallel stage', () => {
    const result = insertImageAtPosition([['a', 'b'], 'c'], 'new', 80);
    expect(result).toEqual([['a', 'b', 'new'], 'c']);
  });
});

describe('clusterStagesByX (drag-reorder result)', () => {
  // Stages laid out at x = 70, 370, 670 for a 3-image sequential order.
  test('returns stages in left-to-right order', () => {
    expect(
      clusterStagesByX([
        { label: 'a', x: 70 },
        { label: 'b', x: 370 },
        { label: 'c', x: 670 },
      ]),
    ).toEqual(['a', 'b', 'c']);
  });

  test('dragging a node before the first stage makes it the new leading stage', () => {
    // 'c' dropped left of 'a' (near the Start terminal), separated by > CLUSTER_THRESHOLD (100).
    const order = clusterStagesByX([
      { label: 'c', x: -50 },
      { label: 'a', x: 70 },
      { label: 'b', x: 370 },
    ]);
    expect(order).toEqual(['c', 'a', 'b']);
  });

  test('dragging a node after the last stage makes it the new trailing stage', () => {
    // 'a' dropped right of 'c' (near the End terminal), separated by > CLUSTER_THRESHOLD.
    const order = clusterStagesByX([
      { label: 'b', x: 370 },
      { label: 'c', x: 670 },
      { label: 'a', x: 820 },
    ]);
    expect(order).toEqual(['b', 'c', 'a']);
  });

  test('dropping a node onto a stage (within threshold) makes them parallel', () => {
    const order = clusterStagesByX([
      { label: 'a', x: 70 },
      { label: 'b', x: 120 },
      { label: 'c', x: 670 },
    ]);
    expect(order).toEqual([['a', 'b'], 'c']);
  });
});

describe('removeImageFromOrder', () => {
  test('remove from single-image stage', () => {
    expect(removeImageFromOrder(['a', 'b', 'c'], 'b')).toEqual(['a', 'c']);
  });

  test('remove from parallel stage leaving one image', () => {
    expect(removeImageFromOrder([['a', 'b'], 'c'], 'a')).toEqual(['b', 'c']);
  });

  test('remove from parallel stage leaving two images', () => {
    expect(removeImageFromOrder([['a', 'b', 'c'], 'd'], 'b')).toEqual([['a', 'c'], 'd']);
  });

  test('remove non-existent image returns same order', () => {
    expect(removeImageFromOrder(['a', 'b'], 'z')).toEqual(['a', 'b']);
  });

  test('remove only image empties the order', () => {
    expect(removeImageFromOrder(['a'], 'a')).toEqual([]);
  });

  test('remove from mixed order', () => {
    expect(removeImageFromOrder([['x', 'y'], 'z', ['a', 'b']], 'y')).toEqual(['x', 'z', ['a', 'b']]);
  });
});

describe('removeImageAtPosition', () => {
  test('remove second of two duplicate sequential stages', () => {
    expect(removeImageAtPosition(['a', 'b', 'a'], 2, 0)).toEqual(['a', 'b']);
  });

  test('remove first of two duplicate sequential stages', () => {
    expect(removeImageAtPosition(['a', 'b', 'a'], 0, 0)).toEqual(['b', 'a']);
  });

  test('remove from parallel group by parallel index', () => {
    expect(removeImageAtPosition([['a', 'b', 'c'], 'd'], 0, 1)).toEqual([['a', 'c'], 'd']);
  });

  test('remove from parallel group collapsing to solo string', () => {
    expect(removeImageAtPosition([['a', 'b'], 'c'], 0, 0)).toEqual(['b', 'c']);
  });

  test('remove only image produces empty order', () => {
    expect(removeImageAtPosition(['a'], 0, 0)).toEqual([]);
  });

  test('remove solo stage from middle of order', () => {
    expect(removeImageAtPosition(['a', 'b', 'c'], 1, 0)).toEqual(['a', 'c']);
  });
});

describe('getImagesInOrder', () => {
  test('extracts from sequential stages', () => {
    expect(getImagesInOrder(['a', 'b', 'c'])).toEqual(new Set(['a', 'b', 'c']));
  });

  test('extracts from parallel stages', () => {
    expect(getImagesInOrder([['a', 'b'], 'c'])).toEqual(new Set(['a', 'b', 'c']));
  });

  test('empty order', () => {
    expect(getImagesInOrder([])).toEqual(new Set());
  });

  test('mixed sequential and parallel', () => {
    expect(getImagesInOrder(['x', ['a', 'b'], 'y'])).toEqual(new Set(['x', 'a', 'b', 'y']));
  });

  test('deduplicates (though duplicates should not occur)', () => {
    expect(getImagesInOrder(['a', ['a', 'b']]).size).toBe(2);
  });
});

describe('estimateNodeWidth', () => {
  test('short label clamps to min-width (120)', () => {
    // "curl" = 4 chars: 4 * 7.2 + 36 = 64.8 → clamped to 120
    expect(estimateNodeWidth('curl')).toBe(120);
  });

  test('medium label uses calculated width', () => {
    // "my-analyzer-tool" = 16 chars: 16 * 7.2 + 36 = 151.2
    expect(estimateNodeWidth('my-analyzer-tool')).toBeCloseTo(151.2);
  });

  test('long label clamps to max-width (240)', () => {
    // "very-long-image-name-here-extra" = 31 chars: 31 * 7.2 + 36 = 259.2 → clamped to 240
    expect(estimateNodeWidth('very-long-image-name-here-extra')).toBe(240);
  });

  test('empty string clamps to min-width', () => {
    // 0 * 7.2 + 36 = 36 → clamped to 120
    expect(estimateNodeWidth('')).toBe(120);
  });
});

describe('vertical handle geometry', () => {
  // Image nodes, terminals, and barriers all derive their height/offset from these so every edge
  // connection point shares one vertical center — otherwise the spine renders as a slight diagonal.
  test('handle center is half the node height', () => {
    expect(HANDLE_CENTER_Y).toBe(NODE_HEIGHT / 2);
  });
});

describe('estimateStageWidth', () => {
  test('single image stage', () => {
    expect(estimateStageWidth('curl')).toBe(120);
  });

  test('parallel stage returns widest', () => {
    // "curl" → 120, "my-analyzer-tool" → 151.2
    expect(estimateStageWidth(['curl', 'my-analyzer-tool'])).toBeCloseTo(151.2);
  });
});
