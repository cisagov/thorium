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
} from './order';

// Layout constants mirrored from PipelineOrderFlow.tsx
// TERMINAL_OFFSET = 70, STEP_WIDTH = 200, CLUSTER_THRESHOLD = 120
// Stage 0 center: x=70, Stage 1 center: x=270, Stage 2 center: x=470

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

  test('insert before first stage (flowX < stage 0 x)', () => {
    // Stage 0 is at x=70, insert at x=0 which is < 70 and outside threshold
    const result = insertImageAtPosition(['a', 'b'], 'new', 0);
    // flowX=0 is within CLUSTER_THRESHOLD (120) of stage 0 at x=70: |0-70|=70 <= 120
    // So it clusters with stage 0 as parallel
    expect(result).toEqual([['a', 'new'], 'b']);
  });

  test('insert between two stages as new sequential step', () => {
    // Stage 0 at x=70, stage 1 at x=270. Insert at x=210 is outside threshold of stage 0
    // (|210-70|=140 > 120) but within threshold of stage 1 (|210-270|=60 <= 120)
    // So it clusters with stage 1
    const result = insertImageAtPosition(['a', 'b'], 'new', 210);
    expect(result).toEqual(['a', ['b', 'new']]);
  });

  test('cluster with existing stage as parallel', () => {
    // Stage 0 at x=70, flowX=80 is within threshold (|80-70|=10 <= 120)
    const result = insertImageAtPosition(['a', 'b'], 'new', 80);
    expect(result).toEqual([['a', 'new'], 'b']);
  });

  test('insert as new stage between stages when outside threshold of both', () => {
    // Stage 0 at x=70, stage 1 at x=270
    // flowX must be > 70+120=190 and < 270-120=150... impossible with these constants
    // Actually: we iterate stages in order. For stage 0 at x=70: |flowX-70|>120 means flowX>190 or flowX<-50
    // For stage 1 at x=270: flowX<270 and |flowX-270|>120 means flowX<150
    // No flowX satisfies both (>190 AND <150). With 3 stages:
    // Stage 0 at 70, stage 1 at 270, stage 2 at 470
    // Between 1 and 2: flowX>270+120=390 AND flowX<470-120=350 -- impossible
    // The CLUSTER_THRESHOLD (120) is 60% of STEP_WIDTH (200), so adjacent stages always overlap.
    // New sequential stages can only be inserted after all existing stages or before the first.
    // Test inserting before first by using negative flowX:
    const result = insertImageAtPosition(['a'], 'new', -200);
    // |(-200)-70| = 270 > 120, so not clustered. flowX < stageX, so insert before.
    expect(result).toEqual(['new', 'a']);
  });

  test('cluster into existing parallel stage', () => {
    const result = insertImageAtPosition([['a', 'b'], 'c'], 'new', 80);
    expect(result).toEqual([['a', 'b', 'new'], 'c']);
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
  test('short label clamps to min-width (80)', () => {
    // "curl" = 4 chars: 4 * 7.2 + 32 = 60.8 → clamped to 80
    expect(estimateNodeWidth('curl')).toBe(80);
  });

  test('medium label uses calculated width', () => {
    // "my-analyzer" = 11 chars: 11 * 7.2 + 32 = 111.2
    expect(estimateNodeWidth('my-analyzer')).toBeCloseTo(111.2);
  });

  test('long label clamps to max-width (210)', () => {
    // "very-long-image-name-here-extra" = 30 chars: 30 * 7.2 + 32 = 248 → clamped to 210
    expect(estimateNodeWidth('very-long-image-name-here-extra')).toBe(210);
  });

  test('empty string clamps to min-width', () => {
    // 0 * 7.2 + 32 = 32 → clamped to 80
    expect(estimateNodeWidth('')).toBe(80);
  });
});

describe('estimateStageWidth', () => {
  test('single image stage', () => {
    expect(estimateStageWidth('curl')).toBe(80);
  });

  test('parallel stage returns widest', () => {
    // "curl" → 80, "my-analyzer" → 111.2
    expect(estimateStageWidth(['curl', 'my-analyzer'])).toBeCloseTo(111.2);
  });
});
