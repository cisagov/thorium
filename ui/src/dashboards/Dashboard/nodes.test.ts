import { describe, it, expect } from 'vitest';

// project imports
import { visibleNodes } from './nodes';
import type { TreeNode } from '@models/trees';

// minimal TreeNode stand-ins; visibleNodes only cares about identity/presence, not shape
const nodeA = { id: 'a' } as unknown as TreeNode;
const nodeB = { id: 'b' } as unknown as TreeNode;
const dataMap: Record<string, TreeNode> = { a: nodeA, b: nodeB };

describe('visibleNodes', () => {
  it('returns every node when there is no visible set (no filter active)', () => {
    expect(visibleNodes(dataMap, null)).toEqual([nodeA, nodeB]);
    expect(visibleNodes(dataMap, undefined)).toEqual([nodeA, nodeB]);
  });

  it('returns only the visible ids when a filter is active', () => {
    expect(visibleNodes(dataMap, new Set(['b']))).toEqual([nodeB]);
  });

  it('drops ids not present in the data_map', () => {
    expect(visibleNodes(dataMap, new Set(['a', 'missing']))).toEqual([nodeA]);
  });

  it('handles an undefined data_map (graph not loaded yet)', () => {
    expect(visibleNodes(undefined, null)).toEqual([]);
    expect(visibleNodes(undefined, new Set(['a']))).toEqual([]);
  });

  it('returns an empty array for an empty visible set', () => {
    expect(visibleNodes(dataMap, new Set())).toEqual([]);
  });
});
