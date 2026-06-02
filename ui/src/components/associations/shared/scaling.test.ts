import { describe, it, expect } from 'vitest';

// project imports
import { getNodeSize, scoreNode } from './scaling';

describe('getNodeSize', () => {
  it('returns base size for small graphs', () => {
    expect(getNodeSize(300, 10)).toBe(30);
    expect(getNodeSize(400, 30)).toBe(40);
  });

  it('scales down for larger graphs', () => {
    const small = getNodeSize(300, 30);
    const medium = getNodeSize(300, 200);
    const large = getNodeSize(300, 1000);

    expect(medium).toBeLessThan(small);
    expect(large).toBeLessThan(medium);
  });

  it('has a minimum scale floor', () => {
    const veryLarge = getNodeSize(300, 100000);
    expect(veryLarge).toBeGreaterThan(0);
    expect(veryLarge).toBeGreaterThanOrEqual(30 * 0.3);
  });

  it('scales monotonically', () => {
    const counts = [30, 50, 100, 200, 500, 1000, 5000];
    const sizes = counts.map((n) => getNodeSize(300, n));

    for (let i = 1; i < sizes.length; i++) {
      expect(sizes[i]).toBeLessThanOrEqual(sizes[i - 1]);
    }
  });
});

describe('scoreNode', () => {
  it('scores Sample nodes based on tags', () => {
    const node = { Sample: { tags: { YaraRuleHit: { rule1: true } } } };
    expect(scoreNode(node)).toBeGreaterThanOrEqual(250);
  });

  it('returns default score for unknown nodes', () => {
    expect(scoreNode({})).toBe(300);
  });

  it('scores Repo nodes', () => {
    expect(scoreNode({ Repo: {} })).toBe(400);
  });

  it('scores Tag nodes', () => {
    expect(scoreNode({ Tag: {} })).toBe(350);
  });
});
