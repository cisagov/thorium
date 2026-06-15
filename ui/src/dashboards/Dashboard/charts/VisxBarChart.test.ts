import { describe, expect, it } from 'vitest';

// project imports
import { truncateTick } from './VisxBarChart';

describe('truncateTick', () => {
  it('returns the label unchanged when at or under the cap', () => {
    expect(truncateTick('PE32', 10)).toBe('PE32');
    expect(truncateTick('1234567890', 10)).toBe('1234567890');
  });
  it('truncates with an ellipsis when longer than the cap', () => {
    // keeps cap-1 chars plus the ellipsis so the result length equals the cap
    expect(truncateTick('FileTypeExtension', 10)).toBe('FileTypeE…');
    expect(truncateTick('FileTypeExtension', 10).length).toBe(10);
  });
  it('respects a larger cap for rotated labels', () => {
    // 14-char cap keeps more of a long value than the 10-char horizontal cap
    expect(truncateTick('FileTypeExtension', 14)).toBe('FileTypeExten…');
    expect(truncateTick('FileTypeExtension', 14).length).toBe(14);
  });
});
