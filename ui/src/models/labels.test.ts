import { describe, it, expect } from 'vitest';

// project imports
import { labelWithFallback } from './labels';

type Kind = 'FooBar' | 'Baz';

const LABELS: Record<Kind, string> = {
  FooBar: 'Foo Bar',
  Baz: 'Custom Baz',
};

describe('labelWithFallback', () => {
  it('returns the mapped label for a known key', () => {
    expect(labelWithFallback(LABELS, 'FooBar')).toBe('Foo Bar');
    expect(labelWithFallback(LABELS, 'Baz')).toBe('Custom Baz');
  });

  it('falls back to humanize for an unmapped key', () => {
    expect(labelWithFallback(LABELS, 'SomeFutureKind')).toBe('Some Future Kind');
  });

  it('humanizes empty input to empty string', () => {
    expect(labelWithFallback(LABELS, '')).toBe('');
  });
});
