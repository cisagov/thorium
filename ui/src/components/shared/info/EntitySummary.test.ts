import { describe, expect, it } from 'vitest';

// project imports
import { resolveFieldLayout } from './EntitySummary';
import { FieldLayout, FieldRender, InfoField } from './info';

const field = (patch: Partial<InfoField>): InfoField => ({ label: 'L', ...patch });

describe('resolveFieldLayout', () => {
  it('honors an explicit layout override over any derivation', () => {
    // a number would auto-derive Inline, but an explicit Stacked wins
    expect(resolveFieldLayout(field({ value: 5, layout: FieldLayout.Stacked }))).toBe(FieldLayout.Stacked);
    // a long string would auto-derive Stacked, but an explicit Inline wins (short word-lists / scalars)
    expect(resolveFieldLayout(field({ value: 'x'.repeat(80), layout: FieldLayout.Inline }))).toBe(FieldLayout.Inline);
    // an override forces a multi-value array inline (Critical Sectors, Groups, Tools, …)
    expect(resolveFieldLayout(field({ value: ['a', 'b', 'c'], layout: FieldLayout.Inline }))).toBe(FieldLayout.Inline);
  });

  it('stacks long prose (Markdown / Text)', () => {
    expect(resolveFieldLayout(field({ value: 'desc', render: FieldRender.Markdown }))).toBe(FieldLayout.Stacked);
    expect(resolveFieldLayout(field({ value: '/a/b/c', render: FieldRender.Text }))).toBe(FieldLayout.Stacked);
  });

  it('stacks multi-value arrays by default', () => {
    expect(resolveFieldLayout(field({ value: ['a', 'b'] }))).toBe(FieldLayout.Stacked);
  });

  it('inlines times, numbers, and booleans', () => {
    expect(resolveFieldLayout(field({ value: '2025-01-01T00:00:00', render: FieldRender.Time }))).toBe(FieldLayout.Inline);
    expect(resolveFieldLayout(field({ value: 1234 }))).toBe(FieldLayout.Inline);
    expect(resolveFieldLayout(field({ value: true }))).toBe(FieldLayout.Inline);
    expect(resolveFieldLayout(field({ value: false }))).toBe(FieldLayout.Inline);
  });

  it('stacks free-form strings and single-element arrays by default (no override)', () => {
    expect(resolveFieldLayout(field({ value: 'some-hash-value' }))).toBe(FieldLayout.Stacked);
    // a lone-element array without an override falls through to the default (Stacked)
    expect(resolveFieldLayout(field({ value: ['only'] }))).toBe(FieldLayout.Stacked);
  });
});
