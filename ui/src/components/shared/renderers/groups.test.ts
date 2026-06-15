import { describe, it, expect } from 'vitest';

// project imports
import { detectRenderGroup, RENDER_LABELS } from './groups';
import { RenderKind } from './types';

const enc = (s: string) => new TextEncoder().encode(s).buffer;
const bin = (...bytes: number[]) => new Uint8Array(bytes).buffer;

describe('detectRenderGroup', () => {
  it('offers image then hex for images, defaulting to image', () => {
    const group = detectRenderGroup('logo.png', bin(0x89, 0x50, 0x4e, 0x47));
    expect(group.options).toEqual([RenderKind.Image, RenderKind.Hex]);
    expect(group.default).toBe(RenderKind.Image);
  });

  it('offers only hex for binary content', () => {
    const group = detectRenderGroup('blob.bin', bin(0x00, 0x01, 0x02, 0xff));
    expect(group.options).toEqual([RenderKind.Hex]);
    expect(group.default).toBe(RenderKind.Hex);
  });

  it('offers json then raw editor for json', () => {
    const group = detectRenderGroup('out.json', enc('{"a":1}'));
    expect(group.options).toEqual([RenderKind.Json, RenderKind.Editor]);
    expect(group.default).toBe(RenderKind.Json);
  });

  it('offers code, decomp, then raw editor for code/yaml', () => {
    const group = detectRenderGroup('config.yaml', enc('a: 1'));
    expect(group.options).toEqual([RenderKind.Code, RenderKind.Decomp, RenderKind.Editor]);
  });

  it('offers yara then raw editor for yara', () => {
    const group = detectRenderGroup('rules.yar', enc('rule x {}'));
    expect(group.options).toEqual([RenderKind.Yara, RenderKind.Editor]);
  });

  it('offers markdown then raw editor for markdown', () => {
    const group = detectRenderGroup('README.md', enc('# Title'));
    expect(group.options).toEqual([RenderKind.Markdown, RenderKind.Editor]);
    expect(group.default).toBe(RenderKind.Markdown);
  });

  it('offers text, decomp, then raw editor for plain text', () => {
    const group = detectRenderGroup('notes.txt', enc('just text'));
    expect(group.options).toEqual([RenderKind.Text, RenderKind.Decomp, RenderKind.Editor]);
  });

  it('always uses options[0] as the default', () => {
    const inputs: [string, ArrayBuffer][] = [
      ['logo.png', bin(0x89, 0x50, 0x4e, 0x47)],
      ['blob.bin', bin(0x00, 0x01)],
      ['out.json', enc('{"a":1}')],
      ['config.yaml', enc('a: 1')],
      ['README.md', enc('# Title')],
      ['notes.txt', enc('text')],
    ];
    for (const [name, bytes] of inputs) {
      const group = detectRenderGroup(name, bytes);
      expect(group.default).toBe(group.options[0]);
    }
  });
});

describe('RENDER_LABELS', () => {
  it('has a label for every render kind', () => {
    for (const kind of Object.values(RenderKind)) {
      expect(RENDER_LABELS[kind]).toBeTruthy();
    }
  });
});
