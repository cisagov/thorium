import { describe, it, expect } from 'vitest';

// project imports
import { prettifiedSeed, prettify, prettifyFormatFor } from './prettify';
import { FormatType } from '@utilities/rules/types';

const enc = (s: string) => new TextEncoder().encode(s).buffer;
const bin = (...bytes: number[]) => new Uint8Array(bytes).buffer;

describe('prettify', () => {
  it('pretty-prints compact JSON with 2-space indent', () => {
    expect(prettify('{"a":1,"b":{"c":2}}', FormatType.JSON)).toBe('{\n  "a": 1,\n  "b": {\n    "c": 2\n  }\n}');
  });

  it('returns invalid JSON unchanged', () => {
    expect(prettify('{not json', FormatType.JSON)).toBe('{not json');
  });

  it('returns empty/whitespace input unchanged', () => {
    expect(prettify('', FormatType.JSON)).toBe('');
    expect(prettify('   ', FormatType.YAML)).toBe('   ');
  });

  it('reformats a flow-style YAML mapping to block style', () => {
    const out = prettify('{a: 1, b: 2}', FormatType.YAML);
    expect(out).not.toBe('{a: 1, b: 2}');
    expect(out).toContain('a: 1');
    expect(out).toContain('b: 2');
  });

  it('preserves YAML comments when reformatting', () => {
    const out = prettify('a: 1 # keep me\nb: 2\n', FormatType.YAML);
    expect(out).toContain('# keep me');
  });

  it('leaves a plain YAML scalar / plain text unchanged', () => {
    expect(prettify('just a plain string', FormatType.YAML)).toBe('just a plain string');
  });

  it('returns unparseable YAML unchanged', () => {
    const bad = 'a:\n  - 1\n b: 2';
    expect(prettify(bad, FormatType.YAML)).toBe(bad);
  });

  it('returns text unchanged for non JSON/YAML (or absent) formats', () => {
    expect(prettify('rule x {}', FormatType.YARA)).toBe('rule x {}');
    expect(prettify('int main() {}', FormatType.Decomp)).toBe('int main() {}');
    expect(prettify('{"a":1}')).toBe('{"a":1}');
  });
});

describe('prettifyFormatFor', () => {
  it('returns JSON for a .json extension', () => {
    expect(prettifyFormatFor('out.json', enc('{"a":1}'))).toBe(FormatType.JSON);
  });

  it('returns JSON for content-detected JSON without an extension', () => {
    expect(prettifyFormatFor('blob', enc('{"a":1}'))).toBe(FormatType.JSON);
  });

  it('returns YAML only for .yaml/.yml extensions', () => {
    expect(prettifyFormatFor('config.yaml', enc('a: 1'))).toBe(FormatType.YAML);
    expect(prettifyFormatFor('config.yml', enc('a: 1'))).toBe(FormatType.YAML);
  });

  it('returns undefined for code/text that only defaults to YAML highlighting', () => {
    expect(prettifyFormatFor('script.py', enc('print(1)'))).toBeUndefined();
    expect(prettifyFormatFor('notes.txt', enc('hello'))).toBeUndefined();
    expect(prettifyFormatFor('', enc('plain text'))).toBeUndefined();
  });

  it('returns undefined for binary content', () => {
    expect(prettifyFormatFor('data.bin', bin(0x00, 0x01, 0x02))).toBeUndefined();
  });
});

describe('prettifiedSeed', () => {
  it('pretty-prints JSON detected by a .json extension', () => {
    expect(prettifiedSeed({ fileName: 'out.json', bytes: enc('{"a":1,"b":2}') })).toBe('{\n  "a": 1,\n  "b": 2\n}');
  });

  it('pretty-prints JSON detected by content when the name is absent', () => {
    expect(prettifiedSeed({ bytes: enc('{"a":1}') })).toBe('{\n  "a": 1\n}');
  });

  it('reformats flow-style YAML for .yaml files', () => {
    const out = prettifiedSeed({ fileName: 'config.yaml', bytes: enc('{a: 1, b: 2}') });
    expect(out).not.toBe('{a: 1, b: 2}');
    expect(out).toContain('a: 1');
  });

  it('leaves plain text untouched when no prettify format applies', () => {
    expect(prettifiedSeed({ fileName: 'notes.txt', bytes: enc('hello world') })).toBe('hello world');
  });

  it('prefers the pre-decoded text over the raw bytes', () => {
    expect(prettifiedSeed({ fileName: 'out.json', bytes: enc('{"stale":true}'), text: '{"a":1}' })).toBe('{\n  "a": 1\n}');
  });
});
