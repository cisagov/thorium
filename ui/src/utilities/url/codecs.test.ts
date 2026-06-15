import { describe, it, expect } from 'vitest';

// project imports
import { boolCodec, listCodec, setCodec, stringCodec } from './codecs';

describe('stringCodec', () => {
  const codec = stringCodec('index');

  it('round-trips a value', () => {
    const params = new URLSearchParams();
    codec.encode('All', params);
    expect(params.toString()).toBe('index=All');
    expect(codec.decode(params)).toBe('All');
  });

  it('treats an empty string as absent', () => {
    const params = new URLSearchParams();
    codec.encode('', params);
    expect(params.toString()).toBe('');
    expect(codec.decode(params)).toBeUndefined();
  });

  it('reports the keys it owns', () => {
    expect(codec.keys(new URLSearchParams('index=All'))).toEqual(['index']);
  });

  it('decodes undefined when the key is missing', () => {
    expect(codec.decode(new URLSearchParams('other=x'))).toBeUndefined();
  });
});

describe('listCodec', () => {
  const codec = listCodec('open');

  it('round-trips a list as repeated params', () => {
    const params = new URLSearchParams();
    codec.encode(['a', 'b'], params);
    expect(params.getAll('open')).toEqual(['a', 'b']);
    expect(codec.decode(params)).toEqual(['a', 'b']);
  });

  it('preserves order and duplicates', () => {
    const params = new URLSearchParams();
    codec.encode(['b', 'a', 'b'], params);
    expect(codec.decode(params)).toEqual(['b', 'a', 'b']);
  });

  it('treats an empty list as absent', () => {
    const params = new URLSearchParams();
    codec.encode([], params);
    expect(params.toString()).toBe('');
    expect(codec.decode(params)).toBeUndefined();
  });
});

describe('boolCodec', () => {
  const codec = boolCodec('flagged');

  it('encodes true as `=1` and decodes it back', () => {
    const params = new URLSearchParams();
    codec.encode(true, params);
    expect(params.toString()).toBe('flagged=1');
    expect(codec.decode(params)).toBe(true);
  });

  it('treats false as absent (not written, decodes undefined)', () => {
    const params = new URLSearchParams();
    codec.encode(false, params);
    expect(params.toString()).toBe('');
    expect(codec.decode(params)).toBeUndefined();
  });

  it('decodes any non-`1` value as undefined', () => {
    expect(codec.decode(new URLSearchParams('flagged=0'))).toBeUndefined();
    expect(codec.decode(new URLSearchParams('flagged=true'))).toBeUndefined();
    expect(codec.decode(new URLSearchParams('other=x'))).toBeUndefined();
  });

  it('reports the key it owns', () => {
    expect(codec.keys(new URLSearchParams('flagged=1'))).toEqual(['flagged']);
  });
});

describe('setCodec', () => {
  const codec = setCodec('hidden');

  it('round-trips a set as repeated params, preserving insertion order', () => {
    const params = new URLSearchParams();
    codec.encode(new Set(['b', 'a', 'c']), params);
    expect(params.getAll('hidden')).toEqual(['b', 'a', 'c']);
    expect(Array.from(codec.decode(params) ?? [])).toEqual(['b', 'a', 'c']);
  });

  it('collapses duplicate URL occurrences into the set', () => {
    expect(Array.from(codec.decode(new URLSearchParams('hidden=a&hidden=a&hidden=b')) ?? [])).toEqual(['a', 'b']);
  });

  it('treats an empty set as absent (not written, decodes undefined)', () => {
    const params = new URLSearchParams();
    codec.encode(new Set(), params);
    expect(params.toString()).toBe('');
    expect(codec.decode(params)).toBeUndefined();
  });

  it('round-trips adversarial ids (sha256/uuid, `&`, `=`, `%`, unicode) through URLSearchParams', () => {
    const ids = ['a'.repeat(64), '11111111-1111-1111-1111-111111111111', 'a&b=c%d', 'ट्री 日本語'];
    const encoded = new URLSearchParams();
    codec.encode(new Set(ids), encoded);
    // re-parse the serialized string so outer percent-encoding is exercised, not just the live object
    const decoded = codec.decode(new URLSearchParams(encoded.toString()));
    expect(Array.from(decoded ?? [])).toEqual(ids);
  });

  it('decodes undefined when the key is missing', () => {
    expect(codec.decode(new URLSearchParams('other=x'))).toBeUndefined();
  });
});
