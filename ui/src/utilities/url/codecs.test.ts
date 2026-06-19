import { describe, it, expect } from 'vitest';

// project imports
import { stringCodec, listCodec } from './codecs';

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
