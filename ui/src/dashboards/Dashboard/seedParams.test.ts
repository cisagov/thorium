import { describe, it, expect } from 'vitest';

// project imports
import { decodeSeedParams, encodeSeedParams } from './seedParams';
import type { Seed } from '@models/trees';

/**
 * Build `URLSearchParams` from a list of `[key, value]` pairs (allows repeated keys, unlike an
 * object literal).
 */
function paramsOf(pairs: [string, string][]): URLSearchParams {
  const params = new URLSearchParams();
  for (const [key, value] of pairs) {
    params.append(key, value);
  }
  return params;
}

describe('decodeSeedParams', () => {
  it('empty params -> empty seed + default depth', () => {
    const { seed, depth } = decodeSeedParams(new URLSearchParams());
    expect(seed).toEqual({});
    expect(depth).toBe(2);
  });

  it('decodes each resource kind', () => {
    const params = paramsOf([
      ['sample', 'a'.repeat(64)],
      ['entity', '11111111-1111-1111-1111-111111111111'],
      ['repo', 'https://github.com/foo/bar'],
      ['tag', 'FileType:PE32'],
    ]);
    const { seed } = decodeSeedParams(params);
    expect(seed.samples).toEqual(['a'.repeat(64)]);
    expect(seed.entities).toEqual(['11111111-1111-1111-1111-111111111111']);
    expect(seed.repos).toEqual(['https://github.com/foo/bar']);
    expect(seed.tags).toEqual({ FileType: ['PE32'] });
  });

  it('routes entity params into Seed.entities and never Seed.devices', () => {
    const { seed } = decodeSeedParams(paramsOf([['entity', 'dev-uuid']]));
    expect(seed.entities).toEqual(['dev-uuid']);
    expect(seed.devices).toBeUndefined();
  });

  it('ignores unrelated (omnibar / hash) params', () => {
    const params = paramsOf([
      ['sample', 'a'.repeat(64)],
      ['query', 'malware'],
      ['groups', 'corp'],
      ['c', 'somegenericclause'],
      ['tags[foo]', 'bar'],
      ['limit', '50'],
    ]);
    const { seed } = decodeSeedParams(params);
    expect(seed).toEqual({ samples: ['a'.repeat(64)] });
  });

  it('ignores the dashboard hidden/flagged state params (owned by their own codecs, not the seed)', () => {
    const params = paramsOf([
      ['sample', 'a'.repeat(64)],
      ['hidden', 'node-1'],
      ['hidden', 'node-2'],
      ['flagged', '1'],
    ]);
    const { seed, depth } = decodeSeedParams(params);
    expect(seed).toEqual({ samples: ['a'.repeat(64)] });
    expect(depth).toBe(2);
  });

  it('drops empty resource values', () => {
    const params = paramsOf([
      ['sample', ''],
      ['entity', ''],
      ['repo', ''],
    ]);
    expect(decodeSeedParams(params).seed).toEqual({});
  });

  describe('depth validation', () => {
    it.each([
      ['absent', undefined, 2],
      ['empty', '', 2],
      ['junk', 'abc', 2],
      ['negative', '-1', 2],
      ['NaN-ish', 'NaN', 2],
      ['float', '2.5', 2],
      ['zero', '0', 0],
      ['in-range', '5', 5],
      ['at-bound', '10', 10],
      ['over-bound clamps', '99', 10],
    ])('%s -> %s', (_label, raw, expected) => {
      const pairs: [string, string][] = raw === undefined ? [] : [['depth', raw]];
      expect(decodeSeedParams(paramsOf(pairs)).depth).toBe(expected);
    });
  });

  describe('tag adversarial inputs', () => {
    it('splits on the first colon only (colons in value)', () => {
      const value = 'ns:with:colons';
      const raw = `${encodeURIComponent('some/key')}:${encodeURIComponent(value)}`;
      const { seed } = decodeSeedParams(paramsOf([['tag', raw]]));
      expect(seed.tags).toEqual({ 'some/key': [value] });
    });

    it('preserves &, =, %, and unicode in key and value', () => {
      const key = 'a&b=c%d';
      const value = 'x&y=z 日本語';
      const raw = `${encodeURIComponent(key)}:${encodeURIComponent(value)}`;
      const { seed } = decodeSeedParams(paramsOf([['tag', raw]]));
      expect(seed.tags).toEqual({ [key]: [value] });
    });

    it('keeps an empty value (tag=key:)', () => {
      const { seed } = decodeSeedParams(paramsOf([['tag', `${encodeURIComponent('empty')}:`]]));
      expect(seed.tags).toEqual({ empty: [''] });
    });

    it('drops a tag with no colon', () => {
      const { seed } = decodeSeedParams(paramsOf([['tag', 'nocolon']]));
      expect(seed.tags).toBeUndefined();
    });

    it('groups multiple values under the same key', () => {
      const params = paramsOf([
        ['tag', `${encodeURIComponent('k')}:${encodeURIComponent('v1')}`],
        ['tag', `${encodeURIComponent('k')}:${encodeURIComponent('v2')}`],
      ]);
      expect(decodeSeedParams(params).seed.tags).toEqual({ k: ['v1', 'v2'] });
    });

    it('does not throw on a malformed percent escape', () => {
      const { seed } = decodeSeedParams(paramsOf([['tag', 'ke%y:va%lue']]));
      // safeDecode falls back to the raw half rather than throwing
      expect(seed.tags).toEqual({ 'ke%y': ['va%lue'] });
    });
  });

  describe('dedup', () => {
    it('dedupes repeated identical resource params', () => {
      const params = paramsOf([
        ['sample', 'dup'],
        ['sample', 'dup'],
        ['entity', 'e'],
        ['entity', 'e'],
        ['repo', 'r'],
        ['repo', 'r'],
      ]);
      const { seed } = decodeSeedParams(params);
      expect(seed.samples).toEqual(['dup']);
      expect(seed.entities).toEqual(['e']);
      expect(seed.repos).toEqual(['r']);
    });

    it('dedupes repeated identical tag values within a key', () => {
      const enc = `${encodeURIComponent('k')}:${encodeURIComponent('v')}`;
      const params = paramsOf([
        ['tag', enc],
        ['tag', enc],
      ]);
      expect(decodeSeedParams(params).seed.tags).toEqual({ k: ['v'] });
    });
  });

  it('round-trips a full repo URL with colons, slashes, query, and fragment', () => {
    const url = 'https://user:pw@host.example.com:8443/a/b?x=1&y=2#frag';
    const { seed } = decodeSeedParams(paramsOf([['repo', url]]));
    expect(seed.repos).toEqual([url]);
  });
});

describe('round-trips', () => {
  const cases: { name: string; seed: Seed; depth: number }[] = [
    { name: 'empty', seed: {}, depth: 2 },
    { name: 'samples only', seed: { samples: ['a'.repeat(64), 'b'.repeat(64)] }, depth: 3 },
    { name: 'entities only', seed: { entities: ['id-1', 'id-2'] }, depth: 0 },
    { name: 'repos with special chars', seed: { repos: ['https://h:8443/a/b?q=1&r=2#f'] }, depth: 5 },
    {
      name: 'tags with adversarial chars',
      seed: { tags: { 'some/key': ['a:b', 'c&d=e'], unicode: ['日本語'], empty: [''] } },
      depth: 10,
    },
    {
      name: 'mixed',
      seed: {
        samples: ['a'.repeat(64)],
        entities: ['e-1'],
        repos: ['https://github.com/x/y'],
        tags: { FileType: ['PE32', 'ELF'] },
      },
      depth: 4,
    },
  ];

  it.each(cases)('encode -> decode preserves seed and depth ($name)', ({ seed, depth }) => {
    const params = encodeSeedParams(seed, depth);
    const decoded = decodeSeedParams(params);
    expect(decoded.seed).toEqual(seed);
    expect(decoded.depth).toBe(depth);
  });

  it('encode always writes a normalized depth', () => {
    // an out-of-range depth handed to encode is clamped so the emitted URL is always valid
    expect(encodeSeedParams({}, 99).get('depth')).toBe('10');
    expect(encodeSeedParams({}, -5).get('depth')).toBe('2');
  });

  it('decode -> encode -> decode is stable (idempotent through the codec)', () => {
    const params = paramsOf([
      ['sample', 'dup'],
      ['sample', 'dup'],
      ['tag', `${encodeURIComponent('k')}:${encodeURIComponent('v')}`],
      ['depth', '3'],
      // an unrelated param that must not survive the seed codec
      ['query', 'ignored'],
    ]);
    const first = decodeSeedParams(params);
    const reencoded = encodeSeedParams(first.seed, first.depth);
    const second = decodeSeedParams(reencoded);
    expect(second.seed).toEqual(first.seed);
    expect(second.depth).toBe(first.depth);
    // the re-encoded params never carry the unrelated key
    expect(reencoded.has('query')).toBe(false);
  });

  it('emits seed keys in a stable order', () => {
    const params = encodeSeedParams({ samples: ['s'], entities: ['e'], repos: ['r'], tags: { k: ['v'] } }, 2);
    expect(Array.from(params.keys())).toEqual(['sample', 'entity', 'repo', 'tag', 'depth']);
  });
});
