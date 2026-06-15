import { describe, it, expect } from 'vitest';

// project imports
import { flattenEntityTags } from './entityTags';

describe('flattenEntityTags', () => {
  it('returns an empty array for an empty tag map', () => {
    expect(flattenEntityTags({})).toEqual([]);
  });

  it('produces one chip per single-valued key', () => {
    expect(flattenEntityTags({ os: ['linux'] })).toEqual(['os: linux']);
  });

  it('produces one chip per value for multi-valued keys', () => {
    expect(flattenEntityTags({ arch: ['x86', 'arm'] })).toEqual(['arch: x86', 'arch: arm']);
  });

  it('flattens multiple keys in key/value order', () => {
    expect(flattenEntityTags({ a: ['1', '2'], b: ['3'] })).toEqual(['a: 1', 'a: 2', 'b: 3']);
  });

  it('omits keys whose value list is empty', () => {
    expect(flattenEntityTags({ empty: [], present: ['x'] })).toEqual(['present: x']);
  });
});
