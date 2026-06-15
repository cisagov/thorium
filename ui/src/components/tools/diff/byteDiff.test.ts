import { describe, it, expect } from 'vitest';

// project imports
import { computeByteDiff, mapSelectionAcross } from './byteDiff';
import { HexByteStatus } from '@components/shared/renderers';

const u8 = (...b: number[]) => new Uint8Array(b);

describe('computeByteDiff', () => {
  it('marks identical buffers as all Same and fully aligned', () => {
    const { baseStatus, compareStatus, baseToCompare } = computeByteDiff(u8(1, 2, 3), u8(1, 2, 3));
    expect(baseStatus).toEqual([HexByteStatus.Same, HexByteStatus.Same, HexByteStatus.Same]);
    expect(compareStatus).toEqual([HexByteStatus.Same, HexByteStatus.Same, HexByteStatus.Same]);
    expect(baseToCompare.get(0)).toBe(0);
    expect(baseToCompare.get(2)).toBe(2);
  });

  it('marks an appended byte as Added on the compare side', () => {
    const { baseStatus, compareStatus } = computeByteDiff(u8(1, 2), u8(1, 2, 3));
    expect(baseStatus).toEqual([HexByteStatus.Same, HexByteStatus.Same]);
    expect(compareStatus).toEqual([HexByteStatus.Same, HexByteStatus.Same, HexByteStatus.Added]);
  });

  it('marks a removed byte as Removed on the base side', () => {
    const { baseStatus, compareStatus } = computeByteDiff(u8(1, 2, 3), u8(1, 3));
    expect(baseStatus[1]).toBe(HexByteStatus.Removed);
    expect(compareStatus).toEqual([HexByteStatus.Same, HexByteStatus.Same]);
  });

  it('aligns unchanged bytes around a middle change', () => {
    // base: 1 2 3 4   compare: 1 9 9 4  -> byte 0 and 3 align
    const { baseToCompare, compareToBase } = computeByteDiff(u8(1, 2, 3, 4), u8(1, 9, 9, 4));
    expect(baseToCompare.get(0)).toBe(0);
    expect(baseToCompare.get(3)).toBe(3);
    expect(compareToBase.get(0)).toBe(0);
    expect(compareToBase.get(3)).toBe(3);
    // the differing middle bytes are not aligned
    expect(baseToCompare.has(1)).toBe(false);
  });
});

describe('mapSelectionAcross', () => {
  it('maps an aligned selection to the other side', () => {
    const map = new Map<number, number>([
      [0, 5],
      [1, 6],
      [2, 7],
    ]);
    expect(mapSelectionAcross({ offset: 0, length: 3 }, map)).toEqual({ offset: 5, length: 3 });
  });

  it('returns the bounding range when only some bytes are aligned', () => {
    const map = new Map<number, number>([
      [0, 5],
      [2, 9],
    ]);
    expect(mapSelectionAcross({ offset: 0, length: 3 }, map)).toEqual({ offset: 5, length: 5 });
  });

  it('returns null when no selected bytes are aligned', () => {
    const map = new Map<number, number>([[10, 10]]);
    expect(mapSelectionAcross({ offset: 0, length: 3 }, map)).toBeNull();
  });
});
