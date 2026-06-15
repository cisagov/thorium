import { describe, it, expect } from 'vitest';

// project imports
import { decodeHexValues } from './decode';

const find = (entries: { label: string; value: string }[], label: string) => entries.find((e) => e.label === label)?.value;

describe('decodeHexValues', () => {
  it('returns empty array for null/empty selection', () => {
    const bytes = new Uint8Array([1, 2, 3]);
    expect(decodeHexValues(bytes, null)).toEqual([]);
    expect(decodeHexValues(bytes, { offset: 0, length: 0 })).toEqual([]);
    expect(decodeHexValues(bytes, { offset: 5, length: 1 })).toEqual([]);
  });

  it('decodes a single byte', () => {
    const bytes = new Uint8Array([0xff]);
    const entries = decodeHexValues(bytes, { offset: 0, length: 1 });
    expect(find(entries, 'uint8')).toBe('255');
    expect(find(entries, 'int8')).toBe('-1');
    expect(find(entries, 'hex')).toBe('ff');
    // no width-dependent rows for a single byte
    expect(find(entries, 'int16 (LE/BE)')).toBeUndefined();
  });

  it('decodes 16-bit values in both endiannesses', () => {
    const bytes = new Uint8Array([0x01, 0x00]); // LE=1, BE=256
    const entries = decodeHexValues(bytes, { offset: 0, length: 2 });
    expect(find(entries, 'uint16 (LE/BE)')).toBe('1 / 256');
  });

  it('decodes 32-bit and float values', () => {
    const bytes = new Uint8Array([0x00, 0x00, 0x80, 0x3f]); // float32 LE = 1.0
    const entries = decodeHexValues(bytes, { offset: 0, length: 4 });
    expect(find(entries, 'float32 (LE/BE)')?.startsWith('1 /')).toBe(true);
    expect(find(entries, 'uint32 (LE/BE)')).toBeDefined();
  });

  it('decodes 64-bit big integers', () => {
    const bytes = new Uint8Array([0x01, 0, 0, 0, 0, 0, 0, 0]); // uint64 LE = 1
    const entries = decodeHexValues(bytes, { offset: 0, length: 8 });
    expect(find(entries, 'uint64 (LE/BE)')?.startsWith('1 /')).toBe(true);
  });

  it('honors the offset and reports it', () => {
    const bytes = new Uint8Array([0xaa, 0x41, 0x42]);
    const entries = decodeHexValues(bytes, { offset: 1, length: 2 });
    expect(find(entries, 'ascii')).toBe('AB');
    expect(find(entries, 'Offset')).toBe('0x1 (1)');
    expect(find(entries, 'Length')).toBe('2 bytes');
  });

  it('clamps a selection that runs past the end', () => {
    const bytes = new Uint8Array([0x41, 0x42]);
    const entries = decodeHexValues(bytes, { offset: 0, length: 10 });
    expect(find(entries, 'ascii')).toBe('AB');
    expect(find(entries, 'Length')).toBe('2 bytes');
  });

  it('renders non-printable bytes as . in the ascii row', () => {
    // 0x00 (NUL) and 0xff are non-printable; 0x41 ("A") is printable
    const bytes = new Uint8Array([0x00, 0x41, 0xff]);
    const entries = decodeHexValues(bytes, { offset: 0, length: 3 });
    expect(find(entries, 'ascii')).toBe('.A.');
  });
});
