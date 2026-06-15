import { describe, it, expect } from 'vitest';

// project imports
import { disassemblyToText, formatAddress, textToDisassembly } from './disassembly';

describe('formatAddress', () => {
  it('formats as 0x-prefixed lowercase hex', () => {
    expect(formatAddress(0x401000)).toBe('0x401000');
    expect(formatAddress(0)).toBe('0x0');
    expect(formatAddress(255)).toBe('0xff');
  });
});

describe('disassemblyToText / textToDisassembly round-trip', () => {
  const instructions = [
    { address: 0x401000, instruction: 'push rbp' },
    { address: 0x401001, instruction: 'mov rbp, rsp' },
  ];

  it('serializes instructions to tab-separated lines', () => {
    expect(disassemblyToText(instructions)).toBe('0x401000\tpush rbp\n0x401001\tmov rbp, rsp');
  });

  it('round-trips back to the same instructions', () => {
    const { instructions: parsed, error } = textToDisassembly(disassemblyToText(instructions));
    expect(error).toBeUndefined();
    expect(parsed).toEqual(instructions);
  });

  it('ignores blank lines', () => {
    const { instructions: parsed, error } = textToDisassembly('\n0x401000\tpush rbp\n\n  \n0x401001\tret\n');
    expect(error).toBeUndefined();
    expect(parsed).toEqual([
      { address: 0x401000, instruction: 'push rbp' },
      { address: 0x401001, instruction: 'ret' },
    ]);
  });

  it('accepts decimal addresses', () => {
    const { instructions: parsed, error } = textToDisassembly('4198400\tpush rbp');
    expect(error).toBeUndefined();
    expect(parsed).toEqual([{ address: 4198400, instruction: 'push rbp' }]);
  });

  it('returns an error for a malformed line (no instruction)', () => {
    const { instructions: parsed, error } = textToDisassembly('0x401000\nvalid line missing addr');
    expect(parsed).toEqual([]);
    expect(error).toContain('Line 1');
  });

  it('reports the correct line number for the first malformed line', () => {
    const { error } = textToDisassembly('0x401000\tpush rbp\nnotanaddress here');
    expect(error).toContain('Line 2');
  });
});
