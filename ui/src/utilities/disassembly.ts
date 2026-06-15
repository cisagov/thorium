// project imports
import { CompiledInstruction } from '@models/entities/functions';

/**
 * Format a virtual address as a `0x`-prefixed lowercase hex string (e.g. `0x401000`).
 *
 * @param address - The numeric address.
 * @returns The `0x`-prefixed hex representation.
 */
export function formatAddress(address: number): string {
  return `0x${address.toString(16)}`;
}

/**
 * Render a compiled function's disassembly as editable text, one instruction per line in the form
 * `0x{address}\t{instruction}`. This is the human-readable editor representation — distinct from the
 * JSON wire format used when submitting the create/update form.
 *
 * @param instructions - The ordered disassembly instructions.
 * @returns The multi-line text representation.
 */
export function disassemblyToText(instructions: CompiledInstruction[]): string {
  return instructions.map((ins) => `${formatAddress(ins.address)}\t${ins.instruction}`).join('\n');
}

/** The result of parsing disassembly text back into structured instructions. */
export interface ParsedDisassembly {
  /** The parsed instructions (empty when `error` is set). */
  instructions: CompiledInstruction[];
  /** A human-readable error describing the first malformed line, or `undefined` when valid. */
  error?: string;
}

// Each line is an address token (hex `0x...` or decimal) followed by whitespace then the instruction.
const LINE_RE = /^(0x[0-9a-fA-F]+|\d+)\s+(.+)$/;

/**
 * Parse disassembly editor text (see {@link disassemblyToText}) back into structured instructions.
 * Blank lines are ignored; the first malformed line aborts parsing with an `error` so the caller can
 * reject the save rather than silently dropping instructions.
 *
 * @param text - The multi-line disassembly text.
 * @returns The parsed instructions, or an `error` describing the first invalid line.
 */
export function textToDisassembly(text: string): ParsedDisassembly {
  const instructions: CompiledInstruction[] = [];
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === '') continue;
    const match = LINE_RE.exec(line);
    if (!match) {
      return {
        instructions: [],
        error: `Line ${i + 1}: expected "0x{address}\\t{instruction}", got "${lines[i]}"`,
      };
    }
    const address = match[1].startsWith('0x') ? parseInt(match[1], 16) : parseInt(match[1], 10);
    instructions.push({ address, instruction: match[2].trim() });
  }
  return { instructions };
}
