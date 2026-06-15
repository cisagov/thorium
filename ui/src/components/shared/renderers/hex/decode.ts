// project imports
import { HexSelection } from './types';
// spec: ../SPEC.md

/** A single decoded interpretation of the selected bytes. */
export interface HexValueEntry {
  label: string;
  value: string;
}

/** ASCII rendering of bytes, with non-printable bytes shown as `.`. */
function asciiOf(bytes: Uint8Array): string {
  let out = '';
  for (const b of bytes) {
    out += b >= 0x20 && b <= 0x7e ? String.fromCharCode(b) : '.';
  }
  return out;
}

/** Hex rendering of bytes (space-separated, lower-case). */
function hexOf(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join(' ');
}

/**
 * Decode a byte selection into a list of typed interpretations (integers of various widths in
 * both endiannesses, floats, ascii, hex). Width-dependent numeric rows are only included when
 * the selection is long enough.
 *
 * @param bytes - The full byte buffer the selection refers to.
 * @param selection - The selected `{ offset, length }`.
 * @returns The decoded entries, or an empty array when the selection is empty/out of range.
 */
export function decodeHexValues(bytes: Uint8Array, selection: HexSelection | null): HexValueEntry[] {
  if (!selection || selection.length <= 0 || selection.offset < 0 || selection.offset >= bytes.length) {
    return [];
  }
  const end = Math.min(selection.offset + selection.length, bytes.length);
  const slice = bytes.subarray(selection.offset, end);
  const len = slice.length;
  const view = new DataView(slice.buffer, slice.byteOffset, len);

  const entries: HexValueEntry[] = [
    { label: 'Offset', value: `0x${selection.offset.toString(16)} (${selection.offset})` },
    { label: 'Length', value: `${len} byte${len === 1 ? '' : 's'}` },
  ];

  if (len >= 1) {
    entries.push({ label: 'int8', value: String(view.getInt8(0)) });
    entries.push({ label: 'uint8', value: String(view.getUint8(0)) });
  }
  if (len >= 2) {
    entries.push({ label: 'int16 (LE/BE)', value: `${view.getInt16(0, true)} / ${view.getInt16(0, false)}` });
    entries.push({ label: 'uint16 (LE/BE)', value: `${view.getUint16(0, true)} / ${view.getUint16(0, false)}` });
  }
  if (len >= 4) {
    entries.push({ label: 'int32 (LE/BE)', value: `${view.getInt32(0, true)} / ${view.getInt32(0, false)}` });
    entries.push({ label: 'uint32 (LE/BE)', value: `${view.getUint32(0, true)} / ${view.getUint32(0, false)}` });
    entries.push({ label: 'float32 (LE/BE)', value: `${view.getFloat32(0, true)} / ${view.getFloat32(0, false)}` });
  }
  if (len >= 8) {
    entries.push({ label: 'int64 (LE/BE)', value: `${view.getBigInt64(0, true)} / ${view.getBigInt64(0, false)}` });
    entries.push({ label: 'uint64 (LE/BE)', value: `${view.getBigUint64(0, true)} / ${view.getBigUint64(0, false)}` });
    entries.push({ label: 'float64 (LE/BE)', value: `${view.getFloat64(0, true)} / ${view.getFloat64(0, false)}` });
  }

  entries.push({ label: 'ascii', value: asciiOf(slice) });
  entries.push({ label: 'hex', value: hexOf(slice) });
  return entries;
}
