import { diffArrays } from 'diff';

// project imports
import { HexByteStatus } from '@components/shared/renderers';

/**
 * Upper bound (per side, in bytes) on inputs the byte-level LCS diff will process. The Myers LCS is
 * O((N+M)·D) time with two boxed-number arrays plus per-index maps, so two large, substantially
 * different binaries are effectively quadratic and can freeze the main thread. Above this cap the diff
 * is skipped and both sides render as plain, uncolored hex (see {@link ByteDiffResult.truncated}).
 */
export const MAX_DIFF_BYTES = 1024 * 1024;

/**
 * The result of aligning two byte buffers: per-byte diff status for each side, plus index maps
 * between aligned (unchanged) bytes used to mirror a selection across the two hex panes.
 */
export interface ByteDiffResult {
  /** Per-index status for the base buffer (`Same` or `Removed`). */
  baseStatus: HexByteStatus[];
  /** Per-index status for the compare buffer (`Same` or `Added`). */
  compareStatus: HexByteStatus[];
  /** base index -> aligned compare index (only for unchanged bytes). */
  baseToCompare: Map<number, number>;
  /** compare index -> aligned base index (only for unchanged bytes). */
  compareToBase: Map<number, number>;
  /**
   * True when either side exceeded {@link MAX_DIFF_BYTES}, so the LCS was skipped: every byte is
   * marked `Same` (no diff coloring) and the alignment maps are empty. Consumers should surface a
   * "diff truncated" notice in this case.
   */
  truncated: boolean;
}

/**
 * Compute a byte-level diff between two buffers using an LCS (jsdiff `diffArrays`).
 *
 * Unchanged bytes are marked `Same` on both sides and cross-linked in the index maps; bytes only
 * in the base are `Removed`, bytes only in the compare are `Added`.
 *
 * When either side exceeds {@link MAX_DIFF_BYTES} the LCS is skipped as a safeguard against
 * pathologically large binaries freezing the tab: both sides are returned entirely `Same` (no diff
 * coloring) with empty alignment maps and `truncated: true`.
 *
 * @param base - The base (left) bytes.
 * @param compare - The compare (right) bytes.
 * @returns Per-side status arrays, the alignment maps, and whether the diff was truncated.
 */
export function computeByteDiff(base: Uint8Array, compare: Uint8Array): ByteDiffResult {
  const baseStatus: HexByteStatus[] = new Array<HexByteStatus>(base.length);
  const compareStatus: HexByteStatus[] = new Array<HexByteStatus>(compare.length);
  const baseToCompare = new Map<number, number>();
  const compareToBase = new Map<number, number>();
  // defensive cap: skip the (potentially quadratic) LCS and render plain hex for oversized inputs
  if (base.length > MAX_DIFF_BYTES || compare.length > MAX_DIFF_BYTES) {
    baseStatus.fill(HexByteStatus.Same);
    compareStatus.fill(HexByteStatus.Same);
    return { baseStatus, compareStatus, baseToCompare, compareToBase, truncated: true };
  }

  const changes = diffArrays(Array.from(base), Array.from(compare));
  let baseIdx = 0;
  let compareIdx = 0;

  for (const change of changes) {
    const count = change.count ?? change.value.length;
    if (change.added) {
      for (let i = 0; i < count; i++) compareStatus[compareIdx++] = HexByteStatus.Added;
    } else if (change.removed) {
      for (let i = 0; i < count; i++) baseStatus[baseIdx++] = HexByteStatus.Removed;
    } else {
      for (let i = 0; i < count; i++) {
        baseStatus[baseIdx] = HexByteStatus.Same;
        compareStatus[compareIdx] = HexByteStatus.Same;
        baseToCompare.set(baseIdx, compareIdx);
        compareToBase.set(compareIdx, baseIdx);
        baseIdx++;
        compareIdx++;
      }
    }
  }

  return { baseStatus, compareStatus, baseToCompare, compareToBase, truncated: false };
}

/**
 * Map a contiguous selection from one side to the aligned range on the other side.
 *
 * Walks the selected indices through the alignment map and returns the bounding range of the
 * mapped indices, or `null` when none of the selected bytes are aligned (e.g. an added/removed
 * region with no counterpart).
 *
 * @param selection - The `{ offset, length }` selection on the source side.
 * @param map - The alignment map for the source side (`baseToCompare` or `compareToBase`).
 * @returns The mirrored `{ offset, length }` on the other side, or `null`.
 */
export function mapSelectionAcross(
  selection: { offset: number; length: number },
  map: Map<number, number>,
): { offset: number; length: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (let i = selection.offset; i < selection.offset + selection.length; i++) {
    const mapped = map.get(i);
    if (mapped !== undefined) {
      if (mapped < min) min = mapped;
      if (mapped > max) max = mapped;
    }
  }
  if (max < min) return null;
  return { offset: min, length: max - min + 1 };
}
