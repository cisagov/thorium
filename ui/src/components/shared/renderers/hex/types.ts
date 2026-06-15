// spec: ../SPEC.md
/** A contiguous byte selection within a hex view (`length` of 0 means no selection). */
export interface HexSelection {
  offset: number;
  length: number;
}

/** Per-byte diff status used to color bytes in the hex diff view. */
export enum HexByteStatus {
  /** Byte is identical on both sides. */
  Same = 'same',
  /** Byte exists only on this (compare) side — rendered green. */
  Added = 'added',
  /** Byte exists only on the other (base) side — rendered red. */
  Removed = 'removed',
}
