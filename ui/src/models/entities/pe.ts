// project imports
import { CreateEntity, Entities, Entity } from './entities';

// ----- PE section -----

/// A single section within a PE/binary (e.g. `.text`, `.rsrc`, `UPX1`). The section name is carried by
/// the entity's `name`, so metadata only holds per-section details. Mirrors the Rust `PeSectionEntity`
/// (`api/src/models/entities/pe.rs`). Sizes are `u64` on the backend but stay within safe-integer range
/// for real sections, so they're modeled as `number`.
export type PeSectionMetaFields = {
  /// The MD5 of this section's raw data
  md5?: string;
  /// The raw (on disk) size of this section in bytes
  raw_size?: number;
  /// The virtual (in memory) size of this section in bytes
  virtual_size?: number;
  /// The Shannon entropy of this section's data
  entropy?: number;
};

export type PeSectionCreateMetaFields = PeSectionMetaFields;

export type PeSectionMeta = {
  PeSection: PeSectionMetaFields;
};

export type PeSectionCreateMeta = {
  PeSection: PeSectionCreateMetaFields;
};

export type PeSection = Entity<Entities.PeSection>;

export type CreatePeSection = CreateEntity<Entities.PeSection>;

export const BlankPeSection: PeSection = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.PeSection,
  metadata: {
    PeSection: {},
  },
  tags: {},
  submitter: '',
  created: '',
  image: null,
};

export const BlankCreatePeSection: CreatePeSection = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.PeSection,
  metadata: {
    PeSection: {},
  },
};

// ----- PE import -----

/// An imported library and the functions imported from it. The DLL/library name is carried by the
/// entity's `name`, so metadata only holds the imported functions. Mirrors the Rust `PeImportEntity`.
export type PeImportMetaFields = {
  /// The functions imported from this library
  functions: string[];
};

export type PeImportCreateMetaFields = PeImportMetaFields;

export type PeImportMeta = {
  PeImport: PeImportMetaFields;
};

export type PeImportCreateMeta = {
  PeImport: PeImportCreateMetaFields;
};

export type PeImport = Entity<Entities.PeImport>;

export type CreatePeImport = CreateEntity<Entities.PeImport>;

export const BlankPeImport: PeImport = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.PeImport,
  metadata: {
    PeImport: {
      functions: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
  image: null,
};

export const BlankCreatePeImport: CreatePeImport = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.PeImport,
  metadata: {
    PeImport: {
      functions: [],
    },
  },
};
