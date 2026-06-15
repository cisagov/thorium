// project imports
import { CreateEntity, Entities, Entity } from './entities';

/// A single disassembled instruction within a compiled function. Mirrors the Rust `CompiledInstruction`
/// (`api/src/models/entities/functions.rs`).
export type CompiledInstruction = {
  /// The virtual address of this instruction
  address: number;
  /// The disassembled instruction text (e.g. `push rbp`)
  instruction: string;
};

// ----- Compiled function (disassembly) -----

/// A compiled function identified during analysis, carrying its disassembly. Mirrors the Rust
/// `CompiledFunction`.
export type CompiledFunctionMetaFields = {
  /// The virtual address of this function
  address: number;
  /// The ordered list of disassembled instructions for this function
  disassembly: CompiledInstruction[];
};

export type CompiledFunctionCreateMetaFields = CompiledFunctionMetaFields;

export type CompiledFunctionMeta = {
  CompiledFunction: CompiledFunctionMetaFields;
};

export type CompiledFunctionCreateMeta = {
  CompiledFunction: CompiledFunctionCreateMetaFields;
};

export type CompiledFunction = Entity<Entities.CompiledFunction>;

export type CreateCompiledFunction = CreateEntity<Entities.CompiledFunction>;

export const BlankCompiledFunction: CompiledFunction = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.CompiledFunction,
  metadata: {
    CompiledFunction: {
      address: 0,
      disassembly: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
  image: null,
};

export const BlankCreateCompiledFunction: CreateCompiledFunction = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.CompiledFunction,
  metadata: {
    CompiledFunction: {
      address: 0,
      disassembly: [],
    },
  },
};

// ----- Decompiled function (decomp) -----

/// A decompiled function, carrying the decompiled source text and the tools that produced it. Mirrors
/// the Rust `DecompiledFunction`.
export type DecompiledFunctionMetaFields = {
  /// The virtual address of this function
  address: number;
  /// The tools that produced this decompilation
  tools: string[];
  /// The decompiled source content
  content: string;
};

export type DecompiledFunctionCreateMetaFields = DecompiledFunctionMetaFields;

export type DecompiledFunctionMeta = {
  DecompiledFunction: DecompiledFunctionMetaFields;
};

export type DecompiledFunctionCreateMeta = {
  DecompiledFunction: DecompiledFunctionCreateMetaFields;
};

export type DecompiledFunction = Entity<Entities.DecompiledFunction>;

export type CreateDecompiledFunction = CreateEntity<Entities.DecompiledFunction>;

export const BlankDecompiledFunction: DecompiledFunction = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.DecompiledFunction,
  metadata: {
    DecompiledFunction: {
      address: 0,
      tools: [],
      content: '',
    },
  },
  tags: {},
  submitter: '',
  created: '',
  image: null,
};

export const BlankCreateDecompiledFunction: CreateDecompiledFunction = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.DecompiledFunction,
  metadata: {
    DecompiledFunction: {
      address: 0,
      tools: [],
      content: '',
    },
  },
};
