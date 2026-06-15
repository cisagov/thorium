// project imports
import { CreateDevice, Device, DeviceCreateMeta, DeviceMeta } from './devices';
import { RequestTags, Tags } from '../tags';
import { CreateVendor, Vendor, VendorCreateMeta, VendorMeta } from './vendors';
import { Collection, CollectionCreateMeta, CollectionMeta, CreateCollection } from './collections';
import { Sample } from './files';
import { Repo } from './repos';
import { CreateFileSystem, FileSystem, FileSystemCreateMeta, FileSystemMeta } from './file_systems';
import { CreateFolder, Folder, FolderCreateMeta, FolderMeta } from './folders';
import { CreateWindowsProcess, WindowsProcess, WindowsProcessCreateMeta, WindowsProcessMeta } from './processes';
import { CreateWindowsProcessTree, WindowsProcessTree, WindowsProcessTreeCreateMeta, WindowsProcessTreeMeta } from './process_trees';
import {
  TransportLayerProtocol,
  CreateNetworkConnection,
  NetworkConnection,
  NetworkConnectionCreateMeta,
  NetworkConnectionMeta,
  NetConState,
} from './network_connections';
import { CreateOther, Other, OtherCreateMeta, OtherMeta } from './other';
import { CreateFlag, Flag, FlagCreateMeta, FlagMeta } from './flag';
import { CreateIncident, Incident, IncidentCreateMeta, IncidentMeta } from './incident';
import {
  CompiledFunction,
  CompiledFunctionCreateMeta,
  CompiledFunctionMeta,
  CompiledInstruction,
  CreateCompiledFunction,
  CreateDecompiledFunction,
  DecompiledFunction,
  DecompiledFunctionCreateMeta,
  DecompiledFunctionMeta,
} from './functions';
import {
  CreatePeImport,
  CreatePeSection,
  PeImport,
  PeImportCreateMeta,
  PeImportMeta,
  PeSection,
  PeSectionCreateMeta,
  PeSectionMeta,
} from './pe';
import { CreateSigmaRule, SigmaActionToTake, SigmaRule, SigmaRuleAppliesTo, SigmaRuleCreateMeta, SigmaRuleMeta } from './rules/sigma';
import { labelWithFallback } from '../labels';

// Entity types
export enum Entities {
  /// A pseudo entity that doesn't follow the standard entity structure
  File = 'File',
  /// A pseudo entity that doesn't follow the standard entity structure
  Repo = 'Repo',
  /// A device entity
  Device = 'Device',
  /// A vendor entity
  Vendor = 'Vendor',
  /// A collection entity
  ///
  /// Collections are dynamic lists of items in Thorium (e.g. samples, repos, etc.)
  /// based on search parameters like tags
  Collection = 'Collection',
  /// A filesystem entity
  FileSystem = 'FileSystem',
  /// A folder within a filesystem entity
  Folder = 'Folder',
  /// A Windows process tree entity
  WindowsProcessTree = 'WindowsProcessTree',
  /// A Windows process
  WindowsProcess = 'WindowsProcess',
  /// A Network connection
  NetworkConnection = 'NetworkConnection',
  /// A sigma rule to apply to data
  SigmaRule = 'SigmaRule',
  /// A flag denoting something interesting, odd, or suspicious
  Flag = 'Flag',
  /// An incident grouping related activity (teams, networks, machines, locations)
  Incident = 'Incident',
  /// A compiled function and its disassembly
  CompiledFunction = 'CompiledFunction',
  /// A decompiled function and its decompiled source
  DecompiledFunction = 'DecompiledFunction',
  /// A section within a PE/binary (e.g. `.text`)
  PeSection = 'PeSection',
  /// A library imported by a PE/binary and the functions imported from it
  PeImport = 'PeImport',
  /// An entity that can't be described by any of the other variants
  Other = 'Other',
}

/**
 * Human-readable display labels for each entity type, colocated with the {@link Entities} enum so the
 * two can't drift. Typed as an exhaustive `Record<Entities, string>`: adding a new `Entities` member
 * without a label here is a compile-time error. Use for DISPLAY only — the raw enum value is still the
 * API/route/key value.
 */
export const ENTITY_LABELS: Record<Entities, string> = {
  [Entities.File]: 'File',
  [Entities.Repo]: 'Repo',
  [Entities.Device]: 'Device',
  [Entities.Vendor]: 'Vendor',
  [Entities.Collection]: 'Collection',
  [Entities.FileSystem]: 'File System',
  [Entities.Folder]: 'Folder',
  [Entities.WindowsProcessTree]: 'Windows Process Tree',
  [Entities.WindowsProcess]: 'Windows Process',
  [Entities.NetworkConnection]: 'Network Connection',
  [Entities.SigmaRule]: 'Sigma Rule',
  [Entities.Flag]: 'Flag',
  [Entities.Incident]: 'Incident',
  [Entities.CompiledFunction]: 'Compiled Function',
  [Entities.DecompiledFunction]: 'Decompiled Function',
  [Entities.PeSection]: 'PE Section',
  [Entities.PeImport]: 'PE Import',
  [Entities.Other]: 'Other',
};

/**
 * Get the human-readable label for an entity kind.
 *
 * @param kind - An {@link Entities} value, or a raw kind string (some call sites carry `string`).
 * @returns The mapped label, falling back to {@link humanize} for an unknown/unmapped kind.
 */
export function entityLabel(kind: Entities | string): string {
  return labelWithFallback(ENTITY_LABELS, kind);
}

// pure entities
export type EntityTypes =
  | CompiledFunction
  | DecompiledFunction
  | Device
  | Collection
  | FileSystem
  | Flag
  | Folder
  | Incident
  | NetworkConnection
  | Other
  | PeImport
  | PeSection
  | SigmaRule
  | Vendor
  | WindowsProcess
  | WindowsProcessTree;

export type EntityMetaTypes =
  | CompiledFunctionMeta
  | DecompiledFunctionMeta
  | DeviceMeta
  | CollectionMeta
  | FileSystemMeta
  | FlagMeta
  | FolderMeta
  | IncidentMeta
  | NetworkConnectionMeta
  | OtherMeta
  | PeImportMeta
  | PeSectionMeta
  | SigmaRuleMeta
  | VendorMeta
  | WindowsProcessMeta
  | WindowsProcessTreeMeta;

// pure create entities
export type EntityCreateTypes =
  | CreateCollection
  | CreateCompiledFunction
  | CreateDecompiledFunction
  | CreateDevice
  | CreateFileSystem
  | CreateFlag
  | CreateFolder
  | CreateIncident
  | CreateNetworkConnection
  | CreateOther
  | CreatePeImport
  | CreatePeSection
  | CreateSigmaRule
  | CreateVendor
  | CreateWindowsProcess
  | CreateWindowsProcessTree;

export type UISupportedEntityCreateTypes =
  | CreateDevice
  | CreateCollection
  | CreateVendor
  | CreateSigmaRule
  | CreateFlag
  | CreateIncident
  | CreateCompiledFunction
  | CreateDecompiledFunction
  | CreatePeSection
  | CreatePeImport;
export type UISupportedEntityCreateKind =
  | Entities.Collection
  | Entities.Device
  | Entities.Vendor
  | Entities.SigmaRule
  | Entities.Flag
  | Entities.Incident
  | Entities.CompiledFunction
  | Entities.DecompiledFunction
  | Entities.PeSection
  | Entities.PeImport;

export type EntityCreateMetaTypes =
  | CompiledFunctionCreateMeta
  | DecompiledFunctionCreateMeta
  | DeviceCreateMeta
  | CollectionCreateMeta
  | FileSystemCreateMeta
  | FlagCreateMeta
  | FolderCreateMeta
  | IncidentCreateMeta
  | NetworkConnectionCreateMeta
  | OtherCreateMeta
  | PeImportCreateMeta
  | PeSectionCreateMeta
  | SigmaRuleCreateMeta
  | VendorCreateMeta
  | WindowsProcessCreateMeta
  | WindowsProcessTreeCreateMeta;

// all possible entity variants including legacy types (file/repo)
export type ExtendedEntityTypes = EntityTypes | Sample | Repo;

export type ByEntityKind<T extends { kind: PropertyKey }> = {
  [E in T as E['kind']]: E;
};

export type SingleKeyObjectToMap<T extends object> = {
  [U in T as keyof U & PropertyKey]: U;
};

export type EntityTypeMap = ByEntityKind<EntityTypes>;
export type EntityCreateTypeMap = ByEntityKind<EntityCreateTypes>;
export type EntityUISupportedCreateTypeMap = ByEntityKind<UISupportedEntityCreateTypes>;

export type EntityMetaMap = SingleKeyObjectToMap<EntityMetaTypes>;
export type EntityCreateMetaMap = SingleKeyObjectToMap<EntityCreateMetaTypes>;

// map with pseudo entity types included (these have a different structure)
export type ExtendedTypeMap = EntityTypeMap & {
  [Entities.File]: Sample;
  [Entities.Repo]: Repo;
};

export type Entity<k extends keyof EntityMetaMap> = {
  id: string; // UUID of entity
  name: string; // name of entity
  kind: k; // type of entity
  metadata: EntityMetaMap[k]; // entity metadata
  description: string | null; // text description of entity
  submitter: string; // Thorium user who created entity
  groups: string[]; // Groups that have permissions to view this entity
  created: string; // Entity creation date
  tags: Tags; // Key/value tags that have been applied to this entity
  image: string | null; // S3 path to entity graphic
};

// Entity Creation type, users don't submit id, submitter or created date.
//    For tags, the format for entity creation does not include a groups
//    permissions array (vector in Rust) which is included when getting an
//    existing entity
export type CreateEntity<K extends keyof EntityMetaMap> = Omit<
  Entity<K>,
  'id' | 'submitter' | 'created' | 'tags' | 'metadata' | 'image'
> & {
  tags: RequestTags;
  metadata: EntityCreateMetaMap[K];
};

// format for updating entity metadata
export type UpdateEntityMetadata = {
  add_urls?: string[];
  remove_urls?: string[];
  add_vendors?: string[];
  remove_vendors?: string[];
  critical_system?: boolean;
  clear_critical_system?: boolean;
  sensitive_location?: boolean;
  clear_sensitive_location?: boolean;
  add_critical_sectors?: string[];
  remove_critical_sectors?: string[];
  add_countries?: string[];
  remove_countries?: string[];
  add_collection_tags?: RequestTags;
  delete_collection_tags?: RequestTags;
  collection_tags_case_insensitive?: boolean;
  collection_ignore_groups?: boolean;
  collection_start?: string;
  collection_end?: string;
  clear_collection_start?: boolean;
  clear_collection_end?: boolean;
  add_tools?: string[];
  remove_tools?: string[];
  name?: string;
  image_path?: string;
  command?: string;
  offset?: bigint;
  threads?: number;
  handles?: number;
  is_wow64?: boolean;
  session_id?: number;
  create_time?: string; // UTC date
  exit_time?: string; // UTC date
  protocol?: TransportLayerProtocol;
  source?: string;
  source_port?: number;
  destination?: string;
  destination_port?: number;
  state?: NetConState;
  pid?: bigint;
  process?: string;
  /// A sigma rule in yaml format
  sigma_rule?: string;
  /// The new things this sigma rule should apply too
  add_sigma_applies_to?: SigmaRuleAppliesTo[];
  /// The things things sigma rule should no longer apply too
  remove_sigma_applies_to?: SigmaRuleAppliesTo[];
  /// The new actions to take when a sigma rule hits
  add_sigma_actions?: SigmaActionToTake[];
  /// The actions to remove by their index in this vec
  remove_sigma_actions?: number[];
  /// The score that a rule applies
  score?: number;
  // ----- Incident -----
  /// The cover term / codename for an incident
  cover_term?: string;
  add_mission_teams?: string[];
  remove_mission_teams?: string[];
  add_networks?: string[];
  remove_networks?: string[];
  add_machines?: string[];
  remove_machines?: string[];
  add_locations?: string[];
  remove_locations?: string[];
  // ----- Compiled/Decompiled functions -----
  /// The virtual address of a function (compiled or decompiled)
  function_address?: number;
  /// The full replacement disassembly for a compiled function
  disassembly?: CompiledInstruction[];
  /// The replacement decompiled content for a decompiled function
  decompilation_content?: string;
  // ----- PE section -----
  /// The MD5 of a PE section's raw data
  md5?: string;
  /// The raw (on disk) size of a PE section in bytes
  raw_size?: number;
  /// The virtual (in memory) size of a PE section in bytes
  virtual_size?: number;
  /// The Shannon entropy of a PE section's data
  entropy?: number;
  // ----- PE import -----
  /// The full replacement list of functions imported by a PE import
  functions?: string[];
};

// entity update format
export type UpdateEntityForm = {
  name?: string;
  add_groups?: string[];
  remove_groups?: string[];
  description?: string;
  clear_description?: boolean;
  metadata?: UpdateEntityMetadata;
};

// blank entity update object is just empty
export const BlankUpdateEntity = {};
