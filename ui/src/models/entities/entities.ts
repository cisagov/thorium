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
import { CreateSigmaRule, SigmaActionToTake, SigmaRule, SigmaRuleAppliesTo, SigmaRuleCreateMeta, SigmaRuleMeta } from './rules/sigma';

// Entity types
export enum Entities {
  // psuedo entity, doesn't follow entity structure
  File = 'File',
  // psuedo entity, doesn't follow entity structure
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
  /// An entity that can't be described by any of the other variants
  Other = 'Other',
}

// pure entities
export type EntityTypes =
  | Device
  | Collection
  | FileSystem
  | Folder
  | NetworkConnection
  | Other
  | SigmaRule
  | Vendor
  | WindowsProcess
  | WindowsProcessTree;

export type EntityMetaTypes =
  | DeviceMeta
  | CollectionMeta
  | FileSystemMeta
  | FolderMeta
  | NetworkConnectionMeta
  | OtherMeta
  | SigmaRuleMeta
  | VendorMeta
  | WindowsProcessMeta
  | WindowsProcessTreeMeta;

// pure create entities
export type EntityCreateTypes =
  | CreateCollection
  | CreateDevice
  | CreateFileSystem
  | CreateFolder
  | CreateNetworkConnection
  | CreateOther
  | CreateSigmaRule
  | CreateVendor
  | CreateWindowsProcess
  | CreateWindowsProcessTree;

export type UISupportedEntityCreateTypes = CreateDevice | CreateCollection | CreateVendor | CreateSigmaRule;
export type UISupportedEntityCreateKind = Entities.Collection | Entities.Device | Entities.Vendor | Entities.SigmaRule;

export type EntityCreateMetaTypes =
  | DeviceCreateMeta
  | CollectionCreateMeta
  | FileSystemCreateMeta
  | FolderCreateMeta
  | NetworkConnectionCreateMeta
  | OtherCreateMeta
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
};

// Entity Creation type, users don't submit id, submitter or created date.
//    For tags, the format for entity creation does not include a groups
//    permissions array (vector in Rust) which is included when getting an
//    existing entity
export type CreateEntity<K extends keyof EntityMetaMap & keyof EntityCreateMetaMap> = Omit<
  Entity<K>,
  'id' | 'submitter' | 'created' | 'tags' | 'metadata'
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
  name?: String;
  image_path?: String;
  command?: String;
  offset?: BigInt;
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
  pid?: BigInt;
  process?: String;
  /// A sigma rule in yaml format
  sigma_rule?: String;
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
