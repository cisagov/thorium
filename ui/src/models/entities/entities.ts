import { Device, DeviceMeta } from './devices';
import { RequestTags, Tags } from '../tags';
import { Vendor, VendorMeta } from './vendors';
import { Collection, CollectionMeta } from './collections';

// an enum of entity type keys
export enum Entities {
  File = 'File',
  Repos = 'Repo',
  Vendor = 'Vendor',
  Device = 'Device',
  Collection = 'Collection',
  Other = 'Other',
}

// entity metadata in the form of { Type: {metadata..}}
export type EntityMeta = DeviceMeta | VendorMeta | CollectionMeta;

export type Entity = {
  id: string; // UUID of entity
  name: string; // name of entity
  kind: Entities; // type of entity
  metadata: any; // entity metadata
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
export type CreateEntity = Omit<Entity, 'id' | 'submitter' | 'created' | 'tags'> & {
  tags: RequestTags;
};

// format for updating entity metadata
export type UpdateEntityMetadata = {
  add_urls?: string[];
  remove_urls?: string[];
  add_vendors?: string[];
  remove_vendors?: string[];
  critical_system?: boolean;
  sensitive_location?: boolean;
  add_critical_sectors?: string[];
  remove_critical_sectors?: string[];
  add_countries?: string[];
  remove_countries?: string[];
  collection_tags_case_insensitive?: boolean;
  collection_start?: string;
  collection_end?: string;
  clear_collection_start?: boolean;
  clear_collection_end?: boolean;
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

// all possible entity variants
export type EntityTypes = Device | Vendor | Collection;

// all possible entity metadata variants
export type EntityMetaTypes = DeviceMeta | VendorMeta | CollectionMeta;
