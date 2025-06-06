import { CreateDevice, Device, DeviceMeta } from './devices';
import { CreateTags, Tags } from '../tags';
import { CreateVendor, Vendor, VendorMeta } from './vendors';

// an enum of entity type keys
export enum Entities {
  File = 'File',
  Repos = 'Repo',
  Vendor = 'Vendor',
  Device = 'Device',
  Other = 'Other',
}

// entity metadata in the form of { Type: {metadata..}}
export type EntityMeta = DeviceMeta | VendorMeta;

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
  tags: CreateTags;
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
export type EntityTypes = Device | Vendor;

// all possible entity metadata variants
export type EntityMetaTypes = DeviceMeta | VendorMeta;
