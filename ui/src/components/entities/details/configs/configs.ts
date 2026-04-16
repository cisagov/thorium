import { JSX } from 'react';

// project imports
import { createEntityDetailsPage, MetadataComponent } from '../EntityDetails';
import CollectionDetailsConfig from './CollectionDetailsConfig';
import DeviceDetailsConfig from './DeviceDetailsConfig';
import FolderDetailsConfig from './FolderDetailsConfig';
import FileSystemDetailsConfig from './FileSystemDetailsConfig';
import VendorDetailsConfig from './VendorDetailsConfig';
import WindowsProcessTreeDetailsConfig from './WindowsProcessTreeDetailsConfig';
import WindowsProcessDetailsConfig from './WindowsProcessDetailsConfig';
import NetworkConnectionsDetailsConfig from './NetworkConnectionDetailsConfig';
import OtherDetailsConfig from './OtherDetailsConfig';
import { Entities, EntityTypeMap } from '@models/entities';
import SigmaRuleDetailsConfig from './SigmaRuleDetailsConfig';

export type EntityDetailsConfig<T extends keyof EntityTypeMap> = {
  getEntityDetails: (entityID: string, setError: (err: string) => void, updateEntity: (entity: EntityTypeMap[T]) => void) => void;
  EntityMetaInfo: MetadataComponent<T>;
  BlankEntity: EntityTypeMap[T];
  icon: (size: number) => JSX.Element;
};

type EntityConfigMap = {
  [K in keyof EntityTypeMap]: EntityDetailsConfig<K>;
};

export const EntitiesDetailsConfig = {
  [Entities.Collection]: CollectionDetailsConfig,
  [Entities.Device]: DeviceDetailsConfig,
  [Entities.FileSystem]: FileSystemDetailsConfig,
  [Entities.Folder]: FolderDetailsConfig,
  [Entities.NetworkConnection]: NetworkConnectionsDetailsConfig,
  [Entities.Other]: OtherDetailsConfig,
  [Entities.SigmaRule]: SigmaRuleDetailsConfig,
  [Entities.Vendor]: VendorDetailsConfig,
  [Entities.WindowsProcessTree]: WindowsProcessTreeDetailsConfig,
  [Entities.WindowsProcess]: WindowsProcessDetailsConfig,
} satisfies EntityConfigMap;

export const EntityDetailsPages = {
  [Entities.Collection]: createEntityDetailsPage(CollectionDetailsConfig),
  [Entities.Device]: createEntityDetailsPage(DeviceDetailsConfig),
  [Entities.FileSystem]: createEntityDetailsPage(FileSystemDetailsConfig),
  [Entities.Folder]: createEntityDetailsPage(FolderDetailsConfig),
  [Entities.NetworkConnection]: createEntityDetailsPage(NetworkConnectionsDetailsConfig),
  [Entities.Other]: createEntityDetailsPage(OtherDetailsConfig),
  [Entities.SigmaRule]: createEntityDetailsPage(SigmaRuleDetailsConfig),
  [Entities.Vendor]: createEntityDetailsPage(VendorDetailsConfig),
  [Entities.WindowsProcessTree]: createEntityDetailsPage(WindowsProcessTreeDetailsConfig),
  [Entities.WindowsProcess]: createEntityDetailsPage(WindowsProcessDetailsConfig),
} satisfies { [K in keyof EntityTypeMap]: React.ComponentType };
