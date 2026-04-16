// project imports
import CollectionBrowsingConfig from './CollectionBrowsingConfig';
import DeviceBrowsingConfig from './DeviceBrowsingConfig';
import FileSystemBrowsingConfig from './FileSystemBrowsingConfig';
import VendorBrowsingConfig from './VendorBrowsingConfig';
import WindowsProcessBrowsingConfig from './WindowsProcessBrowsingConfig';
import WindowsProcessTreeBrowsingConfig from './WindowsProcessTreeBrowsingConfig';
import NetworkConnectionsBrowsingConfig from './NetworkConnectionBrowsingConfig';
import FileBrowsingConfig from './FileBrowsingConfig';
import RepoBrowsingConfig from './RepoBrowsingConfig';
import FolderBrowsingConfig from './FolderBrowsingConfig';
import { Filters } from '@models/search';
import { Entities, ExtendedTypeMap } from '@models/entities/entities';
import OthersBrowsingConfig from './OtherBrowsingConfig';
import { createEntityBrowsingPage } from '../EntityBrowsing';
import CollectionsBrowsingConfig from './CollectionBrowsingConfig';
import SigmaRulesBrowsingConfig from './SigmaRuleBrowsingConfig';

/**
 * Generic browse config for a specific entity type T
 */
export type EntityBrowseConfig<T extends keyof ExtendedTypeMap> = {
  docTitle: string;
  title: string;
  typeLabel: string;
  kind: T;
  creatable?: boolean;
  entityHeaders: React.ReactNode;
  renderEntity: (entity: ExtendedTypeMap[T], idx: number, filters?: Filters) => React.ReactNode;
  fetchEntities: (
    filters: Filters,
    cursor: string | null,
    errorHandler: (error: string) => void,
  ) => Promise<{ entitiesList: ExtendedTypeMap[T][]; entitiesCursor: string | null }>;
};

/**
 * Build a config map where each key gets the correctly typed browse config
 */
type EntityConfigMap = {
  [K in keyof ExtendedTypeMap]: EntityBrowseConfig<K>;
};

export const EntityBrowsingConfig: EntityConfigMap = {
  [Entities.Collection]: CollectionBrowsingConfig,
  [Entities.Device]: DeviceBrowsingConfig,
  [Entities.Folder]: FolderBrowsingConfig,
  [Entities.File]: FileBrowsingConfig,
  [Entities.Repo]: RepoBrowsingConfig,
  [Entities.FileSystem]: FileSystemBrowsingConfig,
  [Entities.SigmaRule]: SigmaRulesBrowsingConfig,
  [Entities.Vendor]: VendorBrowsingConfig,
  [Entities.WindowsProcessTree]: WindowsProcessTreeBrowsingConfig,
  [Entities.WindowsProcess]: WindowsProcessBrowsingConfig,
  [Entities.NetworkConnection]: NetworkConnectionsBrowsingConfig,
  [Entities.Other]: OthersBrowsingConfig,
};

export const EntityBrowsingPages = {
  [Entities.Collection]: createEntityBrowsingPage(CollectionsBrowsingConfig),
  [Entities.Device]: createEntityBrowsingPage(DeviceBrowsingConfig),
  [Entities.FileSystem]: createEntityBrowsingPage(FileSystemBrowsingConfig),
  [Entities.File]: createEntityBrowsingPage(FileBrowsingConfig),
  [Entities.Folder]: createEntityBrowsingPage(FolderBrowsingConfig),
  [Entities.NetworkConnection]: createEntityBrowsingPage(NetworkConnectionsBrowsingConfig),
  [Entities.Other]: createEntityBrowsingPage(OthersBrowsingConfig),
  [Entities.SigmaRule]: createEntityBrowsingPage(SigmaRulesBrowsingConfig),
  [Entities.Vendor]: createEntityBrowsingPage(VendorBrowsingConfig),
  [Entities.Repo]: createEntityBrowsingPage(RepoBrowsingConfig),
  [Entities.WindowsProcessTree]: createEntityBrowsingPage(WindowsProcessTreeBrowsingConfig),
  [Entities.WindowsProcess]: createEntityBrowsingPage(WindowsProcessBrowsingConfig),
} satisfies { [K in keyof ExtendedTypeMap]: React.ComponentType };
