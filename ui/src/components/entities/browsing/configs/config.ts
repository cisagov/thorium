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
import { Entities, ExtendedTypeMap } from '@models/entities/entities';
import OthersBrowsingConfig from './OtherBrowsingConfig';
import FlagBrowsingConfig from './FlagBrowsingConfig';
import IncidentBrowsingConfig from './IncidentBrowsingConfig';
import CompiledFunctionBrowsingConfig from './CompiledFunctionBrowsingConfig';
import DecompiledFunctionBrowsingConfig from './DecompiledFunctionBrowsingConfig';
import PeSectionBrowsingConfig from './PeSectionBrowsingConfig';
import PeImportBrowsingConfig from './PeImportBrowsingConfig';
import { createEntityBrowsingPage } from '../EntityBrowsing';
import SigmaRulesBrowsingConfig from './SigmaRuleBrowsingConfig';
import { EntityBrowseConfig } from './entityFetcher';

// spec: ../EntityBrowsing.spec.md

// `EntityBrowseConfig` and `makeEntityFetcher` live in `./entityFetcher` (which imports no config files) to
// avoid a circular import: `config.ts` imports every per-kind config, and those configs need the fetcher — so
// sourcing it here would leave them accessing a value from a still-initializing module. Re-exported so existing
// `import { EntityBrowseConfig } from './config'` sites keep working.
export type { EntityBrowseConfig };

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
  [Entities.Flag]: FlagBrowsingConfig,
  [Entities.Incident]: IncidentBrowsingConfig,
  [Entities.CompiledFunction]: CompiledFunctionBrowsingConfig,
  [Entities.DecompiledFunction]: DecompiledFunctionBrowsingConfig,
  [Entities.PeSection]: PeSectionBrowsingConfig,
  [Entities.PeImport]: PeImportBrowsingConfig,
};

export const EntityBrowsingPages = {
  [Entities.Collection]: createEntityBrowsingPage(CollectionBrowsingConfig),
  [Entities.Device]: createEntityBrowsingPage(DeviceBrowsingConfig),
  [Entities.FileSystem]: createEntityBrowsingPage(FileSystemBrowsingConfig),
  [Entities.File]: createEntityBrowsingPage(FileBrowsingConfig),
  [Entities.Folder]: createEntityBrowsingPage(FolderBrowsingConfig),
  [Entities.NetworkConnection]: createEntityBrowsingPage(NetworkConnectionsBrowsingConfig),
  [Entities.Other]: createEntityBrowsingPage(OthersBrowsingConfig),
  [Entities.Flag]: createEntityBrowsingPage(FlagBrowsingConfig),
  [Entities.SigmaRule]: createEntityBrowsingPage(SigmaRulesBrowsingConfig),
  [Entities.Vendor]: createEntityBrowsingPage(VendorBrowsingConfig),
  [Entities.Repo]: createEntityBrowsingPage(RepoBrowsingConfig),
  [Entities.WindowsProcessTree]: createEntityBrowsingPage(WindowsProcessTreeBrowsingConfig),
  [Entities.WindowsProcess]: createEntityBrowsingPage(WindowsProcessBrowsingConfig),
  [Entities.Incident]: createEntityBrowsingPage(IncidentBrowsingConfig),
  [Entities.CompiledFunction]: createEntityBrowsingPage(CompiledFunctionBrowsingConfig),
  [Entities.DecompiledFunction]: createEntityBrowsingPage(DecompiledFunctionBrowsingConfig),
  [Entities.PeSection]: createEntityBrowsingPage(PeSectionBrowsingConfig),
  [Entities.PeImport]: createEntityBrowsingPage(PeImportBrowsingConfig),
} satisfies { [K in keyof ExtendedTypeMap]: React.ComponentType };
