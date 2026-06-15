// project imports
import CollectionCreateConfig from './CollectionCreateConfig';
import DeviceCreateConfig from './DeviceCreateConfig';
import VendorCreateConfig from './VendorCreateConfig';
import FlagCreateConfig from './FlagCreateConfig';
import IncidentCreateConfig from './IncidentCreateConfig';
import CompiledFunctionCreateConfig from './CompiledFunctionCreateConfig';
import DecompiledFunctionCreateConfig from './DecompiledFunctionCreateConfig';
import PeSectionCreateConfig from './PeSectionCreateConfig';
import PeImportCreateConfig from './PeImportCreateConfig';
import { Entities, EntityCreateTypeMap, UISupportedEntityCreateKind } from '@models/entities/entities';
import createEntityCreatePage, { CreateMetadataComponent } from '../EntityCreate';
import SigmaRuleCreateConfig from './SigmaRuleCreateConfig';

// spec: ../EntityCreate.spec.md

export type EntityCreateConfig<K extends UISupportedEntityCreateKind> = {
  kind: K;
  EntityMetadata: CreateMetadataComponent<K>;
  BlankCreateEntity: EntityCreateTypeMap[K];
  supportsGraphic?: boolean;
};

export type EntityCreateConfigMap = {
  [K in UISupportedEntityCreateKind]: EntityCreateConfig<K>;
};

export const EntitiesCreateConfig = {
  [Entities.Collection]: CollectionCreateConfig,
  [Entities.SigmaRule]: SigmaRuleCreateConfig,
  [Entities.Device]: DeviceCreateConfig,
  [Entities.Vendor]: VendorCreateConfig,
  [Entities.Flag]: FlagCreateConfig,
  [Entities.Incident]: IncidentCreateConfig,
  [Entities.CompiledFunction]: CompiledFunctionCreateConfig,
  [Entities.DecompiledFunction]: DecompiledFunctionCreateConfig,
  [Entities.PeSection]: PeSectionCreateConfig,
  [Entities.PeImport]: PeImportCreateConfig,
} satisfies EntityCreateConfigMap;

export const EntityCreatePages = {
  [Entities.Collection]: createEntityCreatePage(CollectionCreateConfig),
  [Entities.Device]: createEntityCreatePage(DeviceCreateConfig),
  [Entities.SigmaRule]: createEntityCreatePage(SigmaRuleCreateConfig),
  [Entities.Vendor]: createEntityCreatePage(VendorCreateConfig),
  [Entities.Flag]: createEntityCreatePage(FlagCreateConfig),
  [Entities.Incident]: createEntityCreatePage(IncidentCreateConfig),
  [Entities.CompiledFunction]: createEntityCreatePage(CompiledFunctionCreateConfig),
  [Entities.DecompiledFunction]: createEntityCreatePage(DecompiledFunctionCreateConfig),
  [Entities.PeSection]: createEntityCreatePage(PeSectionCreateConfig),
  [Entities.PeImport]: createEntityCreatePage(PeImportCreateConfig),
} satisfies { [K in UISupportedEntityCreateKind]: React.ComponentType };
