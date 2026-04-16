// project imports
import CollectionCreateConfig from './CollectionCreateConfig';
import DeviceCreateConfig from './DeviceCreateConfig';
import VendorCreateConfig from './VendorCreateConfig';
import { Entities, EntityCreateTypeMap, UISupportedEntityCreateKind } from '@models/entities/entities';
import createEntityCreatePage, { CreateMetadataComponent } from '../EntityCreate';
import SigmaRuleCreateConfig from './SigmaRuleCreateConfig';

export type EntityCreateConfig<K extends UISupportedEntityCreateKind> = {
  kind: K;
  EntityMetadata: CreateMetadataComponent<K>;
  BlankCreateEntity: EntityCreateTypeMap[K];
};

export type EntityCreateConfigMap = {
  [K in UISupportedEntityCreateKind]: EntityCreateConfig<K>;
};

export const EntitiesCreateConfig = {
  [Entities.Collection]: CollectionCreateConfig,
  [Entities.SigmaRule]: SigmaRuleCreateConfig,
  [Entities.Device]: DeviceCreateConfig,
  [Entities.Vendor]: VendorCreateConfig,
} satisfies EntityCreateConfigMap;

export const EntityCreatePages = {
  [Entities.Collection]: createEntityCreatePage(CollectionCreateConfig),
  [Entities.Device]: createEntityCreatePage(DeviceCreateConfig),
  [Entities.SigmaRule]: createEntityCreatePage(SigmaRuleCreateConfig),
  [Entities.Vendor]: createEntityCreatePage(VendorCreateConfig),
} satisfies { [K in UISupportedEntityCreateKind]: React.ComponentType };
