import { CriticalSector } from './sectors';
import { CreateEntity, Entities, Entity } from './entities';
import { BlankCreateVendor, BlankVendor, Vendor } from './vendors';

export type DeviceMetaFields = {
  urls?: string[];
  vendors?: Vendor[];
  critical_system?: boolean;
  sensitive_location?: boolean;
  critical_sectors?: CriticalSector[]; // sectors this device is used in
};

export type DeviceCreateMetaFields = {
  urls?: string[];
  vendors?: string[];
  critical_system?: boolean;
  sensitive_location?: boolean;
  critical_sectors?: CriticalSector[]; // sectors this device is used in
};

export type DeviceMeta = {
  Device: DeviceMetaFields;
};

export type DeviceCreateMeta = {
  Device: DeviceCreateMetaFields;
};

export type Device = Entity & {
  metadata: DeviceMeta;
};

export type CreateDevice = CreateEntity & {
  metadata: DeviceCreateMeta;
};

export const BlankDevice: Device = {
  // need to
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Device,
  metadata: {
    Device: {
      urls: [],
      vendors: [BlankVendor],
      critical_system: false,
      sensitive_location: false,
      critical_sectors: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateDevice: CreateDevice = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.Device,
  metadata: {
    Device: {
      urls: [],
      vendor: '',
      critical_system: false,
      sensitive_location: false,
      critical_sectors: [],
    },
  },
};
