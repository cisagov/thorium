import { Country } from 'country-list';
import { CriticalSector } from './sectors';
import { CreateEntity, Entities, Entity } from './entities';

export type VendorMetaFields = {
  countries: Country[]; // country codes and names type
  critical_sectors: CriticalSector[]; // sectors this vendor's hardware is used in
};

export type VendorCreateMetaFields = {
  countries: string[]; // country code list
  critical_sectors: CriticalSector[]; // sectors this vendor's hardware is used in
};

export type VendorMeta = {
  Vendor: VendorMetaFields;
};

export type VendorCreateMeta = {
  Vendor: VendorCreateMetaFields;
};

export type Vendor = Entity & {
  metadata: VendorMeta;
};

export type CreateVendor = CreateEntity & {
  metadata: VendorCreateMeta;
};

export enum VendorCategory {
  Networking = 'Networking',
  IT = 'Internet Technology',
  Communications = 'Communications',
  Industrial = 'Industrial',
  Defense = 'Defense',
  Medical = 'Medical',
  Agriculture = 'Agriculture',
}

export const BlankVendor: Vendor = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Vendor,
  metadata: {
    Vendor: {
      countries: [
        {
          code: 'US',
          name: 'United States of America',
        },
      ],
      critical_sectors: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateVendor: CreateVendor = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.Vendor,
  metadata: {
    Vendor: {
      countries: ['United States of America'],
      critical_sectors: [],
    },
  },
};
