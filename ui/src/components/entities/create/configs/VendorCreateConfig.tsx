import { JSX } from 'react';
import { Row } from 'react-bootstrap';
import { getNames as getCountryNames, Country } from 'country-list';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { Entities } from '@models/entities/entities';
import { CriticalSector } from '@models/entities/sectors';
import { BlankCreateVendor, VendorCreateMetaFields } from '@models/entities/vendors';

// country names used in SelectInputArray options for vendor country
const CountryNames = getCountryNames();

const VendorMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.Vendor>): JSX.Element => {
  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof VendorCreateMetaFields>(field: T, value: VendorCreateMetaFields[T]): void {
    const updates: VendorCreateMetaFields = structuredClone(entity.metadata.Vendor);
    updates[field] = value;
    onChange('metadata', { Vendor: updates });
  }
  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Countries</InfoHeader>
        <InfoValue>
          <SelectInputArray
            isCreatable={false}
            options={CountryNames}
            values={entity.metadata.Vendor?.countries ? entity.metadata.Vendor.countries : []}
            onChange={(countries) => updatePendingMeta('countries', countries)}
          />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Critical Sectors</InfoHeader>
        <InfoValue>
          <SelectInputArray
            isCreatable={false}
            options={Object.values(CriticalSector)}
            values={entity.metadata.Vendor.critical_sectors ? entity.metadata.Vendor.critical_sectors : []}
            onChange={(sectors) => updatePendingMeta('critical_sectors', sectors as CriticalSector[])}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const VendorCreateConfig: EntityCreateConfig<Entities.Vendor> = {
  kind: Entities.Vendor,
  EntityMetadata: VendorMetaInfo,
  BlankCreateEntity: BlankCreateVendor,
};

export default VendorCreateConfig;
