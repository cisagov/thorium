import { JSX } from 'react';
import { Row } from 'react-bootstrap';
import { getNames as getCountryNames, Country } from 'country-list';

// project imports
import EntityCreate from '@entities/create/EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import SelectInputArray from '@components/shared/selectable/SelectInputArray';
import { Entities } from '@models/entities/entities';
import { CriticalSector } from '@models/entities/sectors';
import { BlankCreateVendor, CreateVendor, VendorMetaFields } from '@models/entities/vendors';

// country names used in SelectInputArray options for vendor country
const CountryNames = getCountryNames();

const VendorMetaInfo = (
  vendor: CreateVendor,
  onChange: <K extends keyof CreateVendor>(field: K, value: CreateVendor[K]) => void,
): JSX.Element => {
  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof VendorMetaFields>(field: T, value: VendorMetaFields[T]): void {
    const updates: VendorMetaFields = structuredClone(vendor.metadata.Vendor);
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
            values={vendor.metadata.Vendor?.countries ? vendor.metadata.Vendor.countries : []}
            onChange={(countries) => updatePendingMeta('countries', countries as unknown as Country[])}
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
            values={vendor.metadata.Vendor.critical_sectors ? vendor.metadata.Vendor.critical_sectors : []}
            onChange={(sectors) => updatePendingMeta('critical_sectors', sectors as CriticalSector[])}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const VendorCreate = () => {
  return <EntityCreate kind={Entities.Vendor} metadata={VendorMetaInfo} blank={BlankCreateVendor} />;
};

export default VendorCreate;
