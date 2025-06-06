import { JSX } from 'react';
import { Row, Col } from 'react-bootstrap';
import { getNames as getCountryNames, Country } from 'country-list';

// project imports
import { SelectInputArray, InfoHeader, InfoValue, EntityCreate } from '@components';
import { BlankCreateVendor, CreateVendor, CriticalSector, Entities, VendorMetaFields } from '@models';

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

const CreateVendorContainer = () => {
  return <EntityCreate kind={Entities.Vendor} metadata={VendorMetaInfo} blank={BlankCreateVendor} />;
};

export default CreateVendorContainer;
