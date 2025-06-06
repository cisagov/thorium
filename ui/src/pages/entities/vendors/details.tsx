import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityDetails, FieldBadge, SelectInputArray, InfoHeader, InfoValue } from '@components';
import { BlankVendor, CriticalSector, Entities, Entity, Vendor, VendorMetaFields } from '@models';
import { getEntity } from '@thorpi';
import { getNames as getCountryNames, getCode as getCountryCode, Country } from 'country-list';
import { MdBusinessCenter } from 'react-icons/md';

// country names used in SelectInputArray options for vendor country
const CountryNames = getCountryNames();

const VendorMetaInfo = (
  vendor: Vendor,
  pendingVendor: Vendor,
  handleUpdate: <K extends keyof Entity>(field: K, value: Entity[K]) => void,
  editing: boolean,
): JSX.Element => {
  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof VendorMetaFields>(field: T, value: VendorMetaFields[T]): void {
    const updates: VendorMetaFields = structuredClone(pendingVendor.metadata.Vendor);
    if (field == 'countries') {
      const countryNames = value as string[];
      updates[field] = countryNames.map((name: string) => {
        return { code: getCountryCode(name), name: name };
      }) as any;
    } else {
      updates[field] = value;
    }
    handleUpdate('metadata', { Vendor: updates });
  }

  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Countries</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={false}
              options={CountryNames}
              values={vendor.metadata.Vendor?.countries ? vendor.metadata.Vendor.countries.map((country: Country) => country.name) : []}
              onChange={(countries) => updatePendingMeta('countries', countries as unknown as Country[])}
            />
          ) : (
            <>
              <FieldBadge color="Gray" noNull={true} field={vendor.metadata.Vendor.countries.map((country: Country) => country.name)} />
            </>
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Critical Sectors</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={false}
              options={Object.values(CriticalSector)}
              values={pendingVendor.metadata.Vendor?.critical_sectors ? pendingVendor.metadata.Vendor.critical_sectors : []}
              onChange={(sectors) => updatePendingMeta('critical_sectors', sectors as CriticalSector[])}
            />
          ) : (
            <>
              <FieldBadge color="DarkRed" noNull={true} field={vendor.metadata.Vendor.critical_sectors}></FieldBadge>
            </>
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const VendorDetailsContainer = () => {
  // Get vendor details from the API
  const getVendorDetails = async (vendorID: string, setError: (err: string) => void, updateEntity: (vendor: Vendor) => void) => {
    getEntity(vendorID, setError).then((data) => {
      if (data && data.kind == Entities.Vendor) {
        // we know that any entity response with kind=Vendor is a Vendor
        const vendor = data as Vendor;
        // need to add spaces for more human friendly suggestions for Critical Sectors
        if (vendor.metadata.Vendor?.critical_sectors) {
          vendor.metadata.Vendor.critical_sectors = vendor.metadata.Vendor.critical_sectors.map((sector: string) =>
            sector.replace(/([A-Z])/g, ' $1').trim(),
          );
        }
        updateEntity(vendor);
      }
    });
  };
  return (
    <EntityDetails
      getEntityDetails={getVendorDetails}
      metadata={VendorMetaInfo}
      blank={BlankVendor}
      icon={(size: number) => <MdBusinessCenter size={size} />}
    />
  );
};

export default VendorDetailsContainer;
