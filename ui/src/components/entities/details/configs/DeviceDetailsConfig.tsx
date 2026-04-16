import { useEffect, useState, JSX } from 'react';
import { Row, Form } from 'react-bootstrap';
import { FaHardDrive } from 'react-icons/fa6';

// project imports
import { DetailsMetadataProps } from '../EntityDetails';
import { EntityDetailsConfig } from './configs';
import InfoValue from '@entities/shared/InfoValue';
import InfoHeader from '@entities/shared/InfoHeader';
import { getAvailableVendors } from '@entities/utilities';
import LinkBadge from '@components/shared/badges/LinkBadge';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities';
import { CriticalSector } from '@models/entities/sectors';
import { BlankVendor, Vendor } from '@models/entities/vendors';
import { BlankDevice, Device, DeviceMetaFields } from '@models/entities/devices';

// handle mapping a lists of vendors to vendor stubs for updating a Device's metadata vendors field
const stubPendingVendors = (vendors: string[], vendorMap: { [key: string]: string }) => {
  return vendors.map((vendorUUID) => {
    const newBlankVendor = structuredClone(BlankVendor);
    newBlankVendor.id = vendorUUID;
    newBlankVendor.name = vendorMap[vendorUUID];
    return newBlankVendor;
  });
};

// sort a list of Vendor entities by their name
const sortVendorsList = (vendors: Vendor[]) => vendors.sort((a: Vendor, b: Vendor) => a.name.localeCompare(b.name));

const DeviceMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.Device>): JSX.Element => {
  const [vendorsMap, setVendorsMap] = useState<{ [key: string]: string }>({});

  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof DeviceMetaFields>(field: T, value: DeviceMetaFields[T]): void {
    const updates: DeviceMetaFields = structuredClone(pendingEntity.metadata.Device);
    updates[field] = value;
    handleUpdate('metadata', { Device: updates });
  }

  // get vendor map on component load
  useEffect(() => {
    getAvailableVendors(setVendorsMap);
  }, []);

  return (
    <>
      <Row className="d-flex flex-row justify-content-center">
        <InfoHeader>Vendors</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={false}
              values={
                pendingEntity.metadata.Device.vendors && pendingEntity.metadata.Device.vendors?.length > 0
                  ? sortVendorsList(pendingEntity.metadata.Device.vendors).map((vendor: Vendor) => vendor.id)
                  : []
              }
              options={Object.keys(vendorsMap)}
              onChange={(vendors) => updatePendingMeta('vendors', stubPendingVendors(vendors, vendorsMap))}
              onCreate={(name) => console.log('Attempting to create a new vendor: ${}')}
              valuesMap={vendorsMap}
            />
          ) : (
            <>
              {entity.metadata.Device.vendors &&
                entity.metadata.Device.vendors?.length > 0 &&
                sortVendorsList(entity.metadata.Device.vendors).map((vendor: Vendor) => (
                  <LinkBadge internal url={`/vendor/${vendor.id}`} label={`${vendor.name} (${vendor.id})`} />
                ))}
            </>
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Critical System</InfoHeader>
        <InfoValue>
          {editing ? (
            <Form.Group>
              <Form.Check
                type="switch"
                id="critical-system-toggle"
                label=""
                checked={pendingEntity.metadata.Device.critical_system}
                onChange={(e) => updatePendingMeta('critical_system', e.target.checked)}
              />
            </Form.Group>
          ) : entity.metadata.Device.critical_system ? (
            'Yes'
          ) : (
            'No'
          )}
        </InfoValue>
      </Row>
      {editing && pendingEntity.metadata.Device.critical_system && (
        <Row className="mt-3">
          <InfoHeader>Critical Sectors</InfoHeader>
          <InfoValue>
            <SelectInputArray
              isCreatable={false}
              options={Object.values(CriticalSector)}
              values={pendingEntity.metadata.Device?.critical_sectors ? pendingEntity.metadata.Device.critical_sectors : []}
              onChange={(sectors) => updatePendingMeta('critical_sectors', sectors as CriticalSector[])}
            />
          </InfoValue>
        </Row>
      )}
      {!editing && entity.metadata.Device.critical_system && (
        <Row className="mt-3">
          <InfoHeader>Critical Sectors</InfoHeader>
          <InfoValue>
            <FieldBadge color="DarkRed" noNull={true} field={entity.metadata.Device.critical_sectors}></FieldBadge>
          </InfoValue>
        </Row>
      )}
      <Row className="mt-3">
        <InfoHeader>Sensitive Location</InfoHeader>
        <InfoValue>
          {editing ? (
            <Form.Group>
              <Form.Check
                type="switch"
                id="sensitive-location-toggle"
                label=""
                checked={pendingEntity.metadata.Device.sensitive_location}
                onChange={(e) => updatePendingMeta('sensitive_location', e.target.checked)}
              />
            </Form.Group>
          ) : entity.metadata.Device.sensitive_location ? (
            'Yes'
          ) : (
            'No'
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Urls</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              values={pendingEntity.metadata.Device.urls ? pendingEntity.metadata.Device.urls : []}
              onChange={(urls) => updatePendingMeta('urls', urls)}
            />
          ) : (
            <>
              {entity.metadata.Device.urls &&
                entity.metadata.Device.urls.map((url: string, idx: number) => <LinkBadge key={`device-url-${idx}`} url={url} />)}
            </>
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const getDeviceDetails = async (vendorID: string, setError: (err: string) => void, updateEntity: (entity: Device) => void) => {
  getEntity(vendorID, setError).then((data) => {
    // check data is not null and is of Device kind
    if (data && data.kind == Entities.Device) {
      // we know that any entity response with kind=Device is a Device
      const device = data as Device;
      // need to add spaces for more human friendly suggestions for Critical Sectors
      if (device.metadata.Device?.critical_sectors) {
        device.metadata.Device.critical_sectors = device.metadata.Device.critical_sectors.map(
          (sector: string) => sector.replace(/([A-Z])/g, ' $1').trim() as CriticalSector,
        );
      }
      updateEntity(device);
    }
  });
};

const DeviceDetailsConfig: EntityDetailsConfig<Entities.Device> = {
  getEntityDetails: getDeviceDetails,
  EntityMetaInfo: DeviceMetaInfo,
  BlankEntity: BlankDevice,
  icon: (size: number) => <FaHardDrive size={size} />,
};

export default DeviceDetailsConfig;
