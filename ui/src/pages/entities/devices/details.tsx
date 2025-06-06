import { useEffect, useState, JSX, useRef } from 'react';
import { Row, Form } from 'react-bootstrap';

// project imports
import { EntityDetails, FieldBadge, SelectInputArray, InfoHeader, InfoValue, LinkBadge } from '@components';
import { BlankDevice, CreateVendor, CriticalSector, Device, DeviceMetaFields, Entities, Entity, Filters, ValueMap, Vendor } from '@models';
import { getEntity, listEntities } from '@thorpi';
import { FaHardDrive } from 'react-icons/fa6';

const DeviceMetaInfo = (
  device: Device,
  pendingDevice: Device,
  handleUpdate: <K extends keyof Entity>(field: K, value: Entity[K]) => void,
  editing: boolean,
): JSX.Element => {
  const vendorsMap = useRef<{ [key: string]: string }>({});
  const vendorInfoMap = useRef<{ [key: string]: Vendor }>({});

  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof DeviceMetaFields>(field: T, value: DeviceMetaFields[T]): void {
    const updates: DeviceMetaFields = structuredClone(pendingDevice.metadata.Device);
    if (field == 'vendors' && value != undefined) {
      // this will be a list of vendor IDs rather than the vendors, we will do a lookup to put the correct values back
      const newVendors: Vendor[] = [];
      const vendorIDs = value as string[];
      vendorIDs.forEach((vendorID: string) => {
        // for newly created vendors, they may not be in the map
        if (vendorInfoMap.current.hasOwnProperty(vendorID)) {
          newVendors.push(vendorInfoMap.current[vendorID]);
        }
      });
      updates[field] = newVendors as any;
    } else {
      updates[field] = value;
    }
    handleUpdate('metadata', { Device: updates });
  }

  const createVendorHandler = (vendorID: string) => {
    // get vendor, add it to the vendor id/name map
    getEntity(vendorID, console.log).then((vendor) => {
      if (vendor != null) {
        const newVendorID = vendor.id;
        vendorsMap.current[newVendorID] = newVendorID;
        // add new vendor info to vendor info map
        vendorInfoMap.current[newVendorID] = vendor;
      }
    });
  };

  const getAvailableVendors = async () => {
    // TODO: actually loop to grab all vendors
    const filters: Filters = { kinds: [Entities.Vendor], limit: 10000 };
    const { entityList } = await listEntities(filters, console.log, true, null);
    if (entityList) {
      entityList.forEach((vendor: Vendor) => {
        vendorsMap.current[vendor.id] = vendor.name;
      });
      // keep track of all vendor info in object with id as a key
      entityList.forEach((vendor: Vendor) => {
        vendorInfoMap.current[vendor.id] = vendor;
      });
    }
  };

  useEffect(() => {
    getAvailableVendors();
  }, []);

  return (
    <>
      <Row className="d-flex flex-row justify-content-center">
        <InfoHeader>Vendors</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={false}
              values={device.metadata.Device.vendors?.length > 0 ? device.metadata.Device.vendors.map((vendor: Vendor) => vendor.id) : []}
              options={Object.keys(vendorsMap.current)}
              onChange={(vendorID) => updatePendingMeta('vendors', vendorID as any)}
              onCreate={(name) => console.log('Attempting to create a new vendor: ${}')}
              valuesMap={vendorsMap.current}
            />
          ) : (
            <>
              {device.metadata.Device.vendors?.length > 0 &&
                device.metadata.Device.vendors.map((vendor: Vendor) => (
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
                checked={pendingDevice.metadata.Device.critical_system}
                onChange={(e) => updatePendingMeta('critical_system', e.target.checked)}
              />
            </Form.Group>
          ) : device.metadata.Device.critical_system ? (
            'Yes'
          ) : (
            'No'
          )}
        </InfoValue>
      </Row>
      {editing && pendingDevice.metadata.Device.critical_system && (
        <Row className="mt-3">
          <InfoHeader>Critical Sectors</InfoHeader>
          <InfoValue>
            <SelectInputArray
              isCreatable={false}
              options={Object.values(CriticalSector)}
              values={pendingDevice.metadata.Device?.critical_sectors ? pendingDevice.metadata.Device.critical_sectors : []}
              onChange={(sectors) => updatePendingMeta('critical_sectors', sectors as CriticalSector[])}
            />
          </InfoValue>
        </Row>
      )}
      {!editing && device.metadata.Device.critical_system && (
        <Row className="mt-3">
          <InfoHeader>Critical Sectors</InfoHeader>
          <InfoValue>
            <FieldBadge color="DarkRed" noNull={true} field={device.metadata.Device.critical_sectors}></FieldBadge>
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
                checked={pendingDevice.metadata.Device.sensitive_location}
                onChange={(e) => updatePendingMeta('sensitive_location', e.target.checked)}
              />
            </Form.Group>
          ) : device.metadata.Device.sensitive_location ? (
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
              values={pendingDevice.metadata.Device.urls ? pendingDevice.metadata.Device.urls : []}
              onChange={(urls) => updatePendingMeta('urls', urls)}
            />
          ) : (
            <>
              {device.metadata.Device.urls &&
                device.metadata.Device.urls.map((url: string, idx: number) => <LinkBadge key={`device-url-${idx}`} url={url} />)}
            </>
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const DeviceDetailsContainer = () => {
  // Get device details from the API
  const getDeviceDetails = async (vendorID: string, setError: (err: string) => void, updateEntity: (entity: Device) => void) => {
    getEntity(vendorID, setError).then((data) => {
      // check data is not null and is of Device kind
      if (data && data.kind == Entities.Device) {
        // we know that any entity response with kind=Device is a Device
        const device = data as Device;
        // need to add spaces for more human friendly suggestions for Critical Sectors
        if (device.metadata.Device?.critical_sectors) {
          device.metadata.Device.critical_sectors = device.metadata.Device.critical_sectors.map((sector: string) =>
            sector.replace(/([A-Z])/g, ' $1').trim(),
          );
        }
        updateEntity(device);
      }
    });
  };
  return (
    <EntityDetails
      getEntityDetails={getDeviceDetails}
      metadata={DeviceMetaInfo}
      blank={BlankDevice}
      icon={(size: number) => <FaHardDrive size={size} />}
    />
  );
};

export default DeviceDetailsContainer;
