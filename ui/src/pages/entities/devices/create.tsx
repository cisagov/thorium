import { JSX, useEffect, useState } from 'react';
import { Row, Col, Form } from 'react-bootstrap';

// project imports
import { SelectInputArray, InfoHeader, InfoValue, EntityCreate } from '@components';
import { BlankCreateDevice, CreateDevice, CriticalSector, DeviceMetaFields, Entities, Vendor } from '@models';
import { getAvailableVendors } from '../utilities';

const DeviceMetaInfo = (
  device: CreateDevice,
  onChange: <K extends keyof CreateDevice>(field: K, value: CreateDevice[K]) => void,
): JSX.Element => {
  // vendor creation state hooks
  const [pendingVendorName, setPendingVendorName] = useState('');
  const [vendorsMap, setVendorsMap] = useState<{ [key: string]: string }>({});

  useEffect(() => {
    getAvailableVendors(setVendorsMap);
  }, []);

  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof DeviceMetaFields>(field: T, value: DeviceMetaFields[T]): void {
    const updates: DeviceMetaFields = structuredClone(device.metadata.Device);
    updates[field] = value;
    onChange('metadata', { Device: updates });
  }

  return (
    <>
      <Row className="d-flex flex-row justify-content-center">
        <InfoHeader>Vendor</InfoHeader>
        <InfoValue>
          <SelectInputArray
            isCreatable={false}
            values={device.metadata.Device.vendors?.length > 0 ? device.metadata.Device.vendors.map((vendor: Vendor) => vendor.id) : []}
            options={Object.keys(vendorsMap)}
            onChange={(vendorID) => updatePendingMeta('vendors', [vendorID] as any)}
            onCreate={(name) => setPendingVendorName(name ? name : '')}
            valuesMap={vendorsMap}
          />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Critical System</InfoHeader>
        <InfoValue>
          <Form.Group>
            <Form.Check
              type="switch"
              id="collect-logs"
              label=""
              checked={device.metadata.Device.critical_system}
              onChange={(e) => updatePendingMeta('critical_system', e.target.checked)}
            />
          </Form.Group>
        </InfoValue>
      </Row>
      {device?.metadata?.Device.critical_system && (
        <>
          <Row className="mt-3">
            <InfoHeader>Critical Sectors</InfoHeader>
            <InfoValue>
              <SelectInputArray
                isCreatable={false}
                options={Object.values(CriticalSector)}
                values={device.metadata.Device.critical_sectors ? device.metadata.Device.critical_sectors : []}
                onChange={(sectors) => updatePendingMeta('critical_sectors', sectors as CriticalSector[])}
              />
            </InfoValue>
          </Row>
        </>
      )}
      <Row>
        <InfoHeader>Sensitive Location</InfoHeader>
        <Col>
          <Form.Group>
            <Form.Check
              type="switch"
              id="collect-logs"
              label=""
              checked={device.metadata.Device.sensitive_location}
              onChange={(e) => updatePendingMeta('sensitive_location', e.target.value == 'on' ? true : false)}
            />
          </Form.Group>
        </Col>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Urls</InfoHeader>
        <InfoValue>
          <SelectInputArray
            values={device.metadata.Device.urls ? device.metadata.Device.urls : []}
            onChange={(urls) => updatePendingMeta('urls', urls)}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const CreateDeviceContainer = () => {
  return <EntityCreate kind={Entities.Device} metadata={DeviceMetaInfo} blank={BlankCreateDevice} />;
};

export default CreateDeviceContainer;
