import { JSX, useEffect, useState } from 'react';
import { Row, Col, Form } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import { getAvailableVendors } from '@entities/utilities';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { Entities } from '@models/entities/entities';
import { CriticalSector } from '@models/entities/sectors';
import { BlankCreateDevice, DeviceCreateMetaFields } from '@models/entities/devices';

const DeviceMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.Device>): JSX.Element => {
  const [vendorsMap, setVendorsMap] = useState<{ [key: string]: string }>({});

  useEffect(() => {
    getAvailableVendors(setVendorsMap);
  }, []);

  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof DeviceCreateMetaFields>(field: T, value: DeviceCreateMetaFields[T]): void {
    const updates: DeviceCreateMetaFields = structuredClone(entity.metadata.Device);
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
            values={
              entity.metadata.Device.vendors && entity.metadata.Device.vendors?.length > 0
                ? entity.metadata.Device.vendors.map((vendorID: string) => vendorID)
                : []
            }
            options={Object.keys(vendorsMap)}
            onChange={(vendorIDs) => updatePendingMeta('vendors', vendorIDs)}
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
              checked={entity.metadata.Device.critical_system}
              onChange={(e) => updatePendingMeta('critical_system', e.target.checked)}
            />
          </Form.Group>
        </InfoValue>
      </Row>
      {entity?.metadata?.Device.critical_system && (
        <>
          <Row className="mt-3">
            <InfoHeader>Critical Sectors</InfoHeader>
            <InfoValue>
              <SelectInputArray
                isCreatable={false}
                options={Object.values(CriticalSector)}
                values={entity.metadata.Device.critical_sectors ? entity.metadata.Device.critical_sectors : []}
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
              checked={entity.metadata.Device.sensitive_location}
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
            values={entity.metadata.Device.urls ? entity.metadata.Device.urls : []}
            onChange={(urls) => updatePendingMeta('urls', urls)}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const DeviceCreateConfig: EntityCreateConfig<Entities.Device> = {
  kind: Entities.Device,
  EntityMetadata: DeviceMetaInfo,
  BlankCreateEntity: BlankCreateDevice,
};

export default DeviceCreateConfig;
