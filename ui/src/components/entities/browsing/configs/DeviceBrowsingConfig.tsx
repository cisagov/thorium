import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';

// project imports
import { EntityBrowseConfig } from './config';
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntityNameWithIcon,
  EntitySecondary,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { listEntities } from '@thorpi/entities';
import { Entities } from '@models/entities/entities';
import { Filters } from '@models/search';
import { Vendor } from '@models/entities/vendors';
import { Device } from '@models/entities/devices';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// spec: ../EntityBrowsing.spec.md

// Fetch devices using search filters and an optional pagination cursor.
const getDevices = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Device];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList as Device[],
    entitiesCursor: entityCursor,
  };
};

const DeviceListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Vendor</EntitySecondary>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface DeviceItemProps {
  device: Device;
}

const DeviceItem: React.FC<DeviceItemProps> = ({ device }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.Device)}/${device.id}`} state={{ device: device }} className="no-decoration">
          <LinkFields>
            <EntityName>
              <EntityNameWithIcon entityId={device.id} hasImage={device.image != null}>
                {device.name}
              </EntityNameWithIcon>
            </EntityName>
            <EntitySecondary>
              {device.metadata.Device.vendors && device.metadata.Device.vendors.length > 0
                ? device.metadata.Device.vendors.map((vendor: Vendor) => vendor.name).join(', ')
                : ''}
            </EntitySecondary>
            <EntityGroups>
              <small>
                <i>
                  {device.groups &&
                    (device.groups.toString().length > 75
                      ? device.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : device.groups.toString().replaceAll(',', ', '))}
                </i>
              </small>
            </EntityGroups>
            <EntitySubmitters>
              {device.submitter ? (
                <small>
                  <i>{device.submitter.length > 75 ? device.submitter.substring(0, 75) + '...' : device.submitter}</i>
                </small>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {device.tags && Object.keys(device.tags).length > 1 && (
          <>
            <hr />
            <CondensedEntityTags resource={Entities.Device} tags={device.tags} />
          </>
        )}
      </BrowsingContents>
    </BrowsingCard>
  );
};

const DeviceBrowsingConfig: EntityBrowseConfig<Entities.Device> = {
  docTitle: 'Devices · Thorium',
  title: 'Devices',
  typeLabel: '',
  kind: Entities.Device,
  creatable: true,
  entityHeaders: <DeviceListHeaders />,
  renderEntity: (entity) => <DeviceItem device={entity} />,
  fetchEntities: getDevices,
};

export default DeviceBrowsingConfig;
