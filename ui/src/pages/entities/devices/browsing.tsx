import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import {
  BrowsingCard,
  BrowsingContents,
  BrowsingFilters,
  CondensedEntityTags,
  EntityGroups,
  EntityList,
  EntityName,
  EntityOrigin,
  EntitySubmitters,
  LinkFields,
  Page,
} from '@components';
import { useAuth } from '@utilities';
import { Device, Entities, Entity, Filters } from '@models';
import { listEntities } from '@thorpi';

// get files using filters and and an optional cursor
const getDevices = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Device];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList,
    entitiesCursor: entityCursor,
  };
};

const DeviceListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntityOrigin>Vendor</EntityOrigin>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface DeviceItemProps {
  device: Device; // device details
}

const DeviceItem: React.FC<DeviceItemProps> = ({ device }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`/device/${device.id}`} state={{ device: device }} className="no-decoration">
          <LinkFields>
            <EntityName>{device.name}</EntityName>
            <EntityOrigin>{device.metadata.Device.vendor ? device.metadata.Device.vendor.substring(0, 30) : ''}</EntityOrigin>
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
        {device.tags != undefined && <hr />}
        <Row>
          {device.tags && Object.keys(device.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.Device} tags={device.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const DeviceBrowsingContainer = () => {
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<Filters>({});
  const { userInfo } = useAuth();

  return (
    <Page title={`Devices · Thorium`}>
      <BrowsingFilters
        title="Devices"
        kind={Entities.Device}
        onChange={setFilters}
        groups={userInfo?.groups ? userInfo.groups : []}
        disabled={loading}
        creatable={true}
      />
      <EntityList
        type="Devices"
        entityHeaders={<DeviceListHeaders />}
        displayEntity={(device) => <DeviceItem device={device} />}
        filters={filters}
        fetchEntities={getDevices}
        setLoading={setLoading}
        loading={loading}
      />
    </Page>
  );
};

export default DeviceBrowsingContainer;
