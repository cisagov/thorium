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
  EntitySecondary,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
import { NetworkConnection, NetworkConnectionMetaFields } from '@models/entities/network_connections';
import { listEntities } from '@thorpi/entities';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// get files using filters and and an optional cursor
const getNetworkConnections = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.NetworkConnection];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList as NetworkConnection[],
    entitiesCursor: entityCursor,
  };
};

const NetworkConnectionListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Source</EntitySecondary>
          <EntityGroups>Destination</EntityGroups>
          <EntitySubmitters>Proto</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface NetworkConnectionItemProps {
  conn: NetworkConnection; // Net conn details
}

const netConnSrc = (conn: NetworkConnectionMetaFields) => {
  const src = conn.source_port ? `${conn.source}:${conn.source_port}` : conn.source;
  return `${src}`;
};

const netConnDest = (conn: NetworkConnectionMetaFields) => {
  const dest = conn.destination_port ? `${conn.destination}:${conn.destination_port}` : conn.destination;
  return `${dest}`;
};

const NetworkConnectionItem: React.FC<NetworkConnectionItemProps> = ({ conn }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.NetworkConnection)}/${conn.id}`} state={{ conn: conn }} className="no-decoration">
          <LinkFields>
            <EntityName>{conn.name}</EntityName>
            <EntitySecondary>
              <i>{netConnSrc(conn.metadata.NetworkConnection)}</i>
            </EntitySecondary>
            <EntityGroups>
              <i>{netConnDest(conn.metadata.NetworkConnection)}</i>
            </EntityGroups>
            <EntitySubmitters>
              {conn.metadata.NetworkConnection.protocol ? (
                <small>
                  <i>
                    {conn.metadata.NetworkConnection.protocol.length > 75
                      ? conn.metadata.NetworkConnection.protocol.substring(0, 75) + '...'
                      : conn.metadata.NetworkConnection.protocol}
                  </i>
                </small>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {conn.tags != undefined && <hr />}
        <Row>
          {conn.tags && Object.keys(conn.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.NetworkConnection} tags={conn.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const NetworkConnectionsBrowsingConfig: EntityBrowseConfig<Entities.NetworkConnection> = {
  docTitle: 'Connections · Thorium',
  title: 'Network Connections',
  typeLabel: '',
  kind: Entities.NetworkConnection,
  creatable: false,
  entityHeaders: <NetworkConnectionListHeaders />,
  renderEntity: (entity) => <NetworkConnectionItem conn={entity} />,
  fetchEntities: getNetworkConnections,
};

export default NetworkConnectionsBrowsingConfig;
