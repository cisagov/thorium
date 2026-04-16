import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';
import styled from 'styled-components';

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
import { listEntities } from '@thorpi/entities';
import { Entities } from '@models/entities/entities';
import { WindowsProcess } from '@models/entities/processes';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// get files using filters and and an optional cursor
const getWindowsProcesses = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.WindowsProcess];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList as WindowsProcess[],
    entitiesCursor: entityCursor,
  };
};

const WindowsProcessListHeader = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Create Time</EntitySecondary>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const Command = styled.div`
  font-size: 0.8rem;
  font-style: italic;
`;

interface WindowsProcessItemProps {
  process: WindowsProcess; // Processes details
}

const WindowsProcessItem: React.FC<WindowsProcessItemProps> = ({ process }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link
          to={`${getDetailsBasePathByEntity(Entities.WindowsProcess)}/${process.id}`}
          state={{ process: process }}
          className="no-decoration"
        >
          <LinkFields>
            <EntityName>{process.name}</EntityName>
            <EntitySecondary>
              {' '}
              <i>
                {process.metadata.WindowsProcess.create_time &&
                  (process.metadata.WindowsProcess.create_time.toString().length > 75
                    ? process.metadata.WindowsProcess.create_time.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : process.metadata.WindowsProcess.create_time.toString().replaceAll(',', ', '))}
              </i>
            </EntitySecondary>
            <EntityGroups>
              <i>
                {process.groups &&
                  (process.groups.toString().length > 75
                    ? process.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                    : process.groups.toString().replaceAll(',', ', '))}
              </i>
            </EntityGroups>
            <EntitySubmitters>
              {process.submitter ? (
                <i>{process.submitter.length > 75 ? process.submitter.substring(0, 75) + '...' : process.submitter}</i>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {process.metadata.WindowsProcess?.command && <Command className="mt-3 mb-2">{process.metadata.WindowsProcess.command}</Command>}
        {process.tags != undefined && <hr />}
        <Row>
          {process.tags && Object.keys(process.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.WindowsProcess} tags={process.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const WindowsProcessesConfig: EntityBrowseConfig<Entities.WindowsProcess> = {
  docTitle: 'Windows Process · Thorium',
  title: 'Windows Processes',
  typeLabel: '',
  kind: Entities.WindowsProcess,
  creatable: true,
  entityHeaders: <WindowsProcessListHeader />,
  renderEntity: (entity) => <WindowsProcessItem process={entity} />,
  fetchEntities: getWindowsProcesses,
};

export default WindowsProcessesConfig;
