import React from 'react';
import { Link } from 'react-router-dom';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { EntityBrowseConfig } from './config';
import { BrowsingCard, BrowsingContents, LinkFields } from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { listEntities } from '@thorpi/entities';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
import { FileSystem } from '@models/entities/file_systems';
import { scaling } from '@styles';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

const Name = styled(Col)`
  white-space: pre-wrap;
  word-break: break-all;
  min-width: 600px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.md}) {
    min-width: 70%;
  }
  @media (max-width: ${scaling.sm}) {
    min-width: 300px;
  }
`;

const Groups = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xl}) {
    display: none !important;
  }
`;

const Tools = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.lg}) {
    display: none !important;
  }
`;

const Sha256 = styled.div`
  font-size: 0.8rem;
  font-style: italic;
`;

// get files using filters and and an optional cursor
const getFileSystems = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.FileSystem];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList as FileSystem[],
    entitiesCursor: entityCursor,
  };
};

const FileSystemListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <Name>File System</Name>
          <Tools>Tools(s)</Tools>
          <Groups>Group(s)</Groups>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface FileSystemItemProps {
  fs: FileSystem; // device details
}

const FileSystemItem: React.FC<FileSystemItemProps> = ({ fs }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.FileSystem)}/${fs.id}`} state={{ filesystem: fs }} className="no-decoration">
          <LinkFields>
            <Name>{fs.name}</Name>
            <Tools>
              {fs.metadata.FileSystem ? (
                <i>
                  {fs.metadata.FileSystem.tools &&
                    (fs.metadata.FileSystem.tools.toString().length > 75
                      ? fs.metadata.FileSystem.tools.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : fs.metadata.FileSystem.tools.toString().replaceAll(',', ', '))}
                </i>
              ) : (
                ''
              )}
            </Tools>
            <Groups>
              <small>
                <i>
                  {fs.groups &&
                    (fs.groups.toString().length > 75
                      ? fs.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : fs.groups.toString().replaceAll(',', ', '))}
                </i>
              </small>
            </Groups>
          </LinkFields>
        </Link>
        <Sha256 className="mt-3 mb-2">{fs.metadata.FileSystem.sha256}</Sha256>
        {fs.tags != undefined && <hr />}
        <Row>
          {fs.tags && Object.keys(fs.tags).length > 1 ? <CondensedEntityTags resource={Entities.FileSystem} tags={fs.tags} /> : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const FileSystemsConfig: EntityBrowseConfig<Entities.FileSystem> = {
  docTitle: 'File Systems · Thorium',
  title: 'File Systems',
  typeLabel: '',
  kind: Entities.FileSystem,
  creatable: false,
  entityHeaders: <FileSystemListHeaders />,
  renderEntity: (entity) => <FileSystemItem fs={entity} />,
  fetchEntities: getFileSystems,
};

export default FileSystemsConfig;
