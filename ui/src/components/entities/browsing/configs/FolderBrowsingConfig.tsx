import React from 'react';
import { Link } from 'react-router-dom';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { EntityBrowseConfig } from './config';
import { BrowsingCard, BrowsingContents, LinkFields } from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { listEntities } from '@thorpi/entities';
import { Folder } from '@models/entities';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
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

const Submitters = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xl}) {
    display: none !important;
  }
`;

const Groups = styled(Col)`
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
const getFolders = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Folder];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList as Folder[],
    entitiesCursor: entityCursor,
  };
};

const FolderListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <Name>Folder</Name>
          <Groups>Group(s)</Groups>
          <Submitters>Submitter(s)</Submitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface FolderItemProps {
  folder: Folder; // folder details
}

const FolderItems: React.FC<FolderItemProps> = ({ folder }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.Folder)}/${folder.id}`} state={{ folder: folder }} className="no-decoration">
          <LinkFields>
            <Name>{folder.name}</Name>
            <Groups>
              <small>
                <i>
                  {folder.groups &&
                    (folder.groups.toString().length > 75
                      ? folder.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : folder.groups.toString().replaceAll(',', ', '))}
                </i>
              </small>
            </Groups>
            <Submitters>
              {folder.submitter ? (
                <small>
                  <i>{folder.submitter.length > 75 ? folder.submitter.substring(0, 75) + '...' : folder.submitter}</i>
                </small>
              ) : null}
            </Submitters>
          </LinkFields>
        </Link>
        <Sha256 className="mt-3 mb-2">{folder.metadata.Folder.all_sha256}</Sha256>
        {folder.tags != undefined && <hr />}
        <Row>
          {folder.tags && Object.keys(folder.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.Folder} tags={folder.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const FolderBrowsingConfig: EntityBrowseConfig<Entities.Folder> = {
  docTitle: 'Folders · Thorium',
  title: 'Folders',
  typeLabel: '',
  kind: Entities.Folder,
  creatable: true,
  entityHeaders: <FolderListHeaders />,
  renderEntity: (entity) => <FolderItems folder={entity} />,
  fetchEntities: getFolders,
};

export default FolderBrowsingConfig;
