import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { BrowsingCard, BrowsingContents, LinkFields } from '@entities/browsing/shared';
import BrowsingFilters from '@entities/browsing/filters/BrowsingFilters';
import EntityList from '@entities/browsing/EntityList';
import Page from '@components/pages/Page';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { useAuth } from '@utilities/auth';
import { FileSystem } from '@models/entities/file_systems';
import { Entities } from '@models/entities/entities';
import { Filters } from '@models/search';
import { listEntities } from '@thorpi/entities';
import { scaling } from '@styles';

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
const getFileSystems = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  // reset cursor when filters have changed, caller must know this
  // get files list from API
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.FileSystem];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return {
    entitiesList: entityList,
    entitiesCursor: entityCursor,
  };
};

const FileSystemListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <Name>File</Name>
          <Groups>Group(s)</Groups>
          <Submitters>Submitter(s)</Submitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface FileSystemItemProps {
  fs: FileSystem; // device details
}

const FileSystemItem: React.FC<FileSystemItemProps> = ({ fs }) => {
  console.log(fs);
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`/filesystem/${fs.id}`} state={{ filesystem: fs }} className="no-decoration">
          <LinkFields>
            <Name>{fs.name}</Name>
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
            <Submitters>
              {fs.submitter ? (
                <small>
                  <i>{fs.submitter.length > 75 ? fs.submitter.substring(0, 75) + '...' : fs.submitter}</i>
                </small>
              ) : null}
            </Submitters>
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

const FileSystemBrowsing = () => {
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<Filters>({});
  const { userInfo } = useAuth();

  return (
    <Page title={`File Systems · Thorium`}>
      <BrowsingFilters
        title="File Systems"
        kind={Entities.FileSystem}
        onChange={setFilters}
        groups={userInfo?.groups ? userInfo.groups : []}
        disabled={loading}
        creatable={false}
      />
      <EntityList
        type="FileSystem"
        entityHeaders={<FileSystemListHeaders />}
        displayEntity={(fs) => <FileSystemItem fs={fs} />}
        filters={filters}
        fetchEntities={getFileSystems}
        setLoading={setLoading}
        loading={loading}
      />
    </Page>
  );
};

export default FileSystemBrowsing;
