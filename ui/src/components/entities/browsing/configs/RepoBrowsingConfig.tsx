import React from 'react';
import { Link } from 'react-router';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { EntityBrowseConfig } from './config';
import { BrowsingCard, BrowsingContents, LinkFields } from '@entities/browsing/shared';
import { listRepos } from '@thorpi/repos';
import { Filters } from '@models/search';
import { Repo } from '@models/entities/repos';
import { Entities } from '@models/entities/entities';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// get repos using filters and and an optional cursor
const getRepos = async (filters: Filters, cursor: string | null) => {
  // get files list from API
  const { entityList, entityCursor } = await listRepos(
    filters,
    console.log,
    true, // details bool
    cursor,
  );
  return {
    entitiesList: entityList,
    entitiesCursor: entityCursor,
  };
};

const Name = styled(Col)`
  white-space: pre-wrap;
  text-align: center;
  flex-wrap: wrap;
  word-break: break-all;
  min-width: 650px;
  color: var(--thorium-text);
`;

const Submissions = styled(Col)`
  min-width: 100px;
  text-align: center;
  color: var(--thorium-text);
`;

const Providers = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
`;

const RepoListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <Name>Repo</Name>
          <Submissions>Submission(s)</Submissions>
          <Providers>Provider(s)</Providers>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface RepoItemProp {
  repo: any; // Repo details
}

const RepoItem: React.FC<RepoItemProp> = ({ repo }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.Repo)}/${repo.url}`} state={{ repo: repo }} className="no-decoration">
          <LinkFields>
            <Name>{repo.name}</Name>
            <Submissions>{JSON.stringify(repo.submissions.length)}</Submissions>
            <Providers>{JSON.stringify(repo.provider)}</Providers>
          </LinkFields>
        </Link>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const RepoBrowsingConfig: EntityBrowseConfig<Entities.Repo> = {
  docTitle: 'Repos · Thorium',
  title: 'Repos',
  typeLabel: '',
  kind: Entities.Repo,
  creatable: true,
  entityHeaders: <RepoListHeaders />,
  renderEntity: (repo) => <RepoItem repo={repo} />,
  fetchEntities: getRepos,
};

export default RepoBrowsingConfig;
