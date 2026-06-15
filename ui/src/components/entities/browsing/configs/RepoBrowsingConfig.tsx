import React from 'react';
import { Link } from 'react-router';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { EntityBrowseConfig } from './config';
import { BrowsingCard, BrowsingContents, LinkFields } from '@entities/browsing/shared';
import { listRepos } from '@thorpi/repos';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';
import { Repo } from '@models/repos';

// spec: ../EntityBrowsing.spec.md

// get repos using filters and an optional cursor
const getRepos = async (filters: Filters, cursor: string | null) => {
  // request full repo details so the listing can render repo metadata
  const { entityList, entityCursor } = await listRepos(filters, console.log, true, cursor);
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
  repo: Repo;
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
  // Repos are ingested, not hand-created in the UI (no create route exists for them)
  creatable: false,
  entityHeaders: <RepoListHeaders />,
  renderEntity: (repo) => <RepoItem repo={repo} />,
  fetchEntities: getRepos,
};

export default RepoBrowsingConfig;
