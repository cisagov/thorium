import React, { useState } from 'react';
import { Link } from 'react-router';
import { Col, Row } from 'react-bootstrap';

// project imports
import { BrowsingCard, BrowsingContents, BrowsingFilters, EntityList, LinkFields, Page } from '@components';
import { useAuth } from '@utilities';
import { listRepos } from '@thorpi';
import { Filters } from '@models';
import styled from 'styled-components';

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
  repo: any; // repo details
}

const RepoItem: React.FC<RepoItemProp> = ({ repo }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`/repo/${repo.url}`} state={{ repo: repo }} className="no-decoration">
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

const RepoBrowsingContainer = () => {
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<Filters>({});
  const { userInfo } = useAuth();
  return (
    <Page title="Repositories · Thorium">
      <BrowsingFilters title="Repos" onChange={setFilters} groups={userInfo ? userInfo.groups : []} disabled={loading} />
      <EntityList
        type="repos"
        entityHeaders={<RepoListHeaders />}
        displayEntity={(repo) => <RepoItem repo={repo} />}
        filters={filters}
        fetchEntities={getRepos}
        setLoading={setLoading}
        loading={loading}
      />
    </Page>
  );
};

export default RepoBrowsingContainer;
