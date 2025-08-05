import React, { useState } from 'react';
import { Link } from 'react-router';
import { Card, Col, Row } from 'react-bootstrap';

// project imports
import { BrowsingFilters, EntityList, Page } from '@components';
import { useAuth } from '@utilities';
import { listRepos } from '@thorpi';
import { Filters } from '@models';

// get repos using filters and an optional cursor
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

const RepoListHeaders = () => {
  return (
    <Card className="basic-card panel">
      <Card.Body>
        <Row>
          <Col className="d-flex justify-content-center">Repo</Col>
          <Col className="d-flex justify-content-center">Submission(s)</Col>
          <Col className="d-flex justify-content-center">Provider(s)</Col>
        </Row>
      </Card.Body>
    </Card>
  );
};

interface RepoItemProp {
  repo: any; // repo details
}

const RepoItem: React.FC<RepoItemProp> = ({ repo }) => {
  return (
    <Card className="basic-card panel">
      <Card.Body>
        <Link to={`/repo/${repo.url}`} state={{ repo: repo }} className="no-decoration">
          <Row className="highlight-card">
            <Col>{repo.name}</Col>
            <Col>{JSON.stringify(repo.submissions.length)}</Col>
            <Col>{JSON.stringify(repo.provider)}</Col>
          </Row>
        </Link>
      </Card.Body>
    </Card>
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
