import React, { createContext, useContext } from 'react';
import { Card } from 'react-bootstrap';
import { useParams } from 'react-router';
import { FaServer } from 'react-icons/fa';
import styled from 'styled-components';

// project imports
const AssociationTree = React.lazy(() => import('../../components/associations/AssociationTree'));
const AssociationGraph = React.lazy(() => import('../../components/associations/graph/AssociationGraph'));
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import Title from '@components/shared/titles/Title';

interface RepoDetailsContextType {
  repo: string | undefined; // full url for repo page is displaying
}

// Page context
const RepoContext = createContext<RepoDetailsContextType | undefined>(undefined);

// custom device create context hook
const useRepoContext = () => {
  const context = useContext(RepoContext);
  if (context === undefined) {
    throw new Error('useRepoContext must be used within a RepoContextProvider');
  }
  return context;
};

const IconTitle = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
`;

const RepoHeader = () => {
  const { repo } = useRepoContext();
  return (
    <Card className="panel">
      <Card.Body>
        <IconTitle>
          <FaServer size="72" className="icon" />
          <Title className="title">{repo}</Title>
        </IconTitle>
      </Card.Body>
    </Card>
  );
};

const RepoDetails = () => {
  const { '*': repo } = useParams<{ '*': string }>();

  return (
    <RepoContext.Provider value={{ repo }}>
      <Page className="full-min-width" title={`Repo · ${repo}`}>
        <RepoHeader />
        <Card className="panel">
          <Card.Body>
            <Subtitle className="text-center">Association Graph</Subtitle>
            <AssociationGraph inView initial={{ repos: [repo ? repo : ''] }} />
          </Card.Body>
        </Card>
        <Card className="panel">
          <Card.Body>
            <Subtitle className="text-center">Association Tree</Subtitle>
            <AssociationTree initial={{ repos: [repo ? repo : ''] }} />
          </Card.Body>
        </Card>
      </Page>
    </RepoContext.Provider>
  );
};

export default RepoDetails;
