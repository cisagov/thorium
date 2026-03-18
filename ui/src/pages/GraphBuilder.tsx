import React, { createContext, useContext, useState } from 'react';
import { Card, Form } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
const AssociationTree = React.lazy(() => import('../components/associations/AssociationTree'));
const AssociationGraph = React.lazy(() => import('../components/associations/graph/AssociationGraph'));
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import { Seed } from '@models/trees';

interface GraphBuilderContextType {
  updateSeed: (seed: Seed) => void | undefined;
}

// Page context
const GraphContext = createContext<GraphBuilderContextType | undefined>(undefined);

// custom device create context hook
const useGraphContext = () => {
  const context = useContext(GraphContext);
  if (context === undefined) {
    throw new Error('useRepoContext must be used within a RepoContextProvider');
  }
  return context;
};

const InputTitle = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
`;

const RepoHeader = () => {
  const { updateSeed } = useGraphContext();
  return (
    <Card className="panel">
      <Card.Body>
        <InputTitle>Graph Seed</InputTitle>
        <Form.Control onChange={(e) => updateSeed(JSON.parse(e.target.value) as unknown as Seed)} />
      </Card.Body>
    </Card>
  );
};

const GraphBuilder = () => {
  const [seed, updateSeed] = useState<Seed | null>(null);
  return (
    <GraphContext.Provider value={{ updateSeed }}>
      <Page className="full-min-width" title={`Graph Builder`}>
        <RepoHeader />
        <Card className="panel">
          <Card.Body>
            <Subtitle className="text-center">Association Graph</Subtitle>
            {seed != null && <AssociationGraph inView initial={seed} />}
          </Card.Body>
        </Card>
        <Card className="panel">
          <Card.Body>
            <Subtitle className="text-center">Association Tree</Subtitle>
            {seed != null && <AssociationTree initial={seed} />}
          </Card.Body>
        </Card>
      </Page>
    </GraphContext.Provider>
  );
};

export default GraphBuilder;
