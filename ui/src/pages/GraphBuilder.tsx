import React, { createContext, useContext, useState } from 'react';
import { Card, Form } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
const AssociationGraph = React.lazy(() => import('../components/associations/graph/AssociationGraph'));
import { GraphDataProvider } from '../components/associations/data/GraphDataContext';
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import { Seed } from '@models/trees';

interface GraphBuilderContextType {
  updateSeed: (seed: Seed) => void;
}

// Page context
const GraphContext = createContext<GraphBuilderContextType | undefined>(undefined);

const useGraphContext = () => {
  const context = useContext(GraphContext);
  if (context === undefined) {
    throw new Error('useGraphContext must be used within a GraphBuilder');
  }
  return context;
};

const InputTitle = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
`;

const GraphHeader = () => {
  const { updateSeed } = useGraphContext();
  return (
    <Card className="panel">
      <Card.Body>
        <InputTitle>Graph Seed</InputTitle>
        <Form.Control
          onChange={(e) => {
            try {
              updateSeed(JSON.parse(e.target.value) as Seed);
            } catch {
              // Wait for valid JSON
            }
          }}
        />
      </Card.Body>
    </Card>
  );
};

const GraphBuilder = () => {
  const [seed, updateSeed] = useState<Seed | null>(null);
  return (
    <GraphContext.Provider value={{ updateSeed }}>
      <Page className="full-min-width" title={`Graph Builder`}>
        <GraphHeader />
        {seed !== null && (
          <GraphDataProvider initial={seed}>
            <Card className="panel">
              <Card.Body>
                <Subtitle className="text-center">Associations</Subtitle>
                <AssociationGraph inView />
              </Card.Body>
            </Card>
          </GraphDataProvider>
        )}
      </Page>
    </GraphContext.Provider>
  );
};

export default GraphBuilder;
