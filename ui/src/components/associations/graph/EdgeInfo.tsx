import React from 'react';

// project imports
import { Edge } from './models';
import Subtitle from '@components/shared/titles/Subtitle';

interface EdgeInfoProps {
  edge: Edge; // arbitrary Thorium node data
}

const EdgeInfo: React.FC<EdgeInfoProps> = ({ edge }) => {
  const source: string = edge.data.source;
  const target: string = edge.data.target;
  return (
    <div className="m-2">
      <Subtitle>{edge.data.label}</Subtitle>
      <hr />
      <Subtitle>Source ID</Subtitle>
      {`${source}`}
      <hr />
      <Subtitle>Target ID</Subtitle>
      {`${target}`}
    </div>
  );
};

export default EdgeInfo;
