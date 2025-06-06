import React from 'react';
import { Node } from './nodes';
import { Row } from 'react-bootstrap';
import { Subtitle } from '@components';

// a cytoscape graph edge
export interface Edge {
  data: {
    source: string;
    target: string;
    label: string;
  };
  classes?: string[];
}

export function isEdge(element: Node | Edge): element is Edge {
  return 'source' in element?.data && 'target' in element?.data;
}

interface EdgeInfoProps {
  edge: Edge; // arbitrary Thorium node data
}

export const EdgeInfo: React.FC<EdgeInfoProps> = ({ edge }) => {
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
