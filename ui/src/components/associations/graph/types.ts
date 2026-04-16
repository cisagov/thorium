import type { NodeObject, LinkObject } from '3d-force-graph';

// project imports
import { NodeType } from '@models/trees';

export type VisualState = 'basic' | 'growable' | 'initial';

export interface GraphNode extends NodeObject {
  id: string;
  label: string;
  nodeType: NodeType;
  visualState: VisualState;
  score: number;
  diameter: number;
  degree: number;
}

export interface GraphLink extends LinkObject<GraphNode> {
  source: string;
  target: string;
  label: string;
  bidirectional: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}
