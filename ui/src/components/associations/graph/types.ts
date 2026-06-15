import type { NodeObject, LinkObject, ForceGraph3DInstance } from '3d-force-graph';
import type * as THREE from 'three';

// project imports
import { NodeType } from '@models/trees';

// spec: ./AssociationGraph.spec.md

export enum VisualState {
  Basic = 'basic',
  Growable = 'growable',
  Initial = 'initial',
}

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

/** Typed alias for the ForceGraph3D instance parameterized with our node/link types */
export type GraphInstance = ForceGraph3DInstance<GraphNode, GraphLink>;

/** Orbit controls interface for the subset of methods used by the graph */
export interface GraphOrbitControls {
  target: THREE.Vector3;
  zoomToCursor: boolean;
  zoomSpeed: number;
  saveState: () => void;
  addEventListener: (event: string, handler: () => void) => void;
  removeEventListener: (event: string, handler: () => void) => void;
  reset: () => void;
}

/** d3 charge force interface for the methods used by the graph */
export interface D3ChargeForce {
  strength: (val?: number) => D3ChargeForce & number;
}

/** d3 link force interface for the methods used by the graph */
export interface D3LinkForce {
  distance: (val?: number | ((link: GraphLink) => number)) => D3LinkForce & number;
  strength: (val?: number | ((link: GraphLink) => number)) => D3LinkForce & number;
}
