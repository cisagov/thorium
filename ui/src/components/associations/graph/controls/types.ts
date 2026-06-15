import type { GraphInstance } from '../types';

// spec: ./GraphControlsToolbar.spec.md

export enum NodeRenderMode {
  Spheres = 'spheres',
  Icons = 'icons',
}

export enum DagMode {
  TopDown = 'td',
  BottomUp = 'bu',
  LeftRight = 'lr',
  RightLeft = 'rl',
  ZOut = 'zout',
  ZIn = 'zin',
  RadialOut = 'radialout',
  RadialIn = 'radialin',
}

export interface GraphControls {
  filterChildless: boolean;
  depth: number;
  showEdgeLabels: boolean;
  showNodeLabels: boolean;
  selectedElement: SelectedElement | null;
  nodeRenderMode: NodeRenderMode;
  focusOnClick: boolean;
  fitNeighborhoodOnFocus: boolean;
  refitOnGrow: boolean;
  nodeLabelScale: number;
  edgeLabelScale: number;
  // edges
  edgeWidth: number;
  edgeLength: number;
  edgeLinkStrength: number;
  edgeOpacity: number;
  arrowLength: number;
  directionalParticles: number;
  particleSpeed: number;
  // nodes
  nodeRelSize: number;
  nodeOpacity: number;
  enableNodeDrag: boolean;
  nodeLabelDensity: number;
  edgeLabelDensity: number;
  // forces
  chargeStrength: number;
  velocityDecay: number;
  warmupTicks: number;
  cooldownTime: number;
  // layout
  dagMode: DagMode | null;
  dagLevelDistance: number | null;
  numDimensions: 2 | 3;
  showGrid: boolean;
  // size-aware auto-scaling: tracks which keys the user has manually changed
  userOverrides: Set<string>;
}

export type SelectedElement = { kind: 'node'; id: string; label: string } | { kind: 'link'; source: string; target: string; label: string };

export type DisplayAction =
  | { type: 'depth'; state: number }
  | {
      type:
        | 'filterChildless'
        | 'showEdgeLabels'
        | 'showNodeLabels'
        | 'focusOnClick'
        | 'fitNeighborhoodOnFocus'
        | 'enableNodeDrag'
        | 'refitOnGrow'
        | 'showGrid';
      state: boolean;
    }
  | { type: 'selected'; state: SelectedElement | null }
  | { type: 'nodeRenderMode'; state: NodeRenderMode }
  | {
      type: 'edgeWidth' | 'edgeLength' | 'edgeLinkStrength' | 'edgeOpacity' | 'arrowLength' | 'directionalParticles' | 'particleSpeed';
      state: number;
    }
  | {
      type: 'nodeRelSize' | 'nodeOpacity' | 'nodeLabelScale' | 'edgeLabelScale' | 'nodeLabelDensity' | 'edgeLabelDensity';
      state: number;
    }
  | { type: 'chargeStrength' | 'velocityDecay' | 'warmupTicks' | 'cooldownTime'; state: number }
  | { type: 'dagMode'; state: DagMode | null }
  | { type: 'dagLevelDistance'; state: number | null }
  | { type: 'numDimensions'; state: 2 | 3 }
  | { type: 'applySizeDefaults'; state: Partial<GraphControls> }
  | { type: 'resetSizeOverrides' };

export enum SectionKey {
  Graph = 'graph',
  Forces = 'forces',
  Nodes = 'nodes',
  Edges = 'edges',
  Export = 'export',
}

export type SectionProps = {
  controls: GraphControls;
  updateControls: React.ActionDispatch<[action: DisplayAction]>;
};

export type GraphSectionProps = SectionProps & {
  graphId: string;
  graphInstance: GraphInstance | null;
  nodeCount: number;
};
