import type { ForceGraph3DInstance } from '3d-force-graph';

export type NodeRenderMode = 'spheres' | 'icons';
export type DagMode = 'td' | 'bu' | 'lr' | 'rl' | 'zout' | 'zin' | 'radialout' | 'radialin' | null;

export interface GraphControls {
  filterChildless: boolean;
  depth: number;
  showEdgeLabels: boolean;
  showNodeLabels: boolean;
  selectedElement: SelectedElement | null;
  showNodeInfo: boolean;
  nodeRenderMode: NodeRenderMode;
  focusOnClick: boolean;
  adjustDistanceOnFocus: boolean;
  refitOnGrow: boolean;
  focusDistanceRatio: number;
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
  nodeLabelMinSize: number;
  edgeLabelDensity: number;
  edgeLabelMinSize: number;
  // forces
  chargeStrength: number;
  velocityDecay: number;
  warmupTicks: number;
  cooldownTime: number;
  // layout
  dagMode: DagMode;
  dagLevelDistance: number | null;
  numDimensions: 2 | 3;
  showGrid: boolean;
}

export type SelectedElement = { kind: 'node'; id: string; label: string } | { kind: 'link'; source: string; target: string; label: string };

export type DisplayAction =
  | { type: 'depth'; state: number }
  | {
      type:
        | 'filterChildless'
        | 'showEdgeLabels'
        | 'showNodeLabels'
        | 'showNodeInfo'
        | 'focusOnClick'
        | 'enableNodeDrag'
        | 'adjustDistanceOnFocus'
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
      type:
        | 'nodeRelSize'
        | 'nodeOpacity'
        | 'focusDistanceRatio'
        | 'nodeLabelScale'
        | 'edgeLabelScale'
        | 'nodeLabelDensity'
        | 'nodeLabelMinSize'
        | 'edgeLabelDensity'
        | 'edgeLabelMinSize';
      state: number;
    }
  | { type: 'chargeStrength' | 'velocityDecay' | 'warmupTicks' | 'cooldownTime'; state: number }
  | { type: 'dagMode'; state: DagMode }
  | { type: 'dagLevelDistance'; state: number | null }
  | { type: 'numDimensions'; state: 2 | 3 };

export type SectionKey = 'graph' | 'forces' | 'nodes' | 'edges' | 'export';

export type SectionProps = {
  controls: GraphControls;
  updateControls: React.ActionDispatch<[action: DisplayAction]>;
};

export type GraphSectionProps = SectionProps & {
  graphId: string;
  graphInstance: ForceGraph3DInstance | null;
};
