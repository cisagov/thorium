// a cytoscape graph edge
export interface Edge {
  data: {
    source: string;
    target: string;
    label: string;
  };
  classes?: string[];
}

// a cytoscape graph node
export interface Node {
  data: {
    id: string;
    label: string;
    diameter: number;
  };
  classes?: string[];
}
