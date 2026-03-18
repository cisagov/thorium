import { Edge, Node } from './models';

export function isEdge(element: Node | Edge): element is Edge {
  return 'source' in element?.data && 'target' in element?.data;
}

export function isNode(element: Edge | Node): element is Node {
  return 'id' in element?.data;
}
