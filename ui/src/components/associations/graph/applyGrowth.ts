import type { ForceGraph3DInstance } from '3d-force-graph';

// project imports
import { getLinkEndpoints } from './data';
import type { GraphData } from './types';
import type { LabelEntry } from './controls/controlsReducer';

export const applyGrowthToInstance = (
  prevData: GraphData,
  newData: GraphData,
  graphInstanceRef: React.RefObject<ForceGraph3DInstance | null>,
  labelSpritesRef: React.RefObject<Map<string, LabelEntry>>,
  graphDataRef: React.RefObject<GraphData>,
  setNodeCount: (count: number) => void,
) => {
  const gi = graphInstanceRef.current;
  if (!gi) return;

  const existingEdgeKeys = new Set(
    prevData.links.map((l) => {
      const { source, target } = getLinkEndpoints(l);
      return `${source}-${target}`;
    }),
  );

  const addedLinks = newData.links.filter((l) => {
    const { source, target } = getLinkEndpoints(l);
    return !existingEdgeKeys.has(`${source}-${target}`);
  });

  const newNodeMap = new Map(newData.nodes.map((n) => [n.id, n]));
  const existingNodeIds = new Set(prevData.nodes.map((n) => n.id));
  const existingNodeMap = new Map(prevData.nodes.map((n) => [n.id, n]));
  const addedNodes = newData.nodes.filter((n) => !existingNodeIds.has(n.id));

  let stateChanged = false;
  const updatedExistingNodes = prevData.nodes.map((n) => {
    const updated = newNodeMap.get(n.id);
    if (updated && updated.visualState !== n.visualState) {
      stateChanged = true;
      return { ...n, visualState: updated.visualState };
    }
    return n;
  });

  if (addedNodes.length === 0 && addedLinks.length === 0 && !stateChanged) return;

  const POSITION_JITTER = 30;
  for (const node of addedNodes) {
    const link = addedLinks.find((l) => {
      const { source, target } = getLinkEndpoints(l);
      return (source === node.id && existingNodeMap.has(target)) || (target === node.id && existingNodeMap.has(source));
    });
    if (link) {
      const { source, target } = getLinkEndpoints(link);
      const parent = existingNodeMap.get(source === node.id ? target : source);
      if (parent?.x !== undefined) {
        node.x = parent.x + (Math.random() - 0.5) * POSITION_JITTER;
        node.y = (parent.y ?? 0) + (Math.random() - 0.5) * POSITION_JITTER;
        node.z = (parent.z ?? 0) + (Math.random() - 0.5) * POSITION_JITTER;
      }
    }
  }

  const normalizedLinks = stateChanged
    ? prevData.links.map((l) => {
        const { source, target } = getLinkEndpoints(l);
        return { ...l, source, target };
      })
    : prevData.links;

  const updatedData: GraphData = {
    nodes: [...updatedExistingNodes, ...addedNodes],
    links: [...normalizedLinks, ...addedLinks],
  };
  graphDataRef.current = updatedData;
  setNodeCount(updatedData.nodes.length);

  gi.graphData(updatedData);

  if (stateChanged) {
    labelSpritesRef.current.clear();
    gi.nodeThreeObject(gi.nodeThreeObject());
    gi.refresh();
  }
};
