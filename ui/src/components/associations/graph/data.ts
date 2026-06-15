// project imports
import { formatSubmissionNames, formatTagNames, getEdgeLabel } from '../utilities';
import { getNodeSize, scoreNode } from '../shared/scaling';
import { isValidSha256 } from '@utilities/files';
import { Graph, BranchNode, Direction, NodeType, TreeNodeKey } from '@models/trees';
import { Entities, ENTITY_LABELS } from '@models/entities/entities';
import { VisualState } from './types';
import type { GraphNode, GraphLink, GraphData } from './types';

// spec: ./AssociationGraph.spec.md

export const getLinkEndpoints = (link: GraphLink): { source: string; target: string } => {
  const source = typeof link.source === 'object' ? (link.source as GraphNode).id : link.source;
  const target = typeof link.target === 'object' ? (link.target as GraphNode).id : link.target;
  return { source, target };
};

export const classifyNode = (
  nodeId: string,
  graph: Graph,
  precomputed?: { growableSet: Set<string>; initialSet: Set<string> },
): { nodeType: NodeType; visualState: VisualState; label: string } => {
  const nodeData = graph.data_map[nodeId];
  const growableSet = precomputed?.growableSet ?? new Set(graph.growable.map((n) => n.toString()));
  const initialSet = precomputed?.initialSet ?? new Set(graph.initial.map((n) => n.toString()));
  const isGrowable = growableSet.has(nodeId);
  const isInitial = initialSet.has(nodeId);
  const visualState: VisualState = isGrowable ? VisualState.Growable : isInitial ? VisualState.Initial : VisualState.Basic;

  if (TreeNodeKey.Sample in nodeData) {
    let label = formatSubmissionNames(nodeData.Sample?.submissions ?? []);
    if (label.length > 30) {
      label = label.substring(0, 15) + '...' + label.substring(label.length - 15);
    }
    return { nodeType: NodeType.File, visualState, label };
  } else if (TreeNodeKey.Repo in nodeData) {
    return { nodeType: NodeType.Repo, visualState, label: nodeData.Repo?.url ?? '' };
  } else if (TreeNodeKey.Tag in nodeData) {
    return { nodeType: NodeType.Tag, visualState, label: formatTagNames(nodeData.Tag?.tags ?? {}, true) };
  } else if (nodeData.Entity?.kind && Object.values(Entities).includes(nodeData.Entity.kind)) {
    let label = nodeData.Entity.name;
    // Windows process tree names are sha256 hashes, which are meaningless in the graph;
    // show the readable type label instead when the name is a bare hash.
    if (nodeData.Entity.kind === Entities.WindowsProcessTree && isValidSha256(label)) {
      label = ENTITY_LABELS[Entities.WindowsProcessTree];
    }
    return { nodeType: nodeData.Entity.kind, visualState, label };
  }
  return { nodeType: NodeType.Other, visualState, label: 'Unknown' };
};

export const buildGraphNode = (
  nodeId: string,
  graph: Graph,
  nodeCount: number,
  degree = 0,
  precomputed?: { growableSet: Set<string>; initialSet: Set<string> },
): GraphNode => {
  const { nodeType, visualState, label } = classifyNode(nodeId, graph, precomputed);
  const score = scoreNode(graph.data_map[nodeId]);
  return {
    id: nodeId,
    label,
    nodeType,
    visualState,
    score,
    diameter: getNodeSize(score, nodeCount),
    degree,
  };
};

export const processInitialGraphData = (graph: Graph): GraphData => {
  const seenNodes = new Set<string>();
  const seenEdges = new Set<string>();
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const nodeCount = Object.keys(graph.data_map).length;

  const precomputed = {
    growableSet: new Set(graph.growable.map((n) => n.toString())),
    initialSet: new Set(graph.initial.map((n) => n.toString())),
  };

  const degreeCounts = new Map<string, number>();
  const incrementDegree = (id: string) => degreeCounts.set(id, (degreeCounts.get(id) ?? 0) + 1);

  Object.keys(graph.branches).forEach((nodeKey) => {
    graph.branches[nodeKey].forEach((descendant: BranchNode) => {
      incrementDegree(nodeKey);
      incrementDegree(descendant.node.toString());
    });
  });

  const addNode = (nodeId: string) => {
    if (seenNodes.has(nodeId)) return;
    seenNodes.add(nodeId);
    nodes.push(buildGraphNode(nodeId, graph, nodeCount, degreeCounts.get(nodeId) ?? 0, precomputed));
  };

  const addEdge = (source: string, target: BranchNode) => {
    const targetNode = target.direction === Direction.To ? target.node.toString() : source;
    const sourceNode = target.direction === Direction.To ? source : target.node.toString();
    // A bidirectional association is stored on both endpoints, so it surfaces as two reverse branches
    // (source->target and target->source) once both nodes are present. Their `relationship_hash` is the
    // same (hashed direction-independently server-side), so canonicalize the endpoint order for the dedup
    // key to collapse the reverse pair into a single link instead of drawing it twice.
    const endpoints =
      target.direction === Direction.Bidirectional ? [sourceNode, targetNode].sort().join('-') : `${sourceNode}-${targetNode}`;
    const edgeKey = `${endpoints}-${target.relationship_hash}`;
    if (seenEdges.has(edgeKey)) return;
    seenEdges.add(edgeKey);
    links.push({
      source: sourceNode,
      target: targetNode,
      label: getEdgeLabel(targetNode, sourceNode, target, graph),
      bidirectional: target.direction === Direction.Bidirectional,
    });
  };

  graph.initial.forEach((initialNode) => addNode(initialNode.toString()));

  Object.keys(graph.branches).forEach((nodeKey) => {
    addNode(nodeKey.toString());
    graph.branches[nodeKey].forEach((descendant: BranchNode) => {
      addNode(descendant.node.toString());
      addEdge(nodeKey, descendant);
    });
  });

  return { nodes, links };
};
