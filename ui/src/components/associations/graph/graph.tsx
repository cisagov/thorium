import React, { useEffect, useReducer, useRef, useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import cytoscape from 'cytoscape';
import styled from 'styled-components';
/* @ts-ignore */
import fcose from 'cytoscape-fcose';
/* @ts-ignore */
import coseBilkent from 'cytoscape-cose-bilkent';
/* @ts-ignore */
import cise from 'cytoscape-cise';
/* @ts-ignore */
import elk from 'cytoscape-elk';
/* @ts-ignore */
import svg from 'cytoscape-svg';

// load cytoscape plugins
cytoscape.use(fcose);
cytoscape.use(coseBilkent);
cytoscape.use(cise);
cytoscape.use(elk);
cytoscape.use(svg);

// project imports
import { Seed, Graph, BranchNode, Direction, BlankGraph } from '@models';
import { RenderErrorAlert } from '@components';
import { getInitialTree, growTree } from '@thorpi';
import { isNode, Node, NodeInfo } from './nodes';
import { Edge, EdgeInfo, isEdge } from './edges';
import { buildStyleSheet } from './styles';
import { formatSubmissionNames, formatTagNames, getEdgeLabel } from '../utilities';
import { getLayout, GraphLayout } from './layout';
import { getNodeSize, scoreNode } from './scaling';
import { DisplayAction, GraphControls } from './controls';

// cytoscape graph elements are either nodes or edges
export type GraphElement = Node | Edge;

// Add node to the graph
const addNode = (
  node: string,
  nodes: string[],
  graph: Graph,
  elements: GraphElement[],
  nodeCount: number,
  showNodeLabels: boolean,
): void => {
  if (!nodes.includes(node)) {
    const nodeData = graph.data_map[node];
    const classes = showNodeLabels ? ['has-node-label'] : [];
    const growable: string[] = graph.growable.map((node) => node.toString());
    const initial: string[] = graph.initial.map((node) => node.toString());
    let isInitialNode = false;

    // initial nodes are max size and have special icon colors
    if (initial.includes(node)) {
      classes.push('root-node');
      isInitialNode = true;
    }
    // Samples are files and have use their submission name rather than hash
    if ('Sample' in nodeData) {
      let submissionNames = formatSubmissionNames(nodeData.Sample?.submissions ? nodeData.Sample.submissions : []);
      if (submissionNames.length > 30) {
        submissionNames = submissionNames.substring(0, 15) + '...' + submissionNames.substring(submissionNames.length - 15);
      }
      if (growable.includes(node)) {
        classes.push('growable-file');
      } else if (isInitialNode) {
        classes.push('initial-file');
      } else {
        classes.push('basic-file');
      }
      elements.push({
        data: {
          id: node,
          label: submissionNames,
          diameter: getNodeSize(scoreNode(graph.data_map[node]), nodeCount),
        },
        classes: classes,
      });
      // Repos are labeled by their URL
    } else if ('Repo' in nodeData) {
      if (growable.includes(node)) {
        classes.push('growable-repo');
      } else if (isInitialNode) {
        classes.push('initial-repo');
      } else {
        classes.push('basic-repo');
      }
      elements.push({
        data: {
          id: node,
          label: nodeData.Repo?.url ? nodeData.Repo?.url : '',
          diameter: getNodeSize(scoreNode(graph.data_map[node]), nodeCount),
        },
        classes: classes,
      });
      // Tag
    } else if ('Tag' in nodeData) {
      if (growable.includes(node)) {
        classes.push('growable-tag');
      } else if (isInitialNode) {
        classes.push('initial-tag');
      } else {
        classes.push('basic-tag');
      }
      elements.push({
        data: {
          id: node,
          label: formatTagNames(nodeData.Tag?.tags ? nodeData.Tag.tags : {}, true),
          diameter: getNodeSize(scoreNode(graph.data_map[node]), nodeCount),
        },
        classes: classes,
      });
    } else if ('Entity' in nodeData && nodeData.Entity?.kind == 'Device') {
      if (growable.includes(node)) {
        classes.push('growable-device');
      } else if (isInitialNode) {
        classes.push('initial-device');
      } else {
        classes.push('basic-device');
      }
      elements.push({
        data: {
          id: node,
          label: nodeData.Entity.name,
          diameter: getNodeSize(scoreNode(graph.data_map[node]), nodeCount),
        },
        classes: classes,
      });
    } else if ('Entity' in nodeData && nodeData.Entity?.kind == 'Vendor') {
      if (growable.includes(node)) {
        classes.push('growable-vendor');
      } else if (isInitialNode) {
        classes.push('initial-vendor');
      } else {
        classes.push('basic-vendor');
      }
      elements.push({
        data: {
          id: node,
          label: nodeData.Entity.name,
          diameter: getNodeSize(scoreNode(graph.data_map[node]), nodeCount),
        },
        classes: classes,
      });
    } else {
      if (growable.includes(node)) {
        classes.push('growable-default');
      } else if (isInitialNode) {
        classes.push('initial-default');
      } else {
        classes.push('basic-default');
      }
      elements.push({
        data: {
          id: node,
          label: 'Unknown',
          diameter: getNodeSize(scoreNode(graph.data_map[node]), nodeCount),
        },
        classes: classes,
      });
    }
    nodes.push(node);
  }
};

const addEdge = (
  source: string,
  target: BranchNode,
  edges: string[],
  elements: GraphElement[],
  initialLabel: boolean,
  graph: Graph,
): void => {
  const targetNode = target.direction == Direction.To ? target.node.toString() : source;
  const sourceNode = target.direction == Direction.To ? source : target.node.toString();
  // check if edge has already been added to the graph
  const edgeKey: string = `${sourceNode}-${targetNode}-${target.relationship_hash}`;
  if (edges.includes(edgeKey) && target.direction != Direction.Bidirectional) {
    return;
  }
  // add edge style css selectors
  const classes: string[] = ['bidirectional-edge'];
  if (target.direction == Direction.Bidirectional) {
    classes.push('bidirectional');
  }
  if (initialLabel) {
    classes.push('has-edge-label');
  }
  console.log(target);
  elements.push({
    data: { source: sourceNode, target: targetNode, label: getEdgeLabel(targetNode, sourceNode, target, graph) },
    classes: classes,
  });
  edges.push(edgeKey);
};

const grow = async (
  graphId: string,
  source: string,
  controls: GraphControls,
  cyInstance: React.RefObject<cytoscape.Core>,
  nodesRef: React.RefObject<string[]>,
  edgesRef: React.RefObject<string[]>,
  graphRef: React.RefObject<Graph>,
  updateNodeCount: (count: number) => void,
) => {
  const growable: string[] = graphRef.current.growable;
  const nodes: string[] = nodesRef.current;
  const edges: string[] = edgesRef.current;
  // if not growable, return
  if (!growable.map((node) => node.toString()).includes(source)) {
    return;
  }
  await growTree(graphId, [source], console.log).then((data) => {
    const newNodes: GraphElement[] = [];
    const newEdges: GraphElement[] = [];
    // update nodes
    if (data?.data_map) {
      Object.keys(data.data_map).map((newNode) => {
        const descendantKey = newNode.toString();
        // add data map node info back to original graph
        graphRef.current.data_map[newNode] = data.data_map[newNode];
        // add nodes to cytoscape graph elements
        addNode(
          descendantKey,
          nodes,
          data as Graph,
          newNodes,
          Object.keys(data.data_map).length + cyInstance.current.nodes().size(),
          controls.showNodeLabels,
        );
      });
      // update rendered node count
      updateNodeCount(Object.keys(graphRef.current.data_map).length);
    }
    // update edges
    if (data?.branches) {
      Object.keys(data.branches).map((branch) => {
        // add branches nodes back into original graph
        graphRef.current.branches[branch] = data.branches[branch];
        data.branches[branch].map((target: BranchNode) => {
          // add branches to cytoscape graph elements
          addEdge(branch.toString(), target, edges, newEdges, controls.showEdgeLabels, graphRef.current);
        });
      });
    }
    // remove any growable classes for source
    if (cyInstance?.current && data?.growable && !data.growable.map((node) => node.toString()).includes(source)) {
      const node = cyInstance.current.$(`#${source}`);
      if (node.hasClass('growable-file')) {
        node.removeClass('growable-file');
        node.addClass('basic-file');
      }
      if (node.hasClass('growable-repo')) {
        node.removeClass('growable-repo');
        node.addClass('basic-repo');
      }
      if (node.hasClass('growable-tag')) {
        node.removeClass('growable-tag');
        node.addClass('basic-tag');
      }
      if (node.hasClass('growable-vendor')) {
        node.removeClass('growable-vendor');
        node.addClass('basic-vendor');
      }
      if (node.hasClass('growable-device')) {
        node.removeClass('growable-device');
        node.addClass('basic-device');
      }
    }
    // update nodes that are no longer growable
    if (data?.growable && data.growable.length > 0) {
      const newGrowable: string[] = [...data.growable];
      newGrowable.push(...growable.filter((growTarget: string) => growTarget != source));
      // update original graph growable list
      graphRef.current.growable = newGrowable;
    } else {
      const newGrowable = growable.filter((growTarget: string) => growTarget != source);
      // update original graph growable list
      graphRef.current.growable = newGrowable;
    }
    // update cytoscape elements
    if (cyInstance?.current && (newNodes.length > 0 || newEdges.length > 0)) {
      cyInstance.current.add([...newNodes, ...newEdges]);
      // run layout after adding elements
      if (controls.autoRunLayout) {
        cyInstance.current.layout(getLayout(controls.layoutAlgorithm)).run();
      }
    }
  });
};

// convert tree data
const processInitialGraphData = (graph: Graph, controls: GraphControls) => {
  const nodes: string[] = [];
  const edges: string[] = [];
  const elements: GraphElement[] = [];
  const nodeCount = Object.keys(graph.data_map).length;
  // add initial nodes to graph in case no branches are present
  graph.initial.map((initialNode) => {
    addNode(initialNode.toString(), nodes, graph, elements, nodeCount, controls.showNodeLabels);
  });

  // iterate through branches and add them to map
  Object.keys(graph.branches).map((nodeKey: string) => {
    // add branch node to the nodes list
    addNode(nodeKey.toString(), nodes, graph, elements, nodeCount, controls.showNodeLabels);
    // Add any descendants of the branch. Descendants can be leafs or other branch nodes.
    graph.branches[nodeKey].map((descendant: BranchNode) => {
      // descendantKeys need to be cast to strings because they are bigints
      const descendantKey = descendant.node.toString();
      addNode(descendantKey, nodes, graph, elements, nodeCount, controls.showNodeLabels);
      addEdge(nodeKey, descendant, edges, elements, controls.showEdgeLabels, graph);
    });
  });
  return { elements, nodes, edges };
};

const getInitialData = async (
  initial: Seed,
  graphRootRef: React.RefObject<string[]>,
  elementsRef: React.RefObject<GraphElement[]>,
  nodesRef: React.RefObject<string[]>,
  edgesRef: React.RefObject<string[]>,
  graphRef: React.RefObject<Graph>,
  controls: GraphControls,
  updateNodeCount: (count: number) => void,
  updateGraphId: (id: string) => void,
) => {
  await getInitialTree(initial, controls.filterChildless, controls.depth, console.log).then((data) => {
    if (data) {
      updateGraphId(data.id);
      // set initial node to one of the first nodes to be reviewed
      if (data?.initial.length > 0) {
        graphRootRef.current = data.initial;
      }
      const { elements, nodes, edges } = processInitialGraphData(data, controls);
      elementsRef.current = elements;
      nodesRef.current = nodes;
      edgesRef.current = edges;
      graphRef.current = data;
      updateNodeCount(nodes.length);
    }
  });
};

const GraphWindow = styled.div`
  position: relative;
  background-color: var(--thorium-panel-bg);
`;

const GraphDiv = styled.div`
  z-index: 200;
  overflow: visible;
  min-height: 90vh;
  max-height: 90vw;
  padding-top: 1rem;
`;

const DataPreview = styled.div`
  position: absolute;
  z-index: 300;
  top: 0px;
  right: 0px;
  background-color: rgba(255, 255, 255, 0.01) !important;
  padding: 10px;
  max-width: 40vw;
  max-height: 30vh;
  overflow-y: auto;
  overflow-x: auto;
`;

interface AssociationGraphProps {
  initial: Seed;
  inView: boolean;
}

const AssociationGraph: React.FC<AssociationGraphProps> = ({ initial, inView }) => {
  // Global data objects
  const cyRef = useRef<HTMLDivElement>(null);
  const cyInstance = useRef<cytoscape.Core | null>(null);
  const graphRootRef = useRef<string[]>(['']);
  const edgesRef = useRef<string[]>([]);
  const nodesRef = useRef<string[]>([]);
  const elementsRef = useRef<GraphElement[]>([]);
  const graphRef = useRef<Graph>(BlankGraph);
  const [nodeCount, setNodeCount] = useState(0);

  // Graph Data
  const [graphId, setGraphId] = useState<string>('');
  // Graph and Display Controls
  const [controls, updateControls] = useReducer(controlsReducer, {
    filterChildless: false,
    depth: 1,
    showEdgeLabels: false,
    showNodeLabels: true,
    selectedElement: null,
    showNodeInfo: true,
    autoRunLayout: true,
    layoutAlgorithm: GraphLayout.Fcose,
  });

  function controlsReducer(state: GraphControls, action: DisplayAction): GraphControls {
    switch (action.type) {
      case 'showEdgeLabels':
        // toggle the "has-edge-label" css class to trigger display of edge label
        cyInstance?.current?.edges().toggleClass('has-edge-label', action.state);
        return { ...state, showEdgeLabels: action.state };
      case 'showNodeLabels':
        // toggle the "has-node-label" css class to trigger display of node label
        cyInstance?.current?.nodes().toggleClass('has-node-label', action.state);
        return { ...state, showNodeLabels: action.state };
      case 'showNodeInfo':
        return { ...state, showNodeInfo: action.state };
      case 'selected':
        return { ...state, selectedElement: action.state };
      case 'depth':
        return { ...state, depth: action.state };
      case 'filterChildless':
        return { ...state, filterChildless: action.state };
      case 'layoutAlgorithm':
        cyInstance?.current?.layout(getLayout(action.state)).run();
        return { ...state, layoutAlgorithm: action.state };
    }
  }

  const handleNodeSelect = async (event: any) => {
    const nodeId = event.target.id();
    // set selected element to node format
    updateControls({
      type: 'selected',
      state: {
        data: {
          id: nodeId,
          label: event.target.data('label'),
          diameter: 0,
        },
      },
    });
    // handle any potential node growth
    if (cyInstance?.current) {
      await grow(
        graphId,
        nodeId.toString(),
        controls,
        cyInstance as React.RefObject<cytoscape.Core>,
        nodesRef,
        edgesRef,
        graphRef,
        setNodeCount,
      );
    }
  };

  // handle selecting a specific edge
  const handleEdgeSelect = async (event: any) => {
    // set selected element to edge format
    updateControls({
      type: 'selected',
      state: { data: { source: event.target.data('source'), target: event.target.data('target'), label: event.target.data('label') } },
    });
  };

  // control graph render properties once, after new graph data is retrieved
  useEffect(() => {
    if (cyRef.current) {
      cyInstance.current = cytoscape({
        //@ts-ignore
        renderer: {
          name: 'canvas',
          webgl: true,
          showFps: false,
        },
        container: cyRef.current,
        elements: elementsRef.current,
        style: buildStyleSheet(graphRootRef.current),
        layout: getLayout(controls.layoutAlgorithm),
      });

      // control zoom settings to keep graphs from disappearing when you zoom out/in
      cyInstance.current.zoomingEnabled(true);
      cyInstance.current.minZoom(0.0001);
      cyInstance.current.maxZoom(5);
      if (elementsRef.current?.length > 100) {
        controls.autoRunLayout = false;
      }
      // event listener for node clicks
      cyInstance.current.on('tap', 'node', handleNodeSelect);
      // event listener for edge clicks
      cyInstance.current.on('tap', 'edge', handleEdgeSelect);
      cyInstance.current.on('zoom', function () {
        let currentZoom = cyInstance.current?.zoom() ? cyInstance.current?.zoom() : 1;
        let zoomFactor = 1.1 / currentZoom;
        let fontSize = zoomFactor * 10; // defaultFontSize is your desired base font size
        cyInstance.current?.style().selector('node').style('font-size', `${fontSize}`).update();
        cyInstance.current?.style().selector('edge').style('font-size', `${fontSize}`).update();
      });
      // cleanup component on unmount
      return () => {
        cyInstance.current?.destroy();
      };
    }
  }, [graphId]);

  // load data when graph must change (due to limit depth change)
  useEffect(() => {
    // must deselect nodes when changing depth in case decreasing depth doesn't include previously selected node
    updateControls({ type: 'selected', state: null });
    // get fresh graph when changing depth, initial resources or controls that affect graph traversal
    getInitialData(initial, graphRootRef, elementsRef, nodesRef, edgesRef, graphRef, controls, setNodeCount, setGraphId);
  }, [controls.filterChildless, controls.depth, initial]);

  // resize when parent says graph in view
  useEffect(() => {
    if (inView && cyInstance.current) {
      cyInstance.current.layout(getLayout(controls.layoutAlgorithm)).run();
    }
  }, [inView]);

  return (
    <GraphWindow>
      <GraphDiv ref={cyRef} />
      <div className="ps-4 ms-2">Node Count: {nodeCount}</div>
      <GraphControls graphId={graphId} controls={controls} updateControls={updateControls} cyInstance={cyInstance} />
      <DataPreview>
        {controls.selectedElement && isNode(controls.selectedElement) && controls.selectedElement.data.id != '' && (
          <NodeInfo node={graphRef.current.data_map[controls.selectedElement.data.id]} />
        )}
        {controls.selectedElement && isEdge(controls.selectedElement) && <EdgeInfo edge={controls.selectedElement} />}
      </DataPreview>
    </GraphWindow>
  );
};

export const AssociationGraphWrapper: React.FC<AssociationGraphProps> = ({ initial, inView }) => {
  return (
    <ErrorBoundary fallback={<RenderErrorAlert page={false} />}>
      {inView && <AssociationGraph inView={inView} initial={initial} />}
    </ErrorBoundary>
  );
};

export default AssociationGraphWrapper;
