import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow,
  Handle,
  Position,
  MarkerType,
  ReactFlowProvider,
  useReactFlow,
  applyNodeChanges,
  type Node,
  type Edge,
  type NodeProps,
  type NodeChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { FaExclamationTriangle } from 'react-icons/fa';
import Select from 'react-select';

// project imports
import {
  FlowContainer,
  StepNodeWrapper,
  StepLabel,
  TerminalNodeWrapper,
  BarrierNodeWrapper,
  BanIcon,
  ContextMenuOverlay,
  ContextMenuItem,
  ImageSelectOverlay,
} from './PipelineOrderFlow.styled';
import { edgeTypes } from './ThemedEdge';
import {
  ordersEqual,
  insertImageAtPosition,
  removeImageAtPosition,
  getImagesInOrder,
  estimateStageWidth,
  STEP_WIDTH,
  TERMINAL_OFFSET,
  CLUSTER_THRESHOLD,
} from './order';
import { createReactSelectStyles } from '@utilities/select';
import { listImages } from '@thorpi/images';

const TERMINAL_WIDTH = 12;
const PARALLEL_GAP = 54;
const STEP_HANDLE_APPROX = 14;
const FIT_VIEW_OPTIONS = { padding: 0.3, maxZoom: 1 };

type ImageStepData = { label: string; isParallel: boolean; stepIndex: number; parallelIndex: number; isBanned: boolean };
type ImageStepNode = Node<ImageStepData, 'imageStep'>;
type TerminalData = { label: string };
type TerminalNode = Node<TerminalData, 'terminal'>;
type BarrierData = { height: number };
type BarrierNode = Node<BarrierData, 'barrier'>;
type FlowNode = ImageStepNode | TerminalNode | BarrierNode;

type ContextMenuState = { x: number; y: number; nodeLabel?: string; stepIndex?: number; parallelIndex?: number } | null;
type SelectMenuState = { x: number; y: number; flowX: number } | null;

const EDGE_STYLE = { stroke: 'var(--thorium-secondary-text)', strokeWidth: 1.5 };
const MARKER_END = { type: MarkerType.ArrowClosed, width: 16, height: 16 };

const selectStyles = createReactSelectStyles('White', 'rgb(160, 162, 163)');

function orderToNodesAndEdges(order: (string | string[])[], bannedImages?: Set<string>): { nodes: FlowNode[]; edges: Edge[] } {
  const nodes: FlowNode[] = [];
  const edges: Edge[] = [];

  nodes.push({
    id: 'start',
    type: 'terminal',
    data: { label: 'Start' },
    position: { x: 0, y: 0 },
    draggable: false,
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
  });

  for (let stepIdx = 0; stepIdx < order.length; stepIdx++) {
    const step = order[stepIdx];
    const images = typeof step === 'string' ? [step] : step;
    const yOffset = (-(images.length - 1) * PARALLEL_GAP) / 2;

    for (let i = 0; i < images.length; i++) {
      const id = `step-${stepIdx}-${i}`;
      nodes.push({
        id,
        type: 'imageStep',
        data: {
          label: images[i],
          isParallel: images.length > 1,
          stepIndex: stepIdx,
          parallelIndex: i,
          isBanned: bannedImages?.has(images[i]) ?? false,
        },
        position: { x: TERMINAL_OFFSET + stepIdx * STEP_WIDTH, y: yOffset + i * PARALLEL_GAP },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      });
    }

    if (stepIdx === 0) {
      const isMultiTarget = images.length > 1;
      for (let ci = 0; ci < images.length; ci++) {
        edges.push({
          id: `e-start-${stepIdx}-${ci}`,
          source: 'start',
          target: `step-${stepIdx}-${ci}`,
          type: 'themedStep',
          animated: true,
          markerEnd: MARKER_END,
          style: EDGE_STYLE,
          data: isMultiTarget ? { routeFlat: 'source' as const } : undefined,
        });
      }
    } else {
      const prevStep = order[stepIdx - 1];
      const prevImages = typeof prevStep === 'string' ? [prevStep] : prevStep;

      if (prevImages.length === 1 && images.length === 1) {
        // 1->1: direct edge, no barrier needed
        edges.push({
          id: `e-${stepIdx - 1}-0-${stepIdx}-0`,
          source: `step-${stepIdx - 1}-0`,
          target: `step-${stepIdx}-0`,
          type: 'themedStep',
          animated: true,
          markerEnd: MARKER_END,
          style: EDGE_STYLE,
        });
      } else {
        // Multi: sync barrier between stages
        const maxP = Math.max(prevImages.length, images.length);
        const barrierHeight = Math.max(20, (maxP - 1) * PARALLEL_GAP + 10);
        const barrierX = TERMINAL_OFFSET + (stepIdx - 1) * STEP_WIDTH + (estimateStageWidth(prevStep) + STEP_WIDTH) / 2;
        const barrierId = `barrier-${stepIdx - 1}`;

        nodes.push({
          id: barrierId,
          type: 'barrier',
          data: { height: barrierHeight },
          position: { x: barrierX, y: STEP_HANDLE_APPROX - barrierHeight / 2 },
          draggable: false,
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        });

        for (let pi = 0; pi < prevImages.length; pi++) {
          edges.push({
            id: `e-${stepIdx - 1}-${pi}-barrier`,
            source: `step-${stepIdx - 1}-${pi}`,
            target: barrierId,
            type: 'themedStep',
            animated: true,
            style: EDGE_STYLE,
            data: { routeFlat: 'target' as const },
          });
        }

        for (let ci = 0; ci < images.length; ci++) {
          edges.push({
            id: `e-barrier-${stepIdx - 1}-${stepIdx}-${ci}`,
            source: barrierId,
            target: `step-${stepIdx}-${ci}`,
            type: 'themedStep',
            animated: true,
            markerEnd: MARKER_END,
            style: EDGE_STYLE,
            data: { routeFlat: 'source' as const },
          });
        }
      }
    }
  }

  const lastStep = order[order.length - 1];
  const lastImages = typeof lastStep === 'string' ? [lastStep] : lastStep;

  const endX = TERMINAL_OFFSET + (order.length - 1) * STEP_WIDTH + estimateStageWidth(lastStep) + (TERMINAL_OFFSET - TERMINAL_WIDTH);
  nodes.push({
    id: 'end',
    type: 'terminal',
    data: { label: 'End' },
    position: { x: endX, y: 0 },
    draggable: false,
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
  });
  const isMultiSource = lastImages.length > 1;
  for (let li = 0; li < lastImages.length; li++) {
    edges.push({
      id: `e-${order.length - 1}-${li}-end`,
      source: `step-${order.length - 1}-${li}`,
      target: 'end',
      type: 'themedStep',
      animated: true,
      markerEnd: MARKER_END,
      style: EDGE_STYLE,
      data: isMultiSource ? { routeFlat: 'target' as const } : undefined,
    });
  }

  return { nodes, edges };
}

// Derive a pipeline order array from node x-positions after a drag
function deriveOrderFromNodes(nodes: FlowNode[]): (string | string[])[] {
  const stepNodes = nodes
    .filter((n) => n.type === 'imageStep')
    .map((n) => ({ label: n.data.label, x: n.position.x }))
    .sort((a, b) => a.x - b.x);

  if (stepNodes.length === 0) return [];

  const stages: string[][] = [[stepNodes[0].label]];
  let stageX = stepNodes[0].x;

  for (let i = 1; i < stepNodes.length; i++) {
    if (stepNodes[i].x - stageX <= CLUSTER_THRESHOLD) {
      stages[stages.length - 1].push(stepNodes[i].label);
    } else {
      stages.push([stepNodes[i].label]);
      stageX = stepNodes[i].x;
    }
  }

  return stages.map((s) => (s.length === 1 ? s[0] : s));
}

const ImageStepNode: React.FC<NodeProps<ImageStepNode>> = ({ data }) => (
  <StepNodeWrapper $parallel={data.isParallel} $banned={data.isBanned} title={data.label}>
    <Handle type="target" position={Position.Left} />
    <StepLabel>
      {data.isBanned && (
        <BanIcon>
          <FaExclamationTriangle />
        </BanIcon>
      )}
      <span>{data.label}</span>
    </StepLabel>
    <Handle type="source" position={Position.Right} />
  </StepNodeWrapper>
);

const TerminalNodeComponent: React.FC<NodeProps<TerminalNode>> = () => (
  <TerminalNodeWrapper>
    <Handle type="target" position={Position.Left} />
    <Handle type="source" position={Position.Right} />
  </TerminalNodeWrapper>
);

const BarrierNodeComponent: React.FC<NodeProps<BarrierNode>> = ({ data }) => (
  <BarrierNodeWrapper $height={data.height}>
    <Handle type="target" position={Position.Left} />
    <Handle type="source" position={Position.Right} />
  </BarrierNodeWrapper>
);

const nodeTypes = { imageStep: ImageStepNode, terminal: TerminalNodeComponent, barrier: BarrierNodeComponent };

interface PipelineOrderFlowProps {
  order: (string | string[])[];
  onOrderChange?: (newOrder: (string | string[])[]) => void;
  bannedImages?: Set<string>;
  group?: string;
}

interface SelectOption {
  readonly label: string;
  readonly value: string;
}

const PipelineOrderFlowInner: React.FC<PipelineOrderFlowProps> = ({ order, onOrderChange, bannedImages, group }) => {
  const { nodes: initialNodes, edges } = useMemo(() => orderToNodesAndEdges(order, bannedImages), [order, bannedImages]);
  const [nodes, setNodes] = useState<FlowNode[]>(initialNodes);
  const { fitView, screenToFlowPosition } = useReactFlow();
  const containerRef = useRef<HTMLDivElement>(null);
  const contextMenuRef = useRef<HTMLDivElement>(null);
  const selectMenuRef = useRef<HTMLDivElement>(null);

  const [contextMenu, setContextMenu] = useState<ContextMenuState>(null);
  const [selectMenu, setSelectMenu] = useState<SelectMenuState>(null);
  const [groupImages, setGroupImages] = useState<string[]>([]);

  // Fetch images for the group when editing is enabled
  useEffect(() => {
    if (!group || !onOrderChange) return;
    let cancelled = false;
    void listImages(group, console.error, false, null, 1000).then((result) => {
      if (cancelled) return;
      if (result && 'names' in result) setGroupImages(result.names);
    });
    return () => {
      cancelled = true;
    };
  }, [group, onOrderChange]);

  // Sync nodes when order prop changes
  useEffect(() => {
    setNodes(initialNodes);
  }, [initialNodes]);

  // Dismiss menus on click outside or Escape
  useEffect(() => {
    if (!contextMenu && !selectMenu) return;
    const handlePointerDown = (e: PointerEvent) => {
      if (contextMenu && contextMenuRef.current && !contextMenuRef.current.contains(e.target as globalThis.Node)) {
        setContextMenu(null);
      }
      if (selectMenu && selectMenuRef.current && !selectMenuRef.current.contains(e.target as globalThis.Node)) {
        setSelectMenu(null);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setContextMenu(null);
        setSelectMenu(null);
      }
    };
    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [contextMenu, selectMenu]);

  const handleNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) => {
      if (!onOrderChange) return;
      setNodes((nds) => applyNodeChanges(changes, nds));
    },
    [onOrderChange],
  );

  const handleNodeDragStop = useCallback(() => {
    if (!onOrderChange) return;
    const newOrder = deriveOrderFromNodes(nodes);
    if (!ordersEqual(newOrder, order)) {
      onOrderChange(newOrder);
    }
  }, [nodes, order, onOrderChange]);

  // Right-click on empty pane
  const handlePaneContextMenu = useCallback(
    (event: MouseEvent | React.MouseEvent) => {
      if (!onOrderChange || !group) return;
      event.preventDefault();
      setSelectMenu(null);
      setContextMenu({ x: event.clientX, y: event.clientY });
    },
    [onOrderChange, group],
  );

  // Right-click on a node
  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: FlowNode) => {
      if (!onOrderChange || !group) return;
      if (node.type !== 'imageStep') return;
      event.preventDefault();
      setSelectMenu(null);
      setContextMenu({
        x: event.clientX,
        y: event.clientY,
        nodeLabel: node.data.label,
        stepIndex: node.data.stepIndex,
        parallelIndex: node.data.parallelIndex,
      });
    },
    [onOrderChange, group],
  );

  // Double-click on empty pane: open image select
  const handlePaneDoubleClick = useCallback(
    (event: React.MouseEvent) => {
      if (!onOrderChange || !group) return;
      const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      setContextMenu(null);
      setSelectMenu({ x: event.clientX, y: event.clientY, flowX: flowPos.x });
    },
    [onOrderChange, group, screenToFlowPosition],
  );

  // Context menu actions
  const handleInsertImageAction = useCallback(() => {
    if (!contextMenu) return;
    const flowPos = screenToFlowPosition({ x: contextMenu.x, y: contextMenu.y });
    setContextMenu(null);
    setSelectMenu({ x: contextMenu.x, y: contextMenu.y, flowX: flowPos.x });
  }, [contextMenu, screenToFlowPosition]);

  const handleRemoveImageAction = useCallback(() => {
    if (contextMenu?.stepIndex == null || contextMenu?.parallelIndex == null || !onOrderChange) return;
    const newOrder = removeImageAtPosition(order, contextMenu.stepIndex, contextMenu.parallelIndex);
    onOrderChange(newOrder);
    setContextMenu(null);
  }, [contextMenu, order, onOrderChange]);

  // Image selected from dropdown
  const handleImageSelected = useCallback(
    (option: SelectOption | null) => {
      if (!option || !selectMenu || !onOrderChange) return;
      const newOrder = insertImageAtPosition(order, option.value, selectMenu.flowX);
      onOrderChange(newOrder);
      setSelectMenu(null);
    },
    [selectMenu, order, onOrderChange],
  );

  // Build select options: images in the group that are not already in the order
  const selectOptions = useMemo(() => {
    const inOrder = getImagesInOrder(order);
    return groupImages.filter((name) => !inOrder.has(name)).map((name) => ({ label: name, value: name }));
  }, [groupImages, order]);

  const maxParallel = Math.max(1, ...order.map((s) => (Array.isArray(s) ? s.length : 1)));
  const height = Math.max(100, maxParallel * PARALLEL_GAP + 50);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver(() => {
      void fitView(FIT_VIEW_OPTIONS);
    });
    observer.observe(el);

    return () => observer.disconnect();
  }, [fitView]);

  return (
    <>
      <FlowContainer ref={containerRef} $height={height}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={handleNodesChange}
          onNodeDragStop={handleNodeDragStop}
          onPaneContextMenu={onOrderChange && group ? handlePaneContextMenu : undefined}
          onNodeContextMenu={onOrderChange && group ? handleNodeContextMenu : undefined}
          onDoubleClick={onOrderChange && group ? handlePaneDoubleClick : undefined}
          fitView
          fitViewOptions={FIT_VIEW_OPTIONS}
          nodesDraggable={!!onOrderChange}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag={false}
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          autoPanOnNodeDrag={false}
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
        />
      </FlowContainer>

      {contextMenu && (
        <ContextMenuOverlay ref={contextMenuRef} $top={contextMenu.y} $left={contextMenu.x}>
          <ContextMenuItem onPointerDown={(e) => e.stopPropagation()} onClick={handleInsertImageAction}>
            Insert Image
          </ContextMenuItem>
          {contextMenu.nodeLabel && (
            <ContextMenuItem onPointerDown={(e) => e.stopPropagation()} onClick={handleRemoveImageAction}>
              Remove Image
            </ContextMenuItem>
          )}
        </ContextMenuOverlay>
      )}

      {selectMenu && (
        <ImageSelectOverlay ref={selectMenuRef} $top={selectMenu.y} $left={selectMenu.x} onPointerDown={(e) => e.stopPropagation()}>
          <Select<SelectOption>
            options={selectOptions}
            onChange={handleImageSelected}
            styles={selectStyles}
            placeholder="Select image..."
            autoFocus
            menuIsOpen
            onBlur={() => setSelectMenu(null)}
          />
        </ImageSelectOverlay>
      )}
    </>
  );
};

const PipelineOrderFlow: React.FC<PipelineOrderFlowProps> = ({ order, onOrderChange, bannedImages, group }) => {
  if (!order || order.length === 0) return null;

  return (
    <ReactFlowProvider>
      <PipelineOrderFlowInner order={order} onOrderChange={onOrderChange} bannedImages={bannedImages} group={group} />
    </ReactFlowProvider>
  );
};

export default PipelineOrderFlow;
