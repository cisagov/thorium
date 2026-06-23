import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
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
import { FaExclamationTriangle, FaGripLines } from 'react-icons/fa';
import Select from 'react-select';

// project imports
import {
  FlowContainer,
  StepNodeWrapper,
  StepLabel,
  TerminalNodeWrapper,
  BarrierNodeWrapper,
  BanIcon,
  NodeTooltip,
  ContextMenuOverlay,
  ContextMenuItem,
  ImageSelectOverlay,
  ResizeHandle,
} from './PipelineOrderFlow.styled';
import { edgeTypes } from './ThemedEdge';
import {
  ordersEqual,
  insertImageAtPosition,
  removeImageAtPosition,
  getImagesInOrder,
  estimateStageWidth,
  clusterStagesByX,
  HANDLE_CENTER_Y,
  NODE_HEIGHT,
  STEP_WIDTH,
  TERMINAL_OFFSET,
} from './order';
import { createReactSelectStyles } from '@utilities/select';
import { listImages } from '@thorpi/images';

const TERMINAL_WIDTH = 12;
const PARALLEL_GAP = 54;
// The canvas auto-grows with parallel count up to MAX_CANVAS_HEIGHT so it can't balloon for highly
// parallel stages. Past the cap the content overflows vertically, which is what lets the camera rules
// engage (zoom out to the readable floor and, when still taller than the canvas, center the added node
// vertically). The user can override the height with the bottom resize handle, between
// MIN_CANVAS_HEIGHT and MAX_USER_CANVAS_HEIGHT; framing recomputes against whatever the live height is.
const MIN_CANVAS_HEIGHT = 140;
const MAX_CANVAS_HEIGHT = 400;
const MAX_USER_CANVAS_HEIGHT = 1000;
// Barrier handle center must match the image/terminal handle center so converging edges stay aligned.
const STEP_HANDLE_APPROX = HANDLE_CENTER_Y;
// Manual zoom limits (what the user can reach with the wheel). The auto-fit floor is separate and
// higher (FIT_MIN_ZOOM) so loading/adding never shrinks the graph to an unreadable size — a graph
// wider than the viewport overflows at a readable size and the user pans (drag background) to it.
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 1.5;
const FIT_MIN_ZOOM = 0.8;
const FIT_PADDING = 0.2;
const FIT_VIEW_OPTIONS = { padding: FIT_PADDING, minZoom: FIT_MIN_ZOOM, maxZoom: 1 };

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

  // Empty (editable) order: render a Start→End canvas with a clickable gap so the
  // first image can be added via double-click / right-click like any other.
  if (order.length === 0) {
    nodes.push({
      id: 'end',
      type: 'terminal',
      data: { label: 'End' },
      position: { x: TERMINAL_OFFSET + STEP_WIDTH, y: 0 },
      draggable: false,
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });
    edges.push({
      id: 'e-start-end',
      source: 'start',
      target: 'end',
      type: 'themedStep',
      animated: true,
      markerEnd: MARKER_END,
      style: EDGE_STYLE,
    });
    return { nodes, edges };
  }

  for (let stepIdx = 0; stepIdx < order.length; stepIdx++) {
    const step = order[stepIdx];
    const images = typeof step === 'string' ? [step] : step;
    const yOffset = (-(images.length - 1) * PARALLEL_GAP) / 2;
    // Size every node in the stage to the stage's width (the widest image). Nodes are left-aligned,
    // so a shared width also aligns their right edges — the source handles all sit at x + stageWidth —
    // so parallel edges converge on a single vertical. The barrier/End x use the same estimate.
    const stageWidth = estimateStageWidth(step);

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
        style: { width: stageWidth },
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
  const stepNodes = nodes.filter((n) => n.type === 'imageStep').map((n) => ({ label: n.data.label, x: n.position.x }));
  return clusterStagesByX(stepNodes);
}

const ImageStepNode: React.FC<NodeProps<ImageStepNode>> = ({ data }) => {
  const wrapperRef = useRef<HTMLDivElement>(null);
  // Tooltip position in screen coords (the node's top-center). Null when not hovered.
  const [tipPos, setTipPos] = useState<{ x: number; y: number } | null>(null);

  const showTip = () => {
    const rect = wrapperRef.current?.getBoundingClientRect();
    if (rect) setTipPos({ x: rect.left + rect.width / 2, y: rect.top });
  };
  const hideTip = () => setTipPos(null);

  return (
    <StepNodeWrapper ref={wrapperRef} $parallel={data.isParallel} $banned={data.isBanned} onMouseEnter={showTip} onMouseLeave={hideTip}>
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
      {/* Portaled to body so the tooltip escapes the canvas's overflow:hidden + zoom transform. */}
      {tipPos && createPortal(<NodeTooltip style={{ left: tipPos.x, top: tipPos.y }}>{data.label}</NodeTooltip>, document.body)}
    </StepNodeWrapper>
  );
};

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

// Bounding box of the whole graph in flow coords, computed from node positions + known sizes (image
// nodes carry an explicit per-stage width; terminals are TERMINAL_WIDTH; barriers carry their height).
// Avoids depending on ReactFlow's async DOM measurement so framing is correct right after a change.
function computeBounds(nodes: FlowNode[]): { x: number; y: number; width: number; height: number } {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const n of nodes) {
    let w = TERMINAL_WIDTH;
    let h = NODE_HEIGHT;
    if (n.type === 'imageStep') {
      w = typeof n.style?.width === 'number' ? n.style.width : NODE_HEIGHT;
    } else if (n.type === 'barrier') {
      w = 2;
      h = n.data.height;
    }
    minX = Math.min(minX, n.position.x);
    minY = Math.min(minY, n.position.y);
    maxX = Math.max(maxX, n.position.x + w);
    maxY = Math.max(maxY, n.position.y + h);
  }
  if (minX === Infinity) return { x: 0, y: 0, width: 0, height: 0 };
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

const PipelineOrderFlowInner: React.FC<PipelineOrderFlowProps> = ({ order, onOrderChange, bannedImages, group }) => {
  const { nodes: initialNodes, edges } = useMemo(() => orderToNodesAndEdges(order, bannedImages), [order, bannedImages]);
  const [nodes, setNodes] = useState<FlowNode[]>(initialNodes);
  const { fitView, screenToFlowPosition, setCenter } = useReactFlow();
  const containerRef = useRef<HTMLDivElement>(null);
  // Tracks the images present on the last render so a newly-added image can be detected and centered.
  const prevImagesRef = useRef<Set<string>>(getImagesInOrder(order));
  // Tracks the previous max parallel count so we can re-frame when the diagram's height changes.
  const prevMaxParallelRef = useRef(1);
  // Label of the image just dragged to a new stage, so the next reframe can focus it once placed.
  const dragFocusRef = useRef<string | null>(null);
  const hasFitRef = useRef(false);
  // Holds the latest frameView so the (once-created) ResizeObserver can re-frame without depending on it.
  const frameViewRef = useRef<(() => void) | null>(null);
  // Debounce for re-framing on container/window resize.
  const reframeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const contextMenuRef = useRef<HTMLDivElement>(null);
  const selectMenuRef = useRef<HTMLDivElement>(null);

  const [contextMenu, setContextMenu] = useState<ContextMenuState>(null);
  const [selectMenu, setSelectMenu] = useState<SelectMenuState>(null);
  const [groupImages, setGroupImages] = useState<string[]>([]);
  // User-chosen canvas height via the bottom resize handle (null = auto height from parallel count).
  const [userHeight, setUserHeight] = useState<number | null>(null);

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
      // Free x/y dragging: the node follows the cursor. Its edges re-route orthogonally (right angles)
      // via the router for any position, so there's no need to constrain the drag axis. Reordering is
      // derived from x-position on drop, and the node is restored to its row when the order re-lays out
      // (see handleNodeDragStop).
      setNodes((nds) => applyNodeChanges(changes, nds));
    },
    [onOrderChange],
  );

  const handleNodeDragStop = useCallback(
    (_event: MouseEvent | TouchEvent, node: FlowNode) => {
      if (!onOrderChange) return;
      const newOrder = deriveOrderFromNodes(nodes);
      if (!ordersEqual(newOrder, order)) {
        // Remember the dragged image so the framing effect can focus it once it's re-laid-out in its
        // final placed position (used only when the resulting graph overflows the canvas).
        if (node.type === 'imageStep') dragFocusRef.current = node.data.label;
        onOrderChange(newOrder);
      } else {
        // Order unchanged (dropped in place): the [initialNodes] effect won't re-run, so restore the
        // clean layout here — otherwise a parallel node dragged-and-dropped stays snapped to y=0.
        setNodes(initialNodes);
      }
    },
    [nodes, order, onOrderChange, initialNodes],
  );

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
  // Auto height grows with parallel count up to the cap; a user resize (handle) overrides it within
  // [MIN_CANVAS_HEIGHT, MAX_USER_CANVAS_HEIGHT].
  const autoHeight = Math.min(MAX_CANVAS_HEIGHT, Math.max(MIN_CANVAS_HEIGHT, maxParallel * PARALLEL_GAP + 50));
  const height = userHeight != null ? Math.min(MAX_USER_CANVAS_HEIGHT, Math.max(MIN_CANVAS_HEIGHT, userHeight)) : autoHeight;

  // Fit once when the container first gets a non-zero size (e.g. mounted inside a collapsed accordion
  // that animates open). On *subsequent* size changes (the user resizing the browser window after the
  // diagram is already open) we re-frame the graph — debounced — so nodes scale back into view rather
  // than drifting off-canvas when the window shrinks. We read frameView through a ref so the observer
  // is created once and isn't torn down/recreated as frameView's identity changes each render.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width === 0 || height === 0) return;
      if (!hasFitRef.current) {
        hasFitRef.current = true;
        void fitView(FIT_VIEW_OPTIONS);
        return;
      }
      clearTimeout(reframeTimer.current);
      reframeTimer.current = setTimeout(() => frameViewRef.current?.(), 150);
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
      clearTimeout(reframeTimer.current);
    };
  }, [fitView]);

  // Frame the camera against the LIVE canvas size (so it's correct after an order change OR a resize).
  // Rules: if the whole graph fits at a readable zoom, center the whole graph; otherwise hold the
  // readable floor zoom and center on `addedCenter` only along the axis/axes whose graph dimension
  // exceeds the canvas — the fitting axis stays graph-centered. `addedCenter` is the newly-added node
  // (when applicable); without it both axes fall back to the graph center.
  const frameView = useCallback(
    (focusCenter?: { x: number; y: number } | null, opts?: { onlyWhenOverflow?: boolean }) => {
      const el = containerRef.current;
      if (!el) return;
      const { width: cw, height: ch } = el.getBoundingClientRect();
      if (cw === 0 || ch === 0) return;

      const bounds = computeBounds(initialNodes);
      if (bounds.width === 0 || bounds.height === 0) {
        if (!opts?.onlyWhenOverflow) void fitView(FIT_VIEW_OPTIONS);
        return;
      }

      const padFactor = 1 + 2 * FIT_PADDING;
      const fitZoom = Math.min(cw / (bounds.width * padFactor), ch / (bounds.height * padFactor), 1);
      if (fitZoom >= FIT_MIN_ZOOM) {
        // Whole graph fits. Skip when the caller only wants to act on overflow (e.g. a plain
        // drag-reorder shouldn't jump the view when everything still fits).
        if (!opts?.onlyWhenOverflow) void fitView(FIT_VIEW_OPTIONS);
        return;
      }

      const zoom = FIT_MIN_ZOOM;
      const overflowX = bounds.width * zoom > cw;
      const overflowY = bounds.height * zoom > ch;
      const graphCx = bounds.x + bounds.width / 2;
      const graphCy = bounds.y + bounds.height / 2;
      const cx = overflowX && focusCenter ? focusCenter.x : graphCx;
      const cy = overflowY && focusCenter ? focusCenter.y : graphCy;
      void setCenter(cx, cy, { zoom, duration: 200 });
    },
    [initialNodes, fitView, setCenter],
  );

  // Reframe after an order change. A structural change (image added, or a stage's parallel count
  // changed) reframes per the full rules (re-centering the whole graph when it fits). A plain
  // drag-reorder focuses the dragged image only when the result overflows the canvas — otherwise the
  // view is left alone so small reorders don't jump. A plain remove leaves the view alone.
  useEffect(() => {
    const current = getImagesInOrder(order);
    const prev = prevImagesRef.current;
    prevImagesRef.current = current;
    const parallelChanged = maxParallel !== prevMaxParallelRef.current;
    prevMaxParallelRef.current = maxParallel;
    const dragFocus = dragFocusRef.current;
    dragFocusRef.current = null;

    let addedLabel: string | null = null;
    for (const name of current) {
      if (!prev.has(name)) {
        addedLabel = name;
        break;
      }
    }

    const structural = !!addedLabel || parallelChanged;
    if (!structural && !dragFocus) return;

    // Focus target: the newly-added node, else the dragged image in its placed position.
    const focusLabel = addedLabel ?? dragFocus;
    let focusCenter: { x: number; y: number } | null = null;
    if (focusLabel) {
      const node = initialNodes.find((n) => n.type === 'imageStep' && n.data.label === focusLabel);
      if (node) {
        const w = typeof node.style?.width === 'number' ? node.style.width : NODE_HEIGHT;
        focusCenter = { x: node.position.x + w / 2, y: node.position.y + NODE_HEIGHT / 2 };
      }
    }
    // Pure drag-reorder (no add / no parallel change): only act when the graph overflows.
    frameView(focusCenter, { onlyWhenOverflow: !structural });
  }, [order, maxParallel, initialNodes, frameView]);

  // Keep the latest frameView reachable from the once-created ResizeObserver without retriggering it.
  frameViewRef.current = () => frameView();

  // Bottom resize handle: drag to expand/contract the canvas height (clamped to
  // [MIN_CANVAS_HEIGHT, MAX_USER_CANVAS_HEIGHT]). Reframe on release so the content recenters for the
  // new height (the camera rules read the live canvas size).
  const handleResizeStart = useCallback(
    (event: React.PointerEvent) => {
      event.preventDefault();
      const startY = event.clientY;
      const startHeight = containerRef.current?.getBoundingClientRect().height ?? height;
      const onMove = (moveEvent: PointerEvent) => {
        const next = Math.min(MAX_USER_CANVAS_HEIGHT, Math.max(MIN_CANVAS_HEIGHT, startHeight + (moveEvent.clientY - startY)));
        setUserHeight(next);
      };
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        frameView();
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    },
    [frameView, height],
  );

  // Reset all positional state (camera + zoom) to the start when the group changes. On the create
  // page the group select also clears the order, so the diagram returns to the empty Start→End canvas.
  useEffect(() => {
    hasFitRef.current = false;
    prevImagesRef.current = new Set();
    prevMaxParallelRef.current = 1;
    void fitView(FIT_VIEW_OPTIONS);
  }, [group, fitView]);

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
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
          nodesDraggable={!!onOrderChange}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag
          zoomOnScroll
          zoomOnPinch
          zoomOnDoubleClick={false}
          autoPanOnNodeDrag={false}
          preventScrolling
          proOptions={{ hideAttribution: true }}
        />
        <ResizeHandle onPointerDown={handleResizeStart} title="Drag to resize the diagram height">
          <FaGripLines />
        </ResizeHandle>
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
  // Nothing to show for an empty order in read-only mode; in edit mode we still render an
  // empty canvas so the first image can be added directly on the diagram.
  if ((!order || order.length === 0) && !onOrderChange) return null;

  return (
    <ReactFlowProvider>
      <PipelineOrderFlowInner order={order} onOrderChange={onOrderChange} bannedImages={bannedImages} group={group} />
    </ReactFlowProvider>
  );
};

export default PipelineOrderFlow;
