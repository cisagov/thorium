import React, { useEffect, useReducer, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ErrorBoundary } from 'react-error-boundary';
import { Popover, Spinner } from 'react-bootstrap';
import ForceGraph3D from '3d-force-graph';
import { FaFolderTree } from 'react-icons/fa6';
import { GoSidebarCollapse } from 'react-icons/go';
import * as THREE from 'three';

// project imports
import { getNodeColor, getEdgeColor, isCrabTheme, buildCrabParticle } from './styles';
import GraphControlsToolbar from './controls/GraphControlsToolbar';
import { NodeRenderMode, DagMode } from './controls/types';
import { createControlsReducer, buildNodeObject, buildEdgeLabelFactory, iconNodeVal } from './controls/controlsReducer';
import { computeSizeDefaults } from './controls/sizeDefaults';
import {
  computeLabelScale,
  EDGE_LABEL_MAX_PX,
  EDGE_LABEL_MIN_PX,
  EDGE_LABEL_TARGET_PX,
  LABEL_MAX_PX,
  LABEL_MIN_PX,
  LABEL_TARGET_PX,
} from './labelScale';
import { EDGE_LABEL_BUDGET, NODE_LABEL_BUDGET, selectVisibleLabels } from './labelVisibility';
import { collectNeighborhoodIds, computeBoundingSphere, computeCentroidRadius, MIN_FOCUS_DISTANCE, sphereFitDistance } from './focusMath';
import type { LabelEntry } from './controls/controlsReducer';
import type { SelectedElement } from './controls/types';
import { processInitialGraphData, getLinkEndpoints } from './data';
import { useGraphData, FocusSource } from '../data/GraphDataContext';
import type { GraphNode, GraphLink, GraphData, GraphInstance, GraphOrbitControls, D3ChargeForce, D3LinkForce } from './types';
import { applyGrowthToInstance } from './applyGrowth';
import DataPreviewPanel from './DataPreviewPanel';
import NavCluster from './NavCluster';
import { AssociationTree } from '../browsing/AssociationTree';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import EntitySummary, { SummaryVariant } from '@components/shared/info/EntitySummary';
import { SummaryPopover } from '@components/shared/info/SummaryPopover';
import { treeNodeToInfo } from '@components/shared/info/info';
import {
  GraphDiv,
  GraphOverlayMessage,
  GraphWindow,
  LoadingOverlay,
  MinimizeButton,
  TreeOverlayHeader,
  TreeOverlayPanel,
  TreeOverlayToggle,
} from './Shared';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';

// spec: ./AssociationGraph.spec.md

interface AssociationGraphProps {
  inView: boolean;
  bordered?: boolean;
}

const MIN_ORBIT_RADIUS = 10;
const ZOOM_SPEED = 1.5;
const PINNED_TIER_BOOST = 1.4;
const HUB_TIER_BOOST = 1.2;
const CAM_DIST_THRESHOLD_RATIO = 0.005;
const CAM_DIST_THRESHOLD_MIN = 0.1;
const SETTLE_COOLDOWN_TICKS = 150;
const FIT_DURATION_MS = 1000;
const FIT_PADDING_PX = 50;
const DOLLY_IN_FACTOR = 0.7;
const DOLLY_OUT_FACTOR = 1.4;
const DOLLY_DURATION_MS = 300;
const DBLCLICK_ZOOM_STEP_RATIO = 0.4;
const SAVE_HOME_BUFFER_MS = 100;
const SETTLE_FIT_FALLBACK_MS = 3500;

/** Which automatic camera fit is armed to run when the force engine settles */
enum PendingFit {
  /** The first fit after mount; skipped if the user already moved the camera */
  Initial = 'initial',
  /** A refit requested by the "Refit on Grow" setting after new nodes arrived */
  Grow = 'grow',
}

const AssociationGraph3DInner: React.FC<{ bordered?: boolean }> = ({ bordered }) => {
  const {
    graph,
    graphId,
    graphVersion,
    loading,
    error,
    growing,
    initialDepth,
    grow,
    growToDepth,
    growable,
    reload,
    focusedNodeId,
    focusSource,
    setFocusedNode,
  } = useGraphData();
  const containerRef = useRef<HTMLDivElement>(null);
  const graphInstanceRef = useRef<GraphInstance | null>(null);
  const graphDataRef = useRef<GraphData>({ nodes: [], links: [] });
  const labelSpritesRef = useRef<Map<string, LabelEntry>>(new Map());
  const edgeLabelSpritesRef = useRef<Map<string, LabelEntry>>(new Map());
  const animFrameRef = useRef<number>(0);
  const cameraAnimRef = useRef<number>(0);
  const mountedVersionRef = useRef<number>(-1);
  // the graphId the live instance was built for; a change means a brand-new server tree
  // (reload/refetch) that must replace the instance data rather than merge as growth
  const mountedGraphIdRef = useRef<string | null>(null);
  const lastCamDistRef = useRef<number>(-1);
  const boundingRadiusRef = useRef(200);
  const radiusVersionRef = useRef(-1);
  const gridRef = useRef<THREE.Group | null>(null);
  const hoverHideTimer = useRef<number | undefined>(undefined);
  const [nodeCount, setNodeCount] = useState(0);
  const [mounted, setMounted] = useState(false);
  const [previewMinimized, setPreviewMinimized] = useState(false);
  const [treeOverlayOpen, setTreeOverlayOpen] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<{ id: string; x: number; y: number } | null>(null);
  const handleNodeSelectRef = useRef<((node: GraphNode) => Promise<void>) | null>(null);
  const handleEdgeSelectRef = useRef<((link: GraphLink) => void) | null>(null);
  const focusSettingsRef = useRef({ focusOnClick: false, fitNeighborhood: true });
  const nodeLabelScaleRef = useRef(1);
  const edgeLabelScaleRef = useRef(1);
  const nodeLabelDensityRef = useRef(0.5);
  const edgeLabelDensityRef = useRef(0.5);
  const refitOnGrowRef = useRef(true);
  // true once the user has moved the camera; blocks the automatic initial fit
  const userInteractedRef = useRef(false);
  // one-shot latch for the settle-aware fit; null means no fit is armed
  const pendingSettleFitRef = useRef<PendingFit | null>(null);
  const settleFitFallbackRef = useRef<number>(0);
  const saveHomeTimerRef = useRef<number>(0);
  const lastSizeDefaultsCountRef = useRef(0);
  // live node-id -> degree lookup for edge label ranking (rebuilt on every data set/growth)
  const nodeDegreesRef = useRef<Map<string, number>>(new Map());
  // importance rankings for the declutter pass, rebuilt only when the label maps change
  const sortedNodeIdsRef = useRef<string[]>([]);
  const sortedEdgeIdsRef = useRef<string[]>([]);
  const initialNodeIdsRef = useRef<Set<string>>(new Set());
  const hubDegreeThresholdRef = useRef(Infinity);
  // selection/focus mirrors readable from the per-frame label loop
  const selectedElementRef = useRef<SelectedElement | null>(null);
  const focusedNodeIdRef = useRef<string | null>(null);

  const controlsReducerRef = useRef(
    createControlsReducer(graphInstanceRef, labelSpritesRef, edgeLabelSpritesRef, lastCamDistRef, nodeDegreesRef),
  );
  const controlsReducer = controlsReducerRef.current;

  const [controls, updateControls] = useReducer(controlsReducer, {
    filterChildless: false,
    // seed the depth display from the depth the graph was fetched to, so it doesn't misleadingly show 1
    depth: initialDepth,
    showEdgeLabels: false,
    showNodeLabels: true,
    selectedElement: null,
    nodeRenderMode: NodeRenderMode.Icons,
    focusOnClick: true,
    fitNeighborhoodOnFocus: true,
    refitOnGrow: false,
    nodeLabelScale: 1,
    edgeLabelScale: 1,
    edgeWidth: 1,
    edgeLength: 30,
    edgeLinkStrength: 0.5,
    edgeOpacity: 0.2,
    arrowLength: 3.5,
    directionalParticles: 1,
    particleSpeed: 0.006,
    nodeRelSize: 4,
    nodeOpacity: 0.75,
    enableNodeDrag: true,
    nodeLabelDensity: 0.5,
    edgeLabelDensity: 0.5,
    chargeStrength: -200,
    velocityDecay: 0.4,
    warmupTicks: 0,
    cooldownTime: 4000,
    dagMode: null as DagMode | null,
    dagLevelDistance: null as number | null,
    numDimensions: 3,
    showGrid: false,
    userOverrides: new Set<string>(),
  });

  focusSettingsRef.current = {
    focusOnClick: controls.focusOnClick,
    fitNeighborhood: controls.fitNeighborhoodOnFocus,
  };
  nodeLabelScaleRef.current = controls.nodeLabelScale;
  edgeLabelScaleRef.current = controls.edgeLabelScale;
  nodeLabelDensityRef.current = controls.nodeLabelDensity;
  edgeLabelDensityRef.current = controls.edgeLabelDensity;
  refitOnGrowRef.current = controls.refitOnGrow;
  selectedElementRef.current = controls.selectedElement;
  focusedNodeIdRef.current = focusedNodeId;

  const cancelCameraAnim = () => {
    if (cameraAnimRef.current) {
      cancelAnimationFrame(cameraAnimRef.current);
      cameraAnimRef.current = 0;
    }
  };

  /**
   * Zooms the camera to frame the whole graph, then records the resulting view as the
   * orbit-controls "home" state so `handleResetView` can restore it later.
   */
  const fitAndSaveHome = () => {
    const gi = graphInstanceRef.current;
    if (!gi) return;
    cancelCameraAnim();
    gi.zoomToFit(FIT_DURATION_MS, FIT_PADDING_PX);
    // save the "home" view only after the fit animation has landed so a later reset
    // restores the framed view rather than a mid-flight camera
    window.clearTimeout(saveHomeTimerRef.current);
    saveHomeTimerRef.current = window.setTimeout(() => {
      const ctrl = graphInstanceRef.current?.controls() as GraphOrbitControls | undefined;
      ctrl?.saveState();
    }, FIT_DURATION_MS + SAVE_HOME_BUFFER_MS);
  };

  /**
   * Consumes an armed settle-aware fit: invoked by `onEngineStop` and the fallback timer,
   * but only the first caller acts because the pending latch is cleared immediately (the
   * engine-stop hook fires on EVERY stop).
   */
  const completeSettleFit = () => {
    const kind = pendingSettleFitRef.current;
    if (!kind) return;
    pendingSettleFitRef.current = null;
    window.clearTimeout(settleFitFallbackRef.current);
    // settled positions can shift the bounding sphere, so force the label loop to
    // re-measure the radius and re-rank against the new layout
    radiusVersionRef.current = -1;
    lastCamDistRef.current = -1;
    // never yank the camera away from a view the user has already taken over; grow
    // refits are exempt because the user opted into them explicitly
    if (kind === PendingFit.Initial && userInteractedRef.current) return;
    fitAndSaveHome();
  };

  /**
   * Arms a one-shot camera fit that runs when the force engine next settles, with a
   * fallback timer in case the engine never reports a stop.
   *
   * @param kind - Whether this fit is the initial framing or a refit-on-grow.
   */
  const armSettleFit = (kind: PendingFit) => {
    pendingSettleFitRef.current = kind;
    window.clearTimeout(settleFitFallbackRef.current);
    settleFitFallbackRef.current = window.setTimeout(completeSettleFit, SETTLE_FIT_FALLBACK_MS);
  };

  /**
   * Restores the camera to the saved "home" view recorded after the last automatic fit.
   * `reset()` fires only the orbit-controls 'change' event — not 'start'/'end' — so the
   * freeze/unfreeze node listeners stay idle during the restore.
   */
  const handleResetView = () => {
    const gi = graphInstanceRef.current;
    if (!gi) return;
    cancelCameraAnim();
    // force the label loop to re-rank and re-scale against the restored camera
    lastCamDistRef.current = -1;
    const ctrl = gi.controls() as GraphOrbitControls | undefined;
    ctrl?.reset();
  };

  /**
   * Dollies the camera along its current view axis, scaling the camera-to-orbit-target
   * distance by `factor` (< 1 moves in, > 1 moves out). The new distance is clamped to
   * `MIN_ORBIT_RADIUS` so the camera never collapses onto the orbit target.
   *
   * @param factor - Multiplier applied to the current orbit distance.
   */
  const handleDolly = (factor: number) => {
    const gi = graphInstanceRef.current;
    if (!gi) return;
    const ctrl = gi.controls() as GraphOrbitControls | undefined;
    const target = ctrl?.target;
    if (!target) return;
    const cam = gi.cameraPosition();
    const dist = Math.hypot(cam.x - target.x, cam.y - target.y, cam.z - target.z);
    // a degenerate distance leaves no view axis to dolly along
    if (!isFinite(dist) || dist <= 0) return;
    // scale the offset from the orbit target, clamped to the minimum orbit radius
    const scale = Math.max(MIN_ORBIT_RADIUS, dist * factor) / dist;
    animateCameraTo(
      gi,
      {
        x: target.x + (cam.x - target.x) * scale,
        y: target.y + (cam.y - target.y) * scale,
        z: target.z + (cam.z - target.z) * scale,
      },
      { x: target.x, y: target.y, z: target.z },
      DOLLY_DURATION_MS,
    );
  };

  /** Dollies the camera toward the orbit target by the standard zoom-in factor. */
  const handleZoomIn = () => handleDolly(DOLLY_IN_FACTOR);

  /** Dollies the camera away from the orbit target by the standard zoom-out factor. */
  const handleZoomOut = () => handleDolly(DOLLY_OUT_FACTOR);

  /** Zooms the camera to frame the whole graph with the standard duration and padding. */
  const handleFitAll = () => {
    const gi = graphInstanceRef.current;
    if (!gi) return;
    cancelCameraAnim();
    gi.zoomToFit(FIT_DURATION_MS, FIT_PADDING_PX);
  };

  /**
   * Keyboard shortcuts scoped to the graph window: +/= zoom in, - zoom out, f fit all,
   * r reset to home, Escape clears the selection and focus. Keystrokes aimed at form
   * fields (inputs in the controls overlays) and chorded keys are left alone.
   *
   * @param event - The keydown event bubbling up from within the graph window.
   */
  const handleGraphKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    // never hijack browser chords or typing inside editable controls
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable) return;
    switch (event.key) {
      case '+':
      case '=':
        handleZoomIn();
        break;
      case '-':
        handleZoomOut();
        break;
      case 'f':
      case 'F':
        handleFitAll();
        break;
      case 'r':
      case 'R':
        handleResetView();
        break;
      case 'Escape':
        // clear both the selected element and the focused node
        updateControls({ type: 'selected', state: null });
        setFocusedNode(null, FocusSource.Graph);
        break;
      default:
        return;
    }
    event.preventDefault();
  };

  const animateCameraTo = (
    gi: GraphInstance,
    targetPos: { x: number; y: number; z: number },
    lookAt: { x: number; y: number; z: number },
    durationMs: number,
  ) => {
    cancelCameraAnim();
    if (durationMs <= 0) {
      gi.cameraPosition(targetPos, lookAt, 0);
      return;
    }
    const startCam = gi.cameraPosition();
    const ctrl = gi.controls() as GraphOrbitControls | undefined;
    const startLookAt = ctrl?.target ? { x: ctrl.target.x, y: ctrl.target.y, z: ctrl.target.z } : lookAt;
    const startTime = performance.now();
    const step = () => {
      const t = Math.min(1, (performance.now() - startTime) / durationMs);
      const ease = 1 - (1 - t) * (1 - t) * (1 - t);
      gi.cameraPosition(
        {
          x: startCam.x + (targetPos.x - startCam.x) * ease,
          y: startCam.y + (targetPos.y - startCam.y) * ease,
          z: startCam.z + (targetPos.z - startCam.z) * ease,
        },
        {
          x: startLookAt.x + (lookAt.x - startLookAt.x) * ease,
          y: startLookAt.y + (lookAt.y - startLookAt.y) * ease,
          z: startLookAt.z + (lookAt.z - startLookAt.z) * ease,
        },
        0,
      );
      if (t < 1) {
        cameraAnimRef.current = requestAnimationFrame(step);
      } else {
        cameraAnimRef.current = 0;
      }
    };
    cameraAnimRef.current = requestAnimationFrame(step);
  };

  /**
   * Animates the camera to look at `target` along its current view direction, either at
   * an explicit distance or keeping the current orbit distance.
   *
   * @param gi - The live ForceGraph3D instance.
   * @param target - The world-space point the camera should aim at.
   * @param dist - Camera-to-target distance in world units; `null` keeps the current
   *   orbit distance (floored so a fully zoomed-out camera still meaningfully focuses).
   * @param durationMs - Camera animation duration in milliseconds.
   */
  const focusCameraOn = (gi: GraphInstance, target: { x: number; y: number; z: number }, dist: number | null = null, durationMs = 2000) => {
    const camPos = gi.cameraPosition();
    const orbitTarget = (gi.controls() as GraphOrbitControls | undefined)?.target;
    const currentDist = orbitTarget ? Math.hypot(camPos.x - orbitTarget.x, camPos.y - orbitTarget.y, camPos.z - orbitTarget.z) : 150;
    // the keep-distance path floors against the graph size so recentering never lands
    // uselessly far away; explicit distances are already floored by their computation
    const minDist = Math.max(MIN_FOCUS_DISTANCE, boundingRadiusRef.current * 0.15);
    const finalDist = dist ?? Math.max(minDist, currentDist);
    // preserve the current view direction so focusing recenters rather than reorients
    const dx = camPos.x - target.x;
    const dy = camPos.y - target.y;
    const dz = camPos.z - target.z;
    const dirLen = Math.hypot(dx, dy, dz);
    let ux: number, uy: number, uz: number;
    if (dirLen > 0.01) {
      ux = dx / dirLen;
      uy = dy / dirLen;
      uz = dz / dirLen;
    } else {
      ux = 0;
      uy = 0;
      uz = 1;
    }
    animateCameraTo(gi, { x: target.x + ux * finalDist, y: target.y + uy * finalDist, z: target.z + uz * finalDist }, target, durationMs);
    // focusing changes the pinned label set and camera framing; force the label loop
    // to re-rank and re-scale on the next frame
    lastCamDistRef.current = -1;
  };

  /**
   * Reads the perspective camera's vertical field of view from the instance, falling back
   * to the three.js default in case the camera is not a `PerspectiveCamera`.
   *
   * @param gi - The live ForceGraph3D instance.
   * @returns The camera fov in degrees.
   */
  const cameraFov = (gi: GraphInstance): number => (gi.camera() as Partial<THREE.PerspectiveCamera>).fov ?? 50;

  /**
   * Focus-on-click camera move shared by graph node clicks and tree-driven focus: with
   * "Fit Neighborhood" on it frames the node's 1-hop neighborhood via its bounding
   * sphere, otherwise it recenters on the node keeping the current distance. The
   * bounding sphere is computed manually because `zoomToFit(nodeFilter)` cannot be
   * used: the underlying `fitToBbox` always aims the camera at the world origin.
   *
   * @param node - The node to focus; ignored unless it has finite layout coordinates.
   */
  const focusOnNode = (node: GraphNode) => {
    const gi = graphInstanceRef.current;
    if (!gi || node.x === undefined || node.y === undefined || !isFinite(node.x) || !isFinite(node.y)) return;
    if (focusSettingsRef.current.fitNeighborhood) {
      const { nodes, links } = graphDataRef.current;
      const sphere = computeBoundingSphere(nodes, collectNeighborhoodIds(links, node.id));
      if (sphere) {
        focusCameraOn(gi, sphere.center, sphereFitDistance(cameraFov(gi), sphere.radius));
        return;
      }
    }
    focusCameraOn(gi, { x: node.x, y: node.y, z: node.z ?? 0 });
  };

  // React to graph changes from the shared context
  useEffect(() => {
    if (!graphId || graphVersion === 0) return;
    const newGraphData = processInitialGraphData(graph);
    // refresh the live degree lookup so edge label ranking sees current degrees
    nodeDegreesRef.current = new Map(newGraphData.nodes.map((n) => [n.id, n.degree]));

    if (!mounted) {
      graphDataRef.current = newGraphData;
      setNodeCount(newGraphData.nodes.length);
      setMounted(true);
      mountedVersionRef.current = graphVersion;
      mountedGraphIdRef.current = graphId;
      return;
    }

    // A new graphId means reload()/fetchInitial() POSTed a brand-new server tree (e.g. a
    // filterChildless refetch): its nodes are unrelated to the ones already rendered, so
    // replace the instance data wholesale instead of merging — otherwise nodes the new
    // fetch dropped would linger. applyGrowthToInstance only ever adds, never removes.
    if (graphId !== mountedGraphIdRef.current) {
      graphDataRef.current = newGraphData;
      // sprites belong to the old tree's objects; drop them so the digest rebuilds cleanly
      labelSpritesRef.current.clear();
      edgeLabelSpritesRef.current.clear();
      sortedNodeIdsRef.current = [];
      sortedEdgeIdsRef.current = [];
      graphInstanceRef.current?.graphData(newGraphData);
      setNodeCount(newGraphData.nodes.length);
      mountedVersionRef.current = graphVersion;
      mountedGraphIdRef.current = graphId;
      lastSizeDefaultsCountRef.current = newGraphData.nodes.length;
      // force the label loop to re-measure the bounding sphere and re-rank the new layout
      radiusVersionRef.current = -1;
      lastCamDistRef.current = -1;
      // re-frame the replacement graph once its fresh layout settles
      if (graphInstanceRef.current) armSettleFit(PendingFit.Initial);
      return;
    }

    if (graphVersion > mountedVersionRef.current) {
      const prevData = graphDataRef.current;
      applyGrowthToInstance(prevData, newGraphData, graphInstanceRef, labelSpritesRef, graphDataRef, setNodeCount);
      mountedVersionRef.current = graphVersion;
      lastCamDistRef.current = -1;

      // Re-apply size defaults if node count grew significantly (50%+)
      const currentCount = newGraphData.nodes.length;
      const lastApplied = lastSizeDefaultsCountRef.current;
      if (lastApplied > 0 && currentCount >= lastApplied * 1.5) {
        lastSizeDefaultsCountRef.current = currentCount;
        updateControls({ type: 'applySizeDefaults', state: computeSizeDefaults(currentCount) });
      }

      // refit once the force engine settles after absorbing the new nodes; this also
      // refreshes the saved "home" view to the newly framed layout
      if (refitOnGrowRef.current && graphInstanceRef.current) {
        armSettleFit(PendingFit.Grow);
      }
    }
  }, [graphId, graphVersion]);

  const handleNodeSelect = async (node: GraphNode) => {
    updateControls({
      type: 'selected',
      state: { kind: 'node', id: node.id, label: node.label },
    });

    if (focusSettingsRef.current.focusOnClick) {
      setFocusedNode(node.id, FocusSource.Graph);
      focusOnNode(node);
    }

    if (!growable.has(node.id) || !graphId) return;
    await grow(node.id);
  };
  handleNodeSelectRef.current = handleNodeSelect;

  const handleEdgeSelect = (link: GraphLink) => {
    const { source, target } = getLinkEndpoints(link);
    updateControls({
      type: 'selected',
      state: { kind: 'link', source, target, label: link.label },
    });

    const settings = focusSettingsRef.current;
    if (settings.focusOnClick) {
      const gi = graphInstanceRef.current;
      if (gi) {
        // frame just the edge's two endpoints (fit mode) or recenter on their midpoint
        const sphere = computeBoundingSphere(graphDataRef.current.nodes, new Set([source, target]));
        if (sphere) {
          focusCameraOn(gi, sphere.center, settings.fitNeighborhood ? sphereFitDistance(cameraFov(gi), sphere.radius) : null);
        }
      }
    }
  };
  handleEdgeSelectRef.current = handleEdgeSelect;

  // Mount the ForceGraph3D instance once data is ready
  useEffect(() => {
    if (!containerRef.current || !mounted) return;

    if (graphInstanceRef.current) {
      graphInstanceRef.current._destructor();
      graphInstanceRef.current = null;
    }
    labelSpritesRef.current.clear();
    edgeLabelSpritesRef.current.clear();

    // ForceGraph3D constructor returns the base generic; cast to our typed alias
    const fg = new ForceGraph3D(containerRef.current, {
      controlType: 'orbit',
    }) as unknown as GraphInstance;

    fg.graphData(graphDataRef.current)
      .backgroundColor('rgba(0,0,0,0)')
      .width(containerRef.current.clientWidth)
      .height(containerRef.current.clientHeight || window.innerHeight * 0.9)
      .nodeVal(controls.nodeRenderMode === NodeRenderMode.Icons ? iconNodeVal(controls.nodeRelSize) : (node: GraphNode) => node.diameter)
      .nodeColor((node: GraphNode) => getNodeColor(node.nodeType, node.visualState))
      .nodeLabel(() => '')
      .nodeThreeObject(
        buildNodeObject(
          controls.nodeRenderMode,
          controls.showNodeLabels,
          controls.nodeRelSize,
          labelSpritesRef.current,
          controls.nodeOpacity,
        ),
      )
      .nodeThreeObjectExtend(controls.nodeRenderMode === NodeRenderMode.Spheres)
      .nodeRelSize(controls.nodeRelSize)
      .nodeOpacity(controls.nodeOpacity)
      .linkDirectionalArrowLength(controls.arrowLength)
      .linkDirectionalArrowRelPos(1)
      .linkThreeObjectExtend(controls.showEdgeLabels)
      .linkThreeObject(
        controls.showEdgeLabels
          ? (((link: GraphLink) =>
              buildEdgeLabelFactory(edgeLabelSpritesRef.current, (id: string) => nodeDegreesRef.current.get(id))(link)) as (
              link: GraphLink,
            ) => THREE.Object3D)
          : (undefined as never),
      )
      .linkPositionUpdate(
        controls.showEdgeLabels
          ? (
              sprite: THREE.Object3D | undefined,
              coords: {
                start: { x: number; y: number; z: number };
                end: { x: number; y: number; z: number };
              },
            ) => {
              if (!sprite) return false;
              sprite.position.set(
                (coords.start.x + coords.end.x) / 2,
                (coords.start.y + coords.end.y) / 2,
                (coords.start.z + coords.end.z) / 2,
              );
              return false;
            }
          : (null as never),
      )
      .linkColor(() => getEdgeColor())
      .linkWidth(controls.edgeWidth)
      .linkOpacity(controls.edgeOpacity)
      .linkCurvature((link: GraphLink) => (link.bidirectional ? 0.2 : 0))
      .linkDirectionalParticles(controls.directionalParticles)
      .linkDirectionalParticleSpeed(controls.particleSpeed)
      .linkDirectionalParticleThreeObject(isCrabTheme() ? () => buildCrabParticle() : (undefined as never))
      .enableNodeDrag(controls.enableNodeDrag)
      .onNodeClick((node: GraphNode) => setTimeout(() => void handleNodeSelectRef.current?.(node), 0))
      .onLinkClick((link: GraphLink) => setTimeout(() => handleEdgeSelectRef.current?.(link), 0))
      .onNodeHover((node: GraphNode | null) => {
        window.clearTimeout(hoverHideTimer.current);
        // a hovered node with finite screen coordinates shows its popover; anything else
        // (leaving all nodes, or coords that can't be projected) starts the hide timer so
        // a previously shown popover doesn't linger with its hide already cancelled
        let shown = false;
        if (node && node.x !== undefined && node.y !== undefined && graphInstanceRef.current && containerRef.current) {
          const screenCoords = graphInstanceRef.current.graph2ScreenCoords(node.x, node.y, node.z ?? 0);
          if (isFinite(screenCoords.x) && isFinite(screenCoords.y)) {
            const rect = containerRef.current.getBoundingClientRect();
            setHoveredNode({
              id: node.id,
              x: rect.left + screenCoords.x + 15,
              y: rect.top + screenCoords.y + 15,
            });
            shown = true;
          }
        }
        if (!shown) {
          hoverHideTimer.current = window.setTimeout(() => setHoveredNode(null), 200);
        }
      })
      .numDimensions(controls.numDimensions)
      .warmupTicks(controls.warmupTicks)
      .cooldownTime(controls.cooldownTime)
      .d3VelocityDecay(controls.velocityDecay);

    if (controls.dagMode) {
      fg.dagMode(controls.dagMode);
      if (controls.dagLevelDistance !== null) {
        fg.dagLevelDistance(controls.dagLevelDistance);
      }
    }

    const chargeForce = fg.d3Force('charge') as D3ChargeForce | undefined;
    if (chargeForce && 'strength' in chargeForce) {
      chargeForce.strength(controls.chargeStrength);
    }

    const linkForce = fg.d3Force('link') as D3LinkForce | undefined;
    if (linkForce && 'distance' in linkForce) {
      linkForce.distance(controls.edgeLength);
      linkForce.strength(controls.edgeLinkStrength);
    }

    graphInstanceRef.current = fg;

    const orbitControls = fg.controls() as GraphOrbitControls | undefined;
    const freezeNodes = () => {
      // manual camera input means the user has taken over; the automatic initial fit
      // checks this so it never rips the camera away mid-interaction
      userInteractedRef.current = true;
      cancelCameraAnim();
      const data = fg.graphData();
      for (const n of data.nodes) {
        if (n.x !== undefined) {
          n.fx = n.x;
          n.fy = n.y;
          n.fz = n.z;
        }
      }
    };
    const unfreezeNodes = () => {
      const data = fg.graphData();
      for (const n of data.nodes) {
        n.fx = undefined;
        n.fy = undefined;
        n.fz = undefined;
      }
    };
    if (orbitControls) {
      orbitControls.zoomToCursor = true;
      orbitControls.zoomSpeed = ZOOM_SPEED;
      orbitControls.addEventListener('start', freezeNodes);
      orbitControls.addEventListener('end', unfreezeNodes);
    }

    const rendererDom = fg.renderer().domElement;
    const swallowPointerCancel = (e: PointerEvent) => {
      e.stopImmediatePropagation();
    };
    rendererDom.addEventListener('pointercancel', swallowPointerCancel, true);

    const gridGroup = new THREE.Group();
    const xzGrid = new THREE.GridHelper(2000, 40);
    (xzGrid.material as THREE.Material).opacity = 0.15;
    (xzGrid.material as THREE.Material).transparent = true;
    gridGroup.add(xzGrid);

    const xyGrid = new THREE.GridHelper(2000, 40);
    (xyGrid.material as THREE.Material).opacity = 0.1;
    (xyGrid.material as THREE.Material).transparent = true;
    xyGrid.rotation.x = Math.PI / 2;
    gridGroup.add(xyGrid);

    gridGroup.visible = controls.showGrid;
    fg.scene().add(gridGroup);
    gridRef.current = gridGroup;

    const enforceMinOrbitRadius = () => {
      const gi = graphInstanceRef.current;
      if (!gi) return;
      const ctrl = gi.controls() as GraphOrbitControls | undefined;
      if (!ctrl?.target) return;
      const cam = gi.camera();
      const { target } = ctrl;
      const radius = cam.position.distanceTo(target);
      if (radius < MIN_ORBIT_RADIUS) {
        const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
        target.set(
          cam.position.x + fwd.x * MIN_ORBIT_RADIUS,
          cam.position.y + fwd.y * MIN_ORBIT_RADIUS,
          cam.position.z + fwd.z * MIN_ORBIT_RADIUS,
        );
      }
    };
    containerRef.current?.addEventListener('wheel', enforceMinOrbitRadius, { capture: true, passive: true });

    // bound every graph to a fixed tick budget so the layout settles in a couple of
    // seconds and onEngineStop fires promptly, instead of running the full cooldownTime
    // wall clock (which left small graphs jiggling for ~15s)
    fg.cooldownTicks(SETTLE_COOLDOWN_TICKS);

    // Apply size-aware defaults based on initial node count
    const initialNodeCount = graphDataRef.current.nodes.length;
    lastSizeDefaultsCountRef.current = initialNodeCount;
    if (initialNodeCount > 30) {
      const sizeDefaults = computeSizeDefaults(initialNodeCount);
      updateControls({ type: 'applySizeDefaults', state: sizeDefaults });
    }

    // last-seen camera/orbit vectors for the label-scaling early-exit gate; start at
    // Infinity so the first frame always recomputes
    const lastCamPos = new THREE.Vector3(Infinity, Infinity, Infinity);
    const lastOrbitTarget = new THREE.Vector3(Infinity, Infinity, Infinity);
    const updateLabelScaling = () => {
      const gi = graphInstanceRef.current;
      const nodeLabels = labelSpritesRef.current;
      const edgeLabels = edgeLabelSpritesRef.current;
      const totalLabels = nodeLabels.size + edgeLabels.size;

      if (!gi || totalLabels === 0) {
        animFrameRef.current = requestAnimationFrame(updateLabelScaling);
        return;
      }

      const camPos = gi.cameraPosition();
      const orbitCtrl = gi.controls() as GraphOrbitControls | undefined;
      const target = orbitCtrl?.target;
      if (!target) {
        animFrameRef.current = requestAnimationFrame(updateLabelScaling);
        return;
      }

      const dist = Math.hypot(camPos.x - target.x, camPos.y - target.y, camPos.z - target.z);

      // Skip recalculation unless the camera position or orbit target actually moved:
      // per-label distances change during pure orbit/pan even when the camera-to-target
      // distance stays constant. -1 in lastCamDistRef forces a recompute (set by control
      // changes and graph growth).
      const camThreshold = dist * CAM_DIST_THRESHOLD_RATIO + CAM_DIST_THRESHOLD_MIN;
      const camMoved =
        Math.hypot(camPos.x - lastCamPos.x, camPos.y - lastCamPos.y, camPos.z - lastCamPos.z) >= camThreshold ||
        Math.hypot(target.x - lastOrbitTarget.x, target.y - lastOrbitTarget.y, target.z - lastOrbitTarget.z) >= camThreshold;
      // label sprites are created lazily during the force-graph digest, so they can
      // appear frames after the -1 sentinel was consumed; a size mismatch against the
      // last ranking catches those late arrivals
      const labelMapsChanged = sortedNodeIdsRef.current.length !== nodeLabels.size || sortedEdgeIdsRef.current.length !== edgeLabels.size;
      if (lastCamDistRef.current >= 0 && !camMoved && !labelMapsChanged) {
        animFrameRef.current = requestAnimationFrame(updateLabelScaling);
        return;
      }
      // -1 marks label-map or pin changes (label toggles, growth, selection/focus),
      // which alongside late sprite creation are the only events that can alter the ranking
      const rankingInvalidated = lastCamDistRef.current === -1 || labelMapsChanged;
      lastCamDistRef.current = dist;
      lastCamPos.set(camPos.x, camPos.y, camPos.z);
      lastOrbitTarget.copy(target);

      if (radiusVersionRef.current !== mountedVersionRef.current) {
        // frame the whole positioned graph; unlike focus fitting this includes every node
        const sphere = computeCentroidRadius(graphDataRef.current.nodes, () => true);
        if (sphere) {
          boundingRadiusRef.current = Math.max(200, sphere.radius);
          radiusVersionRef.current = mountedVersionRef.current;
        }
      }

      // rebuild the global importance rankings only when the label maps or pins changed;
      // a stable global ranking keeps the visible set steady while orbiting
      if (rankingInvalidated) {
        const nodeEntries = [...nodeLabels.entries()];
        // rank by degree desc, id asc so ties resolve deterministically
        nodeEntries.sort((a, b) => b[1].degree - a[1].degree || (a[0] < b[0] ? -1 : 1));
        sortedNodeIdsRef.current = nodeEntries.map(([id]) => id);
        initialNodeIdsRef.current = new Set(nodeEntries.filter(([, entry]) => entry.isInitial).map(([id]) => id));
        // top-decile degree marks "hub" labels for the mid size tier
        hubDegreeThresholdRef.current = nodeEntries.length > 0 ? nodeEntries[Math.floor(nodeEntries.length / 10)][1].degree : Infinity;
        // edge rank (min endpoint degree) is computed at sprite creation in buildEdgeLabelFactory
        const edgeEntries = [...edgeLabels.entries()];
        edgeEntries.sort((a, b) => b[1].degree - a[1].degree || (a[0] < b[0] ? -1 : 1));
        sortedEdgeIdsRef.current = edgeEntries.map(([id]) => id);
      }

      const camVec = new THREE.Vector3(camPos.x, camPos.y, camPos.z);
      // camera projection inputs shared by every label this frame; fall back to the
      // three.js default fov in case the camera is not a PerspectiveCamera
      const fovDeg = (gi.camera() as Partial<THREE.PerspectiveCamera>).fov ?? 50;
      const viewportHeightPx = gi.height();
      // camera distance that would frame the whole graph; the zoom-scaled label budget
      // compares the actual camera distance against it
      const fitDist = boundingRadiusRef.current / Math.tan((fovDeg * Math.PI) / 360);

      // pinned labels bypass the visibility budget: the selection, the focused node,
      // and every initial seed node
      const selectedEl = selectedElementRef.current;
      const pinnedNodeIds = new Set(initialNodeIdsRef.current);
      if (selectedEl?.kind === 'node') pinnedNodeIds.add(selectedEl.id);
      if (focusedNodeIdRef.current) pinnedNodeIds.add(focusedNodeIdRef.current);
      const pinnedEdgeIds = new Set<string>();
      if (selectedEl?.kind === 'link') pinnedEdgeIds.add(`${selectedEl.source}-${selectedEl.target}`);

      const visibleNodeIds = selectVisibleLabels(sortedNodeIdsRef.current, {
        density: nodeLabelDensityRef.current,
        camDist: dist,
        fitDist,
        pinnedIds: pinnedNodeIds,
        baseBudget: NODE_LABEL_BUDGET,
      });
      const visibleEdgeIds = selectVisibleLabels(sortedEdgeIdsRef.current, {
        density: edgeLabelDensityRef.current,
        camDist: dist,
        fitDist,
        pinnedIds: pinnedEdgeIds,
        baseBudget: EDGE_LABEL_BUDGET,
      });

      const nodeScale = nodeLabelScaleRef.current;
      const hubThreshold = hubDegreeThresholdRef.current;
      nodeLabels.forEach((entry, id) => {
        if (!visibleNodeIds.has(id)) {
          entry.sprite.visible = false;
          return;
        }
        entry.sprite.visible = true;
        const parent = entry.sprite.parent;
        const nodeDist = parent ? camVec.distanceTo(parent.position) : dist;
        // size tiers: pinned labels largest, hub (top-decile degree) labels mid, rest base
        const tierBoost = pinnedNodeIds.has(id) ? PINNED_TIER_BOOST : entry.degree >= hubThreshold ? HUB_TIER_BOOST : 1.0;
        const s = computeLabelScale({
          labelDist: nodeDist,
          viewportHeightPx,
          fovDeg,
          baseScaleY: entry.baseScale.y,
          labelScale: nodeScale,
          tierBoost,
          targetPx: LABEL_TARGET_PX,
          minPx: LABEL_MIN_PX,
          maxPx: LABEL_MAX_PX,
        });
        entry.sprite.scale.set(entry.baseScale.x * s, entry.baseScale.y * s, entry.baseScale.z);
      });

      const edgeScale = edgeLabelScaleRef.current;
      const worldPos = new THREE.Vector3();
      edgeLabels.forEach((entry, id) => {
        if (!visibleEdgeIds.has(id)) {
          entry.sprite.visible = false;
          return;
        }
        entry.sprite.visible = true;
        entry.sprite.getWorldPosition(worldPos);
        const edgeDist = camVec.distanceTo(worldPos);
        // edge labels only tier up when pinned (no hub tier for edges)
        const tierBoost = pinnedEdgeIds.has(id) ? PINNED_TIER_BOOST : 1.0;
        const s = computeLabelScale({
          labelDist: edgeDist,
          viewportHeightPx,
          fovDeg,
          baseScaleY: entry.baseScale.y,
          labelScale: edgeScale,
          tierBoost,
          targetPx: EDGE_LABEL_TARGET_PX,
          minPx: EDGE_LABEL_MIN_PX,
          maxPx: EDGE_LABEL_MAX_PX,
        });
        entry.sprite.scale.set(entry.baseScale.x * s, entry.baseScale.y * s, entry.baseScale.z);
      });

      animFrameRef.current = requestAnimationFrame(updateLabelScaling);
    };
    animFrameRef.current = requestAnimationFrame(updateLabelScaling);

    // the engine-stop hook is a single-slot setter that fires on EVERY engine stop, so
    // it services both the initial fit and later refit-on-grow arms; completeSettleFit
    // no-ops unless a fit is currently armed
    fg.onEngineStop(completeSettleFit);
    // every graph settles on the fixed tick budget above, so onEngineStop fires within a
    // couple of seconds for both the initial framing and later refit-on-grow arms; the
    // fallback timer only covers the rare case where the engine never reports a stop
    armSettleFit(PendingFit.Initial);

    const container = containerRef.current;
    const handleDblClick = (event: MouseEvent) => {
      const gi = graphInstanceRef.current;
      if (!gi || !container) return;

      const rect = container.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );

      const camera = gi.cameraPosition();
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, gi.camera());

      const dblCtrl = gi.controls() as GraphOrbitControls;
      const { target } = dblCtrl;
      const dist = Math.hypot(camera.x - target.x, camera.y - target.y, camera.z - target.z);
      const dir = raycaster.ray.direction;
      // shift inverts the step so shift+double-click backs the camera away from the
      // cursor instead of advancing toward it; both land looking at the same ray point
      const step = dist * (event.shiftKey ? -DBLCLICK_ZOOM_STEP_RATIO : DBLCLICK_ZOOM_STEP_RATIO);

      const newPos = { x: camera.x + dir.x * step, y: camera.y + dir.y * step, z: camera.z + dir.z * step };
      const lookDist = dist - step;
      const newLookAt = { x: newPos.x + dir.x * lookDist, y: newPos.y + dir.y * lookDist, z: newPos.z + dir.z * lookDist };

      animateCameraTo(gi, newPos, newLookAt, 500);
    };
    container.addEventListener('dblclick', handleDblClick);

    const resizeObserver = new ResizeObserver((entries) => {
      const gi = graphInstanceRef.current;
      if (!gi) return;
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          gi.width(width).height(height);
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      cancelCameraAnim();
      cancelAnimationFrame(animFrameRef.current);
      window.clearTimeout(settleFitFallbackRef.current);
      window.clearTimeout(saveHomeTimerRef.current);
      window.clearTimeout(hoverHideTimer.current);
      pendingSettleFitRef.current = null;
      if (orbitControls) {
        orbitControls.removeEventListener('start', freezeNodes);
        orbitControls.removeEventListener('end', unfreezeNodes);
      }
      container.removeEventListener('wheel', enforceMinOrbitRadius, { capture: true });
      container.removeEventListener('dblclick', handleDblClick);
      rendererDom.removeEventListener('pointercancel', swallowPointerCancel, true);
      resizeObserver.disconnect();
      fg._destructor();
      graphInstanceRef.current = null;
    };
  }, [mounted]);

  // Grow frontier nodes when depth increases. The provider's growToDepth owns the raise-only, tree-scoped,
  // success-gated guard shared with the entity browser, so a redundant/lower change is a no-op there.
  useEffect(() => {
    if (!graphId || controls.depth <= 1) return;
    void growToDepth(controls.depth);
  }, [controls.depth, graphId]);

  // Re-fetch when filterChildless changes
  useEffect(() => {
    if (!graphId) return;
    void reload({ filterChildless: controls.filterChildless });
  }, [controls.filterChildless]);

  useEffect(() => {
    if (gridRef.current) gridRef.current.visible = controls.showGrid;
  }, [controls.showGrid]);

  // Focus changes pin the focused node's label; invalidate the label loop's camera
  // cache so the declutter pass re-runs on the next frame
  useEffect(() => {
    lastCamDistRef.current = -1;
  }, [focusedNodeId]);

  // Animate camera to focused node when focus originates from tree
  useEffect(() => {
    if (!focusedNodeId || focusSource !== FocusSource.Tree) return;

    const node = graphDataRef.current.nodes.find((n) => n.id === focusedNodeId);
    if (!node || node.x === undefined || node.y === undefined) return;

    updateControls({
      type: 'selected',
      state: { kind: 'node', id: node.id, label: node.label },
    });

    focusOnNode(node);
  }, [focusedNodeId, focusSource]);

  // the info model for the currently hovered node (null for unknown/empty nodes)
  const hoverModel = hoveredNode && graph.data_map[hoveredNode.id] ? treeNodeToInfo(graph.data_map[hoveredNode.id]) : null;

  return (
    // the window is focusable (not just its buttons) so the camera shortcuts work as
    // soon as the user tabs to or clicks anywhere inside the graph
    <GraphWindow $bordered={bordered} tabIndex={0} onKeyDown={handleGraphKeyDown}>
      <GraphDiv ref={containerRef} />
      {hoveredNode &&
        hoverModel &&
        // Cursor-anchored special case: there's no DOM element to anchor to (the node lives in the 3D
        // canvas), so this can't use `EntitySummaryHover` (which needs a ref target). It still reuses the
        // shared `SummaryPopover` + `EntitySummary` and keeps its own keep-open timers so the pointer can
        // move onto the popover and scroll.
        createPortal(
          <SummaryPopover
            id="graph-hover-preview"
            style={{ position: 'fixed', left: hoveredNode.x, top: hoveredNode.y, zIndex: 9999 }}
            onMouseEnter={() => window.clearTimeout(hoverHideTimer.current)}
            onMouseLeave={() => {
              hoverHideTimer.current = window.setTimeout(() => setHoveredNode(null), 200);
            }}
          >
            <Popover.Body>
              <EntitySummary model={hoverModel} variant={SummaryVariant.Compact} />
            </Popover.Body>
          </SummaryPopover>,
          document.body,
        )}
      {loading && (
        <LoadingOverlay>
          <Spinner animation="border" variant="secondary" />
        </LoadingOverlay>
      )}
      {/* surface load failures and empty results so the canvas is never a silent blank. graphId is empty
          until the initial tree loads, so gating the empty message on it avoids a pre-load flash. */}
      {!loading && error && (
        <LoadingOverlay>
          <GraphOverlayMessage>Failed to load graph</GraphOverlayMessage>
        </LoadingOverlay>
      )}
      {!loading && !error && graphId !== '' && nodeCount === 0 && (
        <LoadingOverlay>
          <GraphOverlayMessage>No graph data</GraphOverlayMessage>
        </LoadingOverlay>
      )}
      {treeOverlayOpen ? (
        <TreeOverlayPanel>
          <TreeOverlayHeader>
            <MinimizeButton onClick={() => setTreeOverlayOpen(false)}>
              <GoSidebarCollapse size={14} />
            </MinimizeButton>
          </TreeOverlayHeader>
          <AssociationTree />
        </TreeOverlayPanel>
      ) : (
        <OverlayTipRight tip="File Browser">
          <TreeOverlayToggle onClick={() => setTreeOverlayOpen(true)}>
            <FaFolderTree size={14} />
          </TreeOverlayToggle>
        </OverlayTipRight>
      )}
      <GraphControlsToolbar
        graphId={graphId}
        controls={controls}
        updateControls={updateControls}
        graphInstance={graphInstanceRef.current}
        nodeCount={nodeCount}
        loading={loading}
        growing={growing}
      />
      <NavCluster onZoomIn={handleZoomIn} onZoomOut={handleZoomOut} onFitAll={handleFitAll} onResetView={handleResetView} />
      <DataPreviewPanel
        selectedElement={controls.selectedElement}
        nodeData={controls.selectedElement?.kind === 'node' ? graph.data_map[controls.selectedElement.id] : undefined}
        minimized={previewMinimized}
        onToggleMinimize={() => setPreviewMinimized((m) => !m)}
      />
    </GraphWindow>
  );
};

export const AssociationGraph: React.FC<AssociationGraphProps> = ({ inView, bordered }) => {
  return (
    <ErrorBoundary fallback={<RenderErrorAlert page={false} />}>{inView && <AssociationGraph3DInner bordered={bordered} />}</ErrorBoundary>
  );
};

export default AssociationGraph;
