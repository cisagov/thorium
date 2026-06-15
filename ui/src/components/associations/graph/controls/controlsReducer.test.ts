import { describe, it, expect } from 'vitest';

// project imports
import { createControlsReducer } from './controlsReducer';
import { NodeRenderMode, DagMode } from './types';
import type { GraphControls } from './types';

function defaultControls(): GraphControls {
  return {
    filterChildless: false,
    depth: 1,
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
    dagMode: null,
    dagLevelDistance: null,
    numDimensions: 3,
    showGrid: false,
    userOverrides: new Set<string>(),
  };
}

const nullRef = { current: null };
const labelMapRef = { current: new Map() };
const edgeLabelMapRef = { current: new Map() };

function createTestReducer() {
  return createControlsReducer(nullRef, labelMapRef, edgeLabelMapRef);
}

// Variant with a camera-distance ref so tests can observe the -1 invalidation sentinel
function createTestReducerWithCamRef(camDistRef: { current: number }) {
  return createControlsReducer(nullRef, labelMapRef, edgeLabelMapRef, camDistRef);
}

describe('controlsReducer', () => {
  it('toggles showEdgeLabels', () => {
    const reducer = createTestReducer();
    const state = defaultControls();
    const result = reducer(state, { type: 'showEdgeLabels', state: true });
    expect(result.showEdgeLabels).toBe(true);
  });

  it('toggles showNodeLabels', () => {
    const reducer = createTestReducer();
    const state = defaultControls();
    const result = reducer(state, { type: 'showNodeLabels', state: false });
    expect(result.showNodeLabels).toBe(false);
  });

  it('sets selected element', () => {
    const reducer = createTestReducer();
    const state = defaultControls();
    const selected = { kind: 'node' as const, id: 'n1', label: 'Test' };
    const result = reducer(state, { type: 'selected', state: selected });
    expect(result.selectedElement).toEqual(selected);
  });

  it('clears selected element', () => {
    const reducer = createTestReducer();
    const state = { ...defaultControls(), selectedElement: { kind: 'node' as const, id: 'n1', label: 'Test' } };
    const result = reducer(state, { type: 'selected', state: null });
    expect(result.selectedElement).toBeNull();
  });

  it('sets depth', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'depth', state: 3 });
    expect(result.depth).toBe(3);
  });

  it('toggles filterChildless', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'filterChildless', state: true });
    expect(result.filterChildless).toBe(true);
  });

  it('toggles focusOnClick', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'focusOnClick', state: false });
    expect(result.focusOnClick).toBe(false);
  });

  it('toggles fitNeighborhoodOnFocus off from its default of true', () => {
    const reducer = createTestReducer();
    const state = defaultControls();
    expect(state.fitNeighborhoodOnFocus).toBe(true);
    const result = reducer(state, { type: 'fitNeighborhoodOnFocus', state: false });
    expect(result.fitNeighborhoodOnFocus).toBe(false);
  });

  it('sets edgeWidth and marks user override', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'edgeWidth', state: 3 });
    expect(result.edgeWidth).toBe(3);
    expect(result.userOverrides.has('edgeWidth')).toBe(true);
  });

  it('sets edgeLength and marks user override', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'edgeLength', state: 50 });
    expect(result.edgeLength).toBe(50);
    expect(result.userOverrides.has('edgeLength')).toBe(true);
  });

  it('sets edgeOpacity and marks user override', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'edgeOpacity', state: 0.5 });
    expect(result.edgeOpacity).toBe(0.5);
    expect(result.userOverrides.has('edgeOpacity')).toBe(true);
  });

  it('sets chargeStrength and marks user override', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'chargeStrength', state: -500 });
    expect(result.chargeStrength).toBe(-500);
    expect(result.userOverrides.has('chargeStrength')).toBe(true);
  });

  it('sets nodeRelSize and marks user override', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'nodeRelSize', state: 8 });
    expect(result.nodeRelSize).toBe(8);
    expect(result.userOverrides.has('nodeRelSize')).toBe(true);
  });

  it('sets nodeRenderMode', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'nodeRenderMode', state: NodeRenderMode.Spheres });
    expect(result.nodeRenderMode).toBe(NodeRenderMode.Spheres);
  });

  it('sets dagMode', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'dagMode', state: DagMode.TopDown });
    expect(result.dagMode).toBe(DagMode.TopDown);
  });

  it('sets numDimensions', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'numDimensions', state: 2 });
    expect(result.numDimensions).toBe(2);
  });

  it('toggles showGrid', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'showGrid', state: true });
    expect(result.showGrid).toBe(true);
  });

  it('applySizeDefaults skips user-overridden keys', () => {
    const reducer = createTestReducer();
    const state = { ...defaultControls(), edgeWidth: 5, userOverrides: new Set(['edgeWidth']) };
    const result = reducer(state, {
      type: 'applySizeDefaults',
      state: { edgeWidth: 2, chargeStrength: -100 },
    });
    expect(result.edgeWidth).toBe(5);
    expect(result.chargeStrength).toBe(-100);
  });

  it('applySizeDefaults applies all non-overridden keys', () => {
    const reducer = createTestReducer();
    const state = defaultControls();
    const result = reducer(state, {
      type: 'applySizeDefaults',
      state: { edgeWidth: 2, edgeLength: 40 },
    });
    expect(result.edgeWidth).toBe(2);
    expect(result.edgeLength).toBe(40);
  });

  it('resetSizeOverrides clears all user overrides', () => {
    const reducer = createTestReducer();
    const state = { ...defaultControls(), userOverrides: new Set(['edgeWidth', 'chargeStrength']) };
    const result = reducer(state, { type: 'resetSizeOverrides' });
    expect(result.userOverrides.size).toBe(0);
  });

  it('non-size-scaled keys do not add to userOverrides', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'particleSpeed', state: 0.01 });
    expect(result.particleSpeed).toBe(0.01);
    expect(result.userOverrides.size).toBe(0);
  });

  it('sets nodeLabelDensity without marking a user override', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'nodeLabelDensity', state: 0.3 });
    expect(result.nodeLabelDensity).toBe(0.3);
    expect(result.userOverrides.has('nodeLabelDensity')).toBe(false);
  });

  it('nodeLabelScale updates state, marks a user override, and invalidates the camera cache without a graph instance', () => {
    const camDistRef = { current: 250 };
    const reducer = createTestReducerWithCamRef(camDistRef);
    const result = reducer(defaultControls(), { type: 'nodeLabelScale', state: 1.5 });
    expect(result.nodeLabelScale).toBe(1.5);
    expect(result.userOverrides.has('nodeLabelScale')).toBe(true);
    expect(camDistRef.current).toBe(-1);
  });

  it('edgeLabelScale updates state, marks a user override, and invalidates the camera cache without a graph instance', () => {
    const camDistRef = { current: 250 };
    const reducer = createTestReducerWithCamRef(camDistRef);
    const result = reducer(defaultControls(), { type: 'edgeLabelScale', state: 0.7 });
    expect(result.edgeLabelScale).toBe(0.7);
    expect(result.userOverrides.has('edgeLabelScale')).toBe(true);
    expect(camDistRef.current).toBe(-1);
  });

  it('applySizeDefaults skips an overridden nodeLabelScale', () => {
    const reducer = createTestReducer();
    const state = { ...defaultControls(), nodeLabelScale: 1.8, userOverrides: new Set(['nodeLabelScale']) };
    const result = reducer(state, {
      type: 'applySizeDefaults',
      state: { nodeLabelScale: 0.85, edgeLabelScale: 0.85 },
    });
    expect(result.nodeLabelScale).toBe(1.8);
    expect(result.edgeLabelScale).toBe(0.85);
  });

  it('selected invalidates the camera cache so label pin changes apply next frame', () => {
    const camDistRef = { current: 250 };
    const reducer = createTestReducerWithCamRef(camDistRef);
    const selected = { kind: 'node' as const, id: 'n1', label: 'Test' };
    const result = reducer(defaultControls(), { type: 'selected', state: selected });
    expect(result.selectedElement).toEqual(selected);
    expect(camDistRef.current).toBe(-1);
  });

  it('applySizeDefaults applies label scales and invalidates the camera cache without a graph instance', () => {
    const camDistRef = { current: 250 };
    const reducer = createTestReducerWithCamRef(camDistRef);
    const result = reducer(defaultControls(), {
      type: 'applySizeDefaults',
      state: { nodeLabelScale: 0.9, edgeLabelScale: 0.9 },
    });
    expect(result.nodeLabelScale).toBe(0.9);
    expect(result.edgeLabelScale).toBe(0.9);
    expect(camDistRef.current).toBe(-1);
  });
});
