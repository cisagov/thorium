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
    showNodeInfo: true,
    nodeRenderMode: NodeRenderMode.Icons,
    focusOnClick: true,
    adjustDistanceOnFocus: false,
    refitOnGrow: false,
    focusDistanceRatio: 1,
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
    nodeLabelDensity: 0.7,
    nodeLabelMinSize: 1,
    edgeLabelDensity: 0.7,
    edgeLabelMinSize: 1,
    chargeStrength: -200,
    velocityDecay: 0.4,
    warmupTicks: 0,
    cooldownTime: 15000,
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

  it('toggles showNodeInfo', () => {
    const reducer = createTestReducer();
    const state = defaultControls();
    const result = reducer(state, { type: 'showNodeInfo', state: false });
    expect(result.showNodeInfo).toBe(false);
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

  it('sets focusDistanceRatio', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'focusDistanceRatio', state: 2.5 });
    expect(result.focusDistanceRatio).toBe(2.5);
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

  it('sets nodeLabelDensity and marks user override', () => {
    const reducer = createTestReducer();
    const result = reducer(defaultControls(), { type: 'nodeLabelDensity', state: 0.5 });
    expect(result.nodeLabelDensity).toBe(0.5);
    expect(result.userOverrides.has('nodeLabelDensity')).toBe(true);
  });
});
