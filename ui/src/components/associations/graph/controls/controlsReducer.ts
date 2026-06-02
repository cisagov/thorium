import * as THREE from 'three';
import SpriteText from 'three-spritetext';

import { getNodeColor, getNodeSvg, svgToTexture, getEdgeColor } from '../styles';
import { NodeRenderMode, DagMode } from './types';
import type { GraphControls, DisplayAction } from './types';
import { SIZE_SCALED_KEYS } from './sizeDefaults';
import { VisualState } from '../types';
import type { GraphNode, GraphLink, GraphInstance, D3ChargeForce, D3LinkForce } from '../types';

export type LabelEntry = { sprite: THREE.Object3D; degree: number; isInitial: boolean; baseScale: THREE.Vector3 };

// Directional arrows land at `cbrt(nodeVal) * nodeRelSize` from the node center
// (see three-forcegraph arrow placement). iconHalf is the sprite's bounding-box
// half-width, but our Material icons only fill the ~20/24 "live area" of that box
// (the outer ring is transparent padding). This factor sets where the arrow tip
// lands relative to iconHalf: 0.9 puts it just past the visible glyph edge (~0.83)
// so arrows stop right before the icon instead of floating in the transparent gap.
const ICON_EDGE_PAD = 0.9;

// Per-node depth occluder. The library draws links center-to-center, so the
// segment between the arrow tip (at ICON_EDGE_PAD * iconHalf) and the node center
// keeps rendering. Opaque sphere nodes would hide that stub; our 2D icon sprites
// (depthWrite: false) don't, so it shows through/over the icon. We add an
// invisible sphere that writes depth but not color, sized to the arrow-tip
// radius, so the GPU depth-culls the stub (which lies entirely inside that radius)
// in every camera direction. One geometry/material is shared across all nodes.
const OCCLUDER_GEOMETRY = new THREE.SphereGeometry(1, 8, 6);
// transparent: true keeps this in the transparent pass so renderOrder sequences
// it AFTER the icon sprite; an opaque occluder would render first and hide the icon.
const OCCLUDER_MATERIAL = new THREE.MeshBasicMaterial({
  colorWrite: false,
  depthWrite: true,
  depthTest: true,
  transparent: true,
});

export const iconNodeVal =
  (nodeRelSize: number) =>
  (node: GraphNode): number => {
    const iconHalf = (Math.max(6, node.diameter / 3) * (nodeRelSize / 4)) / 2;
    const r = (iconHalf * ICON_EDGE_PAD) / nodeRelSize;
    return r * r * r;
  };

export const buildNodeObject = (
  renderMode: NodeRenderMode,
  showLabels: boolean,
  nodeRelSize: number,
  labelScale: number,
  labelMap?: Map<string, LabelEntry>,
  nodeOpacity = 1,
) => {
  const sizeFactor = nodeRelSize / 4;
  return (node: GraphNode): THREE.Object3D => {
    const group = new THREE.Group();

    if (renderMode === NodeRenderMode.Icons) {
      const svgString = getNodeSvg(node.nodeType, node.visualState);
      const texture = svgToTexture(svgString, 64);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, opacity: nodeOpacity });
      const sprite = new THREE.Sprite(spriteMaterial);
      const scale = Math.max(6, node.diameter / 3) * sizeFactor;
      sprite.scale.set(scale, scale, 1);
      group.add(sprite);

      // Invisible depth occluder sized to the arrow-tip radius, so the line stub
      // between the arrow and the node center is depth-culled. renderOrder must
      // sit between the icon sprite (0, drawn first so it survives) and the links
      // (10, depth-tested against this and culled where they enter the sphere).
      const occluder = new THREE.Mesh(OCCLUDER_GEOMETRY, OCCLUDER_MATERIAL);
      occluder.scale.setScalar((scale / 2) * ICON_EDGE_PAD);
      occluder.renderOrder = 1;
      group.add(occluder);
    }

    if (showLabels) {
      const labelSprite = new SpriteText(node.label);
      labelSprite.color = getNodeColor(node.nodeType, node.visualState);
      labelSprite.textHeight = 3 * labelScale;
      labelSprite.position.y = renderMode === NodeRenderMode.Icons ? -(node.diameter / 5 + 4) * sizeFactor : -(node.diameter / 5 + 2);
      labelSprite.material.depthWrite = false;
      group.add(labelSprite);
      if (labelMap) {
        const obj = labelSprite as unknown as THREE.Object3D;
        labelMap.set(node.id, {
          sprite: obj,
          degree: node.degree,
          isInitial: node.visualState === VisualState.Initial,
          baseScale: obj.scale.clone(),
        });
      }
    } else if (labelMap) {
      labelMap.delete(node.id);
    }

    return group;
  };
};

export const buildEdgeLabelFactory = (labelScale: number, edgeLabelMap?: Map<string, LabelEntry>) => {
  return (link: GraphLink): THREE.Object3D | undefined => {
    if (!link.label) return undefined;
    const sprite = new SpriteText(link.label);
    sprite.color = getEdgeColor();
    sprite.textHeight = 2.5 * labelScale;
    sprite.material.depthWrite = false;

    if (edgeLabelMap) {
      const src = typeof link.source === 'object' ? (link.source as GraphNode).id : link.source;
      const tgt = typeof link.target === 'object' ? (link.target as GraphNode).id : link.target;
      const obj = sprite as unknown as THREE.Object3D;
      edgeLabelMap.set(`${src}-${tgt}`, { sprite: obj, degree: 1, isInitial: false, baseScale: obj.scale.clone() });
    }

    return sprite;
  };
};

// Impure reducer: mutates the ForceGraph3D instance imperatively via refs.
// This is intentional — the 3d-force-graph API is imperative, and syncing
// control state with graph properties inside the reducer keeps them atomic.
// Do not call this reducer outside of React's useReducer.
export const createControlsReducer = (
  graphInstanceRef: React.RefObject<GraphInstance | null>,
  labelSpritesRef: React.RefObject<Map<string, LabelEntry>>,
  edgeLabelSpritesRef: React.RefObject<Map<string, LabelEntry>>,
  lastCamDistRef?: React.RefObject<number>,
) => {
  const sizeScaledKeySet = new Set<string>(SIZE_SCALED_KEYS);

  // Mark a key as user-overridden so auto-scaling won't touch it
  const markOverride = (state: GraphControls, key: string): Set<string> => {
    if (!sizeScaledKeySet.has(key)) return state.userOverrides;
    const next = new Set(state.userOverrides);
    next.add(key);
    return next;
  };

  return (state: GraphControls, action: DisplayAction): GraphControls => {
    const gi = graphInstanceRef.current;
    switch (action.type) {
      case 'showEdgeLabels': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi) {
          if (action.state) {
            edgeLabelSpritesRef.current.clear();
            gi.linkThreeObjectExtend(true);
            gi.linkThreeObject(((link: GraphLink) => buildEdgeLabelFactory(state.edgeLabelScale, edgeLabelSpritesRef.current)(link)) as (
              link: GraphLink,
            ) => THREE.Object3D);
            gi.linkPositionUpdate(
              (
                sprite: THREE.Object3D | undefined,
                { start, end }: { start: { x: number; y: number; z: number }; end: { x: number; y: number; z: number } },
              ) => {
                if (!sprite) return false;
                sprite.position.set((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2);
                return false;
              },
            );
          } else {
            edgeLabelSpritesRef.current.clear();
            gi.linkThreeObjectExtend(false);
            gi.linkThreeObject(undefined as never);
            gi.linkPositionUpdate(null as never);
          }
          gi.refresh();
        }
        return { ...state, showEdgeLabels: action.state };
      }
      case 'showNodeLabels': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi) {
          labelSpritesRef.current.clear();
          gi.nodeThreeObject(
            buildNodeObject(
              state.nodeRenderMode,
              action.state,
              state.nodeRelSize,
              state.nodeLabelScale,
              labelSpritesRef.current,
              state.nodeOpacity,
            ),
          );
          gi.nodeThreeObjectExtend(state.nodeRenderMode === NodeRenderMode.Spheres);
          gi.refresh();
        }
        return { ...state, showNodeLabels: action.state };
      }
      case 'showNodeInfo':
        return { ...state, showNodeInfo: action.state };
      case 'selected':
        return { ...state, selectedElement: action.state };
      case 'depth':
        return { ...state, depth: action.state };
      case 'filterChildless':
        return { ...state, filterChildless: action.state };
      case 'focusOnClick':
        return { ...state, focusOnClick: action.state };
      case 'adjustDistanceOnFocus':
        return { ...state, adjustDistanceOnFocus: action.state };
      case 'refitOnGrow':
        return { ...state, refitOnGrow: action.state };
      case 'focusDistanceRatio':
        return { ...state, focusDistanceRatio: action.state };
      case 'nodeLabelScale': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi && state.showNodeLabels) {
          labelSpritesRef.current.clear();
          gi.nodeThreeObject(
            buildNodeObject(state.nodeRenderMode, true, state.nodeRelSize, action.state, labelSpritesRef.current, state.nodeOpacity),
          );
          gi.nodeThreeObjectExtend(state.nodeRenderMode === NodeRenderMode.Spheres);
          gi.refresh();
        }
        return { ...state, nodeLabelScale: action.state };
      }
      case 'edgeLabelScale': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi && state.showEdgeLabels) {
          edgeLabelSpritesRef.current.clear();
          gi.linkThreeObject(((link: GraphLink) => buildEdgeLabelFactory(action.state, edgeLabelSpritesRef.current)(link)) as (
            link: GraphLink,
          ) => THREE.Object3D);
          gi.refresh();
        }
        return { ...state, edgeLabelScale: action.state };
      }
      case 'nodeRenderMode': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi) {
          labelSpritesRef.current.clear();
          gi.nodeThreeObject(
            buildNodeObject(
              action.state,
              state.showNodeLabels,
              state.nodeRelSize,
              state.nodeLabelScale,
              labelSpritesRef.current,
              state.nodeOpacity,
            ),
          );
          gi.nodeThreeObjectExtend(action.state === NodeRenderMode.Spheres);
          gi.nodeVal(action.state === NodeRenderMode.Icons ? iconNodeVal(state.nodeRelSize) : (node: GraphNode) => node.diameter);
          gi.refresh();
        }
        return { ...state, nodeRenderMode: action.state };
      }
      case 'edgeWidth': {
        if (gi) gi.linkWidth(action.state);
        return { ...state, edgeWidth: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'edgeLength': {
        if (gi) {
          const lf = gi.d3Force('link') as D3LinkForce | undefined;
          if (lf && 'distance' in lf) lf.distance(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, edgeLength: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'edgeLinkStrength': {
        if (gi) {
          const lf = gi.d3Force('link') as D3LinkForce | undefined;
          if (lf && 'strength' in lf) lf.strength(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, edgeLinkStrength: action.state };
      }
      case 'edgeOpacity': {
        if (gi) gi.linkOpacity(action.state);
        return { ...state, edgeOpacity: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'arrowLength': {
        if (gi) gi.linkDirectionalArrowLength(action.state);
        return { ...state, arrowLength: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'directionalParticles': {
        if (gi) gi.linkDirectionalParticles(action.state);
        return { ...state, directionalParticles: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'particleSpeed': {
        if (gi) gi.linkDirectionalParticleSpeed(action.state);
        return { ...state, particleSpeed: action.state };
      }
      case 'nodeRelSize': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi) {
          gi.nodeRelSize(action.state);
          if (state.nodeRenderMode === NodeRenderMode.Icons) {
            labelSpritesRef.current.clear();
            gi.nodeThreeObject(
              buildNodeObject(
                state.nodeRenderMode,
                state.showNodeLabels,
                action.state,
                state.nodeLabelScale,
                labelSpritesRef.current,
                state.nodeOpacity,
              ),
            );
            gi.nodeVal(iconNodeVal(action.state));
            gi.refresh();
          }
        }
        return { ...state, nodeRelSize: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'nodeOpacity': {
        if (gi) {
          gi.nodeOpacity(action.state);
          if (state.nodeRenderMode === NodeRenderMode.Icons) {
            labelSpritesRef.current.clear();
            gi.nodeThreeObject(
              buildNodeObject(
                state.nodeRenderMode,
                state.showNodeLabels,
                state.nodeRelSize,
                state.nodeLabelScale,
                labelSpritesRef.current,
                action.state,
              ),
            );
            gi.refresh();
          }
        }
        return { ...state, nodeOpacity: action.state };
      }
      case 'enableNodeDrag': {
        if (gi) {
          gi.enableNodeDrag(action.state);
          gi.refresh();
        }
        return { ...state, enableNodeDrag: action.state };
      }
      case 'nodeLabelDensity':
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, nodeLabelDensity: action.state, userOverrides: markOverride(state, action.type) };
      case 'nodeLabelMinSize':
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, nodeLabelMinSize: action.state };
      case 'edgeLabelDensity':
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, edgeLabelDensity: action.state };
      case 'edgeLabelMinSize':
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, edgeLabelMinSize: action.state };
      case 'chargeStrength': {
        if (gi) {
          const charge = gi.d3Force('charge') as D3ChargeForce | undefined;
          if (charge && 'strength' in charge) charge.strength(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, chargeStrength: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'velocityDecay': {
        if (gi) {
          gi.d3VelocityDecay(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, velocityDecay: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'warmupTicks': {
        if (gi) gi.warmupTicks(action.state);
        return { ...state, warmupTicks: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'cooldownTime': {
        if (gi) gi.cooldownTime(action.state);
        return { ...state, cooldownTime: action.state, userOverrides: markOverride(state, action.type) };
      }
      case 'dagMode': {
        if (gi) gi.dagMode(action.state as DagMode);
        return { ...state, dagMode: action.state };
      }
      case 'dagLevelDistance': {
        if (gi) gi.dagLevelDistance(action.state as number);
        return { ...state, dagLevelDistance: action.state };
      }
      case 'numDimensions': {
        if (gi) gi.numDimensions(action.state);
        return { ...state, numDimensions: action.state };
      }
      case 'showGrid':
        return { ...state, showGrid: action.state };
      case 'applySizeDefaults': {
        const defaults = action.state;
        const next = { ...state } as Record<string, unknown> & GraphControls;
        let needsReheat = false;

        for (const [key, value] of Object.entries(defaults)) {
          if (state.userOverrides.has(key)) continue;
          next[key] = value;

          // Apply each setting imperatively to the graph instance
          if (gi) {
            switch (key) {
              case 'chargeStrength': {
                const charge = gi.d3Force('charge') as D3ChargeForce | undefined;
                if (charge && 'strength' in charge) charge.strength(value as number);
                needsReheat = true;
                break;
              }
              case 'edgeLength': {
                const lf = gi.d3Force('link') as D3LinkForce | undefined;
                if (lf && 'distance' in lf) lf.distance(value as number);
                needsReheat = true;
                break;
              }
              case 'velocityDecay':
                gi.d3VelocityDecay(value as number);
                needsReheat = true;
                break;
              case 'edgeWidth':
                gi.linkWidth(value as number);
                break;
              case 'edgeOpacity':
                gi.linkOpacity(value as number);
                break;
              case 'arrowLength':
                gi.linkDirectionalArrowLength(value as number);
                break;
              case 'directionalParticles':
                gi.linkDirectionalParticles(value as number);
                break;
              case 'warmupTicks':
                gi.warmupTicks(value as number);
                break;
              case 'cooldownTime':
                gi.cooldownTime(value as number);
                break;
              case 'nodeRelSize': {
                gi.nodeRelSize(value as number);
                if (state.nodeRenderMode === NodeRenderMode.Icons) {
                  labelSpritesRef.current.clear();
                  gi.nodeThreeObject(
                    buildNodeObject(
                      state.nodeRenderMode,
                      state.showNodeLabels,
                      value as number,
                      state.nodeLabelScale,
                      labelSpritesRef.current,
                      state.nodeOpacity,
                    ),
                  );
                  gi.nodeVal(iconNodeVal(value as number));
                  gi.refresh();
                }
                if (lastCamDistRef) lastCamDistRef.current = -1;
                break;
              }
              case 'nodeLabelDensity':
                if (lastCamDistRef) lastCamDistRef.current = -1;
                break;
            }
          }
        }

        if (gi && needsReheat) gi.d3ReheatSimulation();
        return next;
      }
      case 'resetSizeOverrides':
        return { ...state, userOverrides: new Set<string>() };
    }
  };
};
