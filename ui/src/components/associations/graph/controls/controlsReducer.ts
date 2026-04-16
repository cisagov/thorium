import * as THREE from 'three';
import SpriteText from 'three-spritetext';
import type { ForceGraph3DInstance } from '3d-force-graph';

import { getNodeColor, getNodeSvg, svgToTexture, getEdgeColor } from '../styles';
import type { GraphControls, DisplayAction, NodeRenderMode } from './types';
import type { GraphNode, GraphLink } from '../types';

export type LabelEntry = { sprite: THREE.Object3D; degree: number; isInitial: boolean; baseScale: THREE.Vector3 };

const ICON_EDGE_PAD = 1.3;

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

    if (renderMode === 'icons') {
      const svgString = getNodeSvg(node.nodeType, node.visualState);
      const texture = svgToTexture(svgString, 64);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, opacity: nodeOpacity });
      const sprite = new THREE.Sprite(spriteMaterial);
      const scale = Math.max(6, node.diameter / 3) * sizeFactor;
      sprite.scale.set(scale, scale, 1);
      group.add(sprite);
    }

    if (showLabels) {
      const labelSprite = new SpriteText(node.label);
      labelSprite.color = getNodeColor(node.nodeType, node.visualState);
      labelSprite.textHeight = 3 * labelScale;
      (labelSprite as any).position.y = renderMode === 'icons' ? -(node.diameter / 5 + 4) * sizeFactor : -(node.diameter / 5 + 2);
      // @ts-expect-error depthWrite exists on SpriteMaterial
      labelSprite.material.depthWrite = false;
      group.add(labelSprite);
      if (labelMap) {
        const obj = labelSprite as unknown as THREE.Object3D;
        labelMap.set(node.id, {
          sprite: obj,
          degree: node.degree,
          isInitial: node.visualState === 'initial',
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
    // @ts-expect-error depthWrite exists on SpriteMaterial
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
  graphInstanceRef: React.RefObject<ForceGraph3DInstance | null>,
  labelSpritesRef: React.RefObject<Map<string, LabelEntry>>,
  edgeLabelSpritesRef: React.RefObject<Map<string, LabelEntry>>,
  lastCamDistRef?: React.RefObject<number>,
) => {
  return (state: GraphControls, action: DisplayAction): GraphControls => {
    const gi = graphInstanceRef.current;
    switch (action.type) {
      case 'showEdgeLabels': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi) {
          if (action.state) {
            edgeLabelSpritesRef.current.clear();
            gi.linkThreeObjectExtend(true);
            gi.linkThreeObject(
              (link: any) => buildEdgeLabelFactory(state.edgeLabelScale, edgeLabelSpritesRef.current)(link as GraphLink) as any,
            );
            gi.linkPositionUpdate((sprite: any, { start, end }: any) => {
              if (!sprite) return false;
              sprite.position.set((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2);
              return false;
            });
          } else {
            edgeLabelSpritesRef.current.clear();
            gi.linkThreeObjectExtend(false);
            gi.linkThreeObject(undefined as any);
            gi.linkPositionUpdate(null as any);
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
            ) as any,
          );
          gi.nodeThreeObjectExtend(state.nodeRenderMode === 'spheres');
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
            buildNodeObject(state.nodeRenderMode, true, state.nodeRelSize, action.state, labelSpritesRef.current, state.nodeOpacity) as any,
          );
          gi.nodeThreeObjectExtend(state.nodeRenderMode === 'spheres');
          gi.refresh();
        }
        return { ...state, nodeLabelScale: action.state };
      }
      case 'edgeLabelScale': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi && state.showEdgeLabels) {
          edgeLabelSpritesRef.current.clear();
          gi.linkThreeObject((link: any) => buildEdgeLabelFactory(action.state, edgeLabelSpritesRef.current)(link as GraphLink) as any);
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
            ) as any,
          );
          gi.nodeThreeObjectExtend(action.state === 'spheres');
          gi.nodeVal(action.state === 'icons' ? (iconNodeVal(state.nodeRelSize) as any) : (node: any) => (node as GraphNode).diameter);
          gi.refresh();
        }
        return { ...state, nodeRenderMode: action.state };
      }
      case 'edgeWidth': {
        if (gi) gi.linkWidth(action.state);
        return { ...state, edgeWidth: action.state };
      }
      case 'edgeLength': {
        if (gi) {
          const linkForce = gi.d3Force('link');
          if (linkForce && 'distance' in linkForce) (linkForce as any).distance(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, edgeLength: action.state };
      }
      case 'edgeLinkStrength': {
        if (gi) {
          const linkForce = gi.d3Force('link');
          if (linkForce && 'strength' in linkForce) (linkForce as any).strength(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, edgeLinkStrength: action.state };
      }
      case 'edgeOpacity': {
        if (gi) gi.linkOpacity(action.state);
        return { ...state, edgeOpacity: action.state };
      }
      case 'arrowLength': {
        if (gi) gi.linkDirectionalArrowLength(action.state);
        return { ...state, arrowLength: action.state };
      }
      case 'directionalParticles': {
        if (gi) gi.linkDirectionalParticles(action.state);
        return { ...state, directionalParticles: action.state };
      }
      case 'particleSpeed': {
        if (gi) gi.linkDirectionalParticleSpeed(action.state);
        return { ...state, particleSpeed: action.state };
      }
      case 'nodeRelSize': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi) {
          gi.nodeRelSize(action.state);
          if (state.nodeRenderMode === 'icons') {
            labelSpritesRef.current.clear();
            gi.nodeThreeObject(
              buildNodeObject(
                state.nodeRenderMode,
                state.showNodeLabels,
                action.state,
                state.nodeLabelScale,
                labelSpritesRef.current,
                state.nodeOpacity,
              ) as any,
            );
            gi.nodeVal(iconNodeVal(action.state) as any);
            gi.refresh();
          }
        }
        return { ...state, nodeRelSize: action.state };
      }
      case 'nodeOpacity': {
        if (gi) {
          gi.nodeOpacity(action.state);
          if (state.nodeRenderMode === 'icons') {
            labelSpritesRef.current.clear();
            gi.nodeThreeObject(
              buildNodeObject(
                state.nodeRenderMode,
                state.showNodeLabels,
                state.nodeRelSize,
                state.nodeLabelScale,
                labelSpritesRef.current,
                action.state,
              ) as any,
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
        return { ...state, nodeLabelDensity: action.state };
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
          const charge = gi.d3Force('charge');
          if (charge && 'strength' in charge) (charge as any).strength(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, chargeStrength: action.state };
      }
      case 'velocityDecay': {
        if (gi) {
          gi.d3VelocityDecay(action.state);
          gi.d3ReheatSimulation();
        }
        return { ...state, velocityDecay: action.state };
      }
      case 'warmupTicks': {
        if (gi) gi.warmupTicks(action.state);
        return { ...state, warmupTicks: action.state };
      }
      case 'cooldownTime': {
        if (gi) gi.cooldownTime(action.state);
        return { ...state, cooldownTime: action.state };
      }
      case 'dagMode': {
        if (gi) gi.dagMode(action.state as any);
        return { ...state, dagMode: action.state };
      }
      case 'dagLevelDistance': {
        if (gi) gi.dagLevelDistance(action.state as any);
        return { ...state, dagLevelDistance: action.state };
      }
      case 'numDimensions': {
        if (gi) gi.numDimensions(action.state);
        return { ...state, numDimensions: action.state };
      }
      case 'showGrid':
        return { ...state, showGrid: action.state };
    }
  };
};
