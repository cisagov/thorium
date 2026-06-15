import * as THREE from 'three';
import SpriteText from 'three-spritetext';

// project imports
import { getLinkEndpoints } from '../data';
import { getNodeColor, getNodeSvg, svgToTexture, getEdgeColor } from '../styles';
import { NodeRenderMode, DagMode } from './types';
import type { GraphControls, DisplayAction } from './types';
import { SIZE_SCALED_KEYS } from './sizeDefaults';
import { VisualState } from '../types';
import type { GraphNode, GraphLink, GraphInstance, D3ChargeForce, D3LinkForce } from '../types';

// spec: ./GraphControlsToolbar.spec.md

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

/**
 * Builds the `nodeVal` accessor for icon render mode, sizing each node's collision/arrow
 * volume to match where the icon's visible glyph edge lands (see {@link ICON_EDGE_PAD}).
 *
 * @param nodeRelSize - The current node relative size multiplier from the controls.
 * @returns An accessor that maps a graph node to its `nodeVal` (radius cubed).
 */
export const iconNodeVal =
  (nodeRelSize: number) =>
  (node: GraphNode): number => {
    const iconHalf = (Math.max(6, node.diameter / 3) * (nodeRelSize / 4)) / 2;
    const r = (iconHalf * ICON_EDGE_PAD) / nodeRelSize;
    return r * r * r;
  };

/**
 * Builds the `nodeThreeObject` factory that renders each node as an icon sprite (with a depth
 * occluder) or a bare label group, optionally registering label sprites for the declutter loop.
 *
 * @param renderMode - Whether nodes render as icon sprites or spheres.
 * @param showLabels - Whether to attach a text label sprite to each node.
 * @param nodeRelSize - The node relative size multiplier used to scale icon sprites.
 * @param labelMap - Optional map to register created label sprites in, keyed by node id.
 * @param nodeOpacity - Opacity applied to icon sprites, defaulting to fully opaque.
 * @returns A factory producing the THREE.Object3D group for a graph node.
 */
export const buildNodeObject = (
  renderMode: NodeRenderMode,
  showLabels: boolean,
  nodeRelSize: number,
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
      labelSprite.textHeight = 3;
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

/**
 * Build a link-object factory that renders edge labels as SpriteText billboards
 * and registers them in the edge label map for the per-frame declutter loop.
 *
 * @param edgeLabelMap - Map to register created label sprites in, keyed by `source-target`.
 * @param getDegree - Live lookup from node id to degree; a live lookup is required because
 *   link endpoints may still be unresolved string ids at first digest and the factory
 *   closure outlives graph growths that change degrees.
 * @returns Factory producing a label sprite for a link, or `undefined` for unlabeled links.
 */
export const buildEdgeLabelFactory = (edgeLabelMap?: Map<string, LabelEntry>, getDegree?: (id: string) => number | undefined) => {
  return (link: GraphLink): THREE.Object3D | undefined => {
    if (!link.label) return undefined;
    const sprite = new SpriteText(link.label);
    sprite.color = getEdgeColor();
    sprite.textHeight = 2.5;
    sprite.material.depthWrite = false;

    if (edgeLabelMap) {
      const { source: src, target: tgt } = getLinkEndpoints(link);
      const obj = sprite as unknown as THREE.Object3D;
      // an edge is only as important as its weaker endpoint, so rank by min endpoint degree
      const degree = Math.min(getDegree?.(src) ?? 1, getDegree?.(tgt) ?? 1);
      edgeLabelMap.set(`${src}-${tgt}`, { sprite: obj, degree, isInitial: false, baseScale: obj.scale.clone() });
    }

    return sprite;
  };
};

/**
 * Creates the graph controls reducer. The reducer is intentionally impure: it mutates the
 * ForceGraph3D instance imperatively via the passed refs because the 3d-force-graph API is
 * imperative, and syncing control state with graph properties inside the reducer keeps the two
 * atomic. It must only be driven through React's `useReducer`.
 *
 * @param graphInstanceRef - Ref to the live ForceGraph3D instance, or null before mount.
 * @param labelSpritesRef - Ref to the node label sprite map consumed by the declutter loop.
 * @param edgeLabelSpritesRef - Ref to the edge label sprite map consumed by the declutter loop.
 * @param lastCamDistRef - Optional ref holding the last camera distance; set to -1 to force the
 *   per-frame label loop to recompute rankings after control or growth changes.
 * @param nodeDegreesRef - Optional ref to the live node id -> degree map used to rank edge labels.
 * @returns The reducer function mapping `(state, action)` to the next controls state.
 */
export const createControlsReducer = (
  graphInstanceRef: React.RefObject<GraphInstance | null>,
  labelSpritesRef: React.RefObject<Map<string, LabelEntry>>,
  edgeLabelSpritesRef: React.RefObject<Map<string, LabelEntry>>,
  lastCamDistRef?: React.RefObject<number>,
  nodeDegreesRef?: React.RefObject<Map<string, number>>,
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
            gi.linkThreeObject(((link: GraphLink) =>
              buildEdgeLabelFactory(edgeLabelSpritesRef.current, (id: string) => nodeDegreesRef?.current.get(id))(link)) as (
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
            buildNodeObject(state.nodeRenderMode, action.state, state.nodeRelSize, labelSpritesRef.current, state.nodeOpacity),
          );
          gi.nodeThreeObjectExtend(state.nodeRenderMode === NodeRenderMode.Spheres);
          gi.refresh();
        }
        return { ...state, showNodeLabels: action.state };
      }
      case 'selected':
        // selection pins its label in the declutter pass; invalidate the camera cache
        // so the per-frame label loop applies the new pin on the next frame
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, selectedElement: action.state };
      case 'depth':
        return { ...state, depth: action.state };
      case 'filterChildless':
        return { ...state, filterChildless: action.state };
      case 'focusOnClick':
        return { ...state, focusOnClick: action.state };
      case 'fitNeighborhoodOnFocus':
        return { ...state, fitNeighborhoodOnFocus: action.state };
      case 'refitOnGrow':
        return { ...state, refitOnGrow: action.state };
      case 'nodeLabelScale':
        // sizing is applied per-frame by the label-scaling loop; just invalidate its camera cache
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, nodeLabelScale: action.state, userOverrides: markOverride(state, action.type) };
      case 'edgeLabelScale':
        // sizing is applied per-frame by the label-scaling loop; just invalidate its camera cache
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, edgeLabelScale: action.state, userOverrides: markOverride(state, action.type) };
      case 'nodeRenderMode': {
        if (lastCamDistRef) lastCamDistRef.current = -1;
        if (gi) {
          labelSpritesRef.current.clear();
          gi.nodeThreeObject(
            buildNodeObject(action.state, state.showNodeLabels, state.nodeRelSize, labelSpritesRef.current, state.nodeOpacity),
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
              buildNodeObject(state.nodeRenderMode, state.showNodeLabels, action.state, labelSpritesRef.current, state.nodeOpacity),
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
              buildNodeObject(state.nodeRenderMode, state.showNodeLabels, state.nodeRelSize, labelSpritesRef.current, action.state),
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
      case 'edgeLabelDensity':
        if (lastCamDistRef) lastCamDistRef.current = -1;
        return { ...state, edgeLabelDensity: action.state };
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

          // label scale multipliers are read live by the per-frame label loop,
          // so no object rebuild is needed — just invalidate its camera cache
          if ((key === 'nodeLabelScale' || key === 'edgeLabelScale') && lastCamDistRef) {
            lastCamDistRef.current = -1;
          }

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
