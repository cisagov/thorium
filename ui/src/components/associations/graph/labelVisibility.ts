// spec: ./AssociationGraph.spec.md

/** Ranked node labels shown at fit distance with the baseline density */
export const NODE_LABEL_BUDGET = 40;
/** Ranked edge labels shown at fit distance with the baseline density */
export const EDGE_LABEL_BUDGET = 20;
/** Exponent controlling how quickly the label budget grows as the camera zooms in */
export const ZOOM_EXPONENT = 1.5;

/** Ceiling for the zoom factor so extreme close-ups keep the budget bounded */
const ZOOM_MAX = 8;
/** Density slider value that maps to exactly the base budget */
const DENSITY_BASELINE = 0.5;

/** Per-frame inputs for selecting which labels are visible */
export interface VisibleLabelParams {
  /** Label density slider value; 0.5 maps to exactly the base budget */
  density: number;
  /** Current camera-to-orbit-target distance in world units */
  camDist: number;
  /** Camera distance that frames the whole graph (bounding radius / tan(fov / 2)) */
  fitDist: number;
  /** Ids that are always visible regardless of the budget (selected/focused/initial) */
  pinnedIds: ReadonlySet<string>;
  /** Ranked-label budget at fit distance and baseline density */
  baseBudget: number;
}

/**
 * Select the label ids that should be visible this frame: the pinned ids plus a
 * zoom-scaled top-K prefix of the importance ranking.
 *
 * The budget is `K = round(baseBudget * (density / 0.5) * z ** ZOOM_EXPONENT)` where
 * `z = clamp(fitDist / camDist, 1, ZOOM_MAX)`, so zooming in past the fit distance
 * reveals more labels while zooming out never shows more than the base budget.
 * Because the ranking is global and only K varies with zoom, the visible set is
 * stable while orbiting (no label popping).
 *
 * @param rankedIds - Label ids pre-sorted by importance (degree desc, id asc); the caller memoizes this.
 * @param params - Density, camera geometry, pinned ids, and the base budget.
 * @returns Set of visible label ids: the union of `pinnedIds` and the first K of `rankedIds`.
 */
export function selectVisibleLabels(rankedIds: readonly string[], params: VisibleLabelParams): Set<string> {
  // zoom factor: 1 at/past fit distance, up to ZOOM_MAX when fully zoomed in;
  // degenerate camera geometry (non-finite or zero distances) falls back to the fit baseline
  const usableInputs = Number.isFinite(params.camDist) && Number.isFinite(params.fitDist) && params.camDist > 0;
  const z = usableInputs ? Math.min(ZOOM_MAX, Math.max(1, params.fitDist / params.camDist)) : 1;
  // budget of ranked labels for this frame, scaled linearly by density and superlinearly by zoom
  const budget = Math.round(params.baseBudget * (params.density / DENSITY_BASELINE) * z ** ZOOM_EXPONENT);
  // pinned ids bypass the budget entirely; the top-K ranked ids fill the rest
  const visible = new Set<string>(params.pinnedIds);
  const limit = Math.min(rankedIds.length, Math.max(0, budget));
  for (let i = 0; i < limit; i++) {
    visible.add(rankedIds[i]);
  }
  return visible;
}
