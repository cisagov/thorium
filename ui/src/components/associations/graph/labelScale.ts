// spec: ./AssociationGraph.spec.md

/** Target on-screen height in pixels for node labels */
export const LABEL_TARGET_PX = 12;
/** Minimum on-screen height in pixels for node labels */
export const LABEL_MIN_PX = 8;
/** Maximum on-screen height in pixels for node labels */
export const LABEL_MAX_PX = 24;
/** Target on-screen height in pixels for edge labels */
export const EDGE_LABEL_TARGET_PX = 10;
/** Minimum on-screen height in pixels for edge labels */
export const EDGE_LABEL_MIN_PX = 8;
/** Maximum on-screen height in pixels for edge labels */
export const EDGE_LABEL_MAX_PX = 20;

/** Inputs for computing a pixel-clamped screen-space label scale multiplier */
export interface LabelScaleInput {
  /** World-space distance from the camera to the label */
  labelDist: number;
  /** Viewport height in CSS pixels (from `gi.height()`, not the renderer canvas height) */
  viewportHeightPx: number;
  /** Perspective camera vertical field of view in degrees */
  fovDeg: number;
  /** The label sprite's base scale Y, derived from its fixed textHeight */
  baseScaleY: number;
  /** User slider multiplier applied to the target pixel size (0.5..2.0) */
  labelScale: number;
  /** Importance multiplier boosting the pixel size within the clamp (1.0 | 1.2 | 1.4) */
  tierBoost: number;
  /** Target on-screen pixel height before clamping */
  targetPx: number;
  /** Clamp floor for the on-screen pixel height */
  minPx: number;
  /** Clamp ceiling for the on-screen pixel height */
  maxPx: number;
}

/**
 * Compute the scale multiplier that renders a billboard label sprite at a fixed
 * on-screen pixel height regardless of camera distance, viewport size, or graph size.
 *
 * A billboard sprite of world-height `h` at distance `d` from a perspective camera
 * appears at `h * pxPerWorldAtUnitDist / d` pixels, where `pxPerWorldAtUnitDist`
 * is `viewportHeightPx / (2 * tan(fov / 2))`. Solving for the sprite multiplier
 * that yields the clamped target pixel height gives the returned value; because
 * the resulting world height grows linearly with distance, the on-screen size
 * stays constant at any zoom.
 *
 * @param input - Camera, viewport, sprite, and pixel-target parameters.
 * @returns Multiplier to apply to the sprite's base scale so it renders at the clamped pixel height.
 */
export function computeLabelScale(input: LabelScaleInput): number {
  // clamp the requested pixel height to the readable band
  const finalPx = Math.min(input.maxPx, Math.max(input.minPx, input.targetPx * input.labelScale * input.tierBoost));
  // pixels covered by one world unit at distance 1 for this fov and viewport
  const pxPerWorldAtUnitDist = input.viewportHeightPx / (2 * Math.tan((input.fovDeg * Math.PI) / 360));
  // sprite multiplier whose projected height equals finalPx at labelDist
  return (finalPx * input.labelDist) / (pxPerWorldAtUnitDist * input.baseScaleY);
}
