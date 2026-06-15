// Pure helper functions for pipeline order manipulation.
// Extracted from PipelineOrderFlow.tsx for testability without DOM dependencies.

// spec: ./PipelineOrderFlow.spec.md

// Horizontal spacing between stage left-edges. Must exceed NODE_MAX_WIDTH (240) by a comfortable
// margin so a max-width node leaves room for the connecting edge before the next stage (otherwise
// wide nodes overlap their neighbor). 300 → ~60px edge gap at max node width.
export const STEP_WIDTH = 300;
export const TERMINAL_OFFSET = 70;
// Drag-cluster distance: while reordering, a dropped node joins the nearest stage (becomes parallel)
// only if it lands within this many px of that stage; otherwise it forms its own sequential stage.
// Kept well below STEP_WIDTH and the TERMINAL_OFFSET so a node dropped just left of the first stage
// (near the Start terminal) or just right of the last (near End) becomes a new leading/trailing stage
// rather than clustering into the edge stage — i.e. you can drag images to the start/end to re-insert
// them before/after the first/last image.
export const CLUSTER_THRESHOLD = 100;

const MONO_CHAR_WIDTH = 7.2;
const NODE_PADDING = 36; // 16px padding-left + 16px padding-right + 3px border-left + 1px border-right
const NODE_MIN_WIDTH = 120;
const NODE_MAX_WIDTH = 240;

// Vertical geometry. ReactFlow centers the Left/Right edge handles at 50% of a node's height, so
// every connection point sits at `node.y + height/2`. Image nodes, Start/End terminals, and barriers
// MUST all share the same height (NODE_HEIGHT) so their handle centers line up at HANDLE_CENTER_Y —
// otherwise the spine renders as a slight diagonal and parallel edges converge off-center. The styled
// `StepNodeWrapper`/`TerminalNodeWrapper` heights and the barrier offset all derive from these.
export const NODE_HEIGHT = 34;
export const HANDLE_CENTER_Y = NODE_HEIGHT / 2;

// Estimates a step node's rendered width from its label length (monospace 12px, border-box)
export function estimateNodeWidth(label: string): number {
  return Math.min(NODE_MAX_WIDTH, Math.max(NODE_MIN_WIDTH, label.length * MONO_CHAR_WIDTH + NODE_PADDING));
}

// Estimates the widest node width in a stage
export function estimateStageWidth(stage: string | string[]): number {
  const images = typeof stage === 'string' ? [stage] : stage;
  return Math.max(...images.map(estimateNodeWidth));
}

export function ordersEqual(a: (string | string[])[], b: (string | string[])[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const ai = typeof a[i] === 'string' ? [a[i]] : a[i];
    const bi = typeof b[i] === 'string' ? [b[i]] : b[i];
    if ((ai as string[]).length !== (bi as string[]).length) return false;
    const sortedA = [...(ai as string[])].sort();
    const sortedB = [...(bi as string[])].sort();
    for (let j = 0; j < sortedA.length; j++) {
      if (sortedA[j] !== sortedB[j]) return false;
    }
  }
  return true;
}

export function insertImageAtPosition(order: (string | string[])[], imageName: string, flowX: number): (string | string[])[] {
  if (order.length === 0) return [imageName];

  // A click left of the first stage's node (i.e. in the Start→first-node gap) prepends a new
  // leading stage. Without this, the symmetric cluster zone around stage 0 swallows the entire
  // gap, so it's impossible to place an image in front of the first one.
  if (flowX < TERMINAL_OFFSET) {
    return [imageName, ...order];
  }

  for (let i = 0; i < order.length; i++) {
    const stageX = TERMINAL_OFFSET + i * STEP_WIDTH;
    if (Math.abs(flowX - stageX) <= CLUSTER_THRESHOLD) {
      const stage = order[i];
      const images = typeof stage === 'string' ? [stage] : [...stage];
      images.push(imageName);
      const newOrder = [...order];
      newOrder[i] = images;
      return newOrder;
    }
    if (flowX < stageX) {
      const newOrder = [...order];
      newOrder.splice(i, 0, imageName);
      return newOrder;
    }
  }

  return [...order, imageName];
}

export function removeImageFromOrder(order: (string | string[])[], imageName: string): (string | string[])[] {
  const newOrder: (string | string[])[] = [];
  for (const stage of order) {
    if (typeof stage === 'string') {
      if (stage !== imageName) newOrder.push(stage);
    } else {
      const filtered = stage.filter((img) => img !== imageName);
      if (filtered.length === 1) newOrder.push(filtered[0]);
      else if (filtered.length > 1) newOrder.push(filtered);
    }
  }
  return newOrder;
}

export function removeImageAtPosition(order: (string | string[])[], stepIndex: number, parallelIndex: number): (string | string[])[] {
  const newOrder: (string | string[])[] = [];
  for (let i = 0; i < order.length; i++) {
    if (i !== stepIndex) {
      newOrder.push(order[i]);
      continue;
    }
    const stage = order[i];
    if (typeof stage === 'string') {
      continue;
    }
    const filtered = stage.filter((_, idx) => idx !== parallelIndex);
    if (filtered.length === 1) newOrder.push(filtered[0]);
    else if (filtered.length > 1) newOrder.push(filtered);
  }
  return newOrder;
}

export function getImagesInOrder(order: (string | string[])[]): Set<string> {
  const names = new Set<string>();
  for (const stage of order) {
    if (typeof stage === 'string') names.add(stage);
    else stage.forEach((img) => names.add(img));
  }
  return names;
}

/**
 * Derive a pipeline order from dragged node positions: sort the nodes left→right and group ones that
 * land within `CLUSTER_THRESHOLD` of the current stage into a single parallel stage; a node separated
 * by more than the threshold starts a new sequential stage. Because the threshold is smaller than the
 * gap before the first stage / after the last, a node dropped at the start or end becomes a new
 * leading/trailing stage (insert before/after the first/last image).
 *
 * @param stepNodes - The image nodes with their label and x (left-edge) position, in any order.
 * @returns The pipeline order (string for a single-image stage, string[] for a parallel stage).
 */
export function clusterStagesByX(stepNodes: { label: string; x: number }[]): (string | string[])[] {
  const sorted = [...stepNodes].sort((a, b) => a.x - b.x);
  if (sorted.length === 0) return [];

  const stages: string[][] = [[sorted[0].label]];
  let stageX = sorted[0].x;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].x - stageX <= CLUSTER_THRESHOLD) {
      stages[stages.length - 1].push(sorted[i].label);
    } else {
      stages.push([sorted[i].label]);
      stageX = sorted[i].x;
    }
  }
  return stages.map((s) => (s.length === 1 ? s[0] : s));
}
