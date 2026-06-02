// Pure helper functions for pipeline order manipulation.
// Extracted from PipelineOrderFlow.tsx for testability without DOM dependencies.

export const STEP_WIDTH = 200;
export const TERMINAL_OFFSET = 70;
export const CLUSTER_THRESHOLD = STEP_WIDTH * 0.6;

const MONO_CHAR_WIDTH = 7.2;
const NODE_PADDING = 32; // 14px padding-left + 14px padding-right + 3px border-left + 1px border-right
const NODE_MIN_WIDTH = 80;
const NODE_MAX_WIDTH = 210;

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
