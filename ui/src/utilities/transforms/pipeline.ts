/// Default pipeline SLA in seconds (1 week). Shown as placeholder text on the create form and
/// applied when the SLA field is left blank.
export const DEFAULT_SLA = 604800;

/**
 * Deep-equality check for two values via JSON serialization (with reference/`null` fast paths).
 *
 * @param a - The first value.
 * @param b - The second value.
 * @returns `true` if the values are equal by identity or JSON representation.
 */
function valuesEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === undefined || a === null || b === undefined || b === null) return false;
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Convert a pipeline API object into the plain object shape edited by the code/form editor.
 *
 * Drops server-managed fields (`creator`, `bans`) and coerces a `null` `description` to `''` so it
 * renders as an empty editable input rather than literal `null`.
 *
 * @param pipeline - The pipeline as returned by the API.
 * @returns A new object suitable for populating the editor.
 */
export function pipelineToEditorObject(pipeline: Record<string, unknown>): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  for (const key of Object.keys(pipeline)) {
    if (key === 'creator' || key === 'bans') continue;
    const val = pipeline[key];
    obj[key] = val === null && key === 'description' ? '' : val;
  }
  return obj;
}

/**
 * Build a pipeline-create request from the editor object.
 *
 * Requires `group`, `name`, and `order` (the stage execution order). Returns a shallow copy of the
 * editor object as the payload, defaulting `sla` to `DEFAULT_SLA` when it is left blank/absent.
 *
 * @param obj - The editor object to convert.
 * @returns The create request payload, or `null` if any required field is missing.
 */
export function editorObjectToPipelineCreate(obj: Record<string, unknown>): Record<string, unknown> | null {
  if (!obj['group'] || !obj['name'] || !obj['order']) return null;
  const payload = { ...obj };
  if (payload['sla'] === undefined || payload['sla'] === null || payload['sla'] === '') {
    payload['sla'] = DEFAULT_SLA;
  }
  return payload;
}

/**
 * Compute the trigger diff between a pipeline's new and old trigger maps.
 *
 * A trigger is added/changed if it's new or its config differs (deep-equal); removed if it no longer
 * exists. Empty add/remove collections are omitted from the result.
 *
 * @param newTriggers - The edited trigger map.
 * @param oldTriggers - The original trigger map.
 * @returns A partial update with `triggers` (added/changed) and/or `remove_triggers` as applicable.
 */
function computeTriggerDiffs(
  newTriggers: Record<string, unknown>,
  oldTriggers: Record<string, unknown>,
): { triggers?: Record<string, unknown>; remove_triggers?: string[] } {
  const result: { triggers?: Record<string, unknown>; remove_triggers?: string[] } = {};
  const addedOrChanged: Record<string, unknown> = {};

  for (const [key, val] of Object.entries(newTriggers)) {
    if (!(key in oldTriggers) || !valuesEqual(val, oldTriggers[key])) {
      addedOrChanged[key] = val;
    }
  }

  const removed = Object.keys(oldTriggers).filter((k) => !(k in newTriggers));

  if (Object.keys(addedOrChanged).length > 0) result.triggers = addedOrChanged;
  if (removed.length > 0) result.remove_triggers = removed;
  return result;
}

/**
 * Build a minimal pipeline-update request by diffing the editor object against the original pipeline.
 *
 * Only changed fields are included: `order` and `sla` when they differ, the description (set when
 * present/changed or cleared via `clear_description` when emptied), and trigger add/remove diffs via
 * {@link computeTriggerDiffs}.
 *
 * @param obj - The current editor object.
 * @param originalPipeline - The pipeline as originally loaded from the API.
 * @returns `{ group, name, data }` where `data` is the patch body, or `null` if group/name are missing.
 */
export function editorObjectToPipelineUpdate(
  obj: Record<string, unknown>,
  originalPipeline: Record<string, unknown>,
): { group: string; name: string; data: Record<string, unknown> } | null {
  const group = (originalPipeline['group'] as string) || '';
  const name = (originalPipeline['name'] as string) || '';
  if (!group || !name) return null;

  const data: Record<string, unknown> = {};

  if (!valuesEqual(obj['order'], originalPipeline['order'])) {
    data['order'] = obj['order'];
  }

  if (!valuesEqual(obj['sla'], originalPipeline['sla'])) {
    data['sla'] = obj['sla'];
  }

  const newDesc = obj['description'];
  const origDesc = originalPipeline['description'];
  const hasNewDesc = newDesc && typeof newDesc === 'string' && newDesc.trim() !== '';
  const hasOrigDesc = origDesc && typeof origDesc === 'string' && origDesc.trim() !== '';
  if (hasNewDesc) {
    if (!valuesEqual(newDesc, origDesc)) {
      data['description'] = newDesc;
    }
  } else if (hasOrigDesc) {
    data['clear_description'] = true;
  }

  const newTriggers = (obj['triggers'] ?? {}) as Record<string, unknown>;
  const oldTriggers = (originalPipeline['triggers'] ?? {}) as Record<string, unknown>;
  Object.assign(data, computeTriggerDiffs(newTriggers, oldTriggers));

  return { group, name, data };
}
