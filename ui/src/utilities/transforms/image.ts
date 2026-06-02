const READ_ONLY_FIELDS = ['creator', 'runtime', 'used_by', 'bans'];
const OPTIONAL_STRING_FIELDS = ['image', 'modifiers', 'description'];

const SIMPLE_UPDATE_FIELDS = [
  'version',
  'scaler',
  'image',
  'timeout',
  'lifetime',
  'display_type',
  'spawn_limit',
  'collect_logs',
  'generator',
  'modifiers',
  'resources',
  'args',
  'dependencies',
  'output_collection',
  'security_context',
  'child_filters',
  'clean_up',
  'kvm',
];

/**
 * Convert an image API object into the plain object shape edited by the code/form editor.
 *
 * Drops server-managed read-only fields ({@link READ_ONLY_FIELDS}) and coerces `null` optional
 * string fields to `''` so they render as empty editable inputs rather than literal `null`.
 *
 * @param image - The image as returned by the API.
 * @returns A new object suitable for populating the editor.
 */
export function imageToEditorObject(image: Record<string, unknown>): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  for (const key of Object.keys(image)) {
    if (READ_ONLY_FIELDS.includes(key)) continue;
    const val = image[key];
    obj[key] = val === null && OPTIONAL_STRING_FIELDS.includes(key) ? '' : val;
  }
  return obj;
}

const K8S_ONLY_FIELDS = ['volumes', 'security_context', 'network_policies', 'env'] as const;
const K8S_ONLY_RESOURCE_FIELDS = ['burstable', 'nvidia_gpu', 'amd_gpu'] as const;
const KVM_ONLY_FIELDS = ['kvm'] as const;

/**
 * Remove fields that don't apply to the image's scaler, mutating the object in place.
 *
 * K8s-only fields (and K8s-only resource sub-fields) are stripped for non-K8s scalers, and the
 * `kvm` block is stripped for non-KVM scalers, so the request never carries config the backend
 * would reject for that scaler. An empty/absent scaler is treated as K8s (the default).
 *
 * @param result - The image request object to prune (modified in place).
 */
function stripScalerIrrelevantFields(result: Record<string, unknown>): void {
  const scaler = typeof result['scaler'] === 'string' ? result['scaler'] : '';
  const isK8s = scaler === 'K8s' || scaler === '';
  const isKvm = scaler === 'Kvm';

  if (!isK8s) {
    for (const field of K8S_ONLY_FIELDS) delete result[field];
    if (result['resources'] && typeof result['resources'] === 'object' && !Array.isArray(result['resources'])) {
      const res = result['resources'] as Record<string, unknown>;
      for (const field of K8S_ONLY_RESOURCE_FIELDS) delete res[field];
    }
  }
  if (!isKvm) {
    for (const field of KVM_ONLY_FIELDS) delete result[field];
  }
}

/**
 * Build an image-create request from the editor object.
 *
 * Requires `group` and `name`. Strips out `undefined`/`null` values and empty objects/arrays so
 * the create payload is minimal, then removes scaler-irrelevant fields via
 * {@link stripScalerIrrelevantFields}.
 *
 * @param obj - The editor object to convert.
 * @returns The create request payload, or `null` if `group` or `name` is missing.
 */
export function editorObjectToImageCreate(obj: Record<string, unknown>): Record<string, unknown> | null {
  if (!obj['group'] || !obj['name']) return null;
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue;
    if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0) continue;
    if (Array.isArray(value) && value.length === 0) continue;
    result[key] = value;
  }
  stripScalerIrrelevantFields(result);
  return result;
}

interface VolumeShape {
  name: string;
  [key: string]: unknown;
}

/**
 * Compute the add/remove volume diff between an image's new and old volume lists.
 *
 * Volumes are matched by `name`. A volume is added if it's new or its config changed (deep-equal
 * by JSON), and removed if it no longer exists or changed (the changed case appears in both lists
 * so the backend replaces it). Empty add/remove lists are omitted from the result.
 *
 * @param newVolumes - The edited volume list.
 * @param oldVolumes - The original volume list.
 * @returns A partial update object with `add_volumes`/`remove_volumes` as applicable.
 */
function computeVolumeDiffs(
  newVolumes: VolumeShape[],
  oldVolumes: VolumeShape[],
): { add_volumes?: VolumeShape[]; remove_volumes?: string[] } {
  const result: { add_volumes?: VolumeShape[]; remove_volumes?: string[] } = {};
  const addVolumes: VolumeShape[] = [];
  const removeVolumes: string[] = [];

  for (const nv of newVolumes) {
    const ov = oldVolumes.find((o) => o.name === nv.name);
    if (!ov || JSON.stringify(ov) !== JSON.stringify(nv)) {
      addVolumes.push(nv);
    }
  }

  for (const ov of oldVolumes) {
    const nv = newVolumes.find((n) => n.name === ov.name);
    if (!nv || JSON.stringify(ov) !== JSON.stringify(nv)) {
      removeVolumes.push(ov.name);
    }
  }

  if (addVolumes.length > 0) result.add_volumes = addVolumes;
  if (removeVolumes.length > 0) result.remove_volumes = removeVolumes;
  return result;
}

/**
 * Compute the add/remove environment-variable diff between an image's new and old env maps.
 *
 * A key is added if it's new or its value changed; removed if it no longer exists. Empty add/remove
 * collections are omitted from the result.
 *
 * @param newEnv - The edited environment map.
 * @param oldEnv - The original environment map.
 * @returns A partial update object with `add_env`/`remove_env` as applicable.
 */
function computeEnvDiffs(
  newEnv: Record<string, unknown>,
  oldEnv: Record<string, unknown>,
): { add_env?: Record<string, unknown>; remove_env?: string[] } {
  const result: { add_env?: Record<string, unknown>; remove_env?: string[] } = {};
  const addEnv: Record<string, unknown> = {};
  const removeEnv: string[] = [];

  for (const [key, val] of Object.entries(newEnv)) {
    if (!(key in oldEnv) || oldEnv[key] !== val) {
      addEnv[key] = val;
    }
  }

  for (const key of Object.keys(oldEnv)) {
    if (!(key in newEnv)) {
      removeEnv.push(key);
    }
  }

  if (Object.keys(addEnv).length > 0) result.add_env = addEnv;
  if (removeEnv.length > 0) result.remove_env = removeEnv;
  return result;
}

/**
 * Compute the added/removed network-policy diff between an image's new and old policy lists.
 *
 * @param newPolicies - The edited network policy names.
 * @param oldPolicies - The original network policy names.
 * @returns A partial update with a `network_policies` diff, or an empty object if nothing changed.
 */
function computeNetworkPolicyDiffs(
  newPolicies: string[],
  oldPolicies: string[],
): { network_policies?: { policies_added: string[]; policies_removed: string[] } } {
  const added = newPolicies.filter((p) => !oldPolicies.includes(p));
  const removed = oldPolicies.filter((p) => !newPolicies.includes(p));
  if (added.length === 0 && removed.length === 0) return {};
  return { network_policies: { policies_added: added, policies_removed: removed } };
}

const CLEARABLE_FIELDS = ['version', 'image', 'lifetime'] as const;

/**
 * Whether a value is "present" — i.e. not empty for update-diffing purposes.
 *
 * Treats `undefined`/`null`, whitespace-only strings, and empty plain objects as absent.
 *
 * @param val - The value to test.
 * @returns `true` if the value carries meaningful content.
 */
function hasValue(val: unknown): boolean {
  if (val === undefined || val === null) return false;
  if (typeof val === 'string' && val.trim() === '') return false;
  if (typeof val === 'object' && !Array.isArray(val) && Object.keys(val).length === 0) return false;
  return true;
}

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
 * Build a minimal image-update request by diffing the editor object against the original image.
 *
 * Only changed fields are included. Simple fields are copied when they changed and still have a
 * value; fields cleared by the user emit a `clear_<field>` flag instead. The description is handled
 * separately (set or cleared). For K8s images, volume/env/network-policy diffs are computed via the
 * dedicated `compute*Diffs` helpers; the raw `volumes`/`env` keys are then removed in favor of those
 * add/remove diffs, and scaler-irrelevant fields are stripped.
 *
 * @param obj - The current editor object.
 * @param originalImage - The image as originally loaded from the API.
 * @returns `{ group, name, data }` where `data` is the patch body, or `null` if group/name are missing.
 */
export function editorObjectToImageUpdate(
  obj: Record<string, unknown>,
  originalImage: Record<string, unknown>,
): { group: string; name: string; data: Record<string, unknown> } | null {
  const group = (originalImage['group'] as string) || '';
  const name = (originalImage['name'] as string) || '';
  if (!group || !name) return null;

  const data: Record<string, unknown> = {};

  for (const key of SIMPLE_UPDATE_FIELDS) {
    if (obj[key] === undefined && !hasValue(originalImage[key])) continue;
    if (valuesEqual(obj[key], originalImage[key])) continue;
    if (hasValue(obj[key])) {
      data[key] = obj[key];
    }
  }

  for (const key of CLEARABLE_FIELDS) {
    if (!hasValue(obj[key]) && hasValue(originalImage[key])) {
      data[`clear_${key}`] = true;
      delete data[key];
    }
  }

  if (obj['description'] && typeof obj['description'] === 'string' && obj['description'].trim()) {
    if (!valuesEqual(obj['description'], originalImage['description'])) {
      data['description'] = obj['description'];
    }
  } else if (originalImage['description']) {
    data['clear_description'] = true;
  }

  const scaler =
    typeof obj['scaler'] === 'string' ? obj['scaler'] : typeof originalImage['scaler'] === 'string' ? originalImage['scaler'] : '';
  const isK8s = scaler === 'K8s' || scaler === '';

  if (isK8s) {
    const newVolumes = (obj['volumes'] ?? []) as VolumeShape[];
    const oldVolumes = (originalImage['volumes'] ?? []) as VolumeShape[];
    Object.assign(data, computeVolumeDiffs(newVolumes, oldVolumes));

    const newEnv = (obj['env'] ?? {}) as Record<string, unknown>;
    const oldEnv = (originalImage['env'] ?? {}) as Record<string, unknown>;
    Object.assign(data, computeEnvDiffs(newEnv, oldEnv));

    const newPolicies = (obj['network_policies'] ?? []) as string[];
    const oldPolicies = (originalImage['network_policies'] ?? []) as string[];
    Object.assign(data, computeNetworkPolicyDiffs(newPolicies, oldPolicies));
  }
  delete data['volumes'];
  delete data['env'];

  stripScalerIrrelevantFields(data);

  return { group, name, data };
}
