import {
  ArgStrategy,
  CacheDependencySettings,
  ChildFilters,
  ChildrenDependencySettings,
  Dependencies,
  EphemeralDependencySettings,
  GenericCacheDependencySettings,
  ImageArgs,
  RepoDependencySettings,
  Resources,
  ResultDependencySettings,
  SampleDependencySettings,
  SecurityContext,
  TagDependencySettings,
} from '@models/images';
import {
  CacheDependencySettingsUpdate,
  ChildFiltersUpdate,
  ChildrenDependencySettingsUpdate,
  DependenciesUpdate,
  EphemeralDependencySettingsUpdate,
  GenericCacheDependencySettingsUpdate,
  ImageArgsUpdate,
  ImageUpdate,
  RepoDependencySettingsUpdate,
  ResourcesUpdate,
  ResultDependencySettingsUpdate,
  SampleDependencySettingsUpdate,
  SecurityContextUpdate,
  TagDependencySettingsUpdate,
} from '@models/images_update';

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
  'output_collection',
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
function stripScalerIrrelevantFields(result: Record<string, unknown>, scaler?: string): void {
  const effectiveScaler = scaler ?? (typeof result['scaler'] === 'string' ? result['scaler'] : '');
  const isK8s = effectiveScaler === 'K8s' || effectiveScaler === '';
  const isKvm = effectiveScaler === 'Kvm';

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
): { group: string; name: string; data: ImageUpdate } | null {
  const group = (originalImage['group'] as string) || '';
  const name = (originalImage['name'] as string) || '';
  if (!group || !name) return null;

  const data: ImageUpdate = {};

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

  if (!valuesEqual(obj['resources'], originalImage.resources)) {
    const resources = toResourcesUpdate(obj['resources'] as Resources | Partial<Resources> | undefined);
    if (resources) data.resources = resources;
  }
  const args = toImageArgsUpdate(obj['args'] as Partial<ImageArgs> | undefined, originalImage.args as Partial<ImageArgs> | undefined);
  if (args) data.args = args;

  const securityContext = toSecurityContextUpdate(
    obj['security_context'] as Partial<SecurityContext> | undefined,
    originalImage.security_context as Partial<SecurityContext> | undefined,
  );
  if (securityContext) data.security_context = securityContext;

  if (obj['description'] && typeof obj['description'] === 'string' && obj['description'].trim()) {
    if (!valuesEqual(obj['description'], originalImage['description'])) {
      data['description'] = obj['description'];
    }
  } else if (originalImage['description']) {
    data['clear_description'] = true;
  }

  const childFilters = toChildFiltersUpdate(
    obj['child_filters'] as Partial<ChildFilters> | undefined,
    originalImage.child_filters as Partial<ChildFilters> | undefined,
  );
  if (childFilters) data.child_filters = childFilters;

  const dependencies = toDependenciesUpdate(
    obj['dependencies'] as Partial<Dependencies> | undefined,
    originalImage.dependencies as Partial<Dependencies> | undefined,
  );

  if (dependencies) data.dependencies = dependencies;

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

  stripScalerIrrelevantFields(data as Record<string, unknown>, scaler);

  return { group, name, data };
}

function toResourcesUpdate(resources: Resources | Partial<Resources> | undefined): ResourcesUpdate | undefined {
  if (!resources) return undefined;

  const update: ResourcesUpdate = {};

  if (resources.cpu !== undefined) update.cpu = resources.cpu;
  if (resources.memory !== undefined) update.memory = resources.memory;
  if (resources.ephemeral_storage !== undefined) update.ephemeral_storage = resources.ephemeral_storage;
  if (resources.nvidia_gpu !== undefined) update.nvidia_gpu = resources.nvidia_gpu;
  if (resources.amd_gpu !== undefined) update.amd_gpu = resources.amd_gpu;
  if (resources.burstable !== undefined) update.burstable = resources.burstable;

  return Object.keys(update).length > 0 ? update : undefined;
}

function toImageArgsUpdate(newArgs: Partial<ImageArgs> | undefined, oldArgs: Partial<ImageArgs> | undefined): ImageArgsUpdate | undefined {
  const update: ImageArgsUpdate = {};
  const next = newArgs ?? {};
  const prev = oldArgs ?? {};

  if (!valuesEqual(next.entrypoint, prev.entrypoint)) {
    if (hasValue(next.entrypoint)) update.entrypoint = next.entrypoint;
    else if (hasValue(prev.entrypoint)) update.clear_entrypoint = true;
  }

  if (!valuesEqual(next.command, prev.command)) {
    if (hasValue(next.command)) update.command = next.command;
    else if (hasValue(prev.command)) update.clear_command = true;
  }

  if (!valuesEqual(next.reaction, prev.reaction)) {
    if (hasValue(next.reaction)) update.reaction = next.reaction;
    else if (hasValue(prev.reaction)) update.clear_reaction = true;
  }

  if (!valuesEqual(next.repo, prev.repo)) {
    if (hasValue(next.repo)) update.repo = next.repo;
    else if (hasValue(prev.repo)) update.clear_repo = true;
  }

  if (!valuesEqual(next.commit, prev.commit)) {
    if (hasValue(next.commit)) update.commit = next.commit;
    else if (hasValue(prev.commit)) update.clear_commit = true;
  }

  if (!valuesEqual(next.output, prev.output) && hasValue(next.output)) {
    update.output = next.output as ArgStrategy | undefined;
  }

  if (!valuesEqual(next.output_files, prev.output_files) && hasValue(next.output_files)) {
    update.output_files = next.output_files as ArgStrategy | undefined;
  }

  return Object.keys(update).length > 0 ? update : undefined;
}

function toSecurityContextUpdate(
  next: Partial<SecurityContext> | undefined,
  prev: Partial<SecurityContext> | undefined,
): SecurityContextUpdate | undefined {
  const update: SecurityContextUpdate = {};
  const newSc = next ?? {};
  const oldSc = prev ?? {};

  if (!valuesEqual(newSc.user, oldSc.user)) {
    if (newSc.user !== undefined && newSc.user !== null) update.user = newSc.user;
    else if (oldSc.user !== undefined && oldSc.user !== null) update.clear_user = true;
  }

  if (!valuesEqual(newSc.group, oldSc.group)) {
    if (newSc.group !== undefined && newSc.group !== null) update.group = newSc.group;
    else if (oldSc.group !== undefined && oldSc.group !== null) update.clear_group = true;
  }

  if (!valuesEqual(newSc.allow_privilege_escalation, oldSc.allow_privilege_escalation)) {
    update.allow_privilege_escalation = newSc.allow_privilege_escalation ?? null;
  }

  return Object.keys(update).length > 0 ? update : undefined;
}

function arrayDiff<T>(next: T[] = [], prev: T[] = []): { added: T[]; removed: T[] } {
  return {
    added: next.filter((item) => !prev.includes(item)),
    removed: prev.filter((item) => !next.includes(item)),
  };
}

function toChildFiltersUpdate(
  next: Partial<ChildFilters> | undefined,
  prev: Partial<ChildFilters> | undefined,
): ChildFiltersUpdate | undefined {
  const update: ChildFiltersUpdate = {};
  const newCf = next ?? {};
  const oldCf = prev ?? {};

  const mime = arrayDiff(newCf.mime, oldCf.mime);
  if (mime.added.length > 0) update.add_mime = mime.added;
  if (mime.removed.length > 0) update.remove_mime = mime.removed;

  const fileName = arrayDiff(newCf.file_name, oldCf.file_name);
  if (fileName.added.length > 0) update.add_file_name = fileName.added;
  if (fileName.removed.length > 0) update.remove_file_name = fileName.removed;

  const fileExtension = arrayDiff(newCf.file_extension, oldCf.file_extension);
  if (fileExtension.added.length > 0) update.add_file_extension = fileExtension.added;
  if (fileExtension.removed.length > 0) update.remove_file_extension = fileExtension.removed;

  if (!valuesEqual(newCf.submit_non_matches, oldCf.submit_non_matches)) {
    update.submit_non_matches = newCf.submit_non_matches ?? null;
  }

  return Object.keys(update).length > 0 ? update : undefined;
}

function nonEmpty<T extends object>(obj: T): T | undefined {
  return Object.keys(obj).length > 0 ? obj : undefined;
}

function assignUpdateField<T extends object, K extends keyof T>(update: T, key: K, value: T[K]): void {
  update[key] = value;
}

function setChangedString<T extends object, K extends keyof T>(
  update: T,
  key: K,
  next: string | undefined | null,
  prev: string | undefined | null,
): void {
  if (!valuesEqual(next, prev) && hasValue(next)) {
    assignUpdateField(update, key, next as T[K]);
  }
}

function setChangedValue<T extends object, K extends keyof T>(update: T, key: K, next: T[K] | undefined | null, prev: unknown): void {
  if (!valuesEqual(next, prev) && next !== undefined && next !== null) {
    assignUpdateField(update, key, next as T[K]);
  }
}

function applyStringKwargUpdate(
  update: { kwarg?: string | null; clear_kwarg?: boolean },
  next: string | undefined | null,
  prev: string | undefined | null,
): void {
  if (valuesEqual(next, prev)) return;
  if (hasValue(next)) update.kwarg = next;
  else if (hasValue(prev)) update.clear_kwarg = true;
}

function toSampleDependencySettingsUpdate(
  next: Partial<SampleDependencySettings> | undefined,
  prev: Partial<SampleDependencySettings> | undefined,
): SampleDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: SampleDependencySettingsUpdate = {};

  setChangedString(update, 'location', next.location, prev?.location);
  applyStringKwargUpdate(update, next.kwarg, prev?.kwarg);
  setChangedValue(update, 'strategy', next.strategy, prev?.strategy);
  setChangedValue(update, 'naming', next.naming, prev?.naming);

  return nonEmpty(update);
}

function toEphemeralDependencySettingsUpdate(
  next: Partial<EphemeralDependencySettings> | undefined,
  prev: Partial<EphemeralDependencySettings> | undefined,
): EphemeralDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: EphemeralDependencySettingsUpdate = {};

  setChangedString(update, 'location', next.location, prev?.location);
  applyStringKwargUpdate(update, next.kwarg, prev?.kwarg);
  setChangedValue(update, 'strategy', next.strategy, prev?.strategy);

  const names = arrayDiff(next.names ?? [], prev?.names ?? []);
  if (names.added.length > 0) update.add_names = names.added;
  if (names.removed.length > 0) update.remove_names = names.removed;

  return nonEmpty(update);
}

function toResultDependencySettingsUpdate(
  next: Partial<ResultDependencySettings> | undefined,
  prev: Partial<ResultDependencySettings> | undefined,
): ResultDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: ResultDependencySettingsUpdate = {};

  const images = arrayDiff(next.images ?? [], prev?.images ?? []);
  if (images.added.length > 0) update.add_images = images.added;
  if (images.removed.length > 0) update.remove_images = images.removed;

  setChangedString(update, 'location', next.location, prev?.location);

  if (!valuesEqual(next.kwarg, prev?.kwarg)) {
    if (next.kwarg !== undefined && next.kwarg !== null) {
      update.kwarg = next.kwarg;
    } else if (prev?.kwarg !== undefined && prev.kwarg !== null) {
      update.kwarg = 'None';
    }
  }

  setChangedValue(update, 'strategy', next.strategy, prev?.strategy);

  const names = arrayDiff(next.names ?? [], prev?.names ?? []);
  if (names.added.length > 0) update.add_names = names.added;
  if (names.removed.length > 0) update.remove_names = names.removed;

  return nonEmpty(update);
}

function toRepoDependencySettingsUpdate(
  next: Partial<RepoDependencySettings> | undefined,
  prev: Partial<RepoDependencySettings> | undefined,
): RepoDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: RepoDependencySettingsUpdate = {};

  setChangedString(update, 'location', next.location, prev?.location);
  applyStringKwargUpdate(update, next.kwarg, prev?.kwarg);
  setChangedValue(update, 'strategy', next.strategy, prev?.strategy);

  return nonEmpty(update);
}

function toTagDependencySettingsUpdate(
  next: Partial<TagDependencySettings> | undefined,
  prev: Partial<TagDependencySettings> | undefined,
): TagDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: TagDependencySettingsUpdate = {};

  setChangedValue(update, 'enabled', next.enabled, prev?.enabled);
  setChangedString(update, 'location', next.location, prev?.location);
  applyStringKwargUpdate(update, next.kwarg, prev?.kwarg);
  setChangedValue(update, 'strategy', next.strategy, prev?.strategy);

  return nonEmpty(update);
}

function toChildrenDependencySettingsUpdate(
  next: Partial<ChildrenDependencySettings> | undefined,
  prev: Partial<ChildrenDependencySettings> | undefined,
): ChildrenDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: ChildrenDependencySettingsUpdate = {};

  setChangedValue(update, 'enabled', next.enabled, prev?.enabled);

  const images = arrayDiff(next.images ?? [], prev?.images ?? []);
  if (images.added.length > 0) update.add_images = images.added;
  if (images.removed.length > 0) update.remove_images = images.removed;

  setChangedString(update, 'location', next.location, prev?.location);
  applyStringKwargUpdate(update, next.kwarg, prev?.kwarg);
  setChangedValue(update, 'strategy', next.strategy, prev?.strategy);

  return nonEmpty(update);
}

function toGenericCacheDependencySettingsUpdate(
  next: Partial<GenericCacheDependencySettings> | undefined,
  prev: Partial<GenericCacheDependencySettings> | undefined,
): GenericCacheDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: GenericCacheDependencySettingsUpdate = {};

  applyStringKwargUpdate(update, next.kwarg, prev?.kwarg);
  setChangedValue(update, 'strategy', next.strategy, prev?.strategy);

  return nonEmpty(update);
}

function toCacheDependencySettingsUpdate(
  next: Partial<CacheDependencySettings> | undefined,
  prev: Partial<CacheDependencySettings> | undefined,
): CacheDependencySettingsUpdate | undefined {
  if (!next) return undefined;

  const update: CacheDependencySettingsUpdate = {};

  setChangedString(update, 'location', next.location, prev?.location);
  setChangedValue(update, 'use_parent_cache', next.use_parent_cache, prev?.use_parent_cache);
  setChangedValue(update, 'enabled', next.enabled, prev?.enabled);

  const generic = toGenericCacheDependencySettingsUpdate(next.generic, prev?.generic);
  if (generic) update.generic = generic;

  return nonEmpty(update);
}

function toDependenciesUpdate(
  next: Partial<Dependencies> | undefined,
  prev: Partial<Dependencies> | undefined,
): DependenciesUpdate | undefined {
  if (!next) return undefined;

  const update: DependenciesUpdate = {};

  const samples = toSampleDependencySettingsUpdate(next.samples, prev?.samples);
  if (samples) update.samples = samples;

  const ephemeral = toEphemeralDependencySettingsUpdate(next.ephemeral, prev?.ephemeral);
  if (ephemeral) update.ephemeral = ephemeral;

  const results = toResultDependencySettingsUpdate(next.results, prev?.results);
  if (results) update.results = results;

  const repos = toRepoDependencySettingsUpdate(next.repos, prev?.repos);
  if (repos) update.repos = repos;

  const tags = toTagDependencySettingsUpdate(next.tags, prev?.tags);
  if (tags) update.tags = tags;

  const children = toChildrenDependencySettingsUpdate(next.children, prev?.children);
  if (children) update.children = children;

  const cache = toCacheDependencySettingsUpdate(next.cache, prev?.cache);
  if (cache) update.cache = cache;

  return nonEmpty(update);
}
