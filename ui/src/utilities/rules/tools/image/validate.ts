// Validation spec: see VALIDATION_SPEC.md in this directory
import type { Document } from 'yaml';
import { isSeq } from 'yaml';
import {
  nodeLineCol,
  findMapKey,
  findMapValue,
  nodePosition,
  validateUnknownFields,
  validateEnumField,
  validateNumberField,
  validateBooleanField,
  validateStringField,
  validateObjectField,
  validateSubObject,
  validateVariantField,
} from '../shared';
import { Severity, type Diagnostic } from '../../types';
import { buildLineIndex, type LineIndex } from '../../yaml';
import {
  REQUIRED_IMAGE_FIELDS,
  KNOWN_IMAGE_FIELDS,
  ARG_STRATEGY_VALUES,
  IMAGE_SCALER_VALUES,
  DEPENDENCY_PASS_STRATEGY_VALUES,
  FILE_NAMING_STRATEGY_VALUES,
  OUTPUT_DISPLAY_TYPE_VALUES,
  VOLUME_TYPE_VALUES,
  HOST_PATH_TYPE_VALUES,
  LIFETIME_COUNTER_VALUES,
  OUTPUT_HANDLER_VALUES,
  KNOWN_RESOURCES_FIELDS,
  KNOWN_ARGS_FIELDS,
  KNOWN_DEPENDENCIES_FIELDS,
  KNOWN_SAMPLE_DEP_FIELDS,
  KNOWN_REPO_DEP_FIELDS,
  KNOWN_TAG_DEP_FIELDS,
  KNOWN_CHILDREN_DEP_FIELDS,
  KNOWN_EPHEMERAL_DEP_FIELDS,
  KNOWN_RESULT_DEP_FIELDS,
  KNOWN_CACHE_DEP_FIELDS,
  KNOWN_OUTPUT_COLLECTION_FIELDS,
  KNOWN_FILES_HANDLER_FIELDS,
  KNOWN_CHILD_FILTERS_FIELDS,
  KNOWN_CLEANUP_FIELDS,
  KNOWN_SECURITY_CONTEXT_FIELDS,
  KNOWN_LIFETIME_FIELDS,
  KNOWN_VOLUME_FIELDS,
  KNOWN_KVM_FIELDS,
  KNOWN_BURSTABLE_FIELDS,
  KNOWN_GENERIC_CACHE_FIELDS,
  KNOWN_AUTO_TAG_FIELDS,
  KWARG_DEPENDENCY_SCHEMA,
  AUTO_TAG_LOGIC_SCHEMA,
} from './schema';

function validateSecurityContext(parentMap: unknown, parsed: Record<string, unknown>, lineIndex: LineIndex, diagnostics: Diagnostic[]) {
  const obj = validateObjectField(parentMap, parsed, 'security_context', lineIndex, diagnostics);
  if (!obj) return;
  const subMap = findMapValue(parentMap, 'security_context');
  validateUnknownFields(subMap, KNOWN_SECURITY_CONTEXT_FIELDS, 'security_context', obj, lineIndex, diagnostics);
  validateNumberField(subMap, obj, 'user', lineIndex, diagnostics, true);
  validateNumberField(subMap, obj, 'group', lineIndex, diagnostics, true);
  validateBooleanField(subMap, obj, 'allow_privilege_escalation', lineIndex, diagnostics);
}

function validateResources(
  parentMap: unknown,
  parsed: Record<string, unknown>,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
  isK8s = true,
) {
  const obj = validateObjectField(parentMap, parsed, 'resources', lineIndex, diagnostics);
  if (!obj) return;
  const subMap = findMapValue(parentMap, 'resources');
  validateUnknownFields(subMap, KNOWN_RESOURCES_FIELDS, 'resources', obj, lineIndex, diagnostics);
  const numFields = isK8s
    ? (['cpu', 'memory', 'ephemeral_storage', 'worker_slots', 'nvidia_gpu', 'amd_gpu'] as const)
    : (['cpu', 'memory', 'ephemeral_storage', 'worker_slots'] as const);
  for (const numField of numFields) {
    validateNumberField(subMap, obj, numField, lineIndex, diagnostics);
  }
  if (isK8s && 'burstable' in obj) {
    const burst = validateObjectField(subMap, obj, 'burstable', lineIndex, diagnostics);
    if (burst) {
      const burstMap = findMapValue(subMap, 'burstable');
      validateUnknownFields(burstMap, KNOWN_BURSTABLE_FIELDS, 'burstable', burst, lineIndex, diagnostics);
      validateNumberField(burstMap, burst, 'cpu', lineIndex, diagnostics);
      validateNumberField(burstMap, burst, 'memory', lineIndex, diagnostics);
    }
  }
}

function validateArgStrategyField(
  map: unknown,
  obj: Record<string, unknown>,
  field: string,
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
) {
  if (!(field in obj)) return;
  const val = obj[field];
  if (typeof val === 'string') {
    validateEnumField(map, obj, field, ARG_STRATEGY_VALUES, lineIndex, diagnostics);
  } else if (typeof val !== 'object' || val === null || !('Kwarg' in val)) {
    const node = findMapValue(map, field);
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({
      ...pos,
      severity: Severity.Error,
      message: `'${field}' must be 'None', 'Append', or { Kwarg: <string> }`,
    });
  }
}

function validateArgs(parentMap: unknown, parsed: Record<string, unknown>, lineIndex: LineIndex, diagnostics: Diagnostic[]) {
  const obj = validateObjectField(parentMap, parsed, 'args', lineIndex, diagnostics);
  if (!obj) return;
  const subMap = findMapValue(parentMap, 'args');
  validateUnknownFields(subMap, KNOWN_ARGS_FIELDS, 'args', obj, lineIndex, diagnostics);
  validateArgStrategyField(subMap, obj, 'output', lineIndex, diagnostics);
  validateArgStrategyField(subMap, obj, 'output_files', lineIndex, diagnostics);
}

function validateDepSubSection(
  depMap: unknown,
  deps: Record<string, unknown>,
  field: string,
  knownFields: readonly string[],
  lineIndex: LineIndex,
  diagnostics: Diagnostic[],
) {
  const obj = validateObjectField(depMap, deps, field, lineIndex, diagnostics);
  if (!obj) return;
  const subMap = findMapValue(depMap, field);
  validateUnknownFields(subMap, knownFields, `dependencies.${field}`, obj, lineIndex, diagnostics);
  if ('strategy' in obj) {
    validateEnumField(subMap, obj, 'strategy', DEPENDENCY_PASS_STRATEGY_VALUES, lineIndex, diagnostics);
  }
  if ('naming' in obj) {
    validateEnumField(subMap, obj, 'naming', FILE_NAMING_STRATEGY_VALUES, lineIndex, diagnostics);
  }
  if ('kwarg' in obj) {
    // results.kwarg is the KwargDependency enum; all other deps' kwarg is a plain Option<String>
    if (field === 'results') {
      validateVariantField(subMap, obj, 'kwarg', KWARG_DEPENDENCY_SCHEMA, lineIndex, diagnostics, true);
    } else {
      validateStringField(subMap, obj, 'kwarg', lineIndex, diagnostics, true);
    }
  }
}

function validateDependencies(parentMap: unknown, parsed: Record<string, unknown>, lineIndex: LineIndex, diagnostics: Diagnostic[]) {
  const obj = validateObjectField(parentMap, parsed, 'dependencies', lineIndex, diagnostics);
  if (!obj) return;
  const depMap = findMapValue(parentMap, 'dependencies');
  validateUnknownFields(depMap, KNOWN_DEPENDENCIES_FIELDS, 'dependencies', obj, lineIndex, diagnostics);
  validateDepSubSection(depMap, obj, 'samples', KNOWN_SAMPLE_DEP_FIELDS, lineIndex, diagnostics);
  validateDepSubSection(depMap, obj, 'repos', KNOWN_REPO_DEP_FIELDS, lineIndex, diagnostics);
  validateDepSubSection(depMap, obj, 'tags', KNOWN_TAG_DEP_FIELDS, lineIndex, diagnostics);
  validateDepSubSection(depMap, obj, 'children', KNOWN_CHILDREN_DEP_FIELDS, lineIndex, diagnostics);
  validateDepSubSection(depMap, obj, 'ephemeral', KNOWN_EPHEMERAL_DEP_FIELDS, lineIndex, diagnostics);
  validateDepSubSection(depMap, obj, 'results', KNOWN_RESULT_DEP_FIELDS, lineIndex, diagnostics);
  if ('cache' in obj) {
    const cacheObj = validateObjectField(depMap, obj, 'cache', lineIndex, diagnostics);
    if (cacheObj) {
      const cacheMap = findMapValue(depMap, 'cache');
      validateUnknownFields(cacheMap, KNOWN_CACHE_DEP_FIELDS, 'dependencies.cache', cacheObj, lineIndex, diagnostics);
      validateBooleanField(cacheMap, cacheObj, 'use_parent_cache', lineIndex, diagnostics);
      validateBooleanField(cacheMap, cacheObj, 'enabled', lineIndex, diagnostics);
      if ('generic' in cacheObj) {
        const genObj = validateObjectField(cacheMap, cacheObj, 'generic', lineIndex, diagnostics);
        if (genObj) {
          const genMap = findMapValue(cacheMap, 'generic');
          validateUnknownFields(genMap, KNOWN_GENERIC_CACHE_FIELDS, 'dependencies.cache.generic', genObj, lineIndex, diagnostics);
          if ('strategy' in genObj) {
            validateEnumField(genMap, genObj, 'strategy', DEPENDENCY_PASS_STRATEGY_VALUES, lineIndex, diagnostics);
          }
          validateStringField(genMap, genObj, 'kwarg', lineIndex, diagnostics, true);
        }
      }
    }
  }
}

function validateOutputCollection(parentMap: unknown, parsed: Record<string, unknown>, lineIndex: LineIndex, diagnostics: Diagnostic[]) {
  const obj = validateObjectField(parentMap, parsed, 'output_collection', lineIndex, diagnostics);
  if (!obj) return;
  const subMap = findMapValue(parentMap, 'output_collection');
  validateUnknownFields(subMap, KNOWN_OUTPUT_COLLECTION_FIELDS, 'output_collection', obj, lineIndex, diagnostics);
  validateEnumField(subMap, obj, 'handler', OUTPUT_HANDLER_VALUES, lineIndex, diagnostics);
  if ('files' in obj) {
    validateSubObject(subMap, obj, 'files', KNOWN_FILES_HANDLER_FIELDS, lineIndex, diagnostics);
  }
  validateBooleanField(subMap, obj, 'as_filesystem', lineIndex, diagnostics);
  validateStringField(subMap, obj, 'children', lineIndex, diagnostics);
  if ('auto_tag' in obj) {
    const autoTagVal = obj['auto_tag'];
    if (typeof autoTagVal === 'object' && autoTagVal !== null && !Array.isArray(autoTagVal)) {
      const autoTagMap = autoTagVal as Record<string, unknown>;
      const autoTagMapNode = findMapValue(subMap, 'auto_tag');
      for (const [tagName, tagVal] of Object.entries(autoTagMap)) {
        if (typeof tagVal === 'object' && tagVal !== null && !Array.isArray(tagVal)) {
          const tagObj = tagVal as Record<string, unknown>;
          const tagNode = findMapValue(autoTagMapNode, tagName);
          validateUnknownFields(tagNode, KNOWN_AUTO_TAG_FIELDS, `auto_tag.${tagName}`, tagObj, lineIndex, diagnostics);
          if (!('logic' in tagObj)) {
            const pos = nodePosition(tagNode, lineIndex);
            diagnostics.push({ ...pos, severity: Severity.Error, message: `auto_tag.${tagName} is missing required field: 'logic'` });
          } else {
            validateVariantField(tagNode, tagObj, 'logic', AUTO_TAG_LOGIC_SCHEMA, lineIndex, diagnostics);
          }
          if ('key' in tagObj) {
            validateStringField(tagNode, tagObj, 'key', lineIndex, diagnostics);
          }
        }
      }
    }
  }
}

function validateChildFilters(parentMap: unknown, parsed: Record<string, unknown>, lineIndex: LineIndex, diagnostics: Diagnostic[]) {
  validateSubObject(parentMap, parsed, 'child_filters', KNOWN_CHILD_FILTERS_FIELDS, lineIndex, diagnostics);
}

function validateVolumes(parentMap: unknown, parsed: Record<string, unknown>, lineIndex: LineIndex, diagnostics: Diagnostic[]) {
  if (!('volumes' in parsed)) return;
  if (!Array.isArray(parsed['volumes'])) {
    const node = findMapValue(parentMap, 'volumes');
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({ ...pos, severity: Severity.Error, message: "'volumes' must be an array" });
    return;
  }
  const volsNode = findMapValue(parentMap, 'volumes');
  if (!isSeq(volsNode)) return;
  const volumes = parsed['volumes'] as unknown[];
  for (let i = 0; i < volsNode.items.length; i++) {
    const item = volsNode.items[i];
    const vol = volumes[i];
    if (typeof vol !== 'object' || vol === null || Array.isArray(vol)) {
      const pos = nodePosition(item, lineIndex);
      diagnostics.push({ ...pos, severity: Severity.Error, message: 'Each volume entry must be an object' });
      continue;
    }
    const volObj = vol as Record<string, unknown>;
    validateUnknownFields(item, KNOWN_VOLUME_FIELDS, 'volume', volObj, lineIndex, diagnostics);
    if (!('name' in volObj)) {
      const pos = nodePosition(item, lineIndex);
      diagnostics.push({ ...pos, severity: Severity.Error, message: "Volume is missing required field: 'name'" });
    }
    if (!('mount_path' in volObj)) {
      const pos = nodePosition(item, lineIndex);
      diagnostics.push({ ...pos, severity: Severity.Error, message: "Volume is missing required field: 'mount_path'" });
    }
    if ('archetype' in volObj) {
      validateEnumField(item, volObj, 'archetype', VOLUME_TYPE_VALUES, lineIndex, diagnostics);
    }
    if ('host_path' in volObj && typeof volObj['host_path'] === 'object' && volObj['host_path'] !== null) {
      const hp = volObj['host_path'] as Record<string, unknown>;
      if ('path_type' in hp) {
        const hpMap = findMapValue(item, 'host_path');
        validateEnumField(hpMap, hp, 'path_type', HOST_PATH_TYPE_VALUES, lineIndex, diagnostics);
      }
    }
  }
}

function validateLifetime(parentMap: unknown, parsed: Record<string, unknown>, lineIndex: LineIndex, diagnostics: Diagnostic[]) {
  const obj = validateObjectField(parentMap, parsed, 'lifetime', lineIndex, diagnostics);
  if (!obj) return;
  const subMap = findMapValue(parentMap, 'lifetime');
  validateUnknownFields(subMap, KNOWN_LIFETIME_FIELDS, 'lifetime', obj, lineIndex, diagnostics);
  if ('counter' in obj) {
    validateEnumField(subMap, obj, 'counter', LIFETIME_COUNTER_VALUES, lineIndex, diagnostics);
  }
  validateNumberField(subMap, obj, 'amount', lineIndex, diagnostics);
}

export function validateImageRequest(doc: Document, text: string, parsed: Record<string, unknown>): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const lineIndex = buildLineIndex(text);
  const contents = doc.contents;

  const lastLine = lineIndex.offsets.length;
  for (const field of REQUIRED_IMAGE_FIELDS) {
    if (!(field in parsed)) {
      diagnostics.push({
        line: 1,
        column: 1,
        endLine: lastLine,
        severity: Severity.Error,
        message: `Missing required field: '${field}'`,
      });
    }
  }

  validateUnknownFields(contents, KNOWN_IMAGE_FIELDS, 'image', parsed, lineIndex, diagnostics);

  validateStringField(contents, parsed, 'group', lineIndex, diagnostics);
  validateStringField(contents, parsed, 'name', lineIndex, diagnostics);
  validateStringField(contents, parsed, 'image', lineIndex, diagnostics, true);
  if ('version' in parsed) {
    const ver = parsed['version'];
    if (ver !== null && typeof ver !== 'string' && typeof ver !== 'object') {
      const node = findMapValue(contents, 'version');
      const pos = nodePosition(node, lineIndex);
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `'version' must be a string (e.g. '1.0.0'), or an object with SemVer or Custom key`,
      });
    } else if (typeof ver === 'object' && ver !== null && !Array.isArray(ver)) {
      const vObj = ver as Record<string, unknown>;
      if (!('SemVer' in vObj) && !('Custom' in vObj)) {
        const node = findMapValue(contents, 'version');
        const pos = nodePosition(node, lineIndex);
        diagnostics.push({
          ...pos,
          severity: Severity.Warning,
          message: `'version' object should contain a 'SemVer' or 'Custom' key`,
        });
      }
    }
  }
  validateStringField(contents, parsed, 'modifiers', lineIndex, diagnostics, true);
  validateStringField(contents, parsed, 'description', lineIndex, diagnostics, true);
  validateNumberField(contents, parsed, 'timeout', lineIndex, diagnostics, true);
  validateBooleanField(contents, parsed, 'collect_logs', lineIndex, diagnostics);
  validateBooleanField(contents, parsed, 'generator', lineIndex, diagnostics);

  validateEnumField(contents, parsed, 'scaler', IMAGE_SCALER_VALUES, lineIndex, diagnostics);
  validateEnumField(contents, parsed, 'display_type', OUTPUT_DISPLAY_TYPE_VALUES, lineIndex, diagnostics);

  if ('spawn_limit' in parsed) {
    const sl = parsed['spawn_limit'];
    if (typeof sl !== 'number' && sl !== 'Unlimited' && !(typeof sl === 'object' && sl !== null && 'Basic' in sl)) {
      const node = findMapValue(contents, 'spawn_limit');
      const pos = nodePosition(node, lineIndex);
      diagnostics.push({
        ...pos,
        severity: Severity.Error,
        message: `'spawn_limit' must be a number, 'Unlimited', or { Basic: <number> }`,
      });
    }
  }

  const scaler = typeof parsed['scaler'] === 'string' ? parsed['scaler'] : '';
  const isK8s = scaler === 'K8s' || scaler === '';
  const isKvm = scaler === 'Kvm';

  validateResources(contents, parsed, lineIndex, diagnostics, isK8s);
  validateArgs(contents, parsed, lineIndex, diagnostics);
  validateDependencies(contents, parsed, lineIndex, diagnostics);
  validateOutputCollection(contents, parsed, lineIndex, diagnostics);
  validateChildFilters(contents, parsed, lineIndex, diagnostics);
  if (isK8s) validateVolumes(contents, parsed, lineIndex, diagnostics);
  validateLifetime(contents, parsed, lineIndex, diagnostics);
  if (isK8s) validateSecurityContext(contents, parsed, lineIndex, diagnostics);
  validateSubObject(contents, parsed, 'clean_up', KNOWN_CLEANUP_FIELDS, lineIndex, diagnostics);
  if (isKvm) validateSubObject(contents, parsed, 'kvm', KNOWN_KVM_FIELDS, lineIndex, diagnostics);

  if (isK8s && 'network_policies' in parsed && !Array.isArray(parsed['network_policies'])) {
    const node = findMapValue(contents, 'network_policies');
    const pos = nodePosition(node, lineIndex);
    diagnostics.push({ ...pos, severity: Severity.Error, message: "'network_policies' must be an array of strings" });
  }

  if (isK8s && 'env' in parsed) {
    validateObjectField(contents, parsed, 'env', lineIndex, diagnostics);
  }

  if ('clean_up' in parsed) {
    const obj = parsed['clean_up'];
    if (typeof obj === 'object' && obj !== null && !Array.isArray(obj)) {
      const cu = obj as Record<string, unknown>;
      const cuMap = findMapValue(contents, 'clean_up');
      if (!('script' in cu)) {
        const node = findMapKey(contents, 'clean_up');
        const pos = node ? nodeLineCol(node, lineIndex) : { line: 1, column: 1 };
        diagnostics.push({ ...pos, severity: Severity.Error, message: "clean_up is missing required field: 'script'" });
      }
      validateArgStrategyField(cuMap, cu, 'job_id', lineIndex, diagnostics);
      validateArgStrategyField(cuMap, cu, 'results', lineIndex, diagnostics);
      validateArgStrategyField(cuMap, cu, 'result_files_dir', lineIndex, diagnostics);
    }
  }

  if ('kvm' in parsed) {
    const obj = parsed['kvm'];
    if (typeof obj === 'object' && obj !== null && !Array.isArray(obj)) {
      const kvm = obj as Record<string, unknown>;
      const kvmMap = findMapValue(contents, 'kvm');
      if (!('xml' in kvm)) {
        const pos = nodePosition(kvmMap, lineIndex);
        diagnostics.push({ ...pos, severity: Severity.Error, message: "kvm is missing required field: 'xml'" });
      }
      if (!('qcow2' in kvm)) {
        const pos = nodePosition(kvmMap, lineIndex);
        diagnostics.push({ ...pos, severity: Severity.Error, message: "kvm is missing required field: 'qcow2'" });
      }
    }
  }

  return diagnostics;
}
