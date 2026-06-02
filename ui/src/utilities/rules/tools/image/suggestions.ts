import type { Document } from 'yaml';
import { isMap, isPair, isScalar } from 'yaml';
import { FieldValueType, type Suggestion, type FieldSchema } from '../../types';
import { buildLineIndex, offsetToLineCol, type LineIndex } from '../../yaml';
import { includes } from '../../types';
import { dedupeSuggestions } from '../shared';
import {
  IMAGE_SCALER_VALUES,
  OUTPUT_DISPLAY_TYPE_VALUES,
  DEPENDENCY_PASS_STRATEGY_VALUES,
  FILE_NAMING_STRATEGY_VALUES,
  VOLUME_TYPE_VALUES,
  LIFETIME_COUNTER_VALUES,
  OUTPUT_HANDLER_VALUES,
  IMAGE_FIELD_SCHEMAS,
  REQUIRED_IMAGE_FIELDS,
  KNOWN_IMAGE_FIELDS,
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
  KNOWN_KVM_FIELDS,
  KNOWN_BURSTABLE_FIELDS,
  KNOWN_GENERIC_CACHE_FIELDS,
  KNOWN_AUTO_TAG_FIELDS,
  KNOWN_VOLUME_FIELDS,
  VOLUME_ENTRY_SCHEMA,
  IMAGE_SECTION_ORDER,
  imageFieldCategory,
} from './schema';

function suggestMissingSubFields(
  parentDotPath: string,
  knownFields: readonly string[],
  parsed: Record<string, unknown>,
  parentLine: number,
  schemas: Record<string, FieldSchema>,
  suggestions: Suggestion[],
) {
  for (const field of knownFields) {
    if (!(field in parsed)) {
      const dottedField = `${parentDotPath}.${field}`;
      const schema = lookupSchema(schemas, dottedField);
      suggestions.push({
        line: parentLine,
        field: dottedField,
        message: schema?.description ?? `Consider adding '${field}'`,
        values: schema?.enumValues,
        schema,
      });
    }
  }
}

function suggestUnknownSubFields(
  parentDotPath: string,
  knownFields: readonly string[],
  parsed: Record<string, unknown>,
  parentLine: number,
  suggestions: Suggestion[],
) {
  for (const key of Object.keys(parsed)) {
    if (!includes(knownFields, key)) {
      const unknownKey: string = key;
      suggestions.push({
        line: parentLine,
        field: `${parentDotPath}.${unknownKey}`,
        message: `Remove unknown field '${unknownKey}'`,
        isRemoval: true,
        category: 'Unknown Fields',
      });
    }
  }
}

function suggestNullReplace(
  field: string,
  parsed: Record<string, unknown>,
  line: number,
  schemas: Record<string, FieldSchema>,
  suggestions: Suggestion[],
): boolean {
  if (!(field in parsed) || parsed[field] !== null) return false;
  const schema = schemas[field];
  if (!schema) return false;
  suggestions.push({
    line,
    field,
    message: `Populate '${field}'`,
    schema,
    isReplace: true,
  });
  return true;
}

function lookupSchema(schemas: Record<string, FieldSchema>, dottedField: string): FieldSchema | undefined {
  const parts = dottedField.split('.');
  let schema = schemas[parts[0]];
  for (let i = 1; i < parts.length && schema; i++) {
    if (schema.type === FieldValueType.Object && schema.fields) {
      schema = schema.fields[parts[i]];
    } else {
      return undefined;
    }
  }
  return schema;
}

function findKeyLine(map: unknown, key: string, lineIndex: LineIndex): number {
  if (!isMap(map)) return 1;
  for (const item of map.items) {
    if (isPair(item) && isScalar(item.key) && item.key.value === key) {
      const offset = item.key.range?.[0] ?? 0;
      return offsetToLineCol(lineIndex, offset).line;
    }
  }
  return 1;
}

function lastDocLine(lineIndex: LineIndex): number {
  return lineIndex.offsets.length;
}

export function generateImageSuggestions(doc: Document, text: string, parsed: Record<string, unknown>): Suggestion[] {
  const suggestions: Suggestion[] = [];
  const lineIndex = buildLineIndex(text);
  const endLine = lastDocLine(lineIndex);

  const scaler = typeof parsed['scaler'] === 'string' ? parsed['scaler'] : '';
  const isK8s = scaler === 'K8s' || scaler === '';
  const isKvm = scaler === 'Kvm';

  const nullObjectFields: string[] = ['lifetime', 'resources', 'args', 'dependencies', 'output_collection', 'child_filters', 'clean_up'];
  if (isK8s) {
    nullObjectFields.push('security_context');
    nullObjectFields.push('env');
  }
  if (isKvm) nullObjectFields.push('kvm');
  const handledNullFields = new Set<string>();
  for (const field of nullObjectFields) {
    const line = field in parsed ? findKeyLine(doc.contents, field, lineIndex) : endLine;
    if (suggestNullReplace(field, parsed, line, IMAGE_FIELD_SCHEMAS, suggestions)) {
      handledNullFields.add(field);
    }
  }

  for (const field of KNOWN_IMAGE_FIELDS) {
    if (handledNullFields.has(field)) continue;
    if (!(field in parsed) || parsed[field] !== null) continue;
    const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, field);
    if (!schema) continue;
    const line = findKeyLine(doc.contents, field, lineIndex);
    if (schema.enumValues) {
      suggestions.push({ line, field, message: `Populate '${field}'`, values: schema.enumValues, schema });
    } else {
      suggestions.push({ line, field, message: `Populate '${field}'`, schema, isReplace: true });
    }
  }

  if ('scaler' in parsed) {
    const line = findKeyLine(doc.contents, 'scaler', lineIndex);
    if (typeof parsed['scaler'] !== 'string' || parsed['scaler'] === '') {
      suggestions.push({
        line,
        field: 'scaler',
        message: 'Valid scaler values',
        values: IMAGE_SCALER_VALUES,
        schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'scaler'),
      });
    }
  }

  if ('display_type' in parsed) {
    const line = findKeyLine(doc.contents, 'display_type', lineIndex);
    if (typeof parsed['display_type'] !== 'string' || parsed['display_type'] === '') {
      suggestions.push({
        line,
        field: 'display_type',
        message: 'Valid display type values',
        values: OUTPUT_DISPLAY_TYPE_VALUES,
        schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'display_type'),
      });
    }
  }

  if ('lifetime' in parsed && typeof parsed['lifetime'] === 'object' && parsed['lifetime'] !== null) {
    const lt = parsed['lifetime'] as Record<string, unknown>;
    const lifetimeLine = findKeyLine(doc.contents, 'lifetime', lineIndex);
    if (!('counter' in lt) || lt['counter'] === '' || lt['counter'] === null) {
      suggestions.push({
        line: lifetimeLine,
        field: 'lifetime.counter',
        message: 'Valid lifetime counter values',
        values: LIFETIME_COUNTER_VALUES,
        schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'lifetime.counter'),
      });
    }
  }

  if ('dependencies' in parsed && typeof parsed['dependencies'] === 'object' && parsed['dependencies'] !== null) {
    const deps = parsed['dependencies'] as Record<string, unknown>;
    const depLine = findKeyLine(doc.contents, 'dependencies', lineIndex);

    const strategySubSections = ['samples', 'ephemeral', 'results', 'repos', 'tags', 'children'] as const;
    for (const section of strategySubSections) {
      if (section in deps && typeof deps[section] === 'object' && deps[section] !== null) {
        const sub = deps[section] as Record<string, unknown>;
        if ('strategy' in sub && (typeof sub['strategy'] !== 'string' || sub['strategy'] === '')) {
          suggestions.push({
            line: depLine,
            field: `dependencies.${section}.strategy`,
            message: 'Valid dependency pass strategy values',
            values: DEPENDENCY_PASS_STRATEGY_VALUES,
            schema: { type: FieldValueType.Enum, enumValues: DEPENDENCY_PASS_STRATEGY_VALUES },
          });
        }
      }
    }

    if ('samples' in deps && typeof deps['samples'] === 'object' && deps['samples'] !== null) {
      const samples = deps['samples'] as Record<string, unknown>;
      if ('naming' in samples && (typeof samples['naming'] !== 'string' || samples['naming'] === '')) {
        suggestions.push({
          line: depLine,
          field: 'dependencies.samples.naming',
          message: 'Valid file naming strategy values',
          values: FILE_NAMING_STRATEGY_VALUES,
          schema: { type: FieldValueType.Enum, enumValues: FILE_NAMING_STRATEGY_VALUES },
        });
      }
    }
  }

  if ('output_collection' in parsed && typeof parsed['output_collection'] === 'object' && parsed['output_collection'] !== null) {
    const oc = parsed['output_collection'] as Record<string, unknown>;
    const ocLine = findKeyLine(doc.contents, 'output_collection', lineIndex);
    if ('handler' in oc && (typeof oc['handler'] !== 'string' || oc['handler'] === '')) {
      suggestions.push({
        line: ocLine,
        field: 'output_collection.handler',
        message: 'Valid output handler values',
        values: OUTPUT_HANDLER_VALUES,
        schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'output_collection.handler'),
      });
    }
  }

  if (isK8s && 'volumes' in parsed && Array.isArray(parsed['volumes'])) {
    const volLine = findKeyLine(doc.contents, 'volumes', lineIndex);
    for (const vol of parsed['volumes']) {
      if (typeof vol === 'object' && vol !== null) {
        const v = vol as Record<string, unknown>;
        if ('archetype' in v && (typeof v['archetype'] !== 'string' || v['archetype'] === '')) {
          suggestions.push({
            line: volLine,
            field: 'volumes[].archetype',
            message: 'Valid volume type values',
            values: VOLUME_TYPE_VALUES,
            schema: { type: FieldValueType.Enum, enumValues: VOLUME_TYPE_VALUES },
          });
          break;
        }
      }
    }
  }

  // Conditional suggestions: X implies Y should be present
  if (parsed['scaler'] === 'Kvm' && !('kvm' in parsed)) {
    suggestions.push({
      line: endLine,
      field: 'kvm',
      message: "Required when scaler is 'Kvm'",
      schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'kvm'),
    });
  }

  if (parsed['generator'] === true && !('child_filters' in parsed)) {
    suggestions.push({
      line: endLine,
      field: 'child_filters',
      message: 'Recommended for generator images to filter child submissions',
      schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'child_filters'),
    });
  }

  if ('output_collection' in parsed && typeof parsed['output_collection'] === 'object' && parsed['output_collection'] !== null) {
    const oc = parsed['output_collection'] as Record<string, unknown>;
    if ((oc['handler'] === 'Files' || !('handler' in oc)) && !('files' in oc)) {
      const ocLine = findKeyLine(doc.contents, 'output_collection', lineIndex);
      suggestions.push({
        line: ocLine,
        field: 'output_collection.files',
        message: "Recommended when handler is 'Files'",
        schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'output_collection.files'),
      });
    }
  }

  if ('dependencies' in parsed && typeof parsed['dependencies'] === 'object' && parsed['dependencies'] !== null) {
    const deps = parsed['dependencies'] as Record<string, unknown>;
    if (typeof deps['children'] === 'object' && deps['children'] !== null) {
      const children = deps['children'] as Record<string, unknown>;
      if (children['enabled'] === true && !('images' in children)) {
        const depLine = findKeyLine(doc.contents, 'dependencies', lineIndex);
        suggestions.push({
          line: depLine,
          field: 'dependencies.children.images',
          message: 'Specify which images to pull children from when enabled',
          schema: lookupSchema(IMAGE_FIELD_SCHEMAS, 'dependencies.children.images'),
        });
      }
    }
  }

  // Suggest missing sub-fields within existing objects
  if ('resources' in parsed && typeof parsed['resources'] === 'object' && parsed['resources'] !== null) {
    const res = parsed['resources'] as Record<string, unknown>;
    const resLine = findKeyLine(doc.contents, 'resources', lineIndex);
    const K8S_ONLY_RESOURCE_FIELDS = ['nvidia_gpu', 'amd_gpu', 'burstable'] as const;
    const activeResourceFields = isK8s
      ? KNOWN_RESOURCES_FIELDS
      : KNOWN_RESOURCES_FIELDS.filter((f) => !(K8S_ONLY_RESOURCE_FIELDS as readonly string[]).includes(f));
    suggestMissingSubFields('resources', activeResourceFields, res, resLine, IMAGE_FIELD_SCHEMAS, suggestions);
    suggestUnknownSubFields('resources', KNOWN_RESOURCES_FIELDS, res, resLine, suggestions);
    if (isK8s && 'burstable' in res && typeof res['burstable'] === 'object' && res['burstable'] !== null) {
      const burst = res['burstable'] as Record<string, unknown>;
      suggestMissingSubFields('resources.burstable', KNOWN_BURSTABLE_FIELDS, burst, resLine, IMAGE_FIELD_SCHEMAS, suggestions);
      suggestUnknownSubFields('resources.burstable', KNOWN_BURSTABLE_FIELDS, burst, resLine, suggestions);
    }
  }

  if ('args' in parsed && typeof parsed['args'] === 'object' && parsed['args'] !== null) {
    const args = parsed['args'] as Record<string, unknown>;
    const argsLine = findKeyLine(doc.contents, 'args', lineIndex);
    suggestMissingSubFields('args', KNOWN_ARGS_FIELDS, args, argsLine, IMAGE_FIELD_SCHEMAS, suggestions);
    suggestUnknownSubFields('args', KNOWN_ARGS_FIELDS, args, argsLine, suggestions);
  }

  if ('lifetime' in parsed && typeof parsed['lifetime'] === 'object' && parsed['lifetime'] !== null) {
    const lt = parsed['lifetime'] as Record<string, unknown>;
    const ltLine = findKeyLine(doc.contents, 'lifetime', lineIndex);
    for (const field of KNOWN_LIFETIME_FIELDS) {
      if (!(field in lt)) {
        const dottedField = `lifetime.${field}`;
        const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, dottedField);
        suggestions.push({
          line: ltLine,
          field: dottedField,
          message: schema?.required ? `Required: '${field}'` : (schema?.description ?? `Consider adding '${field}'`),
          values: schema?.enumValues,
          schema,
        });
      }
    }
    suggestUnknownSubFields('lifetime', KNOWN_LIFETIME_FIELDS, lt, ltLine, suggestions);
  }

  if ('dependencies' in parsed && typeof parsed['dependencies'] === 'object' && parsed['dependencies'] !== null) {
    const deps = parsed['dependencies'] as Record<string, unknown>;
    const depLine = findKeyLine(doc.contents, 'dependencies', lineIndex);
    // Suggest missing dependency sub-sections
    for (const field of KNOWN_DEPENDENCIES_FIELDS) {
      if (!(field in deps)) {
        const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, `dependencies.${field}`);
        suggestions.push({
          line: depLine,
          field: `dependencies.${field}`,
          message: schema?.description ?? `Consider adding '${field}' dependency config`,
          schema,
        });
      }
    }
    suggestUnknownSubFields('dependencies', KNOWN_DEPENDENCIES_FIELDS, deps, depLine, suggestions);
    // Sub-field suggestions within each dependency sub-section
    const depSubConfigs: Array<{ key: string; fields: readonly string[] }> = [
      { key: 'samples', fields: KNOWN_SAMPLE_DEP_FIELDS },
      { key: 'repos', fields: KNOWN_REPO_DEP_FIELDS },
      { key: 'tags', fields: KNOWN_TAG_DEP_FIELDS },
      { key: 'children', fields: KNOWN_CHILDREN_DEP_FIELDS },
      { key: 'ephemeral', fields: KNOWN_EPHEMERAL_DEP_FIELDS },
      { key: 'results', fields: KNOWN_RESULT_DEP_FIELDS },
      { key: 'cache', fields: KNOWN_CACHE_DEP_FIELDS },
    ];
    for (const { key, fields } of depSubConfigs) {
      if (key in deps && typeof deps[key] === 'object' && deps[key] !== null) {
        const sub = deps[key] as Record<string, unknown>;
        for (const f of fields) {
          if (!(f in sub)) {
            const dottedField = `dependencies.${key}.${f}`;
            // Use the canonical schema directly so list fields (images/names) stay StringArray,
            // booleans stay boolean, and results.kwarg keeps its KwargDependency variant schema.
            const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, dottedField);
            suggestions.push({
              line: depLine,
              field: dottedField,
              message: schema?.description ?? `Consider adding '${f}'`,
              values: schema?.enumValues,
              schema,
            });
          }
        }
        suggestUnknownSubFields(`dependencies.${key}`, fields, sub, depLine, suggestions);
      }
    }
    if ('cache' in deps && typeof deps['cache'] === 'object' && deps['cache'] !== null) {
      const cache = deps['cache'] as Record<string, unknown>;
      if ('generic' in cache && typeof cache['generic'] === 'object' && cache['generic'] !== null) {
        const generic = cache['generic'] as Record<string, unknown>;
        for (const f of KNOWN_GENERIC_CACHE_FIELDS) {
          if (!(f in generic)) {
            const dottedField = `dependencies.cache.generic.${f}`;
            const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, dottedField);
            suggestions.push({
              line: depLine,
              field: dottedField,
              message: schema?.description ?? `Consider adding '${f}'`,
              values: schema?.enumValues,
              schema,
            });
          }
        }
        suggestUnknownSubFields('dependencies.cache.generic', KNOWN_GENERIC_CACHE_FIELDS, generic, depLine, suggestions);
      }
    }
  }

  if ('output_collection' in parsed && typeof parsed['output_collection'] === 'object' && parsed['output_collection'] !== null) {
    const oc = parsed['output_collection'] as Record<string, unknown>;
    const ocLine = findKeyLine(doc.contents, 'output_collection', lineIndex);
    for (const field of KNOWN_OUTPUT_COLLECTION_FIELDS) {
      if (!(field in oc)) {
        const dottedField = `output_collection.${field}`;
        const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, dottedField);
        suggestions.push({
          line: ocLine,
          field: dottedField,
          message: schema?.required ? `Required: '${field}'` : (schema?.description ?? `Consider adding '${field}'`),
          values: schema?.enumValues,
          schema,
        });
      }
    }
    suggestUnknownSubFields('output_collection', KNOWN_OUTPUT_COLLECTION_FIELDS, oc, ocLine, suggestions);
    if ('files' in oc && typeof oc['files'] === 'object' && oc['files'] !== null) {
      const files = oc['files'] as Record<string, unknown>;
      for (const field of KNOWN_FILES_HANDLER_FIELDS) {
        if (!(field in files)) {
          const dottedField = `output_collection.files.${field}`;
          const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, dottedField);
          suggestions.push({
            line: ocLine,
            field: dottedField,
            message: schema?.description ?? `Consider adding '${field}'`,
            schema,
          });
        }
      }
      suggestUnknownSubFields('output_collection.files', KNOWN_FILES_HANDLER_FIELDS, files, ocLine, suggestions);
    }
    if ('auto_tag' in oc && typeof oc['auto_tag'] === 'object' && oc['auto_tag'] !== null) {
      const autoTagMap = oc['auto_tag'] as Record<string, unknown>;
      for (const [tagName, tagVal] of Object.entries(autoTagMap)) {
        if (typeof tagVal === 'object' && tagVal !== null) {
          const tagObj = tagVal as Record<string, unknown>;
          for (const f of KNOWN_AUTO_TAG_FIELDS) {
            if (!(f in tagObj)) {
              const subSchema = IMAGE_FIELD_SCHEMAS.output_collection.fields!.auto_tag.fields![f];
              suggestions.push({
                line: ocLine,
                field: `output_collection.auto_tag.${tagName}.${f}`,
                message: subSchema?.required ? `Required: '${f}'` : (subSchema?.description ?? `Consider adding '${f}'`),
                values: subSchema?.enumValues,
                schema: subSchema,
              });
            }
          }
        }
      }
    }
    suggestions.push({
      line: ocLine,
      field: 'output_collection.auto_tag.tag-name',
      message: 'Add auto_tag rule',
      isMapEntry: true,
      schema: IMAGE_FIELD_SCHEMAS.output_collection.fields!.auto_tag,
    });
  }

  if ('child_filters' in parsed && typeof parsed['child_filters'] === 'object' && parsed['child_filters'] !== null) {
    const cf = parsed['child_filters'] as Record<string, unknown>;
    const cfLine = findKeyLine(doc.contents, 'child_filters', lineIndex);
    suggestMissingSubFields('child_filters', KNOWN_CHILD_FILTERS_FIELDS, cf, cfLine, IMAGE_FIELD_SCHEMAS, suggestions);
    suggestUnknownSubFields('child_filters', KNOWN_CHILD_FILTERS_FIELDS, cf, cfLine, suggestions);
  }

  if ('clean_up' in parsed && typeof parsed['clean_up'] === 'object' && parsed['clean_up'] !== null) {
    const cu = parsed['clean_up'] as Record<string, unknown>;
    const cuLine = findKeyLine(doc.contents, 'clean_up', lineIndex);
    for (const field of KNOWN_CLEANUP_FIELDS) {
      if (!(field in cu)) {
        const dottedField = `clean_up.${field}`;
        const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, dottedField);
        suggestions.push({
          line: cuLine,
          field: dottedField,
          message: schema?.required ? `Required: '${field}'` : (schema?.description ?? `Consider adding '${field}'`),
          values: schema?.enumValues,
          schema,
        });
      }
    }
    suggestUnknownSubFields('clean_up', KNOWN_CLEANUP_FIELDS, cu, cuLine, suggestions);
  }

  if (isK8s && 'security_context' in parsed && typeof parsed['security_context'] === 'object' && parsed['security_context'] !== null) {
    const sc = parsed['security_context'] as Record<string, unknown>;
    const scLine = findKeyLine(doc.contents, 'security_context', lineIndex);
    suggestMissingSubFields('security_context', KNOWN_SECURITY_CONTEXT_FIELDS, sc, scLine, IMAGE_FIELD_SCHEMAS, suggestions);
    suggestUnknownSubFields('security_context', KNOWN_SECURITY_CONTEXT_FIELDS, sc, scLine, suggestions);
  }

  if (isKvm && 'kvm' in parsed && typeof parsed['kvm'] === 'object' && parsed['kvm'] !== null) {
    const kvm = parsed['kvm'] as Record<string, unknown>;
    const kvmLine = findKeyLine(doc.contents, 'kvm', lineIndex);
    for (const field of KNOWN_KVM_FIELDS) {
      if (!(field in kvm)) {
        const dottedField = `kvm.${field}`;
        const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, dottedField);
        suggestions.push({
          line: kvmLine,
          field: dottedField,
          message: schema?.required ? `Required: '${field}'` : (schema?.description ?? `Consider adding '${field}'`),
          values: schema?.enumValues,
          schema,
        });
      }
    }
    suggestUnknownSubFields('kvm', KNOWN_KVM_FIELDS, kvm, kvmLine, suggestions);
  }

  if (isK8s) {
    const envLine = 'env' in parsed ? findKeyLine(doc.contents, 'env', lineIndex) : endLine;
    suggestions.push({
      line: envLine,
      field: 'env.VAR_NAME',
      message: 'Add environment variable',
      isMapEntry: true,
      schema: { type: FieldValueType.String, placeholder: 'value' },
    });
  }

  if (isK8s && 'volumes' in parsed && Array.isArray(parsed['volumes'])) {
    const volLine = findKeyLine(doc.contents, 'volumes', lineIndex);
    for (const vol of parsed['volumes']) {
      if (typeof vol === 'object' && vol !== null) {
        const v = vol as Record<string, unknown>;
        suggestMissingSubFields('volumes[]', KNOWN_VOLUME_FIELDS, v, volLine, { 'volumes[]': VOLUME_ENTRY_SCHEMA }, suggestions);
        break;
      }
    }
  }

  const K8S_ONLY_FIELDS = ['volumes', 'security_context', 'network_policies', 'env'] as const;
  const KVM_ONLY_FIELDS = ['kvm'] as const;
  for (const field of KNOWN_IMAGE_FIELDS) {
    if (field in parsed) continue;
    if (field === 'env') continue;
    if (!isK8s && (K8S_ONLY_FIELDS as readonly string[]).includes(field)) continue;
    if (!isKvm && (KVM_ONLY_FIELDS as readonly string[]).includes(field)) continue;
    const schema = lookupSchema(IMAGE_FIELD_SCHEMAS, field);
    const isRequired = includes(REQUIRED_IMAGE_FIELDS, field);
    // volumes is a list of structured Volume objects — offer the full object schema as a list entry
    if (field === 'volumes') {
      suggestions.push({
        line: endLine,
        field: 'volumes',
        message: schema?.description ?? "Consider adding 'volumes'",
        schema: VOLUME_ENTRY_SCHEMA,
        isList: true,
      });
      continue;
    }
    suggestions.push({
      line: endLine,
      field,
      message: isRequired ? `Required: '${field}'` : (schema?.description ?? `Consider adding '${field}'`),
      values: schema?.enumValues,
      schema,
    });
  }

  // Removal suggestions for unknown fields and scaler-inappropriate fields
  for (const key of Object.keys(parsed)) {
    if (!includes(KNOWN_IMAGE_FIELDS, key)) {
      const keyLine = findKeyLine(doc.contents, key, lineIndex);
      suggestions.push({
        line: keyLine,
        field: key,
        message: `Remove unknown field '${key}'`,
        isRemoval: true,
        category: 'Unknown Fields',
      });
    } else if (!isK8s && (K8S_ONLY_FIELDS as readonly string[]).includes(key)) {
      const keyLine = findKeyLine(doc.contents, key, lineIndex);
      suggestions.push({
        line: keyLine,
        field: key,
        message: `'${key}' only applies to scaler 'K8s'`,
        isRemoval: true,
        category: 'Invalid Fields',
      });
    } else if (!isKvm && (KVM_ONLY_FIELDS as readonly string[]).includes(key)) {
      const keyLine = findKeyLine(doc.contents, key, lineIndex);
      suggestions.push({
        line: keyLine,
        field: key,
        message: `'${key}' only applies to scaler 'Kvm'`,
        isRemoval: true,
        category: 'Invalid Fields',
      });
    }
  }

  const deduped = dedupeSuggestions(suggestions);
  for (const s of deduped) s.category = s.category ?? imageFieldCategory(s.field);
  const sectionIndex = (cat: string) => {
    const idx = (IMAGE_SECTION_ORDER as readonly string[]).indexOf(cat);
    return idx >= 0 ? idx : IMAGE_SECTION_ORDER.length;
  };
  deduped.sort((a, b) => {
    const sa = sectionIndex(a.category!);
    const sb = sectionIndex(b.category!);
    if (sa !== sb) return sa - sb;
    return a.field.localeCompare(b.field);
  });
  return deduped;
}
