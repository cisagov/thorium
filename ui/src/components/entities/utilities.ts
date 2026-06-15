import { Country } from 'country-list';

// project imports
import { getCountryCode } from '@entities/shared/countries';
import { diffTagUpdate } from '@utilities/tags';
import { listEntities } from '@thorpi/entities';
import { RequestTags } from '@models/tags';
import { Filters } from '@models/search';
import { CriticalSector } from '@models/entities/sectors';
import { Vendor } from '@models/entities/vendors';
import { Device, DeviceMetaFields } from '@models/entities/devices';
import { VendorMetaFields } from '@models/entities/vendors';
import { Collection, CollectionMetaFields } from '@models/entities/collections';
import { FileSystem } from '@models/entities/file_systems';
import { WindowsProcessTree } from '@models/entities/process_trees';
import { SigmaRule, SigmaRuleMetaFields } from '@models/entities/rules/sigma';
import { SigmaActionToTake } from '@models/entities/rules/sigma';
import { Incident, IncidentMetaFields } from '@models/entities/incident';
import { CompiledFunction, CompiledFunctionMetaFields, DecompiledFunction, DecompiledFunctionMetaFields } from '@models/entities/functions';
import { PeImport, PeImportMetaFields, PeSection, PeSectionMetaFields } from '@models/entities/pe';
import { Entities, EntityCreateTypes, EntityTypes, UISupportedEntityCreateTypes } from '@models/entities/entities';
import { EntityRequest, entityRequestKind } from '@models/entities/requests';

// spec: ./utilities.spec.md

export const DEFAULT_LIST_LIMIT = 25;

/**
 * Recursively append a value to a `FormData` using PHP/Rails-style bracket keys
 * (`prefix`, `prefix[]`, `prefix[key]`, `prefix[key][]`).
 *
 * Used to serialize an arbitrary, kind-agnostic entity metadata object into the `metadata[...]`
 * multipart fields the entity create route expects.
 *
 * @param form - The form to append to.
 * @param prefix - The current bracketed key prefix (e.g. `metadata[urls]`).
 * @param value - The value to encode (scalar, array, or nested object); `null`/`undefined` are skipped.
 */
export function appendBracketed(form: FormData, prefix: string, value: unknown): void {
  if (value === null || value === undefined) return;
  if (Array.isArray(value)) {
    value.forEach((item) => appendBracketed(form, `${prefix}[]`, item));
  } else if (typeof value === 'object') {
    Object.entries(value as Record<string, unknown>).forEach(([key, child]) => appendBracketed(form, `${prefix}[${key}]`, child));
  } else if (typeof value === 'string') {
    form.append(prefix, value);
  } else if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
    form.append(prefix, String(value));
  } else {
    // any other primitive (symbol/function shouldn't appear in parsed JSON) — serialize defensively
    form.append(prefix, JSON.stringify(value));
  }
}

/**
 * Build the multipart `FormData` for `createEntity` from a parsed {@link EntityRequest} (as returned
 * by the per-kind entities result-file).
 *
 * Serializes `name`, `kind` (derived from the metadata variant), `groups[]`, `tags[k][]`,
 * `description`, and the metadata fields via {@link appendBracketed}. Unit-variant metadata (a bare
 * string kind, e.g. `Other`) contributes only `kind`.
 *
 * @param req - The parsed entity request.
 * @returns The `FormData` to POST to `/entities/`.
 */
export function buildEntityRequestForm(req: EntityRequest): FormData {
  const form = new FormData();
  form.set('name', req.name);
  form.set('kind', entityRequestKind(req));
  (req.groups ?? []).forEach((group) => form.append('groups[]', group));
  if (req.description) {
    form.set('description', req.description);
  }
  Object.entries(req.tags ?? {}).forEach(([key, values]) => {
    (values ?? []).forEach((value) => form.append(`tags[${key}][]`, value));
  });
  // object-variant metadata carries the kind's fields; serialize them under metadata[...]
  if (req.metadata && typeof req.metadata === 'object') {
    const inner = req.metadata[entityRequestKind(req)];
    if (inner && typeof inner === 'object') {
      Object.entries(inner as Record<string, unknown>).forEach(([field, value]) => appendBracketed(form, `metadata[${field}]`, value));
    }
  }
  return form;
}

const reformatCriticalSectors = (sector: string): string => {
  return sector.replaceAll(' and ', '').replaceAll(',', '').replaceAll(' ', '');
};

export const getAvailableVendors = async (updateVendors: (vendorsMap: { [key: string]: string }) => void) => {
  const filters: Filters = { kinds: [Entities.Vendor], limit: 10000 };
  const vendorsMap: { [key: string]: string } = {};
  const { entityList } = await listEntities(filters, console.log, true, null);
  if (entityList) {
    entityList.forEach((entity: EntityTypes) => {
      if ('kind' in entity && entity.kind == Entities.Vendor) {
        vendorsMap[entity.id] = entity.name;
      }
    });
  }
  updateVendors(vendorsMap);
  return entityList;
};

// Append shared entity-level fields (name, groups, description, image) to the form
function appendEntityFields(
  updateForm: FormData,
  entity: EntityTypes,
  pendingEntity: EntityTypes,
  imageFile?: File | null,
  clearImage?: boolean,
): void {
  if (imageFile) {
    updateForm.append('image', imageFile);
  } else if (clearImage) {
    updateForm.set('clear_image', 'true');
  }
  if (entity.name != pendingEntity.name) {
    updateForm.set('name', pendingEntity.name);
  }
  const addGroups = pendingEntity.groups.filter((g) => !entity.groups.includes(g));
  const removeGroups = entity.groups.filter((g) => !pendingEntity.groups.includes(g));
  addGroups.forEach((group) => updateForm.append('add_groups[]', group));
  removeGroups.forEach((group) => updateForm.append('remove_groups[]', group));
  if (entity.description != pendingEntity.description) {
    if (pendingEntity === null || pendingEntity.description === '') {
      updateForm.set('clear_description', 'true');
    } else if (typeof pendingEntity.description == 'string') {
      updateForm.set('description', pendingEntity.description);
    }
  }
}

// Diff urls between current and pending metadata
function appendUrlDiffs(form: FormData, currentUrls: string[] | undefined, pendingUrls: string[] | undefined): void {
  const addUrls = pendingUrls?.filter((url) => !currentUrls?.includes(url));
  const removeUrls = currentUrls?.filter((url) => !pendingUrls?.includes(url));
  addUrls?.forEach((url) => form.append('metadata[add_urls][]', url));
  removeUrls?.forEach((url) => form.append('metadata[remove_urls][]', url));
}

// Diff vendors between current and pending metadata
function appendVendorDiffs(form: FormData, currentVendors: Vendor[] | undefined, pendingVendors: Vendor[] | undefined): void {
  pendingVendors
    ?.filter((pv) => !currentVendors?.map((iv) => iv.id).includes(pv.id))
    .forEach((v) => form.append('metadata[add_vendors][]', v.id));
  currentVendors
    ?.filter((iv) => !pendingVendors?.map((pv) => pv.id).includes(iv.id))
    .forEach((v) => form.append('metadata[remove_vendors][]', v.id));
}

// Diff critical sectors between current and pending metadata
function appendCriticalSectorDiffs(
  form: FormData,
  currentSectors: CriticalSector[] | undefined,
  pendingSectors: CriticalSector[] | undefined,
): void {
  const addSectors = pendingSectors?.filter((s) => !currentSectors?.includes(s));
  const removeSectors = currentSectors?.filter((s) => !pendingSectors?.includes(s));
  addSectors?.forEach((s) => form.append('metadata[add_critical_sectors][]', reformatCriticalSectors(s)));
  removeSectors?.forEach((s) => form.append('metadata[remove_critical_sectors][]', reformatCriticalSectors(s)));
}

// Diff countries between current and pending metadata
function appendCountryDiffs(form: FormData, currentCountries: Country[] | undefined, pendingCountries: Country[] | undefined): void {
  pendingCountries
    ?.filter((pc) => !currentCountries?.map((c) => c.code).includes(pc.code))
    .forEach((c) => {
      if (c.code !== undefined) form.append('metadata[add_countries][]', c.code);
    });
  currentCountries
    ?.filter((ic) => !pendingCountries?.map((c) => c.code).includes(ic.code))
    .forEach((c) => {
      if (c.code !== undefined) form.append('metadata[remove_countries][]', c.code);
    });
}

function appendDeviceMetaUpdates(form: FormData, meta: DeviceMetaFields, pending: DeviceMetaFields): void {
  appendUrlDiffs(form, meta.urls, pending.urls);
  appendVendorDiffs(form, meta.vendors, pending.vendors);
  if (meta.critical_system != pending.critical_system) {
    form.set('metadata[critical_system]', `${pending.critical_system}`);
  }
  if (pending.critical_system === false) {
    meta.critical_sectors?.forEach((sector) => {
      form.append('metadata[remove_critical_sectors][]', reformatCriticalSectors(sector));
    });
  }
  if (pending.critical_system === true) {
    appendCriticalSectorDiffs(form, meta.critical_sectors, pending.critical_sectors);
  }
  if (meta.sensitive_location != pending.sensitive_location && typeof pending.sensitive_location == 'boolean') {
    form.set('metadata[sensitive_location]', `${pending.sensitive_location}`);
  }
}

function appendVendorMetaUpdates(form: FormData, meta: VendorMetaFields, pending: VendorMetaFields): void {
  appendCriticalSectorDiffs(form, meta.critical_sectors, pending.critical_sectors);
  appendCountryDiffs(form, meta.countries, pending.countries);
}

function appendCollectionMetaUpdates(form: FormData, meta: CollectionMetaFields, pending: CollectionMetaFields): void {
  if (meta.tags_case_insensitive != pending.tags_case_insensitive && typeof pending.tags_case_insensitive == 'boolean') {
    form.set('metadata[collection_tags_case_insensitive]', `${pending.tags_case_insensitive}`);
  }
  if (meta.ignore_groups != pending.ignore_groups && typeof pending.ignore_groups == 'boolean') {
    form.set('metadata[collection_ignore_groups]', `${pending.ignore_groups}`);
  }
  if (meta.start !== pending.start) {
    if (!pending.start) {
      form.set('metadata[clear_collection_start]', 'true');
    } else {
      form.set('metadata[collection_start]', pending.start);
    }
  }
  if (meta.end !== pending.end) {
    if (!pending.end) {
      form.set('metadata[clear_collection_end]', 'true');
    } else {
      form.set('metadata[collection_end]', pending.end);
    }
  }
  const { toAdd, toDelete } = diffTagUpdate(meta.collection_tags, pending.collection_tags);
  Object.entries(toAdd).forEach(([k, vals]) => {
    vals.forEach((v) => form.append(`metadata[add_collection_tags][${k}][]`, v));
  });
  Object.entries(toDelete).forEach(([k, vals]) => {
    vals.forEach((v) => form.append(`metadata[delete_collection_tags][${k}][]`, v));
  });
}

function appendToolsDiffs(form: FormData, currentTools: string[], pendingTools: string[]): void {
  pendingTools.filter((t) => !currentTools.includes(t)).forEach((t) => form.append('metadata[add_tools][]', t));
  currentTools.filter((t) => !pendingTools.includes(t)).forEach((t) => form.append('metadata[remove_tools][]', t));
}

// Generic add/remove diff for a plain string list, appending to the given add/remove form keys.
function appendStringListDiff(form: FormData, addKey: string, removeKey: string, current: string[], pending: string[]): void {
  pending.filter((v) => !current.includes(v)).forEach((v) => form.append(addKey, v));
  current.filter((v) => !pending.includes(v)).forEach((v) => form.append(removeKey, v));
}

function appendIncidentMetaUpdates(form: FormData, meta: IncidentMetaFields, pending: IncidentMetaFields): void {
  // the API's incident update has no clear for cover_term, so only send it when set to a new value
  if (meta.cover_term !== pending.cover_term && pending.cover_term) {
    form.set('metadata[cover_term]', pending.cover_term);
  }
  appendStringListDiff(
    form,
    'metadata[add_mission_teams][]',
    'metadata[remove_mission_teams][]',
    meta.mission_teams,
    pending.mission_teams,
  );
  appendStringListDiff(form, 'metadata[add_networks][]', 'metadata[remove_networks][]', meta.networks, pending.networks);
  appendStringListDiff(form, 'metadata[add_machines][]', 'metadata[remove_machines][]', meta.machines, pending.machines);
  appendStringListDiff(form, 'metadata[add_locations][]', 'metadata[remove_locations][]', meta.locations, pending.locations);
}

function appendCompiledFunctionMetaUpdates(form: FormData, meta: CompiledFunctionMetaFields, pending: CompiledFunctionMetaFields): void {
  if (meta.address !== pending.address) {
    form.set('metadata[function_address]', String(pending.address));
  }
  // disassembly is replaced wholesale (the API update takes the full instruction list)
  if (JSON.stringify(meta.disassembly) !== JSON.stringify(pending.disassembly)) {
    pending.disassembly.forEach((ins) => form.append('metadata[disassembly][]', JSON.stringify(ins)));
  }
}

function appendDecompiledFunctionMetaUpdates(
  form: FormData,
  meta: DecompiledFunctionMetaFields,
  pending: DecompiledFunctionMetaFields,
): void {
  if (meta.address !== pending.address) {
    form.set('metadata[function_address]', String(pending.address));
  }
  if (meta.content !== pending.content) {
    form.set('metadata[decompilation_content]', pending.content);
  }
  // NOTE: the API's decompiled-function update form has no `tools` field — tools are set at creation only.
}

function appendPeSectionMetaUpdates(form: FormData, meta: PeSectionMetaFields, pending: PeSectionMetaFields): void {
  // scalar set-if-changed; the API update has no clear for these, so undefined is left untouched
  if (meta.md5 !== pending.md5 && pending.md5 !== undefined) {
    form.set('metadata[md5]', pending.md5);
  }
  if (meta.raw_size !== pending.raw_size && pending.raw_size !== undefined) {
    form.set('metadata[raw_size]', String(pending.raw_size));
  }
  if (meta.virtual_size !== pending.virtual_size && pending.virtual_size !== undefined) {
    form.set('metadata[virtual_size]', String(pending.virtual_size));
  }
  if (meta.entropy !== pending.entropy && pending.entropy !== undefined) {
    form.set('metadata[entropy]', String(pending.entropy));
  }
}

function appendPeImportMetaUpdates(form: FormData, meta: PeImportMetaFields, pending: PeImportMetaFields): void {
  // functions are replaced wholesale; the API ignores an empty list, so only send when changed and non-empty
  if (JSON.stringify(meta.functions) !== JSON.stringify(pending.functions)) {
    pending.functions.forEach((fn) => form.append('metadata[functions][]', fn));
  }
}

function appendSigmaRuleMetaUpdates(form: FormData, meta: SigmaRuleMetaFields, pending: SigmaRuleMetaFields): void {
  if (pending.rule !== '') {
    form.append('metadata[sigma_rule]', String(pending.rule));
  }
  if (pending.score) {
    form.append(`metadata[score]`, String(pending.score));
  }
  if (pending.actions && Array.isArray(pending.actions)) {
    pending.actions
      .filter((pa: SigmaActionToTake) => !meta.actions.map((ia) => JSON.stringify(ia)).includes(JSON.stringify(pa)))
      .forEach((a) => form.append('metadata[add_sigma_actions][]', JSON.stringify(a)));
    let numberRemovedActions = 0;
    meta.actions?.forEach((initialAction: SigmaActionToTake, index: number) => {
      if (JSON.stringify(initialAction) != JSON.stringify(pending.actions[index - numberRemovedActions])) {
        form.append('metadata[remove_sigma_actions][]', String(index));
        numberRemovedActions++;
      }
    });
  }
  if (meta.applies_to && Array.isArray(meta.applies_to)) {
    pending.applies_to?.filter((pe) => !meta.applies_to.includes(pe)).forEach((ae) => form.append('metadata[add_sigma_applies_to][]', ae));
    meta.applies_to
      ?.filter((ie) => !pending.applies_to.map((e) => JSON.stringify(e)).includes(JSON.stringify(ie)))
      .forEach((re) => form.append('metadata[remove_sigma_applies_to][]', re));
  }
}

export function buildUpdateEntityForm(
  entity: EntityTypes,
  pendingEntity: EntityTypes,
  imageFile?: File | null,
  clearImage?: boolean,
): FormData {
  const updateForm = new FormData();
  appendEntityFields(updateForm, entity, pendingEntity, imageFile, clearImage);

  switch (entity.kind) {
    case Entities.Device: {
      const meta = entity.metadata.Device;
      const pending = (pendingEntity as Device).metadata.Device;
      appendDeviceMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.Vendor: {
      const meta = entity.metadata.Vendor;
      const pending = (pendingEntity as import('@models/entities/vendors').Vendor).metadata.Vendor;
      appendVendorMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.Collection: {
      const meta = entity.metadata.Collection;
      const pending = (pendingEntity as Collection).metadata.Collection;
      appendCollectionMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.FileSystem: {
      const meta = entity.metadata.FileSystem;
      const pending = (pendingEntity as FileSystem).metadata.FileSystem;
      appendToolsDiffs(updateForm, meta.tools, pending.tools);
      break;
    }
    case Entities.WindowsProcessTree: {
      const meta = entity.metadata.WindowsProcessTree;
      const pending = (pendingEntity as WindowsProcessTree).metadata.WindowsProcessTree;
      appendToolsDiffs(updateForm, meta.tools, pending.tools);
      break;
    }
    case Entities.SigmaRule: {
      const meta = entity.metadata.SigmaRule;
      const pending = (pendingEntity as SigmaRule).metadata.SigmaRule;
      appendSigmaRuleMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.Incident: {
      const meta = entity.metadata.Incident;
      const pending = (pendingEntity as Incident).metadata.Incident;
      appendIncidentMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.CompiledFunction: {
      const meta = entity.metadata.CompiledFunction;
      const pending = (pendingEntity as CompiledFunction).metadata.CompiledFunction;
      appendCompiledFunctionMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.DecompiledFunction: {
      const meta = entity.metadata.DecompiledFunction;
      const pending = (pendingEntity as DecompiledFunction).metadata.DecompiledFunction;
      appendDecompiledFunctionMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.PeSection: {
      const meta = entity.metadata.PeSection;
      const pending = (pendingEntity as PeSection).metadata.PeSection;
      appendPeSectionMetaUpdates(updateForm, meta, pending);
      break;
    }
    case Entities.PeImport: {
      const meta = entity.metadata.PeImport;
      const pending = (pendingEntity as PeImport).metadata.PeImport;
      appendPeImportMetaUpdates(updateForm, meta, pending);
      break;
    }
    // Folder, WindowsProcess, NetworkConnection, Other: no metadata update logic
  }

  return updateForm;
}

export function buildCreateEntityForm(entity: EntityCreateTypes, imageFile?: File): FormData {
  const createForm = new FormData();
  if (imageFile) {
    createForm.append('image', imageFile);
  }
  const kind = entity.kind;
  createForm.set('name', entity.name);
  createForm.set('kind', kind);
  entity.groups.forEach((group: string) => createForm.append('groups[]', group));
  if (entity.description && entity.description !== '') {
    createForm.set('description', entity.description);
  }
  Object.keys(entity.tags).forEach((key: string) => {
    entity.tags[key].forEach((value: string) => createForm.append(`tags[${key}][]`, value));
  });

  switch (kind) {
    case Entities.Device: {
      const meta = entity.metadata.Device;
      if (meta.urls && meta.urls.length > 0) {
        meta.urls.forEach((url) => createForm.append('metadata[urls][]', url));
      }
      if (typeof meta.sensitive_location == 'boolean') {
        createForm.set('metadata[sensitive_location]', String(meta.sensitive_location));
      }
      if (typeof meta.critical_system == 'boolean') {
        createForm.set('metadata[critical_system]', String(meta.critical_system));
        if (meta.critical_system === true && meta.critical_sectors && meta.critical_sectors.length > 0) {
          meta.critical_sectors.forEach((s) => createForm.append('metadata[critical_sectors][]', reformatCriticalSectors(s)));
        }
      }
      if (meta.vendors && meta.vendors.length > 0) {
        meta.vendors.forEach((v) => createForm.append('metadata[vendors][]', v));
      }
      break;
    }
    case Entities.Vendor: {
      const meta = entity.metadata.Vendor;
      if (meta.critical_sectors && meta.critical_sectors.length > 0) {
        meta.critical_sectors.forEach((s) => createForm.append('metadata[critical_sectors][]', reformatCriticalSectors(s)));
      }
      if (meta.countries && meta.countries.length > 0) {
        meta.countries.forEach((name: string) => {
          const code = getCountryCode(name);
          if (code !== undefined) createForm.append('metadata[countries][]', code);
        });
      }
      break;
    }
    case Entities.Collection: {
      const meta = entity.metadata.Collection;
      if (meta.collection_kind) {
        createForm.append('metadata[collection_kind]', meta.collection_kind);
      }
      if (meta.tags_case_insensitive) {
        createForm.append('metadata[collection_tags_case_insensitive]', 'true');
      }
      if (meta.ignore_groups) {
        createForm.append('metadata[collection_ignore_groups]', 'true');
      }
      if (meta.start) {
        createForm.append('metadata[collection_start]', meta.start);
      }
      if (meta.end) {
        createForm.append('metadata[collection_end]', meta.end);
      }
      if (meta.collection_tags && typeof meta.collection_tags === 'object') {
        Object.entries(meta.collection_tags).forEach(([tagKey, tagVals]) => {
          if (Array.isArray(tagVals)) {
            tagVals.forEach((val) => createForm.append(`metadata[collection_tags][${tagKey}][]`, val));
          }
        });
      }
      break;
    }
    case Entities.SigmaRule: {
      const meta = entity.metadata.SigmaRule;
      if (meta.rule !== '') {
        createForm.append('metadata[sigma_rule]', String(meta.rule));
      }
      if (meta.score) {
        createForm.append(`metadata[score]`, String(meta.score));
      }
      if (meta.actions && Array.isArray(meta.actions)) {
        meta.actions.forEach((action: SigmaActionToTake) => {
          if (action.Flag) {
            createForm.append(`metadata[sigma_actions][]`, JSON.stringify(action));
          }
        });
      }
      if (meta.applies_to && Array.isArray(meta.applies_to)) {
        meta.applies_to.forEach((e) => createForm.append(`metadata[sigma_applies_to][]`, e));
      }
      break;
    }
    case Entities.Flag: {
      const meta = entity.metadata.Flag;
      createForm.set('metadata[suspicion]', String(meta.suspicion));
      createForm.set('metadata[confidence]', meta.confidence);
      createForm.set('metadata[reasoning]', meta.reasoning);
      if (meta.content) {
        createForm.set('metadata[content]', meta.content);
      }
      break;
    }
    case Entities.Incident: {
      const meta = entity.metadata.Incident;
      if (meta.cover_term) {
        createForm.set('metadata[cover_term]', meta.cover_term);
      }
      // incident lists are plain-text list fields (server pushes each value verbatim)
      meta.mission_teams.forEach((t) => createForm.append('metadata[mission_teams][]', t));
      meta.networks.forEach((n) => createForm.append('metadata[networks][]', n));
      meta.machines.forEach((m) => createForm.append('metadata[machines][]', m));
      meta.locations.forEach((l) => createForm.append('metadata[locations][]', l));
      break;
    }
    case Entities.CompiledFunction: {
      const meta = entity.metadata.CompiledFunction;
      createForm.set('metadata[function_address]', String(meta.address));
      // each instruction is JSON-serialized (server deserializes each disassembly[] entry)
      meta.disassembly.forEach((ins) => createForm.append('metadata[disassembly][]', JSON.stringify(ins)));
      break;
    }
    case Entities.DecompiledFunction: {
      const meta = entity.metadata.DecompiledFunction;
      createForm.set('metadata[function_address]', String(meta.address));
      createForm.set('metadata[decompilation_content]', meta.content);
      // tools are plain-text list entries (server pushes each value verbatim)
      meta.tools.forEach((t) => createForm.append('metadata[tools][]', t));
      break;
    }
    case Entities.PeSection: {
      const meta = entity.metadata.PeSection;
      // all section scalars are optional; only send the ones that were set
      if (meta.md5 !== undefined) createForm.set('metadata[md5]', meta.md5);
      if (meta.raw_size !== undefined) createForm.set('metadata[raw_size]', String(meta.raw_size));
      if (meta.virtual_size !== undefined) createForm.set('metadata[virtual_size]', String(meta.virtual_size));
      if (meta.entropy !== undefined) createForm.set('metadata[entropy]', String(meta.entropy));
      break;
    }
    case Entities.PeImport: {
      const meta = entity.metadata.PeImport;
      // imported functions are plain-text list entries (server pushes each value verbatim)
      meta.functions.forEach((fn) => createForm.append('metadata[functions][]', fn));
      break;
    }
    // Other entity kinds: no special create metadata handling
  }
  return createForm;
}

export function copyEntityFields(existingEntity: EntityTypes, blank: UISupportedEntityCreateTypes): UISupportedEntityCreateTypes {
  const newEntity = structuredClone(blank);
  newEntity.name = `${existingEntity.name} - copy`;
  newEntity.description = existingEntity.description;
  newEntity.groups = [...existingEntity.groups];
  newEntity.kind = existingEntity.kind as UISupportedEntityCreateTypes['kind'];

  // Copy and transform metadata per kind
  switch (existingEntity.kind) {
    case Entities.Device: {
      const srcMeta = existingEntity.metadata.Device;
      newEntity.metadata = {
        Device: {
          ...structuredClone(srcMeta),
          vendors: srcMeta.vendors ? srcMeta.vendors.map((v: Vendor) => v.id) : [],
        },
      };
      break;
    }
    case Entities.Vendor: {
      const srcMeta = existingEntity.metadata.Vendor;
      newEntity.metadata = {
        Vendor: {
          ...structuredClone(srcMeta),
          countries: srcMeta.countries ? srcMeta.countries.map((c: Country) => c.name) : [],
        },
      };
      break;
    }
    case Entities.Collection: {
      const srcMeta = existingEntity.metadata.Collection;
      newEntity.metadata = { Collection: structuredClone(srcMeta) };
      break;
    }
    case Entities.SigmaRule: {
      const srcMeta = existingEntity.metadata.SigmaRule;
      newEntity.metadata = { SigmaRule: structuredClone(srcMeta) };
      break;
    }
    case Entities.Flag: {
      const srcMeta = existingEntity.metadata.Flag;
      newEntity.metadata = { Flag: structuredClone(srcMeta) };
      break;
    }
    case Entities.PeSection: {
      const srcMeta = existingEntity.metadata.PeSection;
      newEntity.metadata = { PeSection: structuredClone(srcMeta) };
      break;
    }
    case Entities.PeImport: {
      const srcMeta = existingEntity.metadata.PeImport;
      newEntity.metadata = { PeImport: structuredClone(srcMeta) };
      break;
    }
  }

  // Copy tags from Tags format (key -> {value -> groups[]}) to RequestTags format (key -> values[])
  const newTags: RequestTags = {};
  Object.keys(existingEntity.tags).forEach((key: string) => {
    if (!(key in newTags)) {
      newTags[key] = [];
    }
    Object.keys(existingEntity.tags[key]).forEach((value: string) => {
      newTags[key].push(value);
    });
  });
  newEntity.tags = newTags;
  return newEntity;
}
