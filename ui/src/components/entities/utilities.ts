import { getCode as getCountryCode, Country } from 'country-list';

// project imports
import { diffTagUpdate } from '@utilities/tags';
import { listEntities } from '@thorpi/entities';
import { RequestTags } from '@models/tags';
import { Filters } from '@models/search';
import { CriticalSector } from '@models/entities/sectors';
import { Vendor } from '@models/entities/vendors';
import {
  Entities,
  EntityCreateTypeMap,
  EntityMetaMap,
  EntityTypeMap,
  EntityTypes,
  EntityUISupportedCreateTypeMap,
} from '@models/entities/entities';
import { SigmaActionToTake } from '@models/entities/rules/sigma';

// default number of results to render when listing resources
export const DEFAULT_LIST_LIMIT = 25;

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

export function buildUpdateEntityForm<T extends keyof EntityTypeMap>(entity: EntityTypeMap[T], pendingEntity: EntityTypeMap[T]): FormData {
  const updateForm = new FormData();
  // name
  if (entity.name != pendingEntity.name) {
    updateForm.set('name', pendingEntity.name);
  }
  // groups
  const addGroups = pendingEntity.groups.filter((pendingGroup) => !entity.groups.includes(pendingGroup));
  const removeGroups = entity.groups.filter((entityGroup) => !pendingEntity.groups.includes(entityGroup));
  addGroups.map((group) => {
    updateForm.append('add_groups[]', group);
  });
  removeGroups.map((group) => {
    updateForm.append('remove_groups[]', group);
  });
  // description
  if (entity.description != pendingEntity.description) {
    if (pendingEntity === null || pendingEntity.description === '') {
      updateForm.set('clear_description', 'true');
    } else if (typeof pendingEntity.description == 'string') {
      updateForm.set('description', pendingEntity.description);
    }
  }
  // metadata
  const metadata = entity.metadata[entity.kind];
  const pendingMeta = pendingEntity.metadata[entity.kind];
  // metadata: urls
  const addUrls: string[] | undefined = pendingMeta?.urls?.filter((url: string) => !metadata.urls.includes(url));
  const removeUrls: string[] | undefined = metadata?.urls?.filter((url: string) => !pendingMeta.urls.includes(url));
  addUrls?.map((url: string) => {
    updateForm.append('metadata[add_urls][]', url);
  });
  removeUrls?.map((url: string) => {
    updateForm.append('metadata[remove_urls][]', url);
  });
  // metadata: vendors
  pendingMeta?.vendors
    ?.filter((pendingVendor: Vendor) => !metadata.vendors.map((initialVendor: Vendor) => initialVendor.id).includes(pendingVendor.id))
    .forEach((addedVendor: Vendor) => updateForm.append('metadata[add_vendors][]', addedVendor.id));
  metadata?.vendors
    ?.filter((initialVendor: Vendor) => !pendingMeta.vendors.map((pendingVendor: Vendor) => pendingVendor.id).includes(initialVendor.id))
    .forEach((removedVendor: Vendor) => updateForm.append('metadata[remove_vendors][]', removedVendor.id));
  // metadata: critical_system/critical_sectors
  if (metadata?.critical_system != pendingMeta?.critical_system) {
    updateForm.set('metadata[critical_system]', `${pendingMeta.critical_system}`);
  }
  // critical system is false, clear any critical_sectors listed
  if (entity.kind != 'Vendor' && pendingMeta.critical_system === false) {
    metadata.critical_sectors.map((sector: CriticalSector) => {
      updateForm.append('metadata[remove_critical_sectors][]', reformatCriticalSectors(sector));
    });
  }
  if (entity.kind == 'Vendor' || pendingMeta.critical_system === true) {
    const addSectors: string[] | undefined = pendingMeta?.critical_sectors?.filter(
      (sector: string) => !metadata.critical_sectors.includes(sector),
    );
    addSectors?.map((sector: string) => {
      updateForm.append('metadata[add_critical_sectors][]', reformatCriticalSectors(sector));
    });
    const removeSectors: string[] | undefined = metadata?.critical_sectors?.filter(
      (sector: string) => !pendingMeta.critical_sectors.includes(sector),
    );
    removeSectors?.map((sector: string) => {
      updateForm.append('metadata[remove_critical_sectors][]', reformatCriticalSectors(sector));
    });
  }
  // metadata: sensitive_location
  if (metadata?.sensitive_location != pendingMeta.sensitive_location && typeof pendingMeta?.sensitive_location == 'boolean') {
    updateForm.set('metadata[sensitive_location]', `${pendingMeta.sensitive_location}`);
  }
  // metadata: countries
  pendingMeta?.countries
    ?.filter((country: Country) => !metadata.countries.map((country: Country) => country.code).includes(country.code))
    .forEach((country: Country) => {
      if (country.code !== undefined) {
        updateForm.append('metadata[add_countries][]', country.code);
      }
    });
  metadata?.countries
    ?.filter((country: Country) => !pendingMeta.countries.map((country: Country) => country.code).includes(country.code))
    .forEach((country: any) => {
      if (country.code !== undefined) {
        updateForm.append('metadata[remove_countries][]', country.code);
      }
    });
  // Collection specific metadata
  if (entity.kind == Entities.Collection) {
    if (metadata?.tags_case_insensitive != pendingMeta.tags_case_insensitive && typeof pendingMeta?.tags_case_insensitive == 'boolean') {
      updateForm.set('metadata[collection_tags_case_insensitive]', `${pendingMeta.tags_case_insensitive}`);
    }
    if (metadata?.ignore_groups != pendingMeta.ignore_groups && typeof pendingMeta?.ignore_groups == 'boolean') {
      updateForm.set('metadata[collection_ignore_groups]', `${pendingMeta.ignore_groups}`);
    }
    if (metadata?.start !== pendingMeta?.start) {
      if (pendingMeta.start === null || pendingMeta.start === '') {
        updateForm.set('metadata[clear_collection_start]', 'true');
      } else {
        updateForm.set('metadata[collection_start]', pendingMeta?.start);
      }
    }
    if (metadata?.end !== pendingMeta?.end) {
      if (pendingMeta.end === null || pendingMeta.end === '') {
        updateForm.set('metadata[clear_collection_end]', 'true');
      } else {
        updateForm.set('metadata[collection_end]', pendingMeta?.end);
      }
    }
    // collection_tags add/delete
    const { toAdd, toDelete } = diffTagUpdate(metadata?.collection_tags, pendingMeta?.collection_tags);
    Object.entries(toAdd).forEach(([k, vals]) => {
      vals.forEach((v) => {
        updateForm.append(`metadata[add_collection_tags][${k}][]`, v);
      });
    });
    Object.entries(toDelete).forEach(([k, vals]) => {
      vals.forEach((v) => {
        updateForm.append(`metadata[delete_collection_tags][${k}][]`, v);
      });
    });
  }
  // FileSystem specific metadata
  if ([Entities.FileSystem, Entities.WindowsProcessTree].includes(entity.kind)) {
    // metadata.tools
    pendingMeta?.tools
      ?.filter((pendingTool: string) => !metadata.tools.includes(pendingTool))
      .forEach((addedTool: string) => updateForm.append('metadata[add_tools][]', addedTool));
    metadata?.tools
      ?.filter((initialTool: string) => !pendingMeta.tools.includes(initialTool))
      .forEach((removedTool: string) => updateForm.append('metadata[remove_tools][]', removedTool));
  }

  if (entity.kind == Entities.SigmaRule) {
    // SigmaRule-specific metadata
    if (pendingMeta && pendingMeta?.rule !== '') {
      updateForm.append('metadata[sigma_rule]', String(pendingMeta?.rule));
    }
    if (pendingMeta?.score) {
      updateForm.append(`metadata[score]`, pendingMeta.score);
    }
    if (pendingMeta?.actions && Array.isArray(pendingMeta.actions)) {
      pendingMeta?.actions
        ?.filter(
          (pendingAction: SigmaActionToTake) =>
            !metadata.actions.map((initialAction: string) => JSON.stringify(initialAction)).includes(JSON.stringify(pendingAction)),
        )
        .forEach((addedAction: string) => updateForm.append('metadata[add_sigma_actions][]', JSON.stringify(addedAction)));
      let numberRemovedActions = 0;
      metadata?.actions?.forEach((initialAction: SigmaActionToTake, index: number) => {
        if (JSON.stringify(initialAction) != JSON.stringify(pendingMeta.actions[index - numberRemovedActions])) {
          updateForm.append('metadata[remove_sigma_actions][]', String(index));
          numberRemovedActions++;
        }
      });
    }
    if (metadata?.applies_to && Array.isArray(metadata.applies_to)) {
      pendingMeta?.applies_to
        ?.filter((pendingEntity: Entities) => !metadata.applies_to.includes(pendingEntity))
        .forEach((addedEntity: Entities) => updateForm.append('metadata[add_sigma_applies_to][]', addedEntity));
      metadata?.applies_to
        ?.filter((initialEntity: Entities) => !pendingMeta.applies_to.includes(JSON.stringify(initialEntity)))
        .forEach((removedEntity: Entities) => updateForm.append('metadata[remove_sigma_applies_to][]', removedEntity));
    }
  }
  return updateForm;
}

export function buildCreateEntityForm<T extends keyof EntityCreateTypeMap>(entity: EntityCreateTypeMap[T]): FormData {
  const createForm = new FormData();
  const kind = entity.kind;
  createForm.set('name', entity.name);
  createForm.set('kind', kind);
  // add groups to form
  entity.groups.map((group: string) => {
    createForm.append('groups[]', group);
  });
  if (entity.description && entity.description !== '') {
    createForm.set('description', entity.description);
  }
  // add create tags to form
  Object.keys(entity.tags).map((key: string) => {
    entity.tags[key].map((value: string) => {
      createForm.append(`tags[${key}][]`, value);
    });
  });
  // metadata
  const metadata = entity.metadata[kind];
  // metadata: urls
  if (metadata && 'urls' in metadata && metadata.urls?.length > 0) {
    metadata.urls.map((url: string) => {
      createForm.append('metadata[urls][]', url);
    });
  }
  // metadata: sensitive_location
  if (metadata && 'sensitive_location' in metadata && typeof metadata.sensitive_location == 'boolean') {
    createForm.set('metadata[sensitive_location]', String(metadata.sensitive_location));
  }
  // metadata: critical_system
  if (metadata && 'critical_system' in metadata && typeof metadata.critical_system == 'boolean') {
    createForm.set('metadata[critical_system]', String(metadata.critical_system));
  }
  // metadata: critical_sectors
  if (
    metadata &&
    'critical_sectors' in metadata &&
    metadata.critical_sectors?.length > 0 &&
    (kind == Entities.Vendor ||
      ('critical_system' in metadata && typeof metadata.critical_system == 'boolean' && metadata.critical_system === true))
  ) {
    metadata.critical_sectors.map((sector: string) => {
      createForm.append('metadata[critical_sectors][]', reformatCriticalSectors(sector));
    });
  }
  // metadata: vendor
  if (metadata && 'vendors' in metadata && metadata.vendors.length > 0) {
    metadata.vendors.map((vendorUUID: string) => {
      createForm.append('metadata[vendors][]', vendorUUID);
    });
  }
  // metadata: countries
  if (metadata && 'countries' in metadata && metadata.countries.length > 0) {
    metadata.countries.map((name: string) => {
      const code = getCountryCode(name);
      if (code !== undefined) {
        createForm.append('metadata[countries][]', code);
      }
    });
  }
  // Collection-specific metadata
  if (entity.kind == Entities.Collection) {
    if (metadata?.collection_kind && metadata?.collection_kind !== '') {
      createForm.append('metadata[collection_kind]', metadata?.collection_kind);
    }
    if (metadata?.tags_case_insensitive) {
      createForm.append('metadata[collection_tags_case_insensitive]', 'true');
    }
    if (metadata?.ignore_groups) {
      createForm.append('metadata[collection_ignore_groups]', 'true');
    }

    if (metadata?.start && metadata?.start !== '') {
      createForm.append('metadata[collection_start]', metadata?.start);
    }
    if (metadata?.end && metadata?.end !== '') {
      createForm.append('metadata[collection_end]', metadata?.end);
    }
    if (metadata?.collection_tags && typeof metadata?.collection_tags === 'object') {
      Object.entries(metadata.collection_tags).forEach(([tagKey, tagVals]) => {
        if (Array.isArray(tagVals)) {
          tagVals.forEach((val) => {
            // e.g. metadata[collection_tags][key][]=value
            createForm.append(`metadata[collection_tags][${tagKey}][]`, val);
          });
        }
      });
    }
  }
  if (entity.kind == Entities.SigmaRule) {
    // SigmaRule-specific metadata
    if (metadata && metadata?.rule !== '') {
      createForm.append('metadata[sigma_rule]', String(metadata?.rule));
    }
    if (metadata?.score) {
      createForm.append(`metadata[score]`, metadata.score);
    }
    if (metadata?.actions && Array.isArray(metadata.actions)) {
      metadata.actions.map((action: SigmaActionToTake) => {
        if (action.Flag) {
          createForm.append(`metadata[sigma_actions][]`, JSON.stringify(action));
        }
      });
    }
    if (metadata?.applies_to && Array.isArray(metadata.applies_to)) {
      console.log(`sigma_applies_to: ${metadata.applies_to}`);
      console.log(metadata.applies_to);
      metadata.applies_to.map((entity: Entities) => {
        createForm.append(`metadata[sigma_applies_to][]`, entity);
      });
    }
  }
  return createForm;
}

export function copyEntityFields<T extends keyof EntityUISupportedCreateTypeMap>(
  existingEntity: EntityTypeMap[T],
  blank: EntityCreateTypeMap[T],
): EntityCreateTypeMap[T] {
  const newEntity = blank;
  newEntity.name = `${existingEntity.name} - copy`;
  newEntity.description = existingEntity.description;
  newEntity.groups = [...existingEntity.groups];
  newEntity.kind = existingEntity.kind;
  // need to handle countries/vendors
  const newMeta: any = structuredClone(existingEntity.metadata[existingEntity.kind]);
  if (newMeta && newMeta.countries && newMeta.countries.length > 0) {
    newMeta.countries = newMeta.countries.map((country: Country) => country.name);
  }
  if (newMeta && newMeta.vendors && newMeta.vendors.length > 0) {
    newMeta.vendors = newMeta.vendors.map((vendor: Vendor) => vendor.id);
  }
  newEntity.metadata[existingEntity.kind] = newMeta;
  //tags
  const newTags: RequestTags = {};
  Object.keys(existingEntity.tags).map((key: string) => {
    if (!(key in newTags)) {
      newTags[key] = [];
    }
    Object.keys(existingEntity.tags[key]).map((value: string) => {
      newTags[key].push(value);
    });
  });
  newEntity.tags = newTags;
  return newEntity;
}
