import { getCode as getCountryCode, Country } from 'country-list';

// project imports
import { RequestTags, CreateEntity, CriticalSector, Entities, Entity, Vendor } from '@models';
import { diffTagUpdate } from '@utilities';

const reformatCriticalSectors = (sector: string): string => {
  return sector.replaceAll(' and ', '').replaceAll(',', '').replaceAll(' ', '');
};

export function buildUpdateEntityForm(entity: Entity, pendingEntity: Entity): FormData {
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
  console.log(metadata.vendors);
  const addVendors: string[] | undefined = pendingMeta?.vendors?.filter(
    (pendingVendor: string) => !metadata.vendors.map((vendor: Vendor) => vendor.id).includes(pendingVendor),
  );
  const removeVendors: string[] | undefined = metadata?.vendors?.filter(
    (vendor: Vendor) => !pendingMeta.vendors.map((pendingVendor: string) => pendingVendor).includes(vendor.id),
  );
  addVendors?.map((vendor: string) => {
    updateForm.append('metadata[add_vendors][]', vendor);
  });
  removeVendors?.map((vendor: string) => {
    updateForm.append('metadata[remove_vendors][]', vendor);
  });
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
  const addCountries: Country[] | undefined = pendingMeta?.countries?.filter(
    (country: Country) => !metadata.countries.map((country: Country) => country.code).includes(country.code),
  );
  const removeCountries: Country[] | undefined = metadata?.countries?.filter(
    (country: Country) => !pendingMeta.countries.map((country: Country) => country.code).includes(country.code),
  );
  addCountries?.map((country: Country) => {
    if (country.code !== undefined) {
      updateForm.append('metadata[add_countries][]', country.code);
    }
  });
  removeCountries?.map((country: any) => {
    if (country.code !== undefined) {
      updateForm.append('metadata[remove_countries][]', country.code);
    }
  });
  // Collection-specific metadata
  if (entity.kind == 'Collection') {
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
  return updateForm;
}

export function buildCreateEntityForm(entity: CreateEntity, kind: Entities): FormData {
  const createForm = new FormData();
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
    createForm.set('metadata[sensitive_location]', metadata.sensitive_location);
  }
  // metadata: critical_system
  if (metadata && 'critical_system' in metadata && typeof metadata.critical_system == 'boolean') {
    createForm.set('metadata[critical_system]', metadata.critical_system);
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
  if (metadata && 'vendor' in metadata && metadata.vendor.length > 0) {
    metadata.vendor.map((vendor: string) => {
      createForm.append('metadata[vendor][]', vendor);
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
  if (entity.kind == 'Collection') {
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
  return createForm;
}

export function copyEntityFields<T extends CreateEntity, K extends Entity>(existingEntity: K, blank: T): T {
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
