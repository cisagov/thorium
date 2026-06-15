import type React from 'react';

// project imports
import { listEntities } from '@thorpi/entities';
import { ExtendedTypeMap } from '@models/entities/entities';
import { Filters } from '@models/search';

// spec: ../EntityBrowsing.spec.md

// This module deliberately imports NO per-kind `*BrowsingConfig` file. `config.ts` imports every config, and
// each config needs `makeEntityFetcher`; if that factory lived in `config.ts`, the configs would import a value
// back from a module still mid-evaluation, hitting the temporal dead zone ("Cannot access 'makeEntityFetcher'
// before initialization"). Keeping the type + factory here breaks that cycle.

/**
 * Generic browse config for a specific entity type T
 */
export type EntityBrowseConfig<T extends keyof ExtendedTypeMap> = {
  docTitle: string;
  title: string;
  typeLabel: string;
  kind: T;
  creatable?: boolean;
  entityHeaders: React.ReactNode;
  renderEntity: (entity: ExtendedTypeMap[T], idx: number, filters?: Filters) => React.ReactNode;
  fetchEntities: (
    filters: Filters,
    cursor: string | null,
    errorHandler: (error: string) => void,
  ) => Promise<{ entitiesList: ExtendedTypeMap[T][]; entitiesCursor: string | null }>;
};

/**
 * Build a browse-config `fetchEntities` function that lists a single entity kind.
 *
 * Every per-kind browse config fetches the same way: clone the incoming filters, pin
 * `kinds` to the one kind, and page through `listEntities`. This factory captures that
 * shared shape so each config only needs to name its kind.
 *
 * @template T - The entity kind key into `ExtendedTypeMap`.
 * @param kind - The entity kind to restrict the listing to.
 * @returns A `fetchEntities` function matching `EntityBrowseConfig<T>`.
 */
export const makeEntityFetcher =
  <T extends keyof ExtendedTypeMap>(kind: T): EntityBrowseConfig<T>['fetchEntities'] =>
  async (filters, cursor, errorHandler) => {
    const listFilters = structuredClone(filters);
    listFilters.kinds = [kind];
    const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
    return { entitiesList: entityList as ExtendedTypeMap[T][], entitiesCursor: entityCursor };
  };
