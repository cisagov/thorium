// EntityBrowsing.tsx
import { useState } from 'react';
import Page from '@components/pages/Page';
import EntityList from './EntityList';
import BrowsingFilters from '@entities/browsing/filters/BrowsingFilters';
import { Filters } from '@models/search';
import { ExtendedTypeMap } from '@models/entities/entities';
import { EntityBrowseConfig } from './configs/config';

// spec: ./EntityBrowsing.spec.md

export function createEntityBrowsingPage<T extends keyof ExtendedTypeMap>(config: EntityBrowseConfig<T>) {
  const BoundEntityBrowsingPage = () => {
    const [loading, setLoading] = useState(false);
    const [filters, setFilters] = useState<Filters>({});

    return (
      <Page title={config.docTitle}>
        <BrowsingFilters
          title={config.title}
          kind={config.kind}
          onChange={setFilters}
          disabled={loading}
          creatable={config.creatable ?? true}
        />

        <EntityList<ExtendedTypeMap[T]>
          type={config.typeLabel}
          entityHeaders={config.entityHeaders}
          displayEntity={config.renderEntity}
          filters={filters}
          fetchEntities={config.fetchEntities}
          setLoading={setLoading}
          loading={loading}
        />
      </Page>
    );
  };

  return BoundEntityBrowsingPage;
}
