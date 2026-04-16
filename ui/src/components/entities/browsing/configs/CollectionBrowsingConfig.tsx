import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';

// project imports
import { EntityBrowseConfig } from './config';
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntitySecondary,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { listEntities } from '@thorpi/entities';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
import { Collection, CollectionMeta } from '@models/entities/collections';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

interface CollectionItemProps {
  collection: Collection;
}

const CollectionItem: React.FC<CollectionItemProps> = ({ collection }) => {
  const kind = collection.metadata.Collection.collection_kind ?? '';
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link
          to={`${getDetailsBasePathByEntity(Entities.Collection)}/${collection.id}`}
          state={{ collection: collection }}
          className="no-decoration"
        >
          <LinkFields>
            <EntityName>{collection.name}</EntityName>
            <EntitySecondary>{kind}</EntitySecondary>
            <EntityGroups>
              {Object.keys((collection.metadata as CollectionMeta).Collection.collection_tags ?? {})
                .sort()
                .map((key) =>
                  (((collection.metadata as CollectionMeta).Collection.collection_tags ?? {})[key] ?? [])
                    .slice()
                    .sort()
                    .map((value) => <FieldBadge key={`${key}_${value}`} color="Gray" field={`${key}: ${value}`} />),
                )}
            </EntityGroups>
            <EntitySubmitters>
              <small>
                <i>
                  {collection.groups &&
                    (collection.groups.toString().length > 75
                      ? collection.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : collection.groups.toString().replaceAll(',', ', '))}
                </i>
              </small>
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {collection.tags != undefined && <hr />}
        <Row>
          {collection.tags && Object.keys(collection.tags).length > 1 ? (
            <CondensedEntityTags resource={Entities.Collection} tags={collection.tags} />
          ) : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const CollectionListHeaders = () => (
  <BrowsingCard>
    <BrowsingContents>
      <Row>
        <EntityName>Name</EntityName>
        <EntitySecondary>Kind</EntitySecondary>
        <EntityGroups>Tag(s)</EntityGroups>
        <EntitySubmitters>Groups(s)</EntitySubmitters>
      </Row>
    </BrowsingContents>
  </BrowsingCard>
);

const getCollections = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Collection];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return { entitiesList: entityList as Collection[], entitiesCursor: entityCursor };
};

const CollectionsBrowsingConfig: EntityBrowseConfig<Entities.Collection> = {
  docTitle: 'Collections · Thorium',
  title: 'Collections',
  typeLabel: '',
  kind: Entities.Collection,
  creatable: true,
  entityHeaders: <CollectionListHeaders />,
  renderEntity: (entity) => <CollectionItem collection={entity} />,
  fetchEntities: getCollections,
};

export default CollectionsBrowsingConfig;
