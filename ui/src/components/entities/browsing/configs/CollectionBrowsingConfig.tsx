import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { EntityBrowseConfig } from './config';
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntityNameWithIcon,
  EntitySecondary,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { listEntities } from '@thorpi/entities';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
import { Collection } from '@models/entities/collections';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

const TagRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
`;

interface CollectionItemProps {
  collection: Collection;
}

const CollectionItem: React.FC<CollectionItemProps> = ({ collection }) => {
  const kind = collection.metadata.Collection.collection_kind ?? '';
  const collectionTags = collection.metadata.Collection.collection_tags ?? {};
  const hasCollectionTags = Object.keys(collectionTags).length > 0;
  const hasEntityTags = collection.tags && Object.keys(collection.tags).length > 0;
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link
          to={`${getDetailsBasePathByEntity(Entities.Collection)}/${collection.id}`}
          state={{ collection: collection }}
          className="no-decoration"
        >
          <LinkFields>
            <EntityName>
              <EntityNameWithIcon entityId={collection.id} hasImage={collection.image != null}>
                {collection.name}
              </EntityNameWithIcon>
            </EntityName>
            <EntitySecondary>{kind}</EntitySecondary>
            <EntityGroups>
              <small>
                <i>
                  {collection.groups &&
                    (collection.groups.toString().length > 75
                      ? collection.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : collection.groups.toString().replaceAll(',', ', '))}
                </i>
              </small>
            </EntityGroups>
          </LinkFields>
        </Link>
        {(hasCollectionTags || hasEntityTags) && (
          <>
            <hr />
            <TagRow>
              {hasCollectionTags &&
                Object.keys(collectionTags)
                  .sort()
                  .map((key) =>
                    (collectionTags[key] ?? [])
                      .slice()
                      .sort()
                      .map((value) => <FieldBadge key={`${key}_${value}`} color="Gray" field={`${key}: ${value}`} />),
                  )}
              {hasEntityTags && <CondensedEntityTags resource={Entities.Collection} tags={collection.tags} />}
            </TagRow>
          </>
        )}
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
        <EntityGroups>Group(s)</EntityGroups>
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
