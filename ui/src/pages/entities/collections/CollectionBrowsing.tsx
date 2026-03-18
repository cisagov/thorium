import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';

// project imports
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntityOrigin,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import BrowsingFilters from '@entities/browsing/filters/BrowsingFilters';
import EntityList from '@entities/browsing/EntityList';
import Page from '@components/pages/Page';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { useAuth } from '@utilities/auth';
import { Collection } from '@models/entities/collections';
import { Entities } from '@models/entities/entities';
import { Filters } from '@models/search';
import { listEntities } from '@thorpi/entities';

const getCollections = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Collection];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return { entitiesList: entityList, entitiesCursor: entityCursor };
};

const CollectionListHeaders = () => (
  <BrowsingCard>
    <BrowsingContents>
      <Row>
        <EntityName>Name</EntityName>
        <EntityOrigin>Kind</EntityOrigin>
        <EntityGroups>Group(s)</EntityGroups>
        <EntitySubmitters>Submitter(s)</EntitySubmitters>
      </Row>
    </BrowsingContents>
  </BrowsingCard>
);

interface CollectionItemProps {
  collection: Collection;
}

const CollectionItem: React.FC<CollectionItemProps> = ({ collection }) => {
  const kind = collection.metadata.Collection.collection_kind ?? '';
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`/collection/${collection.id}`} state={{ collection: collection }} className="no-decoration">
          <LinkFields>
            <EntityName>{collection.name}</EntityName>
            <EntityOrigin>{kind}</EntityOrigin>
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
            <EntitySubmitters>
              {collection.submitter ? (
                <small>
                  <i>{collection.submitter.length > 75 ? collection.submitter.substring(0, 75) + '...' : collection.submitter}</i>
                </small>
              ) : null}
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

const CollectionBrowsing = () => {
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<Filters>({});
  const { userInfo } = useAuth();
  return (
    <Page title="Collections · Thorium">
      <BrowsingFilters
        title="Collections"
        kind={Entities.Collection}
        onChange={setFilters}
        groups={userInfo?.groups ?? []}
        disabled={loading}
        creatable={true}
      />
      <EntityList
        type="Collections"
        entityHeaders={<CollectionListHeaders />}
        displayEntity={(c) => <CollectionItem collection={c} />}
        filters={filters}
        fetchEntities={getCollections}
        setLoading={setLoading}
        loading={loading}
      />
    </Page>
  );
};

export default CollectionBrowsing;
