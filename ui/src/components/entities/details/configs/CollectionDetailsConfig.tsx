import { JSX } from 'react';
import { Row, Form } from 'react-bootstrap';
import { FaFolder } from 'react-icons/fa6';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import InfoValue from '@entities/shared/InfoValue';
import EntityDetailsLabel from '@entities/details/EntityDetailsLabel';
import FilterDatePicker from '@entities/browsing/filters/FilterDatePicker';
import { buildCollectionsBrowsingUrl } from '@entities/details/ListCollectionsButton';
import { TagSelect } from '@components/shared/inputs/tags/TagSelect';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { safeDateToStringConversion } from '@utilities/inputs';
import { requestTagsToTagEntryList, tagEntriesToRequestTags } from '@utilities/tags';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities';
import { BlankCollection, Collection, CollectionMeta, CollectionMetaFields } from '@models/entities/collections';

const CollectionTips = {
  kind: `The type of items this collection contains`,
  collectionTags: `The tags on items in this collection; all items in the collection
    must have at least one of the given tags`,
  tagsCaseInsensitive: `If true, tags on items can match tags regardless of case`,
  ignoreGroups: `If true, items from all of the user's groups will be included in the collection.
  Otherwise, collection items are restricted to the groups of the collection itself.`,
  start: `The most recent time an item must have been uploaded for it to be included`,
  end: `The oldest time an item must have been uploaded for it to be included`,
  link: `Navigate to view the files in this collection`,
};

const CollectionMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.Collection>): JSX.Element => {
  // current date is the latest you can set for start/end
  const maxDate = new Date();
  // handle any updates to Collection entity metadata
  function updatePendingMeta<T extends keyof CollectionMetaFields>(field: T, value: CollectionMetaFields[T]) {
    const updates: CollectionMetaFields = structuredClone(pendingEntity.metadata.Collection);
    updates[field] = value;
    handleUpdate('metadata', { Collection: updates });
  }
  return (
    <>
      {!editing && (
        <Row>
          <EntityDetailsLabel label="Kind" tip={CollectionTips.kind} />
          <InfoValue>{entity.metadata.Collection.collection_kind}</InfoValue>
        </Row>
      )}
      <Row className="mt-3">
        <EntityDetailsLabel label="Collection Tags" tip={CollectionTips.collectionTags} />
        <InfoValue>
          {editing ? (
            <TagSelect
              tags={requestTagsToTagEntryList((pendingEntity.metadata as CollectionMeta).Collection.collection_tags ?? {})}
              setTags={(updatedTags) => updatePendingMeta('collection_tags', tagEntriesToRequestTags(updatedTags))}
              placeholderText="Add Tags"
            />
          ) : (
            <>
              {Object.keys((entity.metadata as CollectionMeta).Collection.collection_tags ?? {})
                .sort()
                .map((key) =>
                  (((entity.metadata as CollectionMeta).Collection.collection_tags ?? {})[key] ?? [])
                    .slice()
                    .sort()
                    .map((value) => <FieldBadge key={`${key}_${value}`} color="Gray" field={`${key}: ${value}`} />),
                )}
            </>
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <EntityDetailsLabel label="Case‑Insensitive" tip={CollectionTips.tagsCaseInsensitive} />
        <InfoValue>
          {editing ? (
            <Form.Check
              type="switch"
              id="case-insensitive-toggle"
              checked={pendingEntity.metadata.Collection.tags_case_insensitive ?? false}
              onChange={(e) => updatePendingMeta('tags_case_insensitive', e.target.checked)}
            />
          ) : entity.metadata.Collection.tags_case_insensitive ? (
            'Yes'
          ) : (
            'No'
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <EntityDetailsLabel label="Ignore Groups" tip={CollectionTips.ignoreGroups} />
        <InfoValue>
          {editing ? (
            <Form.Check
              type="switch"
              id="case-insensitive-toggle"
              checked={pendingEntity.metadata.Collection.ignore_groups ?? false}
              onChange={(e) => updatePendingMeta('ignore_groups', e.target.checked)}
            />
          ) : entity.metadata.Collection.ignore_groups ? (
            'Yes'
          ) : (
            'No'
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <EntityDetailsLabel label="Newest" tip={CollectionTips.start} />
        <InfoValue>
          {editing ? (
            <FilterDatePicker
              max={maxDate}
              min={pendingEntity.metadata.Collection.end}
              selected={pendingEntity.metadata.Collection.start}
              disabled={false}
              onChange={(e) => updatePendingMeta('start', safeDateToStringConversion(e))}
            />
          ) : entity.metadata.Collection.start ? (
            new Date(entity.metadata.Collection.start).toLocaleString()
          ) : (
            ''
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <EntityDetailsLabel label="Oldest" tip={CollectionTips.end} />
        <InfoValue>
          {editing ? (
            <FilterDatePicker
              max={pendingEntity.metadata.Collection.start ? pendingEntity.metadata.Collection.start : maxDate}
              selected={pendingEntity.metadata.Collection.end}
              disabled={false}
              onChange={(e) => updatePendingMeta('end', safeDateToStringConversion(e))}
            />
          ) : entity.metadata.Collection.end ? (
            new Date(entity.metadata.Collection.end).toLocaleString()
          ) : (
            ''
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <EntityDetailsLabel label={`${entity.metadata.Collection.collection_kind}`} tip={CollectionTips.link} />
        <InfoValue>
          <a href={buildCollectionsBrowsingUrl(entity)}>View</a>
        </InfoValue>
      </Row>
    </>
  );
};

const getCollectionDetails = async (collectionId: string, setError: (e: string) => void, updateEntity: (c: Collection) => void) => {
  getEntity(collectionId, setError).then((data) => {
    if (data && data.kind === Entities.Collection) {
      updateEntity(data as Collection);
    }
  });
};

const CollectionDetailsConfig: EntityDetailsConfig<Entities.Collection> = {
  getEntityDetails: getCollectionDetails,
  EntityMetaInfo: CollectionMetaInfo,
  BlankEntity: BlankCollection,
  icon: (size: number) => <FaFolder size={size} />,
};

export default CollectionDetailsConfig;
