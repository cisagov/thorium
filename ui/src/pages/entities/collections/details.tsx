import { JSX } from 'react';
import { Row, Form } from 'react-bootstrap';
import { FaFolder } from 'react-icons/fa6';

// project imports
import {
  EntityDetails,
  FieldBadge,
  InfoValue,
  FilterDatePicker,
  EntityDetailsLabel,
  TagSelect,
  buildCollectionsBrowsingUrl,
} from '@components';
import { BlankCollection, Collection, CollectionMeta, CollectionMetaFields, Entities } from '@models';
import { requestTagsToTagEntryList, safeDateToStringConversion, tagEntriesToRequestTags } from '@utilities';
import { getEntity } from '@thorpi';

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

const CollectionMetaInfo = (
  collection: Collection,
  pendingCollection: Collection,
  handleUpdate: <K extends keyof Collection>(field: K, value: Collection[K]) => void,
  editing: boolean,
): JSX.Element => {
  // current date is the latest you can set for start/end
  const maxDate = new Date();
  // handle any updates to Collection entity metadata
  function updatePendingMeta<T extends keyof CollectionMetaFields>(field: T, value: CollectionMetaFields[T]) {
    const updates: CollectionMetaFields = structuredClone(pendingCollection.metadata.Collection);
    updates[field] = value;
    handleUpdate('metadata', { Collection: updates });
  }
  return (
    <>
      {!editing && (
        <Row>
          <EntityDetailsLabel label="Kind" tip={CollectionTips.kind} />
          <InfoValue>{collection.metadata.Collection.collection_kind}</InfoValue>
        </Row>
      )}
      <Row className="mt-3">
        <EntityDetailsLabel label="Collection Tags" tip={CollectionTips.collectionTags} />
        <InfoValue>
          {editing ? (
            <TagSelect
              tags={requestTagsToTagEntryList((pendingCollection.metadata as CollectionMeta).Collection.collection_tags ?? {})}
              setTags={(updatedTags) => updatePendingMeta('collection_tags', tagEntriesToRequestTags(updatedTags))}
              placeholderText="Add Tags"
            />
          ) : (
            <>
              {Object.keys((collection.metadata as CollectionMeta).Collection.collection_tags ?? {})
                .sort()
                .map((key) =>
                  (((collection.metadata as CollectionMeta).Collection.collection_tags ?? {})[key] ?? [])
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
              checked={pendingCollection.metadata.Collection.tags_case_insensitive ?? false}
              onChange={(e) => updatePendingMeta('tags_case_insensitive', e.target.checked)}
            />
          ) : collection.metadata.Collection.tags_case_insensitive ? (
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
              checked={pendingCollection.metadata.Collection.ignore_groups ?? false}
              onChange={(e) => updatePendingMeta('ignore_groups', e.target.checked)}
            />
          ) : collection.metadata.Collection.ignore_groups ? (
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
              min={pendingCollection.metadata.Collection.end}
              selected={pendingCollection.metadata.Collection.start}
              disabled={false}
              onChange={(e) => updatePendingMeta('start', safeDateToStringConversion(e))}
            />
          ) : collection.metadata.Collection.start ? (
            new Date(collection.metadata.Collection.start).toLocaleString()
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
              max={pendingCollection.metadata.Collection.start ? pendingCollection.metadata.Collection.start : maxDate}
              selected={pendingCollection.metadata.Collection.end}
              disabled={false}
              onChange={(e) => updatePendingMeta('end', safeDateToStringConversion(e))}
            />
          ) : collection.metadata.Collection.end ? (
            new Date(collection.metadata.Collection.end).toLocaleString()
          ) : (
            ''
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <EntityDetailsLabel label={`${collection.metadata.Collection.collection_kind}`} tip={CollectionTips.link} />
        <InfoValue>
          <a href={buildCollectionsBrowsingUrl(collection)}>View</a>
        </InfoValue>
      </Row>
    </>
  );
};

const CollectionDetailsContainer = () => {
  const getCollectionDetails = async (collectionId: string, setError: (e: string) => void, updateEntity: (c: Collection) => void) => {
    getEntity(collectionId, setError).then((data) => {
      if (data && data.kind === Entities.Collection) {
        updateEntity(data as Collection);
      }
    });
  };

  return (
    <EntityDetails
      getEntityDetails={getCollectionDetails}
      metadata={CollectionMetaInfo}
      blank={BlankCollection}
      icon={(size: number) => <FaFolder size={size} />}
    />
  );
};

export default CollectionDetailsContainer;
