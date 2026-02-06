import { JSX } from 'react';
import { Row, Form } from 'react-bootstrap';
import { FaFolder } from 'react-icons/fa6';

import { EntityDetails, FieldBadge, InfoValue, FilterDatePicker, EditableCollectionTags, EntityDetailsLabel } from '@components';
import { BlankCollection, Collection, CollectionMeta, CollectionMetaFields, Entities } from '@models';
import { safeDateToStringConversion } from '@utilities';
import { getEntity } from '@thorpi';

const CollectionTips = {
  kind: `The type of items this collection contains`,
  collectionTags: `The tags on items in this collection; all items in the collection
    must have at least one of the given tags`,
  tagsCaseInsensitive: `If set, tags on items can match the above tags regardless of case`,
  ignoreGroups: `If set, items from all of the user's groups may be included in the collection;
    otherwise, items are restricted to the groups the collection itself is in`,
  start: `The most recent time an item must have been uploaded for it to be included`,
  end: `The oldest time an item must have been uploaded for it to be included`,
};

const CollectionMetaInfo = (
  collection: Collection,
  pendingCollection: Collection,
  handleUpdate: <K extends keyof Collection>(field: K, value: Collection[K]) => void,
  editing: boolean,
): JSX.Element => {
  function updatePendingMeta<T extends keyof CollectionMetaFields>(field: T, value: CollectionMetaFields[T]) {
    const updates: CollectionMetaFields = structuredClone(pendingCollection.metadata.Collection);
    updates[field] = value;
    handleUpdate('metadata', { Collection: updates });
  }
  // current date is the latest you can set for start/end
  const maxDate = new Date();

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
            <EditableCollectionTags
              value={(pendingCollection.metadata as CollectionMeta).Collection.collection_tags ?? {}}
              onChange={(next) => updatePendingMeta('collection_tags', next)}
            />
          ) : (
            <>
              {Object.keys((collection.metadata as CollectionMeta).Collection.collection_tags ?? {})
                .sort()
                .map((key) =>
                  (((collection.metadata as CollectionMeta).Collection.collection_tags ?? {})[key] ?? [])
                    .slice()
                    .sort()
                    .map((value) => <FieldBadge key={key} color="Gray" field={`${key}: ${value}`} />),
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
        <EntityDetailsLabel label="Start" tip={CollectionTips.start} />
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
        <EntityDetailsLabel label="End" tip={CollectionTips.end} />
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
