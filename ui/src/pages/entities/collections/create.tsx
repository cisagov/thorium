import { JSX, useEffect, useState } from 'react';
import { Row, Form } from 'react-bootstrap';

import { EntityCreate, InfoHeader, InfoValue, SelectInput, FilterDatePicker, SelectableDictionary } from '@components';
import {
  BlankCreateCollection,
  CreateCollection,
  CollectionKind,
  CollectionMetaFields,
  Entities,
  TagEntry,
  CreateCollectionPreprocessor,
  CreateEntityPreprocessor,
} from '@models';
import { safeDateToStringConversion } from '@utilities';

const CollectionMetaInfo = (
  collection: CreateCollection,
  onChange: <K extends keyof CreateCollection>(field: K, value: CreateCollection[K]) => void,
  preprocessor?: CreateEntityPreprocessor<CreateCollection>,
): JSX.Element => {
  // helper to update nested metadata
  const updatePendingMeta = <T extends keyof CollectionMetaFields>(field: T, value: CollectionMetaFields[T]) => {
    const updates: CollectionMetaFields = structuredClone(collection.metadata.Collection);
    updates[field] = value;
    onChange('metadata', { Collection: updates });
  };
  // make sure we got a collection preprocessor; initialize one if we didn't
  const collectionPreprocessor = ((preprocessor as CreateCollectionPreprocessor | undefined) ??= new CreateCollectionPreprocessor());

  const [tags, setTags] = useState<TagEntry[]>([{ key: '', value: '' }]);

  useEffect(() => {
    collectionPreprocessor.editbleTags = tags;
  }, [tags, collectionPreprocessor]);

  // current date is the latest you can set for start/end
  const maxDate = new Date();

  return (
    <>
      <Row>
        <InfoHeader>
          Kind<sub>*</sub>
        </InfoHeader>
        <InfoValue>
          <SelectInput
            options={Object.values(CollectionKind)}
            value={collection.metadata.Collection.collection_kind ?? CollectionKind.Files}
            onChange={(k) => updatePendingMeta('collection_kind', k as CollectionKind)}
            disabled={false}
          />
        </InfoValue>
      </Row>
      <Row>
        <InfoHeader>Collection Tags</InfoHeader>
        <InfoValue>
          <SelectableDictionary
            entries={tags}
            setEntries={setTags}
            keyPlaceholder={'Add Tag Key'}
            valuePlaceholder={'Add Tag Value'}
            disabled={false}
            setDeleted={void 0}
            keys={null}
            deleted={null}
            trim={true}
          />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Case‑Insensitive</InfoHeader>
        <InfoValue>
          <Form.Check
            type="switch"
            id="case-insensitive-toggle"
            checked={collection.metadata.Collection.tags_case_insensitive ?? false}
            onChange={(e) => updatePendingMeta('tags_case_insensitive', e.target.checked)}
          />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Ignore Groups</InfoHeader>
        <InfoValue>
          <Form.Check
            type="switch"
            id="ignore-groups-toggle"
            checked={collection.metadata.Collection.ignore_groups ?? false}
            onChange={(e) => updatePendingMeta('ignore_groups', e.target.checked)}
          />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Start</InfoHeader>
        <InfoValue>
          <FilterDatePicker
            max={maxDate}
            min={collection.metadata.Collection.end}
            selected={collection.metadata.Collection.start}
            disabled={false}
            onChange={(e) => updatePendingMeta('start', safeDateToStringConversion(e))}
          />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>End</InfoHeader>
        <InfoValue>
          <FilterDatePicker
            max={collection.metadata.Collection.start ? collection.metadata.Collection.start : maxDate}
            selected={collection.metadata.Collection.end}
            disabled={false}
            onChange={(e) => updatePendingMeta('end', safeDateToStringConversion(e))}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const CreateCollectionContainer = () => {
  return (
    <EntityCreate
      kind={Entities.Collection}
      metadata={CollectionMetaInfo}
      blank={BlankCreateCollection}
      preprocessor={new CreateCollectionPreprocessor()}
    />
  );
};

export default CreateCollectionContainer;
