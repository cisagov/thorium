import { JSX } from 'react';
import { Row, Form } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import FilterDatePicker from '@entities/browsing/filters/FilterDatePicker';
import { TagSelect } from '@components/shared/inputs/tags/TagSelect';
import SelectInput from '@components/shared/inputs/selectable/SelectInput';
import { safeDateToStringConversion } from '@utilities/inputs';
import { requestTagsToTagEntryList, tagEntriesToRequestTags } from '@utilities/tags';
import { Entities } from '@models/entities/entities';
import { BlankCreateCollection, CollectionKind, CollectionMetaFields, CollectionMeta } from '@models/entities/collections';

const CollectionMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.Collection>): JSX.Element => {
  // current date is the latest you can set for start/end
  const maxDate = new Date();
  // helper to update nested metadata
  const updatePendingMeta = <T extends keyof CollectionMetaFields>(field: T, value: CollectionMetaFields[T]) => {
    const updates: CollectionMetaFields = structuredClone(entity.metadata.Collection);
    updates[field] = value;
    onChange('metadata', { Collection: updates });
  };
  return (
    <>
      <Row>
        <InfoHeader>
          Kind<sub>*</sub>
        </InfoHeader>
        <InfoValue>
          <SelectInput
            options={Object.values(CollectionKind)}
            value={entity.metadata.Collection.collection_kind ?? CollectionKind.Files}
            onChange={(k) => updatePendingMeta('collection_kind', k as CollectionKind)}
            disabled={false}
          />
        </InfoValue>
      </Row>
      <Row>
        <InfoHeader>Collection Tags</InfoHeader>
        <InfoValue className="mt-2">
          <TagSelect
            tags={requestTagsToTagEntryList((entity.metadata as CollectionMeta).Collection.collection_tags ?? {})}
            setTags={(updatedTags) => updatePendingMeta('collection_tags', tagEntriesToRequestTags(updatedTags))}
            placeholderText="Add Tags"
          />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Case‑Insensitive</InfoHeader>
        <InfoValue>
          <Form.Check
            type="switch"
            id="case-insensitive-toggle"
            checked={entity.metadata.Collection.tags_case_insensitive ?? false}
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
            checked={entity.metadata.Collection.ignore_groups ?? false}
            onChange={(e) => updatePendingMeta('ignore_groups', e.target.checked)}
          />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Newest</InfoHeader>
        <InfoValue>
          <FilterDatePicker
            max={maxDate}
            min={entity.metadata.Collection.end}
            selected={entity.metadata.Collection.start}
            disabled={false}
            onChange={(e) => updatePendingMeta('start', safeDateToStringConversion(e))}
          />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Oldest</InfoHeader>
        <InfoValue>
          <FilterDatePicker
            max={entity.metadata.Collection.start ? entity.metadata.Collection.start : maxDate}
            selected={entity.metadata.Collection.end}
            disabled={false}
            onChange={(e) => updatePendingMeta('end', safeDateToStringConversion(e))}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const CollectionCreateConfig: EntityCreateConfig<Entities.Collection> = {
  kind: Entities.Collection,
  EntityMetadata: CollectionMetaInfo,
  BlankCreateEntity: BlankCreateCollection,
};

export default CollectionCreateConfig;
