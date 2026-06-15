import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityDetailsConfig, makeGetEntityDetails } from './factory';
import { DetailsMetadataProps } from '../EntityDetails';
import { CreateOnlyNote } from './shared';
import EntityTypeIcon from '@entities/shared/EntityTypeIcon';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import TextInput from '@components/shared/inputs/TextInput';
import { Entities } from '@models/entities';
import { BlankIncident, IncidentMetaFields } from '@models/entities/incident';

// spec: ../EntityDetails.spec.md

// The incident list fields rendered identically as edit (multi-select) / view (badges) rows.
const LIST_FIELDS: { field: keyof Pick<IncidentMetaFields, 'mission_teams' | 'networks' | 'machines' | 'locations'>; label: string }[] = [
  { field: 'mission_teams', label: 'Mission Teams' },
  { field: 'networks', label: 'Networks' },
  { field: 'machines', label: 'Machines' },
  { field: 'locations', label: 'Locations' },
];

const IncidentMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.Incident>): JSX.Element => {
  // apply a single metadata field change and hand the updated metadata back to the entity update
  function updatePendingMeta<T extends keyof IncidentMetaFields>(field: T, value: IncidentMetaFields[T]): void {
    const updates: IncidentMetaFields = structuredClone(pendingEntity.metadata.Incident);
    updates[field] = value;
    handleUpdate('metadata', { Incident: updates });
  }

  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Cover Term</InfoHeader>
        <InfoValue>
          {editing ? (
            <>
              <TextInput
                type="text"
                value={pendingEntity.metadata.Incident.cover_term ?? ''}
                onChange={(e) => updatePendingMeta('cover_term', e.target.value === '' ? null : e.target.value)}
              />
              {/* the API's incident update only sets cover_term when non-empty, so clearing it won't persist */}
              <CreateOnlyNote>Clearing the cover term won't be saved.</CreateOnlyNote>
            </>
          ) : (
            (entity.metadata.Incident.cover_term ?? '')
          )}
        </InfoValue>
      </Row>
      {LIST_FIELDS.map(({ field, label }) => (
        <div key={field}>
          <hr className="my-3" />
          <Row>
            <InfoHeader>{label}</InfoHeader>
            <InfoValue>
              {editing ? (
                <SelectInputArray values={pendingEntity.metadata.Incident[field]} onChange={(values) => updatePendingMeta(field, values)} />
              ) : (
                <FieldBadge color="Gray" noNull field={entity.metadata.Incident[field]} />
              )}
            </InfoValue>
          </Row>
        </div>
      ))}
    </>
  );
};

const IncidentDetailsConfig: EntityDetailsConfig<Entities.Incident> = {
  getEntityDetails: makeGetEntityDetails(Entities.Incident),
  EntityMetaInfo: IncidentMetaInfo,
  BlankEntity: BlankIncident,
  icon: (size: number) => <EntityTypeIcon kind={Entities.Incident} size={size} />,
};

export default IncidentDetailsConfig;
