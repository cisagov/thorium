import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import TextInput from '@components/shared/inputs/TextInput';
import { Entities } from '@models/entities/entities';
import { BlankCreateIncident, IncidentCreateMetaFields } from '@models/entities/incident';

// spec: ../EntityCreate.spec.md

const LIST_FIELDS: {
  field: keyof Pick<IncidentCreateMetaFields, 'mission_teams' | 'networks' | 'machines' | 'locations'>;
  label: string;
}[] = [
  { field: 'mission_teams', label: 'Mission Teams' },
  { field: 'networks', label: 'Networks' },
  { field: 'machines', label: 'Machines' },
  { field: 'locations', label: 'Locations' },
];

const IncidentMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.Incident>): JSX.Element => {
  function updatePendingMeta<T extends keyof IncidentCreateMetaFields>(field: T, value: IncidentCreateMetaFields[T]): void {
    const updates: IncidentCreateMetaFields = structuredClone(entity.metadata.Incident);
    updates[field] = value;
    onChange('metadata', { Incident: updates });
  }

  return (
    <>
      <Row className="d-flex flex-row justify-content-center">
        <InfoHeader>Cover Term</InfoHeader>
        <InfoValue>
          <TextInput
            type="text"
            value={entity.metadata.Incident.cover_term ?? ''}
            onChange={(e) => updatePendingMeta('cover_term', e.target.value === '' ? null : e.target.value)}
          />
        </InfoValue>
      </Row>
      {LIST_FIELDS.map(({ field, label }) => (
        <div key={field}>
          <hr className="my-3" />
          <Row>
            <InfoHeader>{label}</InfoHeader>
            <InfoValue>
              <SelectInputArray values={entity.metadata.Incident[field]} onChange={(values) => updatePendingMeta(field, values)} />
            </InfoValue>
          </Row>
        </div>
      ))}
    </>
  );
};

const IncidentCreateConfig: EntityCreateConfig<Entities.Incident> = {
  kind: Entities.Incident,
  EntityMetadata: IncidentMetaInfo,
  BlankCreateEntity: BlankCreateIncident,
};

export default IncidentCreateConfig;
