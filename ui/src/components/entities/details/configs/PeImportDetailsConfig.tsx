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
import { Entities } from '@models/entities';
import { BlankPeImport, PeImportMetaFields } from '@models/entities/pe';

// spec: ../EntityDetails.spec.md

const PeImportMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.PeImport>): JSX.Element => {
  // apply a single metadata field change and hand the updated metadata back to the entity update
  function updatePendingMeta<T extends keyof PeImportMetaFields>(field: T, value: PeImportMetaFields[T]): void {
    const updates: PeImportMetaFields = structuredClone(pendingEntity.metadata.PeImport);
    updates[field] = value;
    handleUpdate('metadata', { PeImport: updates });
  }

  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Functions</InfoHeader>
        <InfoValue>
          {editing ? (
            <>
              <SelectInputArray
                values={pendingEntity.metadata.PeImport.functions}
                onChange={(functions) => updatePendingMeta('functions', functions)}
              />
              {/* the API's PeImport update ignores an empty list, so clearing every function won't persist */}
              <CreateOnlyNote>Removing every function won't be saved — at least one must remain.</CreateOnlyNote>
            </>
          ) : (
            <FieldBadge color="Gray" noNull field={entity.metadata.PeImport.functions} />
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const PeImportDetailsConfig: EntityDetailsConfig<Entities.PeImport> = {
  getEntityDetails: makeGetEntityDetails(Entities.PeImport),
  EntityMetaInfo: PeImportMetaInfo,
  BlankEntity: BlankPeImport,
  icon: (size: number) => <EntityTypeIcon kind={Entities.PeImport} size={size} />,
};

export default PeImportDetailsConfig;
