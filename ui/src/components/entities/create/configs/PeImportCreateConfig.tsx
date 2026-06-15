import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { Entities } from '@models/entities/entities';
import { BlankCreatePeImport, PeImportCreateMetaFields } from '@models/entities/pe';

// spec: ../EntityCreate.spec.md

const PeImportMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.PeImport>): JSX.Element => {
  function updatePendingMeta<T extends keyof PeImportCreateMetaFields>(field: T, value: PeImportCreateMetaFields[T]): void {
    const updates: PeImportCreateMetaFields = structuredClone(entity.metadata.PeImport);
    updates[field] = value;
    onChange('metadata', { PeImport: updates });
  }

  return (
    <Row>
      <InfoHeader>Functions</InfoHeader>
      <InfoValue>
        <SelectInputArray values={entity.metadata.PeImport.functions} onChange={(functions) => updatePendingMeta('functions', functions)} />
      </InfoValue>
    </Row>
  );
};

const PeImportCreateConfig: EntityCreateConfig<Entities.PeImport> = {
  kind: Entities.PeImport,
  EntityMetadata: PeImportMetaInfo,
  BlankCreateEntity: BlankCreatePeImport,
};

export default PeImportCreateConfig;
