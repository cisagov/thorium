import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import NumberInput from '@components/shared/inputs/NumberInput';
import TextInput from '@components/shared/inputs/TextInput';
import { Entities } from '@models/entities/entities';
import { BlankCreatePeSection, PeSectionCreateMetaFields } from '@models/entities/pe';

// spec: ../EntityCreate.spec.md

const PeSectionMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.PeSection>): JSX.Element => {
  function updatePendingMeta<T extends keyof PeSectionCreateMetaFields>(field: T, value: PeSectionCreateMetaFields[T]): void {
    const updates: PeSectionCreateMetaFields = structuredClone(entity.metadata.PeSection);
    updates[field] = value;
    onChange('metadata', { PeSection: updates });
  }
  const meta = entity.metadata.PeSection;

  return (
    <>
      <Row>
        <InfoHeader>MD5</InfoHeader>
        <InfoValue>
          <TextInput
            type="text"
            value={meta.md5 ?? ''}
            onChange={(e) => updatePendingMeta('md5', e.target.value === '' ? undefined : e.target.value)}
          />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Raw Size</InfoHeader>
        <InfoValue>
          {/* optional field: undefined ↔ null and required=false so an empty box stays unset, not 0 */}
          <NumberInput
            value={meta.raw_size ?? null}
            onChange={(v) => updatePendingMeta('raw_size', v ?? undefined)}
            min={0}
            required={false}
          />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Virtual Size</InfoHeader>
        <InfoValue>
          <NumberInput
            value={meta.virtual_size ?? null}
            onChange={(v) => updatePendingMeta('virtual_size', v ?? undefined)}
            min={0}
            required={false}
          />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Entropy</InfoHeader>
        <InfoValue>
          <NumberInput
            value={meta.entropy ?? null}
            onChange={(v) => updatePendingMeta('entropy', v ?? undefined)}
            min={0}
            step={0.01}
            required={false}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const PeSectionCreateConfig: EntityCreateConfig<Entities.PeSection> = {
  kind: Entities.PeSection,
  EntityMetadata: PeSectionMetaInfo,
  BlankCreateEntity: BlankCreatePeSection,
};

export default PeSectionCreateConfig;
