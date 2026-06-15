import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityDetailsConfig, makeGetEntityDetails } from './factory';
import { DetailsMetadataProps } from '../EntityDetails';
import EntityTypeIcon from '@entities/shared/EntityTypeIcon';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import { Entities } from '@models/entities';
import { BlankFlag } from '@models/entities/flag';

// spec: ../EntityDetails.spec.md

// Read-only display of a flag's metadata fields.
const FlagMetaInfo = ({ entity }: DetailsMetadataProps<Entities.Flag>): JSX.Element => {
  const meta = entity.metadata.Flag;
  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Suspicion</InfoHeader>
        <InfoValue>{meta.suspicion}</InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Confidence</InfoHeader>
        <InfoValue>{meta.confidence}</InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Reasoning</InfoHeader>
        <InfoValue>{meta.reasoning}</InfoValue>
      </Row>
      {meta.content && (
        <>
          <hr className="my-3" />
          <Row className="mt-3">
            <InfoHeader>Content</InfoHeader>
            <InfoValue>{meta.content}</InfoValue>
          </Row>
        </>
      )}
    </>
  );
};

const FlagDetailsConfig: EntityDetailsConfig<Entities.Flag> = {
  getEntityDetails: makeGetEntityDetails(Entities.Flag),
  EntityMetaInfo: FlagMetaInfo,
  BlankEntity: BlankFlag,
  icon: (size: number) => <EntityTypeIcon kind={Entities.Flag} size={size} />,
};

export default FlagDetailsConfig;
