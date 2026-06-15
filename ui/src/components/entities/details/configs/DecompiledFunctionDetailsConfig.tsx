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
import NumberInput from '@components/shared/inputs/NumberInput';
import CodeEditor from '@components/shared/inputs/code/CodeEditor/CodeEditor';
import CodeRenderer from '@components/shared/renderers/CodeRenderer';
import { stringToRenderableInput } from '@components/shared/renderers/detect';
import { formatAddress } from '@utilities/disassembly';
import { FormatType } from '@utilities/rules/types';
import { Entities } from '@models/entities';
import { BlankDecompiledFunction, DecompiledFunctionMetaFields } from '@models/entities/functions';

// spec: ../EntityDetails.spec.md

const DecompiledFunctionMetaInfo = ({
  entity,
  pendingEntity,
  handleUpdate,
  editing,
}: DetailsMetadataProps<Entities.DecompiledFunction>): JSX.Element => {
  // apply a single metadata field change and hand the updated metadata back to the entity update
  function updatePendingMeta<T extends keyof DecompiledFunctionMetaFields>(field: T, value: DecompiledFunctionMetaFields[T]): void {
    const updates: DecompiledFunctionMetaFields = structuredClone(pendingEntity.metadata.DecompiledFunction);
    updates[field] = value;
    handleUpdate('metadata', { DecompiledFunction: updates });
  }

  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Address</InfoHeader>
        <InfoValue>
          {editing ? (
            <NumberInput
              value={pendingEntity.metadata.DecompiledFunction.address}
              onChange={(v) => updatePendingMeta('address', v ?? 0)}
              min={0}
            />
          ) : (
            formatAddress(entity.metadata.DecompiledFunction.address)
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Tools</InfoHeader>
        <InfoValue>
          <FieldBadge color="Gray" noNull field={entity.metadata.DecompiledFunction.tools} />
          {editing && <CreateOnlyNote>Tools are set at creation and can't be changed here.</CreateOnlyNote>}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Content</InfoHeader>
        <InfoValue>
          {editing ? (
            <CodeEditor
              value={pendingEntity.metadata.DecompiledFunction.content}
              onChange={(text) => updatePendingMeta('content', text)}
              format={FormatType.Decomp}
              height="400px"
            />
          ) : (
            // view-only reuses the shared decomp renderer (same viewer as results / file preview)
            <CodeRenderer
              input={stringToRenderableInput(entity.metadata.DecompiledFunction.content)}
              format={FormatType.Decomp}
              height="auto"
            />
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const DecompiledFunctionDetailsConfig: EntityDetailsConfig<Entities.DecompiledFunction> = {
  getEntityDetails: makeGetEntityDetails(Entities.DecompiledFunction),
  EntityMetaInfo: DecompiledFunctionMetaInfo,
  BlankEntity: BlankDecompiledFunction,
  icon: (size: number) => <EntityTypeIcon kind={Entities.DecompiledFunction} size={size} />,
};

export default DecompiledFunctionDetailsConfig;
