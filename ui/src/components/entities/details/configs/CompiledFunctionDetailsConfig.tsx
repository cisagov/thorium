import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityDetailsConfig, makeGetEntityDetails } from './factory';
import { DetailsMetadataProps } from '../EntityDetails';
import DisassemblyEditor from '@entities/shared/DisassemblyEditor';
import EntityTypeIcon from '@entities/shared/EntityTypeIcon';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import NumberInput from '@components/shared/inputs/NumberInput';
import CodeRenderer from '@components/shared/renderers/CodeRenderer';
import { stringToRenderableInput } from '@components/shared/renderers/detect';
import { disassemblyToText, formatAddress } from '@utilities/disassembly';
import { FormatType } from '@utilities/rules/types';
import { Entities } from '@models/entities';
import { BlankCompiledFunction, CompiledFunctionMetaFields } from '@models/entities/functions';

// spec: ../EntityDetails.spec.md

const CompiledFunctionMetaInfo = ({
  entity,
  pendingEntity,
  handleUpdate,
  editing,
}: DetailsMetadataProps<Entities.CompiledFunction>): JSX.Element => {
  // apply a single metadata field change and hand the updated metadata back to the entity update
  function updatePendingMeta<T extends keyof CompiledFunctionMetaFields>(field: T, value: CompiledFunctionMetaFields[T]): void {
    const updates: CompiledFunctionMetaFields = structuredClone(pendingEntity.metadata.CompiledFunction);
    updates[field] = value;
    handleUpdate('metadata', { CompiledFunction: updates });
  }
  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Address</InfoHeader>
        <InfoValue>
          {editing ? (
            <NumberInput
              value={pendingEntity.metadata.CompiledFunction.address}
              onChange={(v) => updatePendingMeta('address', v ?? 0)}
              min={0}
            />
          ) : (
            formatAddress(entity.metadata.CompiledFunction.address)
          )}
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Disassembly</InfoHeader>
        <InfoValue>
          {editing ? (
            <DisassemblyEditor
              disassembly={pendingEntity.metadata.CompiledFunction.disassembly}
              onCommit={(instructions) => updatePendingMeta('disassembly', instructions)}
              resetSignal={entity}
            />
          ) : (
            // view-only reuses the shared read-only renderer with the assembly highlighter
            <CodeRenderer
              input={stringToRenderableInput(disassemblyToText(entity.metadata.CompiledFunction.disassembly))}
              format={FormatType.Disassembly}
              height="auto"
            />
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const CompiledFunctionDetailsConfig: EntityDetailsConfig<Entities.CompiledFunction> = {
  getEntityDetails: makeGetEntityDetails(Entities.CompiledFunction),
  EntityMetaInfo: CompiledFunctionMetaInfo,
  BlankEntity: BlankCompiledFunction,
  icon: (size: number) => <EntityTypeIcon kind={Entities.CompiledFunction} size={size} />,
};

export default CompiledFunctionDetailsConfig;
