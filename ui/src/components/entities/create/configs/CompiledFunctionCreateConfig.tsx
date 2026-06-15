import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import DisassemblyEditor from '@entities/shared/DisassemblyEditor';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import NumberInput from '@components/shared/inputs/NumberInput';
import { Entities } from '@models/entities/entities';
import { BlankCreateCompiledFunction, CompiledFunctionCreateMetaFields } from '@models/entities/functions';

// spec: ../EntityCreate.spec.md

const CompiledFunctionMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.CompiledFunction>): JSX.Element => {
  function updatePendingMeta<T extends keyof CompiledFunctionCreateMetaFields>(field: T, value: CompiledFunctionCreateMetaFields[T]): void {
    const updates: CompiledFunctionCreateMetaFields = structuredClone(entity.metadata.CompiledFunction);
    updates[field] = value;
    onChange('metadata', { CompiledFunction: updates });
  }
  return (
    <>
      <Row>
        <InfoHeader>Address</InfoHeader>
        <InfoValue>
          <NumberInput value={entity.metadata.CompiledFunction.address} onChange={(v) => updatePendingMeta('address', v ?? 0)} min={0} />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Disassembly</InfoHeader>
        <InfoValue>
          <DisassemblyEditor
            disassembly={entity.metadata.CompiledFunction.disassembly}
            onCommit={(instructions) => updatePendingMeta('disassembly', instructions)}
          />
        </InfoValue>
      </Row>
    </>
  );
};

const CompiledFunctionCreateConfig: EntityCreateConfig<Entities.CompiledFunction> = {
  kind: Entities.CompiledFunction,
  EntityMetadata: CompiledFunctionMetaInfo,
  BlankCreateEntity: BlankCreateCompiledFunction,
};

export default CompiledFunctionCreateConfig;
