import { JSX } from 'react';
import { Row } from 'react-bootstrap';

// project imports
import { EntityCreateConfig } from './config';
import { CreateMetadataProps } from '../EntityCreate';
import InfoHeader from '@entities/shared/InfoHeader';
import InfoValue from '@entities/shared/InfoValue';
import NumberInput from '@components/shared/inputs/NumberInput';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import CodeEditor from '@components/shared/inputs/code/CodeEditor/CodeEditor';
import { FormatType } from '@utilities/rules/types';
import { Entities } from '@models/entities/entities';
import { BlankCreateDecompiledFunction, DecompiledFunctionCreateMetaFields } from '@models/entities/functions';

// spec: ../EntityCreate.spec.md

const DecompiledFunctionMetaInfo = ({ entity, onChange }: CreateMetadataProps<Entities.DecompiledFunction>): JSX.Element => {
  function updatePendingMeta<T extends keyof DecompiledFunctionCreateMetaFields>(
    field: T,
    value: DecompiledFunctionCreateMetaFields[T],
  ): void {
    const updates: DecompiledFunctionCreateMetaFields = structuredClone(entity.metadata.DecompiledFunction);
    updates[field] = value;
    onChange('metadata', { DecompiledFunction: updates });
  }

  return (
    <>
      <Row>
        <InfoHeader>Address</InfoHeader>
        <InfoValue>
          <NumberInput value={entity.metadata.DecompiledFunction.address} onChange={(v) => updatePendingMeta('address', v ?? 0)} min={0} />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Tools</InfoHeader>
        <InfoValue>
          <SelectInputArray values={entity.metadata.DecompiledFunction.tools} onChange={(tools) => updatePendingMeta('tools', tools)} />
        </InfoValue>
      </Row>
      <hr className="my-3" />
      <Row>
        <InfoHeader>Content</InfoHeader>
        <InfoValue>
          <CodeEditor
            value={entity.metadata.DecompiledFunction.content}
            onChange={(text) => updatePendingMeta('content', text)}
            format={FormatType.Decomp}
            height="400px"
          />
        </InfoValue>
      </Row>
    </>
  );
};

const DecompiledFunctionCreateConfig: EntityCreateConfig<Entities.DecompiledFunction> = {
  kind: Entities.DecompiledFunction,
  EntityMetadata: DecompiledFunctionMetaInfo,
  BlankCreateEntity: BlankCreateDecompiledFunction,
};

export default DecompiledFunctionCreateConfig;
