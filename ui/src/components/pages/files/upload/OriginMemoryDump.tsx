import React from 'react';
import { Col, Row } from 'react-bootstrap';
import Subtitle from '@components/shared/titles/Subtitle';
import SelectableArray from '@components/shared/inputs/selectable/SelectableArray';
import { useUpload } from './UploadContext';
import OriginField from './OriginField';

// spec: ./upload.spec.md

const OriginMemoryDump: React.FC = () => {
  const { originState, origin } = useUpload();
  const { memoryType, parentFile, reconstructed, baseAddress } = originState.memoryDump;

  return (
    <>
      <OriginField
        label="Memory Type"
        value={memoryType}
        onChange={(v) => origin.setMemoryDumpField('memoryType', v)}
        placeholder="type"
        isInvalid={!memoryType && (!!parentFile || reconstructed.length > 0 || !!baseAddress)}
        feedback="Please enter a Memory Type."
      />
      <OriginField label="Parent" value={parentFile} onChange={(v) => origin.setMemoryDumpField('parentFile', v)} />
      <br />
      <Row>
        <Col m={12} lg={3} xxl={2}>
          <Subtitle>Reconstructed</Subtitle>
        </Col>
        <Col m={12} lg={9} xxl={10}>
          <SelectableArray
            initialEntries={[]}
            setEntries={(entries: string[]) => origin.setMemoryDumpField('reconstructed', entries.join(','))}
            disabled={false}
            placeholder="optional"
            trim={false}
          />
        </Col>
      </Row>
      <OriginField label="Base Address" value={baseAddress} onChange={(v) => origin.setMemoryDumpField('baseAddress', v)} />
    </>
  );
};

export default OriginMemoryDump;
