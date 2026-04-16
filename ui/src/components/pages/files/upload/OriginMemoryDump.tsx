import React from 'react';
import { Col, Row } from 'react-bootstrap';
import Subtitle from '@components/shared/titles/Subtitle';
import SelectableArray from '@components/shared/inputs/selectable/SelectableArray';
import { useUpload } from './UploadContext';
import OriginField from './OriginField';

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
        <Col className="name-width" xs={2}>
          <Subtitle>Reconstructed</Subtitle>
        </Col>
        <Col xs={5}>
          <SelectableArray
            initialEntries={[]}
            setEntries={(val: string) => origin.setMemoryDumpField('reconstructed', val)}
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
