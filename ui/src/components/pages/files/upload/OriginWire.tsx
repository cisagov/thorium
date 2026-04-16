import React from 'react';
import { useUpload } from './UploadContext';
import OriginField from './OriginField';

const OriginWire: React.FC = () => {
  const { originState, origin } = useUpload();
  const { sniffer, source, destination } = originState.wire;

  return (
    <>
      <OriginField
        label="Sniffer"
        value={sniffer}
        onChange={(v) => origin.setWireField('sniffer', v)}
        placeholder="name"
        isInvalid={!sniffer && (!!source || !!destination)}
        feedback="Please enter a name for the Sniffer."
      />
      <OriginField label="Source" value={source} onChange={(v) => origin.setWireField('source', v)} />
      <OriginField label="Destination" value={destination} onChange={(v) => origin.setWireField('destination', v)} />
    </>
  );
};

export default OriginWire;
