import React from 'react';
import { useUpload } from './UploadContext';
import OriginField from './OriginField';

interface OriginParentToolFlagsProps {
  variant: 'transformed' | 'unpacked';
}

const OriginParentToolFlags: React.FC<OriginParentToolFlagsProps> = ({ variant }) => {
  const { originState, origin } = useUpload();
  const { parentFile, tool, toolFlags } = originState[variant];

  return (
    <>
      <OriginField
        label="Parent"
        value={parentFile}
        onChange={(v) => origin.setParentToolField(variant, 'parentFile', v)}
        placeholder="SHA256"
        isInvalid={!parentFile && (!!tool || !!toolFlags)}
        feedback="Please enter a SHA256 value for the Parent."
      />
      <OriginField label="Tool" value={tool} onChange={(v) => origin.setParentToolField(variant, 'tool', v)} />
      <OriginField label="Flags" value={toolFlags} onChange={(v) => origin.setParentToolField(variant, 'toolFlags', v)} />
    </>
  );
};

export default OriginParentToolFlags;
