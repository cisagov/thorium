import React from 'react';
import { useUpload } from './UploadContext';
import OriginField from './OriginField';

const OriginDownloaded: React.FC = () => {
  const { originState, origin } = useUpload();
  const { url, name } = originState.downloaded;

  return (
    <>
      <OriginField
        label="URL"
        value={url}
        onChange={(v) => origin.setDownloadedField('url', v)}
        placeholder="badsite.xyz"
        isInvalid={!url && !!name}
        feedback="Please enter a site URL."
      />
      <OriginField label="Site Name" value={name} onChange={(v) => origin.setDownloadedField('name', v)} />
    </>
  );
};

export default OriginDownloaded;
