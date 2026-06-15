import { useLocation } from 'react-router';

import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import { UploadProvider, useUpload, UploadForm, UploadStatusDashboard } from '@components/pages/files/upload';
import { EntityTypes } from '@models/entities/entities';

// navigation state passed when the upload page is opened from an entity page
interface FileUploadLocationState {
  entity?: EntityTypes;
}

const UploadContent: React.FC = () => {
  const { showUploadStatus } = useUpload();
  return showUploadStatus ? <UploadStatusDashboard /> : <UploadForm />;
};

const FileUpload = () => {
  const state = useLocation().state as FileUploadLocationState | null;
  const entity = state?.entity;

  return (
    <Page title="Upload Files · Thorium">
      <div className="d-flex justify-content-center">
        <Title>Upload</Title>
      </div>
      <UploadProvider entity={entity}>
        <UploadContent />
      </UploadProvider>
    </Page>
  );
};

export default FileUpload;
