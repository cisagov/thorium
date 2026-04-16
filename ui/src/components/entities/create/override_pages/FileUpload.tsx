import { useLocation } from 'react-router';

// project imports
import { UploadProvider, useUpload, UploadForm, UploadStatusDashboard } from '@components/pages/files/upload';
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';

const UploadContent: React.FC = () => {
  const { showUploadStatus } = useUpload();
  return showUploadStatus ? <UploadStatusDashboard /> : <UploadForm />;
};

const FileUpload = () => {
  // grab state in case entity was passed in, entity context allows us to associate files with that entity
  const { state } = useLocation();
  return (
    <Page title="Upload Files · Thorium">
      <div className="d-flex justify-content-center">
        <Title>Upload</Title>
      </div>
      <UploadProvider entity={state?.entity}>
        <UploadContent />
      </UploadProvider>
    </Page>
  );
};

export default FileUpload;
