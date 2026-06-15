import { useLocation } from 'react-router';

// project imports
import { UploadProvider, useUpload, UploadForm, UploadStatusDashboard } from '@components/pages/files/upload';
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import type { EntityTypes } from '@models/entities/entities';

// spec: ../EntityCreate.spec.md

const UploadContent: React.FC = () => {
  const { showUploadStatus } = useUpload();
  return showUploadStatus ? <UploadStatusDashboard /> : <UploadForm />;
};

const FileUpload = () => {
  // grab state in case entity was passed in, entity context allows us to associate files with that entity
  const { state } = useLocation() as { state: { entity?: EntityTypes } | null };
  return (
    <Page title="Analyze · Thorium">
      <div className="d-flex justify-content-center">
        <Title>Analyze</Title>
      </div>
      <UploadProvider entity={state?.entity}>
        <UploadContent />
      </UploadProvider>
    </Page>
  );
};

export default FileUpload;
