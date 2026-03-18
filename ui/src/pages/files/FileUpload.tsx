import { useLocation } from 'react-router';

// project imports
import UploadFile from '@components/pages/files/UploadFile';
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';

const FileUpload = () => {
  // grab state in case entity was passed in, entity context allows us to associate files with that entity
  const { state } = useLocation();
  return (
    <Page title="Upload Files · Thorium">
      <div className="d-flex justify-content-center">
        <Title>Upload</Title>
      </div>
      <UploadFile entity={state?.entity} />
    </Page>
  );
};

export default FileUpload;
