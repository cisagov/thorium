// project imports
import { Title, Page, Upload } from '@components';
import { useLocation } from 'react-router';

const UploadFilesContainer = () => {
  // grab state in case entity was passed in, entity context allows us to associate files with that entity
  const { state } = useLocation();
  return (
    <Page title="Upload Files · Thorium">
      <div className="d-flex justify-content-center">
        <Title>Upload</Title>
      </div>
      <Upload entity={state?.entity} />
    </Page>
  );
};

export default UploadFilesContainer;
