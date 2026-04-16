import React, { Fragment } from 'react';
import { Button, Card, Col, Row } from 'react-bootstrap';
import { FaRedo } from 'react-icons/fa';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import ProgressBarContainer from './ProgressBarContainer';
import UploadStatusTable from './UploadStatusTable';
import { useUpload } from './UploadContext';

const UploadStatusDashboard: React.FC = () => {
  const {
    uploadInProgress,
    activeUploads,
    uploadStatus,
    uploadFailures,
    uploadReactionRes,
    uploadReactionFailures,
    uploadError,
    totalProgress,
    retryAllFileUploads,
    retryAllReactionSubmissions,
    handleBack,
    cancelUpload,
  } = useUpload();

  return (
    <Fragment>
      Total
      <Row className="upload-bar">
        <Col>
          <ProgressBarContainer name={'Total'} value={totalProgress} error={uploadError.length > 0} />
        </Col>
      </Row>
      {uploadInProgress && (
        <Row className="upload-bar">
          {Object.values(activeUploads).map((key) => {
            const status = uploadStatus[key];
            if (!status) return null;
            return (
              <OverlayTipTop key={key} tip={status.msg}>
                {key}
                <ProgressBarContainer name={key} value={status.progress} error={status.fileFail} />
              </OverlayTipTop>
            );
          })}
        </Row>
      )}
      {!uploadInProgress && (
        <Card className="stats-container panel">
          <Card.Body>
            <div>{Object.keys(uploadStatus).length - Object.keys(uploadFailures).length} Files Uploaded Successfully</div>
            {Object.keys(uploadFailures).length > 0 && (
              <div>
                {Object.keys(uploadFailures).length} File Upload Failure(s)
                <Button size={'xsm' as 'sm'} variant="no-outline-secondary" className="retry-button" onClick={retryAllFileUploads}>
                  {' '}
                  <FaRedo />
                </Button>
              </div>
            )}
            <div>{uploadReactionRes.length - uploadReactionFailures} Reaction(s) Submitted Successfully</div>
            {uploadReactionFailures > 0 && (
              <div>
                {uploadReactionFailures} Reaction Submission(s) Failed
                <Button size={'xsm' as 'sm'} variant="no-outline-secondary" className="retry-button" onClick={retryAllReactionSubmissions}>
                  {' '}
                  <FaRedo />
                </Button>
              </div>
            )}
          </Card.Body>
        </Card>
      )}
      <UploadStatusTable />
      {!uploadInProgress ? (
        <Col className="d-flex justify-content-center close-button">
          <Button className="ok-btn" onClick={handleBack}>
            Back
          </Button>
        </Col>
      ) : (
        <Col className="d-flex justify-content-center close-button">
          <Button className="warning-btn" onClick={cancelUpload}>
            Cancel
          </Button>
        </Col>
      )}
    </Fragment>
  );
};

export default UploadStatusDashboard;
