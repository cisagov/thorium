import React, { Fragment, useMemo } from 'react';
import { Button, Card, Col, Row } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { FaRedo } from 'react-icons/fa';
import { FaChartColumn } from 'react-icons/fa6';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import ProgressBarContainer from './ProgressBarContainer';
import UploadStatusTable from './UploadStatusTable';
import { useUpload } from './UploadContext';
import { encodeSeedParams } from '@dashboards/Dashboard/seedParams';

// spec: ./upload.spec.md

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

  // the sha256s of files that uploaded successfully, deduped — used to seed a dashboard of this batch
  const uploadedShas = useMemo(
    () =>
      Array.from(
        new Set(
          Object.values(uploadStatus)
            .map((status) => status.sha256)
            .filter((sha): sha is string => Boolean(sha)),
        ),
      ),
    [uploadStatus],
  );
  // link to a dashboard pre-seeded with this upload batch (default crawl depth 2, matching the builder)
  const dashboardHref = `/dashboard/view?${encodeSeedParams({ samples: uploadedShas }, 2).toString()}`;

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
      {!uploadInProgress && uploadedShas.length > 0 && (
        <Row className="mt-2">
          <Col className="d-flex justify-content-center">
            <OverlayTipTop tip="Open a dashboard seeded with the files you just uploaded">
              <Link to={dashboardHref} className="ok-btn">
                <FaChartColumn className="me-2" />
                Open a dashboard to view these items
              </Link>
            </OverlayTipTop>
          </Col>
        </Row>
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
