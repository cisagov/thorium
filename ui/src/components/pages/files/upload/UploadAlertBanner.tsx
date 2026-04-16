import React, { Fragment } from 'react';
import { Row } from 'react-bootstrap';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { Link } from 'react-router-dom';
import RunReactionAlerts from '../reactions/RunReactionAlerts';
import { ReactionSubmitResult } from './types';

interface UploadAlertBannerProps {
  uploadSHA256: string[];
  uploadError: string[];
  runReactionsRes: ReactionSubmitResult[];
}

const UploadAlertBanner: React.FC<UploadAlertBannerProps> = ({ uploadSHA256, uploadError, runReactionsRes }) => {
  return (
    <Fragment>
      <Row>
        {uploadSHA256 &&
          uploadSHA256.map((sha256) => (
            <AlertBanner severity={Severity.Success} key={sha256}>
              File uploaded successfully: <pre> </pre>
              <Link className="link-text" to={'/file/' + sha256} target="_blank">
                {sha256}
              </Link>
            </AlertBanner>
          ))}
      </Row>
      <Row>{uploadError && uploadError.map((message) => <AlertBanner key={message}>{message}</AlertBanner>)}</Row>
      <RunReactionAlerts responses={runReactionsRes} />
    </Fragment>
  );
};

export default UploadAlertBanner;
