import React, { useState } from 'react';
import { Alert } from 'react-bootstrap';
import styled from 'styled-components';

const PageAlertWrapper = styled(Alert)`
  top: 60px;
  margin-left: 10rem;
  margin-right: 0.75rem;
  display: flex;
  justify-content: center;
`;

const ComponentAlertWrapper = styled(Alert)`
  text-align: center;
`;

interface RenderErrorProps {
  message?: string; // message to display within alert
  page?: boolean; // whether this alert is for a page or within an component
}

export const RenderErrorAlert: React.FC<RenderErrorProps> = ({ message = '', page = true }) => {
  let errorMessage =
    'Uh oh! An error occurred while rendering. If this persists after refreshing the page, please report it to your Thorium Admins.';
  if (message) {
    errorMessage = message;
  }
  if (page) {
    return (
      <PageAlertWrapper variant="danger">
        <pre>{errorMessage}</pre>
      </PageAlertWrapper>
    );
  }
  return (
    <ComponentAlertWrapper variant="danger">
      <pre>{errorMessage}</pre>
    </ComponentAlertWrapper>
  );
};

interface AlertBannerProps {
  prefix: string; // prefix to add to api error message
  errorStatus: string; // error status message from api request response
}

export const AlertBanner: React.FC<AlertBannerProps> = ({ prefix, errorStatus }) => {
  const [show, setShow] = useState(true);
  return (
    <>
      {show && (
        <Alert className="d-flex justify-content-center" onClose={() => setShow(false)} variant="danger">
          {prefix}: {errorStatus}
        </Alert>
      )}
    </>
  );
};
