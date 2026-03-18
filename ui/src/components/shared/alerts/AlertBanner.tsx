import React, { useState } from 'react';
import { Alert } from 'react-bootstrap';
import styled from 'styled-components';

interface AlertBannerProps {
  prefix: string; // prefix to add to api error message
  errorStatus: string; // error status message from api request response
  variant: string; // type of alert banner
}

const AlertBox = styled(Alert)`
  word-break: break-all;
`;

const AlertBanner: React.FC<AlertBannerProps> = ({ prefix = '', errorStatus, variant = 'danger' }) => {
  const [show, setShow] = useState(true);
  return (
    <>
      {show && (
        <AlertBox className="d-flex justify-content-center" onClose={() => setShow(false)} variant={`${variant}`}>
          {prefix != '' ? `${prefix}: ${errorStatus}` : `${errorStatus}`}
        </AlertBox>
      )}
    </>
  );
};

export default AlertBanner;
