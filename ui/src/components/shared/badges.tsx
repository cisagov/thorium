import React, { useState } from 'react';
import { Button, Modal } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { OverlayTipBottom } from '@components';

interface LinkBadgeProps {
  url: string; // redirect url
  label?: string;
  internal?: boolean; // whether link is on same site
}

const BadgeContent = styled.div`
  display: inline-flex;
  border-radius: 4px;
  font-size: 12px;
  padding: 0.1rem 0.4rem 0.1rem 0.4rem;
  cursor: pointer;
  max-width: 20rem;
  overflow: hidden;
  background-color: gray;
  margin-right: 0.1rem;
  text-decoration: underline;
`;

export const LinkBadge: React.FC<LinkBadgeProps> = ({ url, label, internal = false }) => {
  const [showRedirectModal, setShowRedirectModal] = useState(false);

  // on click function to redirect to external URL
  const redirectToExternal = () => {
    window.open(url, '_blank');
  };

  // redirect to current site with url as path
  const redirectToInternal = () => {
    window.open(`${window.location.protocol}//${window.location.host}${url}`, '_blank');
  };

  const handleBadgeClick = () => {
    if (!internal) {
      setShowRedirectModal(true);
    } else {
      redirectToInternal();
    }
  };

  return (
    <OverlayTipBottom className="" tip={`Click to navigate to url: ${url}`}>
      <Modal show={showRedirectModal} onHide={() => setShowRedirectModal(false)}>
        <Modal.Header closeButton>
          <h3>Navigate to an external site?</h3>
        </Modal.Header>
        <Modal.Body className="d-flex justify-content-center">
          <i>{url}</i>
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          {/* @ts-ignore */}
          <Button
            variant=""
            className="warning-btn"
            onClick={() => {
              redirectToExternal();
              setShowRedirectModal(false);
            }}
          >
            Confirm
          </Button>
        </Modal.Footer>
      </Modal>
      <a onClick={handleBadgeClick}>
        <BadgeContent>{label ? label : url}</BadgeContent>
      </a>
    </OverlayTipBottom>
  );
};
