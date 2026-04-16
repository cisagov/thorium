import React from 'react';
import styled from 'styled-components';
import AlertBanner from './AlertBanner';

const PageWrapper = styled.div`
  top: 60px;
  margin-left: 10rem;
  margin-right: 0.75rem;
  display: flex;
  justify-content: center;
`;

interface RenderErrorProps {
  message?: string;
  page?: boolean;
}

const RenderErrorAlert: React.FC<RenderErrorProps> = ({ message = '', page = true }) => {
  let errorMessage =
    'Uh oh! An error occurred while rendering. If this persists after refreshing the page, please report it to your Thorium Admins.';
  if (message) {
    errorMessage = message;
  }
  if (page) {
    return (
      <PageWrapper>
        <AlertBanner>
          <pre>{errorMessage}</pre>
        </AlertBanner>
      </PageWrapper>
    );
  }
  return (
    <AlertBanner>
      <pre>{errorMessage}</pre>
    </AlertBanner>
  );
};

export default RenderErrorAlert;
