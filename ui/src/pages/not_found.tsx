import styled from 'styled-components';

// project imports
import { Banner, Page } from '@components';

const NotFoundWrapper = styled.div`
  min-height: 320px;
  height: 100vh;
  margin-top: -4.5rem;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
`;

const NotFoundContainer = () => {
  return (
    <Page className="d-flex justify-content-center" title="Not Found · Thorium">
      <NotFoundWrapper>
        <img src="/ferris-scientist.png" className="pe-4 icon-separator-end" alt="FerrisScientist" height="200px" />
        <div className="d-flex flex-column justify-content-center ms-4">
          <Banner>Uh Oh!</Banner>
          <Banner>{window.location.pathname}</Banner>
          <Banner>Not Found</Banner>
        </div>
      </NotFoundWrapper>
    </Page>
  );
};

export default NotFoundContainer;
