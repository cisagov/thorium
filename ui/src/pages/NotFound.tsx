import styled from 'styled-components';

// project imports
import Page from '@components/pages/Page';
import Banner from '@components/shared/titles/Banner';

const NotFoundWrapper = styled.div`
  min-height: 320px;
  height: 100vh;
  margin-top: -4.5rem;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
`;

const Image = styled.img`
  border-right: 2px solid;
`;

const NotFound = () => {
  return (
    <Page className="d-flex justify-content-center" title="Not Found · Thorium">
      <NotFoundWrapper>
        <Image src="/ferris-scientist.png" className="pe-4" alt="FerrisScientist" height="200px" />
        <div className="d-flex flex-column justify-content-center ms-4">
          <Banner>Uh Oh!</Banner>
          <Banner>{window.location.pathname}</Banner>
          <Banner>Not Found</Banner>
        </div>
      </NotFoundWrapper>
    </Page>
  );
};

export default NotFound;
