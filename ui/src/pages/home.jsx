import React from 'react';
import styled from 'styled-components';

// project imports
import { Banner, Search, Page } from '@components';

const Stack = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const HomeContainer = () => {
  return (
    <Page title="Thorium">
      <Stack>
        <img src="/ferris-scientist.png" alt="FerrisScientist" width="125px" />
        <Banner>Thorium</Banner>
        <Search />
      </Stack>
    </Page>
  );
};

export default HomeContainer;
