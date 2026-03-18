import styled from 'styled-components';

// project imports
import Page from '@components/pages/Page';
import Banner from '@components/shared/titles/Banner';
import Search from '@components/pages/search/Search';

const Stack = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const Home = () => {
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

export default Home;
