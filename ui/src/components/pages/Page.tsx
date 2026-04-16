import React, { useEffect } from 'react';
import styled from 'styled-components';

// project imports
import { RequireAdmin, RequireAuth, useAuth } from '@utilities/auth';
import { scaling } from '@styles';

interface PageProps {
  auth?: boolean;
  admin?: boolean;
  Contents: React.LazyExoticComponent<React.ComponentType<any>>;
}

export const PageWrapper: React.FC<PageProps> = ({ auth = true, admin = false, Contents }) => {
  const { refreshUserInfo } = useAuth();
  useEffect(() => {
    void refreshUserInfo();
  }, []);

  if (admin) {
    return (
      <RequireAuth>
        <RequireAdmin>
          <Contents />
        </RequireAdmin>
      </RequireAuth>
    );
  } else if (auth) {
    return (
      <RequireAuth>
        <Contents />
      </RequireAuth>
    );
  } else {
    return (
      <div>
        <Contents />
      </div>
    );
  }
};

interface ThoriumPageProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  id?: string;
}

const PageContainer = styled.div`
  width: 100%;
  max-width: 85%;
  margin: 0 auto;
  padding: 4rem 15px 0.5rem 15px;
  @media (min-width: ${scaling.sm}) {
    padding: 4rem 0rem 0.5rem 3.5rem;
  }
  @media (min-width: ${scaling.lg}) {
    padding: 4rem 0rem 0.5rem 6rem;
  }
  @media (min-width: ${scaling.xxxl}) {
    max-width: 80%;
  }
  @media (min-width: ${scaling.fivexl}) {
    max-width: ${scaling.fourxl};
  }
`;

const Page: React.FC<ThoriumPageProps> = ({ children, className, title, id }) => {
  return (
    <PageContainer className={className} id={id}>
      <title>{title}</title>
      {children}
    </PageContainer>
  );
};

export default Page;
