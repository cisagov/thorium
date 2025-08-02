import React, { useEffect } from 'react';
import { Container } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { RequireAdmin, RequireAuth, useAuth } from '@utilities';

interface PageProps {
  auth?: boolean; // whether page requires validated user auth
  admin?: boolean; // whether page required Thorium admin role
  Contents: React.LazyExoticComponent<React.ComponentType<any>>; // page contents
}

// A basic page content wrapper
export const PageWrapper: React.FC<PageProps> = ({ auth = true, admin = false, Contents }) => {
  const { refreshUserInfo } = useAuth();
  // Check to see if user info is stale on first page load
  useEffect(() => {
    refreshUserInfo();
  });

  // Require an authed user with Admin role
  if (admin) {
    return (
      <RequireAuth>
        <RequireAdmin>
          <Contents />
        </RequireAdmin>
      </RequireAuth>
    );
    // Require user to be authed
  } else if (auth) {
    return (
      <RequireAuth>
        <Contents />
      </RequireAuth>
    );
    // No page auth or Admin role required
  } else {
    return (
      <div>
        <Contents />
      </div>
    );
  }
};

// properties for our Thorium page wrapper
interface ThoriumPageProps {
  children: React.ReactNode; // For the children
  className?: string; // Optional class name for wrapper styles
  title?: string; // Optional helmet title
  id?: string; // Optional id for page container
}

// @ts-ignore
const PageContainer = styled(Container)`
  padding: 4rem 0rem 0.5rem 6rem;
  width: 100%;
`;

export const Page: React.FC<ThoriumPageProps> = ({ children, className, title, id }) => {
  return (
    <PageContainer className={className} id={id}>
      <title>{title}</title>
      {children}
    </PageContainer>
  );
};
