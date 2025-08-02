import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// project imports
import { AuthProvider } from '@utilities';
import { PageWrapper, RenderErrorAlert, NavBanner, SidebarColumn } from '@components';
import '@styles/main.scss';
import styled from 'styled-components';

// import pages lazily
const Home = lazy(() => import('./pages/home'));
const NotFound = lazy(async () => import('./pages/not_found'));
const FileDetails = lazy(async () => import('./pages/files/details'));
const RepoDetails = lazy(() => import('./pages/repos/details'));
const FilesBrowsing = lazy(() => import('./pages/files/browsing'));
const RepoBrowsing = lazy(() => import('./pages/repos/browsing'));
const CreateImage = lazy(() => import('./pages/images/create'));
const UploadFiles = lazy(() => import('./pages/files/upload'));
const Pipelines = lazy(() => import('./pages/pipelines'));
const Images = lazy(() => import('./pages/images/browsing'));
const Groups = lazy(() => import('./pages/users/groups'));
const Users = lazy(() => import('./pages/users/browsing'));
const ReactionStatus = lazy(() => import('./pages/reactions/reaction_status'));
const ReactionStageLogs = lazy(() => import('./pages/reactions/reaction_stage_logs'));
const Login = lazy(() => import('./pages/login'));
const Profile = lazy(() => import('./pages/users/profile'));
const SystemStats = lazy(() => import('./pages/system/system_stats'));
const SystemSettings = lazy(() => import('./pages/system/system_settings'));

// Data loading ui empty for now
const FallbackView = <h1 />;

const Resources = () => {
  return (
    <Routes>
      <Route path="/files" element={<PageWrapper Contents={FilesBrowsing} />} />
      <Route path="/file" element={<PageWrapper Contents={FileDetails} />} />
      <Route path="/file/:sha256" element={<PageWrapper Contents={FileDetails} />} />
      <Route path="/files/:sha256" element={<PageWrapper Contents={FileDetails} />} />
      <Route path="/create/image" element={<PageWrapper Contents={CreateImage} />} />
      <Route path="/upload" element={<PageWrapper Contents={UploadFiles} />} />
      <Route path="/repos" element={<PageWrapper Contents={RepoBrowsing} />} />
      <Route path="/repo/*" element={<PageWrapper Contents={RepoDetails} />} />
      <Route path="/reaction/:group/:reactionID" element={<PageWrapper Contents={ReactionStatus} />} />
      <Route path="/reactions/:group/:reactionID" element={<PageWrapper Contents={ReactionStatus} />} />
      <Route path="/reaction/logs/:group/:reactionID/:stage" element={<PageWrapper Contents={ReactionStageLogs} />} />
      <Route path="/reactions/logs/:group/:reactionID/:stage" element={<PageWrapper Contents={ReactionStageLogs} />} />
      <Route path="/profile" element={<PageWrapper Contents={Profile} />} />
      <Route path="/pipelines" element={<PageWrapper Contents={Pipelines} />} />
      <Route path="/images" element={<PageWrapper Contents={Images} />} />
      <Route path="/groups" element={<PageWrapper Contents={Groups} />} />
      <Route path="/users" element={<PageWrapper admin Contents={Users} />} />
      <Route path="/settings" element={<PageWrapper admin Contents={SystemSettings} />} />
      <Route path="/stats" element={<PageWrapper Contents={SystemStats} />} />
      <Route path="/auth" element={<PageWrapper auth={false} Contents={Login} />} />
      <Route path="/" element={<PageWrapper Contents={Home} />} />
      <Route path="*" element={<PageWrapper Contents={NotFound} />} />
      <Route index element={<PageWrapper Contents={Home} />} />
    </Routes>
  );
};

const Body = styled.div`
  color: var(--thorium-text);
  background-color: var(--thorium-body-bg);
  min-height: 100vh;
  // removed this to enable position: sticky
  // this is needed for the results table of contents
  // overflow-x: hidden;
`;

const Thorium = () => {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Body>
          <ToastContainer
            position="top-right"
            autoClose={2000}
            hideProgressBar={true}
            newestOnTop={true}
            closeOnClick={true}
            pauseOnHover={false}
            draggable={false}
            rtl={false}
            theme="dark"
          />
          <NavBanner />
          <SidebarColumn />
          <Suspense fallback={FallbackView}>
            <ErrorBoundary fallback={<RenderErrorAlert />}>
              <Resources />
            </ErrorBoundary>
          </Suspense>
        </Body>
      </AuthProvider>
    </BrowserRouter>
  );
};

export default Thorium;
