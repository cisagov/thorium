import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';
import { ToastContainer } from 'react-toastify';
import styled from 'styled-components';
import 'react-toastify/dist/ReactToastify.css';

// project imports
import NavBanner from './components/pages/NavBanner';
import SidebarColumn from './components/pages/SidebarColumn';
import RenderErrorAlert from './components/shared/alerts/RenderErrorAlert';
import { PageWrapper } from './components/pages/Page';
import { WindowManager } from './components/shared/windows/WindowManager';
import { Auth } from './utilities/auth';
import './styles/main.scss';
import { CanvasMargin } from './styles/margin';

// import pages lazily
const Home = lazy(() => import('./pages/Home'));
const NotFound = lazy(async () => import('./pages/NotFound'));
const FileDetails = lazy(async () => import('./pages/files/FileDetails'));
const DeviceDetails = lazy(async () => await import('./pages/entities/devices/DeviceDetails'));
const VendorDetails = lazy(async () => await import('./pages/entities/vendors/VendorDetails'));
const CollectionDetails = lazy(async () => await import('./pages/entities/collections/CollectionDetails'));
const FileSystemDetails = lazy(async () => await import('./pages/entities/file_systems/FileSystemDetails'));
const RepoDetails = lazy(() => import('./pages/repos/RepoDetails'));
const FilesBrowsing = lazy(() => import('./pages/files/FilesBrowsing'));
const RepoBrowsing = lazy(() => import('./pages/repos/RepoBrowsing'));
const DeviceBrowsing = lazy(async () => await import('./pages/entities/devices/DeviceBrowsing'));
const VendorBrowsing = lazy(async () => await import('./pages/entities/vendors/VendorBrowsing'));
const CollectionBrowsing = lazy(async () => await import('./pages/entities/collections/CollectionBrowsing'));
const FileSystemBrowsing = lazy(async () => await import('./pages/entities/file_systems/FileSystemBrowsing'));
const CreateDevice = lazy(async () => await import('./pages/entities/devices/DeviceCreate'));
const CreateVendor = lazy(async () => await import('./pages/entities/vendors/VendorCreate'));
const CreateCollection = lazy(async () => await import('./pages/entities/collections/CollectionCreate'));
const CreateImage = lazy(() => import('./pages/images/ImageCreate'));
const GraphBuilder = lazy(async () => await import('./pages/GraphBuilder'));
const UploadFiles = lazy(() => import('./pages/files/FileUpload'));
const Pipelines = lazy(() => import('./pages/Pipelines'));
const Images = lazy(() => import('./pages/images/ImageBrowsing'));
const Groups = lazy(() => import('./pages/users/Groups'));
const Users = lazy(() => import('./pages/users/UserBrowsing'));
const ReactionStatus = lazy(() => import('./pages/reactions/ReactionStatus'));
const ReactionStageLogs = lazy(() => import('./pages/reactions/ReactionStageLogs'));
const Login = lazy(() => import('./pages/Login'));
const Profile = lazy(() => import('./pages/users/UserProfile'));
const SystemStats = lazy(() => import('./pages/system/SystemStats'));
const SystemSettings = lazy(() => import('./pages/system/SystemSettings'));

// Data loading ui empty for now
const FallbackView = <h1 />;

const Resources = () => (
  <Routes>
    <Route path="/files" element={<PageWrapper Contents={FilesBrowsing} />} />
    <Route path="/file" element={<PageWrapper Contents={FileDetails} />} />
    <Route path="/file/:sha256" element={<PageWrapper Contents={FileDetails} />} />
    <Route path="/files/:sha256" element={<PageWrapper Contents={FileDetails} />} />
    <Route path="/devices" element={<PageWrapper Contents={DeviceBrowsing} />} />
    <Route path="/devices/" element={<PageWrapper Contents={DeviceBrowsing} />} />
    <Route path="/device" element={<PageWrapper Contents={DeviceDetails} />} />
    <Route path="/device/:entityID" element={<PageWrapper Contents={DeviceDetails} />} />
    <Route path="/vendors" element={<PageWrapper Contents={VendorBrowsing} />} />
    <Route path="/vendor" element={<PageWrapper Contents={VendorBrowsing} />} />
    <Route path="/vendor/:entityID" element={<PageWrapper Contents={VendorDetails} />} />
    <Route path="/collections" element={<PageWrapper Contents={CollectionBrowsing} />} />
    <Route path="/collections/" element={<PageWrapper Contents={CollectionBrowsing} />} />
    <Route path="/collection/:entityID" element={<PageWrapper Contents={CollectionDetails} />} />
    <Route path="/filesystems" element={<PageWrapper Contents={FileSystemBrowsing} />} />
    <Route path="/filesystem" element={<PageWrapper Contents={FileSystemBrowsing} />} />
    <Route path="/filesystem/:entityID" element={<PageWrapper Contents={FileSystemDetails} />} />
    <Route path="/create/vendor" element={<PageWrapper Contents={CreateVendor} />} />
    <Route path="/create/device" element={<PageWrapper Contents={CreateDevice} />} />
    <Route path="/create/collection" element={<PageWrapper Contents={CreateCollection} />} />
    <Route path="/create/image" element={<PageWrapper Contents={CreateImage} />} />
    <Route path="/graph" element={<PageWrapper Contents={GraphBuilder} />} />
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

const Body = styled.div`
  color: var(--thorium-text);
  background-color: var(--thorium-body-bg);
  min-height: 100vh;
  // removed this to enable position: sticky
  // this is needed for the results table of contents
  // overflow-x: hidden;
`;

const Site = () => (
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
);

const Thorium = () => (
  <BrowserRouter>
    <Auth>
      <WindowManager name="thorium" zRange={{ start: 1000, end: 4000, step: 5 }} canvasMargin={CanvasMargin}>
        <Site />
      </WindowManager>
    </Auth>
  </BrowserRouter>
);

export default Thorium;
