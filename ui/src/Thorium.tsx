import { lazy, Suspense } from 'react';
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
import { EntityBrowsingRoutes } from '@components/entities/browsing/EntityBrowsingRoutes';
import { EntityDetailsRoutes } from '@components/entities/details/EntityDetailsRoutes';
import { EntityCreateRoutes } from '@components/entities/create/EntityCreateRoutes';
import { Auth } from './utilities/auth';
import '@styles/main.scss';
import { CanvasMargin } from './styles/margin';

// import pages lazily
const Home = lazy(async () => await import('./pages/Home'));
const NotFound = lazy(async () => await import('./pages/NotFound'));
const EntityBrowsing = lazy(async () => await import('./pages/entities/EntityBrowsing'));
const EntityDetails = lazy(async () => await import('./pages/entities/EntityDetails'));
const EntityCreate = lazy(async () => await import('./pages/entities/EntityCreate'));
const CreateImage = lazy(async () => await import('./pages/images/ImageCreate'));
const GraphBuilder = lazy(async () => await import('./pages/GraphBuilder'));
const Pipelines = lazy(async () => await import('./pages/pipelines/PipelineBrowsing'));
const Images = lazy(async () => await import('./pages/images/ImageBrowsing'));
const Groups = lazy(async () => await import('./pages/users/Groups'));
const Users = lazy(async () => await import('./pages/users/UserBrowsing'));
const ReactionStatus = lazy(async () => await import('./pages/reactions/ReactionStatus'));
const ReactionStageLogs = lazy(async () => await import('./pages/reactions/ReactionStageLogs'));
const Login = lazy(async () => await import('./pages/Login'));
const UserRegistration = lazy(async () => await import('./pages/users/UserRegistration'));
const OAuthCallback = lazy(async () => await import('./pages/oauth/OAuthCallback'));
const OAuthLink = lazy(async () => await import('./pages/oauth/OAuthLink'));
const OAuthLinked = lazy(async () => await import('./pages/oauth/OAuthLinked'));
const EmailVerify = lazy(async () => await import('./pages/users/EmailVerify'));
const EmailVerified = lazy(async () => await import('./pages/users/EmailVerified'));
const Profile = lazy(async () => await import('./pages/users/UserProfile'));
const SystemStats = lazy(async () => await import('./pages/system/SystemStats'));
const SystemSettings = lazy(async () => await import('./pages/system/SystemSettings'));
// test pages
const SigmaTest = lazy(() => import('./pages/test/code/SigmaTest'));
const YaraTest = lazy(() => import('./pages/test/code/YaraTest'));
const AlertBannerTest = lazy(() => import('./pages/test/AlertBannerTest'));
const ImagePipelineTest = lazy(() => import('./pages/test/code/ImagePipelineTest'));
const OverlayWindowTest = lazy(() => import('./pages/test/OverlayWindowTest'));
const ButtonTest = lazy(() => import('./pages/test/ButtonTest'));

// Data loading ui empty for now
const FallbackView = <h1 />;

const Resources = () => (
  <Routes>
    // Entities
    {Object.keys(EntityBrowsingRoutes).map((path) => (
      <Route path={path} element={<PageWrapper Contents={EntityBrowsing} />} />
    ))}
    {Object.keys(EntityDetailsRoutes).map((path) => (
      <Route
        path={path}
        element={
          <PageWrapper Contents={EntityDetailsRoutes[path].override_page ? EntityDetailsRoutes[path].override_page : EntityDetails} />
        }
      />
    ))}
    {Object.keys(EntityCreateRoutes).map((path) => (
      <Route
        path={path}
        element={<PageWrapper Contents={EntityCreateRoutes[path].override_page ? EntityCreateRoutes[path].override_page : EntityCreate} />}
      />
    ))}
    // Graph Builder
    <Route path="/graph" element={<PageWrapper Contents={GraphBuilder} />} />
    // Reactions
    <Route path="/reaction/:group/:reactionID" element={<PageWrapper Contents={ReactionStatus} />} />
    <Route path="/reactions/:group/:reactionID" element={<PageWrapper Contents={ReactionStatus} />} />
    <Route path="/reaction/logs/:group/:reactionID/:stage" element={<PageWrapper Contents={ReactionStageLogs} />} />
    <Route path="/reactions/logs/:group/:reactionID/:stage" element={<PageWrapper Contents={ReactionStageLogs} />} />
    <Route path="/stats" element={<PageWrapper Contents={SystemStats} />} />
    // Users
    <Route path="/profile" element={<PageWrapper Contents={Profile} />} />
    <Route path="/users" element={<PageWrapper admin Contents={Users} />} />
    <Route path="/groups" element={<PageWrapper Contents={Groups} />} />
    <Route path="/auth" element={<PageWrapper auth={false} Contents={Login} />} />
    <Route path="/register" element={<PageWrapper auth={false} Contents={UserRegistration} />} />
    // OAuth (browser-redirect flow): IdP redirects here, then the SPA calls /api/oauth/...
    <Route path="/oauth/:provider/callback" element={<PageWrapper auth={false} Contents={OAuthCallback} />} />
    <Route path="/oauth/:provider/link" element={<PageWrapper auth={false} Contents={OAuthLink} />} />
    <Route path="/oauth/:provider/linked" element={<PageWrapper auth={false} Contents={OAuthLinked} />} />
    // Account-creation email verification (browser-redirect catch page, like the OAuth link page)
    <Route path="/users/verify/:username/email/:token" element={<PageWrapper auth={false} Contents={EmailVerify} />} />
    <Route path="/users/verify/:username/verified" element={<PageWrapper auth={false} Contents={EmailVerified} />} />
    // Pipelines and Image settings
    <Route path="/pipelines" element={<PageWrapper Contents={Pipelines} />} />
    <Route path="/images" element={<PageWrapper Contents={Images} />} />
    <Route path="/create/image" element={<PageWrapper Contents={CreateImage} />} />
    // System
    <Route path="/settings" element={<PageWrapper admin Contents={SystemSettings} />} />
    // Testing
    <Route path="/test/sigma" element={<PageWrapper Contents={SigmaTest} />} />
    <Route path="/test/yara" element={<PageWrapper Contents={YaraTest} />} />
    <Route path="/test/alerts" element={<PageWrapper Contents={AlertBannerTest} />} />
    <Route path="/test/image-pipeline" element={<PageWrapper Contents={ImagePipelineTest} />} />
    <Route path="/test/overlay-window" element={<PageWrapper Contents={OverlayWindowTest} />} />
    <Route path="/test/buttons" element={<PageWrapper Contents={ButtonTest} />} />
    // Basic
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
