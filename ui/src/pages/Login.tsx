import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import styled from 'styled-components';

// project imports
import Page from '@components/pages/Page';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import { VerifyEmailNotice } from '@components/shared/auth';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import TextInput from '@components/shared/inputs/TextInput';
import { OAuthProviderButtons } from '@components/shared/oauth';
import SimpleSubtitle from '@components/shared/titles/SimpleSubtitle';
import SimpleTitle from '@components/shared/titles/SimpleTitle';
import { getBanner } from '@thorpi/base';
import { buildOAuthAuthUrl, listOAuthProviders } from '@thorpi/oauth';
import { LoginOutcome, useAuth } from '@utilities/auth';
import { stashOAuthReturn } from '@utilities/oauthReturn';

interface LocationState {
  path?: string;
}

const LoginLayout = styled.div`
  display: flex;
  justify-content: center;
  padding-top: 2rem;
`;

// Sizes to its content so the ASCII banner (up to .banner's 50rem) isn't squished; the interactive
// controls are constrained separately by FormColumn.
const LoginCard = styled.div`
  max-width: 50rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 2rem;
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

// Keeps the inputs/buttons at the legacy login width (.login = 25rem) even when the banner is wider.
const FormColumn = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  width: 25rem;
  max-width: 100%;
`;

const FieldStack = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  width: 100%;
`;

const OrDivider = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  color: var(--thorium-secondary-text);
  font-size: 0.85rem;
  text-transform: uppercase;

  &::before,
  &::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid var(--thorium-panel-border);
  }
`;

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginErr, setLoginErr] = useState('');
  const [banner, setBanner] = useState('');
  const [loggingIn, setLoggingIn] = useState(false);
  const [providers, setProviders] = useState<string[]>([]);
  // set to the username once a login is rejected for an unverified email; shows the verify-email notice
  const [unverifiedUser, setUnverifiedUser] = useState<string | null>(null);
  const navigate = useNavigate();
  const { state } = useLocation() as { state: LocationState | null };
  const { login } = useAuth();

  // login to Thorium and redirect if successful
  const handleAuthFormSubmit = async (username: string, password: string, handleAuthErr: (error: string) => void) => {
    setLoggingIn(true);
    setLoginErr('');
    const outcome = await login(username, password, handleAuthErr);
    if (outcome === LoginOutcome.LoggedIn) {
      void navigate(state?.path || '/');
    } else if (outcome === LoginOutcome.VerifyEmail) {
      // valid credentials but the email is unverified — surface the verify-email notice (with resend)
      setUnverifiedUser(username);
      setLoggingIn(false);
    } else {
      setLoggingIn(false);
    }
  };

  // handle all key presses
  const handleFormKeyPress = async (e: React.KeyboardEvent<HTMLElement>) => {
    // key code 13 is enter
    if (e.keyCode === 13) {
      await handleAuthFormSubmit(username, password, setLoginErr);
    }
  };

  // start an OAuth login: stash where to return, then hand the browser off to the IdP.
  // Must be a full-page navigation (not an XHR) so the 303 redirect to the provider is followed.
  const handleOAuthSelect = (provider: string) => {
    stashOAuthReturn(state?.path);
    window.location.assign(buildOAuthAuthUrl(provider));
  };

  // fetch banner and set state
  const fetchBanner = async () => {
    const req = await getBanner(setBanner);
    if (req) {
      setBanner(req);
    }
  };

  // grab the banner and any configured OAuth providers on page load
  useEffect(() => {
    void fetchBanner();
    void listOAuthProviders(console.log).then((list) => {
      if (list) {
        setProviders(list);
      }
    });
  }, []);

  return (
    <Page title="Login · Thorium">
      <LoginLayout>
        <LoginCard>
          <SimpleTitle>Welcome to Thorium!</SimpleTitle>
          {banner != null && banner != '' && (
            <pre className="banner">
              <center>{String(banner)}</center>
            </pre>
          )}
          <FormColumn>
            {unverifiedUser !== null ? (
              <VerifyEmailNotice username={unverifiedUser} onLogin={() => setUnverifiedUser(null)} />
            ) : (
              <>
                {providers.length > 0 && (
                  <>
                    <OAuthProviderButtons providers={providers} onSelect={handleOAuthSelect} disabled={loggingIn} />
                    <OrDivider>or</OrDivider>
                  </>
                )}
                <FieldStack>
                  <TextInput
                    type="text"
                    value={username}
                    placeholder="username"
                    autoComplete="username"
                    onChange={(e) => setUsername(String(e.target.value))}
                    onKeyDown={(e) => {
                      void handleFormKeyPress(e);
                    }}
                  />
                  <TextInput
                    type="password"
                    value={password}
                    placeholder="password"
                    autoComplete="current-password"
                    onChange={(e) => setPassword(String(e.target.value))}
                    onKeyDown={(e) => {
                      void handleFormKeyPress(e);
                    }}
                  />
                </FieldStack>
                {loggingIn ? (
                  <LoadingSpinner loading={loggingIn} />
                ) : (
                  <>
                    <SimpleSubtitle>
                      New user? Create an&nbsp;
                      <Link to="/register" state={state}>
                        account
                      </Link>
                      .
                    </SimpleSubtitle>
                    {loginErr != '' && <AlertBanner>{loginErr}</AlertBanner>}
                    <Button
                      variant={ButtonVariant.Primary}
                      size={ButtonSize.Large}
                      onClick={() => {
                        void handleAuthFormSubmit(username, password, setLoginErr);
                      }}
                    >
                      Login
                    </Button>
                  </>
                )}
              </>
            )}
          </FormColumn>
        </LoginCard>
      </LoginLayout>
    </Page>
  );
};

export default Login;
