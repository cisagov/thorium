import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

// project imports
import { CardBox, Centered, Message } from './shared.styled';
import Page from '@components/pages/Page';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import { VerifyEmailNotice } from '@components/shared/auth';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { OAuthRegisterForm } from '@components/shared/oauth';
import SimpleTitle from '@components/shared/titles/SimpleTitle';
import { exchangeOAuthCallback } from '@thorpi/oauth';
import { whoami } from '@thorpi/users';
import { useAuth } from '@utilities/auth';
import { consumeOAuthReturn } from '@utilities/oauthReturn';
import { isAuthed, isNewUser, isVerifyEmail } from '@models/oauth';
import { UserAuthResponse } from '@models/users';

// Where the callback flow currently is. The Authed (login) outcome navigates away, so it has no state.
enum CallbackStatus {
  Loading = 'loading',
  NeedsRegistration = 'needs_registration',
  PendingVerification = 'pending_verification',
  Error = 'error',
}

/**
 * Landing page the IdP redirects back to after authorization. Exchanges the `code`/`state` for a
 * session (existing user) or a registration token (new user), then either logs the user in or
 * shows the OAuth registration form.
 */
const OAuthCallback = () => {
  const { provider = '' } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { completeOAuth } = useAuth();

  const [status, setStatus] = useState<CallbackStatus>(CallbackStatus.Loading);
  const [errorMsg, setErrorMsg] = useState('');
  const [formError, setFormError] = useState('');
  const [sessionToken, setSessionToken] = useState('');
  // username/email for the verify-email notice (username known only when we just registered here)
  const [verifyInfo, setVerifyInfo] = useState<{ username?: string; email?: string }>({});
  // Guard against the mount effect running the (single-use) exchange twice under StrictMode.
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    const code = searchParams.get('code');
    const state = searchParams.get('state');
    const providerError = searchParams.get('error');

    // The IdP redirects back with ?error=... and no code when the user cancels or is denied.
    if (providerError || !code || !state) {
      setErrorMsg(
        providerError ? 'Sign-in was cancelled or denied by the provider.' : 'This sign-in link is missing required information.',
      );
      setStatus(CallbackStatus.Error);
      return;
    }

    void (async () => {
      const result = await exchangeOAuthCallback(provider, code, state, setErrorMsg);
      if (!result) {
        setStatus(CallbackStatus.Error);
        return;
      }
      if (isAuthed(result)) {
        completeOAuth(result.Authed.token, result.Authed.expires);
        void navigate(consumeOAuthReturn(), { replace: true });
        return;
      }
      // an existing linked account whose email isn't verified yet — no token is issued
      if (isVerifyEmail(result)) {
        setVerifyInfo({ email: result.VerifyEmail });
        setStatus(CallbackStatus.PendingVerification);
        return;
      }
      if (isNewUser(result)) {
        setSessionToken(result.NewUser);
        setStatus(CallbackStatus.NeedsRegistration);
      }
    })();
    // Run exactly once on mount; provider/searchParams are read synchronously above.
  }, []);

  // After registration, the issued token may be unusable until the account's email is verified.
  // Probe whoami to distinguish "logged in" from "must verify first".
  const handleRegisterComplete = async (auth: UserAuthResponse, account: { username: string; email: string }) => {
    completeOAuth(auth.token, auth.expires);
    const user = await whoami();
    if (user) {
      void navigate(consumeOAuthReturn(), { replace: true });
    } else {
      setVerifyInfo(account);
      setStatus(CallbackStatus.PendingVerification);
    }
  };

  const renderBody = () => {
    switch (status) {
      case CallbackStatus.Loading:
        return (
          <>
            <LoadingSpinner loading={true} />
            <Message>Completing sign-in…</Message>
          </>
        );
      case CallbackStatus.NeedsRegistration:
        return (
          <>
            {formError != '' && <AlertBanner>{formError}</AlertBanner>}
            <OAuthRegisterForm
              provider={provider}
              sessionToken={sessionToken}
              onComplete={(auth, account) => void handleRegisterComplete(auth, account)}
              onVerifyEmail={(account) => {
                setVerifyInfo(account);
                setStatus(CallbackStatus.PendingVerification);
              }}
              onError={setFormError}
            />
          </>
        );
      case CallbackStatus.PendingVerification:
        return <VerifyEmailNotice username={verifyInfo.username} email={verifyInfo.email} />;
      case CallbackStatus.Error:
      default:
        return (
          <>
            <AlertBanner>{errorMsg || 'Sign-in failed. Please try again.'}</AlertBanner>
            <Button variant={ButtonVariant.Primary} size={ButtonSize.Large} onClick={() => void navigate('/auth', { replace: true })}>
              Try again
            </Button>
          </>
        );
    }
  };

  return (
    <Page title="Sign in · Thorium">
      <Centered>
        <CardBox>
          <SimpleTitle>Sign in to Thorium</SimpleTitle>
          {renderBody()}
        </CardBox>
      </Centered>
    </Page>
  );
};

export default OAuthCallback;
