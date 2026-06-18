import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

// project imports
import { getProviderMeta } from './providerMeta';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import TextInput from '@components/shared/inputs/TextInput';
import Subtitle from '@components/shared/titles/Subtitle';
import { checkOAuthUsername, registerOAuthUser } from '@thorpi/oauth';
import { OAuthRegisterStatus } from '@models/oauth';
import { UserAuthResponse } from '@models/users';

interface OAuthRegisterFormProps {
  /// The provider that authenticated the user
  provider: string;
  /// The registration session token from the callback `NewUser` response (kept in memory only)
  sessionToken: string;
  /// Called once a new, auto-verified account is created, with the issued auth response and the
  /// account's username/email (so a follow-up verify-email step can resend to the right account).
  onComplete: (auth: UserAuthResponse, account: { username: string; email: string }) => void;
  /// Called when the account was created but its email must be verified before sign-in.
  onVerifyEmail: (account: { username: string; email: string }) => void;
  /// Called with a formatted message if registration fails (e.g. email belongs to a different user)
  onError: (msg: string) => void;
}

// Live username-availability state. A "taken" username switches the form into account-link mode.
enum Availability {
  Idle = 'idle',
  Checking = 'checking',
  Available = 'available',
  Taken = 'taken',
}

const FormWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
`;

const Field = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
`;

const Hint = styled.span<{ $tone: 'ok' | 'muted' }>`
  font-size: 0.85rem;
  color: ${({ $tone }) => ($tone === 'ok' ? 'var(--thorium-ok-bg)' : 'var(--thorium-secondary-text)')};
`;

const Actions = styled.div`
  display: flex;
  justify-content: center;
  padding-top: 0.5rem;
`;

const USERNAME_CHECK_DEBOUNCE_MS = 400;

/**
 * First-time OAuth identity form. If the chosen username is available it creates a new Thorium account;
 * if it is already taken it switches to "link mode", letting an existing local/LDAP user connect this
 * provider to their account via the email-link flow. Role/verification fields are not exposed (the
 * backend forces role=User).
 */
const OAuthRegisterForm: React.FC<OAuthRegisterFormProps> = ({ provider, sessionToken, onComplete, onVerifyEmail, onError }) => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [availability, setAvailability] = useState<Availability>(Availability.Idle);
  const [submitting, setSubmitting] = useState(false);
  // Terminal state after an account-link email has been sent.
  const [linkSent, setLinkSent] = useState(false);
  // Monotonic id so stale debounced availability responses are ignored.
  const checkIdRef = useRef(0);
  const label = getProviderMeta(provider).label;
  const isLinkMode = availability === Availability.Taken;

  // Debounced username availability check against the backend.
  useEffect(() => {
    const trimmed = username.trim();
    if (!trimmed) {
      setAvailability(Availability.Idle);
      return;
    }
    setAvailability(Availability.Checking);
    const myId = ++checkIdRef.current;
    const timer = setTimeout(() => {
      void checkOAuthUsername(provider, { username: trimmed, session_token: sessionToken }, onError).then((available) => {
        // Ignore responses superseded by a newer keystroke.
        if (myId !== checkIdRef.current) return;
        setAvailability(available ? Availability.Available : Availability.Taken);
      });
    }, USERNAME_CHECK_DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // onError identity is not stable across renders; intentionally excluded from deps.
  }, [username, provider, sessionToken]);

  const canSubmit = username.trim() !== '' && email.trim() !== '' && availability !== Availability.Checking && !submitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    const result = await registerOAuthUser(provider, { session_token: sessionToken, username: username.trim(), email: email.trim() });
    switch (result.status) {
      case OAuthRegisterStatus.Created:
        onComplete(result.auth, { username: username.trim(), email: email.trim() });
        break;
      case OAuthRegisterStatus.VerifyEmail:
        onVerifyEmail({ username: username.trim(), email: email.trim() });
        break;
      case OAuthRegisterStatus.LinkEmailSent:
        setLinkSent(true);
        setSubmitting(false);
        break;
      default:
        onError(result.message);
        setSubmitting(false);
        break;
    }
  };

  // Terminal success: the server emailed an account-link link.
  if (linkSent) {
    return (
      <FormWrapper>
        <AlertBanner severity={Severity.Success}>
          Check your email to finish linking your {label} sign-in to &quot;{username.trim()}&quot;.
        </AlertBanner>
      </FormWrapper>
    );
  }

  const usernameHint = () => {
    switch (availability) {
      case Availability.Checking:
        return <Hint $tone="muted">Checking availability…</Hint>;
      case Availability.Available:
        return <Hint $tone="ok">Username is available</Hint>;
      default:
        return null;
    }
  };

  return (
    <FormWrapper>
      <Subtitle>Link {label} to a Thorium account</Subtitle>
      {isLinkMode && (
        <AlertBanner severity={Severity.Warning}>
          An account named &quot;{username.trim()}&quot; already exists. If it&apos;s yours, enter the email on that account and we&apos;ll
          send a link to connect your {label} sign-in.
        </AlertBanner>
      )}
      <Field>
        <label htmlFor="oauth-reg-username">Username</label>
        <TextInput
          id="oauth-reg-username"
          type="text"
          placeholder="Choose a username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
        />
        {usernameHint()}
      </Field>
      <Field>
        <label htmlFor="oauth-reg-email">Email</label>
        <TextInput
          id="oauth-reg-email"
          type="email"
          placeholder="Enter your email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
        />
      </Field>
      <Actions>
        {submitting ? (
          <LoadingSpinner loading={submitting} />
        ) : (
          <Button
            variant={isLinkMode ? ButtonVariant.Warning : ButtonVariant.Ok}
            size={ButtonSize.Large}
            disabled={!canSubmit}
            onClick={() => void handleSubmit()}
          >
            {isLinkMode ? 'Link Account' : 'Create account'}
          </Button>
        )}
      </Actions>
    </FormWrapper>
  );
};

export default OAuthRegisterForm;
