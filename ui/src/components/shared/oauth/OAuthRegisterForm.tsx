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
  /// Whether the user must supply their own email because the provider returned no email claim
  emailRequired: boolean;
  /// Called once a new, auto-verified account is created, with the issued auth response and the
  /// account's username (the email is derived server-side from the verified OIDC claims).
  onComplete: (auth: UserAuthResponse, account: { username: string }) => void;
  /// Called when the account was created but its email must be verified before sign-in. The email
  /// (when known) comes from the server's verify-email response, not from user input.
  onVerifyEmail: (account: { username: string; email?: string }) => void;
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
const OAuthRegisterForm: React.FC<OAuthRegisterFormProps> = ({
  provider,
  sessionToken,
  emailRequired,
  onComplete,
  onVerifyEmail,
  onError,
}) => {
  const [username, setUsername] = useState('');
  // the user-supplied email, only collected when the provider returned no email claim
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

  // when the provider returned no email claim the user must supply one before we can submit
  const emailSatisfied = !emailRequired || email.trim() !== '';
  const canSubmit = username.trim() !== '' && emailSatisfied && availability !== Availability.Checking && !submitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    // only send the user-supplied email when the provider requires one; otherwise the backend
    // derives it from the verified OIDC claims
    const result = await registerOAuthUser(provider, {
      session_token: sessionToken,
      username: username.trim(),
      ...(emailRequired ? { email: email.trim() } : {}),
    });
    switch (result.status) {
      case OAuthRegisterStatus.Created:
        onComplete(result.auth, { username: username.trim() });
        break;
      case OAuthRegisterStatus.VerifyEmail:
        // the email (if any) comes from the server's verify-email response, not user input
        onVerifyEmail({ username: username.trim(), email: result.email });
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
          An account named &quot;{username.trim()}&quot; already exists. If it&apos;s yours and your {label} email matches
          it, we&apos;ll send a link to that account&apos;s email to connect your {label} sign-in.
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
      {emailRequired && (
        <Field>
          <label htmlFor="oauth-reg-email">Email</label>
          <TextInput
            id="oauth-reg-email"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />
          <Hint $tone="muted">
            Your {label} sign-in didn&apos;t provide an email, so enter one here. You&apos;ll need to
            verify it before you can sign in.
          </Hint>
        </Field>
      )}
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
