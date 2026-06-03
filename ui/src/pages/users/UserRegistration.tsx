// spec: ./SPEC.md
import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styled from 'styled-components';

// project imports
import Page from '@components/pages/Page';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { VerifyEmailNotice } from '@components/shared/auth';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import TextInput from '@components/shared/inputs/TextInput';
import SimpleTitle from '@components/shared/titles/SimpleTitle';
import Subtitle from '@components/shared/titles/Subtitle';
import { RegisterOutcome, useAuth } from '@utilities/auth';

interface LocationState {
  path?: string;
}

const Layout = styled.div`
  display: flex;
  justify-content: center;
  padding-top: 2rem;
`;

const Card = styled.div`
  width: 34rem;
  max-width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 2rem;
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

const FieldStack = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  width: 100%;
`;

/**
 * Standalone account-registration page (replaces the former login modal). On success it either logs
 * the user in (auto-verified deployments) or shows the shared {@link VerifyEmailNotice} so the user
 * can verify their email — with a resend button + cooldown — and then sign in.
 */
const UserRegistration = () => {
  const navigate = useNavigate();
  const { state } = useLocation() as { state: LocationState | null };
  const { register } = useAuth();

  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  // true once a submit found the two password entries didn't match; highlights both fields in red
  const [passwordMismatch, setPasswordMismatch] = useState(false);
  const [warning, setWarning] = useState('');
  const [error, setError] = useState('');
  const [registering, setRegistering] = useState(false);
  // set to the registered email once registration requires email verification
  const [verifyEmail, setVerifyEmail] = useState<string | null>(null);

  const handleRegister = async () => {
    setWarning('');
    setError('');
    setPasswordMismatch(false);
    // require all fields; email is needed so the account can be verified
    if (!username || !email || !password) {
      setWarning('You must specify a username, email, and password!');
      return;
    }
    // both password entries must match before we create the account
    if (password !== confirmPassword) {
      setPasswordMismatch(true);
      setWarning('The entered passwords do not match.');
      return;
    }
    setRegistering(true);
    const outcome = await register(username, password, setError, email, 'User');
    if (outcome === RegisterOutcome.LoggedIn) {
      // auto-verified deployment: the account is usable immediately
      void navigate(state?.path || '/');
    } else if (outcome === RegisterOutcome.VerifyEmail) {
      // account created but the user must verify their email before signing in
      setVerifyEmail(email);
      setRegistering(false);
    } else {
      // registration failed; the error is already surfaced via setError
      setRegistering(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLElement>) => {
    // key code 13 is enter
    if (e.keyCode === 13) {
      void handleRegister();
    }
  };

  return (
    <Page title="Register · Thorium">
      <Layout>
        <Card>
          <SimpleTitle>Create an account</SimpleTitle>
          {verifyEmail !== null ? (
            <VerifyEmailNotice username={username} email={verifyEmail} />
          ) : (
            <>
              <FieldStack>
                <Subtitle>Username</Subtitle>
                <TextInput
                  type="text"
                  placeholder="Enter username"
                  value={username}
                  autoComplete="username"
                  onChange={(e) => setUsername(String(e.target.value))}
                  onKeyDown={handleKeyDown}
                />
                <Subtitle>Email</Subtitle>
                <TextInput
                  type="email"
                  placeholder="Enter email"
                  value={email}
                  autoComplete="email"
                  onChange={(e) => setEmail(String(e.target.value))}
                  onKeyDown={handleKeyDown}
                />
                <Subtitle>Password</Subtitle>
                <TextInput
                  type="password"
                  placeholder="Enter password"
                  value={password}
                  autoComplete="new-password"
                  $invalid={passwordMismatch}
                  onChange={(e) => {
                    setPassword(String(e.target.value));
                    setPasswordMismatch(false);
                  }}
                  onKeyDown={handleKeyDown}
                />
                <Subtitle>Confirm password</Subtitle>
                <TextInput
                  type="password"
                  placeholder="Re-enter password"
                  value={confirmPassword}
                  autoComplete="new-password"
                  $invalid={passwordMismatch}
                  onChange={(e) => {
                    setConfirmPassword(String(e.target.value));
                    setPasswordMismatch(false);
                  }}
                  onKeyDown={handleKeyDown}
                />
              </FieldStack>
              {warning != '' && <AlertBanner severity={Severity.Warning}>{warning}</AlertBanner>}
              {error != '' && <AlertBanner>{error}</AlertBanner>}
              {registering ? (
                <LoadingSpinner loading={registering} />
              ) : (
                <Button variant={ButtonVariant.Ok} size={ButtonSize.Large} onClick={() => void handleRegister()}>
                  Create account
                </Button>
              )}
            </>
          )}
        </Card>
      </Layout>
    </Page>
  );
};

export default UserRegistration;
