import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';

// project imports
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import Button from '@components/shared/buttons/Button';
import { BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import { resendVerificationEmail } from '@thorpi/users';
import { ResendVerificationStatus } from '@models/users';

const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
`;

const Actions = styled.div`
  display: flex;
  gap: ${BUTTON_BAR_GAP};
  width: 100%;

  /* equal-width buttons so Resend and Login line up uniformly */
  & > * {
    flex: 1 1 0;
  }
`;

/**
 * Decrement a resend cooldown by one tick, clamped at zero. Exported for unit testing.
 *
 * @param seconds - The current cooldown in seconds.
 * @returns The cooldown after one second elapses, never below zero.
 */
export function tickCooldown(seconds: number): number {
  return Math.max(0, seconds - 1);
}

/**
 * Build the resend button label for the current cooldown / in-flight state. Exported for unit testing.
 *
 * @param cooldownSecs - Seconds remaining before another resend is allowed (0 = allowed now).
 * @param sending - Whether a resend request is currently in flight.
 * @returns The label to show on the resend button.
 */
export function resendButtonLabel(cooldownSecs: number, sending: boolean): string {
  if (sending) {
    return 'Sending…';
  }
  if (cooldownSecs > 0) {
    return `Resend in ${cooldownSecs}s`;
  }
  return 'Resend';
}

interface VerifyEmailNoticeProps {
  /// The username whose verification email can be resent. When omitted the resend control is hidden
  /// (e.g. an OAuth login where only the email is known, not the username).
  username?: string;
  /// The email address the verification link was sent to; shown in the message when known.
  email?: string;
  /// Overrides the Login button's default `navigate('/auth')`. Required when this notice is
  /// rendered ON the login page itself: same-route navigation doesn't remount the page, so the
  /// parent must reset its own state to bring the login form back.
  onLogin?: () => void;
}

/**
 * Shared "verify your email" view used wherever a user lands in the unverified state (after
 * registration, after an unverified password login, or after an OAuth flow). Shows the verification
 * message and — when the username is known — a Resend button that respects the server's resend
 * cooldown via a live countdown (seeded from the `Retry-After` header on both 200 and 429 responses).
 */
const VerifyEmailNotice: React.FC<VerifyEmailNoticeProps> = ({ username, email, onLogin }) => {
  const navigate = useNavigate();
  const [cooldown, setCooldown] = useState(0);
  const [sending, setSending] = useState(false);
  // the latest resend outcome to surface; null falls back to the standing "check your inbox" notice.
  // Tracking a single slot (not separate info + error) guarantees only the most recent message shows.
  const [message, setMessage] = useState<{ text: string; severity: Severity } | null>(null);
  // set once the API reports the email is already verified — resend is no longer offered
  const [alreadyVerified, setAlreadyVerified] = useState(false);

  // tick the resend cooldown down to zero
  useEffect(() => {
    if (cooldown <= 0) {
      return;
    }
    const timer = setInterval(() => setCooldown((prev) => tickCooldown(prev)), 1000);
    return () => clearInterval(timer);
  }, [cooldown]);

  const handleResend = async () => {
    if (!username || sending || cooldown > 0 || alreadyVerified) {
      return;
    }
    setSending(true);
    // clear the previous outcome up front so this request's response fully replaces it — a success
    // after a transient error must not be masked by the stale error.
    setMessage(null);
    const result = await resendVerificationEmail(username, (text) => setMessage({ text, severity: Severity.Error }));
    if (result.status === ResendVerificationStatus.Sent) {
      // confirm the fresh send and seed the resend cooldown from the Retry-After window
      setCooldown(result.retryAfterSecs);
      setMessage({
        text: email ? `We sent a new verification link to ${email}.` : 'We sent a new verification link. Check your inbox.',
        severity: Severity.Success,
      });
    } else if (result.status === ResendVerificationStatus.Cooldown) {
      // rate-limited: just seed the countdown; the error message is already set via the handler above
      setCooldown(result.retryAfterSecs);
    } else if (result.status === ResendVerificationStatus.AlreadyVerified) {
      // stop offering resend; the "already verified" message is surfaced via the error handler above
      setAlreadyVerified(true);
    }
    // Failed: the error is already surfaced via the error handler above
    setSending(false);
  };

  return (
    <Wrapper>
      {/* Only the latest message shows: the most recent resend outcome (success or error) replaces the
          standing "check your inbox" notice rather than stacking with it. */}
      {message ? (
        <AlertBanner severity={message.severity}>{message.text}</AlertBanner>
      ) : (
        <AlertBanner severity={Severity.Success}>
          {email
            ? `We sent a verification link to ${email}. Verify your email, then please sign in.`
            : 'Your email address needs to be verified. Check your inbox for the verification link, then sign in.'}
        </AlertBanner>
      )}
      <Actions>
        {username && (
          <Button
            variant={ButtonVariant.Secondary}
            size={ButtonSize.Large}
            disabled={sending || cooldown > 0 || alreadyVerified}
            onClick={() => void handleResend()}
          >
            {resendButtonLabel(cooldown, sending)}
          </Button>
        )}
        <Button variant={ButtonVariant.Primary} size={ButtonSize.Large} onClick={() => (onLogin ? onLogin() : void navigate('/auth'))}>
          Login
        </Button>
      </Actions>
    </Wrapper>
  );
};

export default VerifyEmailNotice;
