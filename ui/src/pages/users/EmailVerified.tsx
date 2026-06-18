import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

// project imports
import { CardBox, Centered } from '../oauth/shared.styled';
import Page from '@components/pages/Page';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import SimpleTitle from '@components/shared/titles/SimpleTitle';

/**
 * Terminal landing page the catch page navigates to after its verify XHR: `?status=ok` confirms the
 * email is verified; `?status=expired` covers an invalid/used token. Mirrors the OAuth linked page.
 */
const EmailVerified = () => {
  const { username = '' } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const expired = searchParams.get('status') === 'expired';

  return (
    <Page title="Email verified · Thorium">
      <Centered>
        <CardBox>
          <SimpleTitle>Email verification</SimpleTitle>
          {expired ? (
            <AlertBanner>This verification link has expired or was already used. Sign in to request a new one.</AlertBanner>
          ) : (
            <AlertBanner severity={Severity.Success}>
              Your email is verified{username ? ` for "${username}"` : ''}. You can sign in now.
            </AlertBanner>
          )}
          <Button variant={ButtonVariant.Primary} size={ButtonSize.Large} onClick={() => void navigate('/auth', { replace: true })}>
            Go to sign in
          </Button>
        </CardBox>
      </Centered>
    </Page>
  );
};

export default EmailVerified;
