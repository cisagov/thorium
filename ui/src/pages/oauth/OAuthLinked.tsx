// spec: ./SPEC.md
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

// project imports
import { CardBox, Centered } from './shared.styled';
import Page from '@components/pages/Page';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import { getProviderMeta } from '@components/shared/oauth';
import SimpleTitle from '@components/shared/titles/SimpleTitle';

/**
 * Terminal landing page after the account-link API call. The catch page navigates here after its
 * confirm XHR with `?status=ok` on success or `?status=expired` when the link token is invalid/used, so
 * the user lands on a themed page instead of a blank 204.
 */
const OAuthLinked = () => {
  const { provider = '' } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const label = getProviderMeta(provider).label;
  const expired = searchParams.get('status') === 'expired';

  return (
    <Page title="Account linked · Thorium">
      <Centered>
        <CardBox>
          <SimpleTitle>Link a sign-in provider</SimpleTitle>
          {expired ? (
            <AlertBanner>This account-link has expired or was already used. Please request a new one.</AlertBanner>
          ) : (
            <AlertBanner severity={Severity.Success}>{label} is now linked to your account. You can sign in with it next time.</AlertBanner>
          )}
          <Button variant={ButtonVariant.Primary} size={ButtonSize.Large} onClick={() => void navigate('/auth', { replace: true })}>
            Go to sign in
          </Button>
        </CardBox>
      </Centered>
    </Page>
  );
};

export default OAuthLinked;
