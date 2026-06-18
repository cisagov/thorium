import { useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import styled from 'styled-components';

// project imports
import { CardBox, Centered, Message } from './shared.styled';
import Page from '@components/pages/Page';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import Button from '@components/shared/buttons/Button';
import { BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { getProviderMeta } from '@components/shared/oauth';
import SimpleTitle from '@components/shared/titles/SimpleTitle';
import { confirmOAuthLink, revokeOAuthLink } from '@thorpi/oauth';
import { OAuthLinkConfirmStatus } from '@models/oauth';

// Outcome of the emailed account-link confirmation.
enum LinkStatus {
  Idle = 'idle',
  Revoked = 'revoked',
  Invalid = 'invalid',
  Error = 'error',
}

// Horizontal button row using the shared toolbar gap (matches the accordion toolbar / other
// horizontal-button locations).
const Actions = styled.div`
  display: flex;
  gap: ${BUTTON_BAR_GAP};
  justify-content: center;
`;

/**
 * Landing page for the account-link URL emailed when a new OAuth sign-up collides with an existing
 * account's email. This page is intentionally inert until the user clicks: email security scanners
 * that pre-fetch the link won't consume the single-use token. Confirming calls the `/api` link endpoint
 * via XHR (which links the alias and returns 204, or a uniform 401 for an expired/used token) and then
 * navigates to the themed success page; cancelling revokes the pending token via XHR.
 */
const OAuthLink = () => {
  const { provider = '' } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const username = searchParams.get('username') ?? '';
  const token = searchParams.get('token') ?? '';
  const label = getProviderMeta(provider).label;

  const [status, setStatus] = useState<LinkStatus>(username && token ? LinkStatus.Idle : LinkStatus.Invalid);
  const [working, setWorking] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Call the API link endpoint via XHR, then land the user on the themed success page. A 204 (Linked)
  // or a uniform 401 (Expired) both navigate to the terminal page with the matching status; only an
  // unexpected failure (e.g. the network is down) keeps the user here with an error so a transient
  // outage isn't misreported as a consumed token. Only fires on an explicit user click.
  const handleConfirm = async () => {
    setWorking(true);
    setErrorMsg('');
    const result = await confirmOAuthLink(provider, username, token, setErrorMsg);
    if (result === OAuthLinkConfirmStatus.Error) {
      setWorking(false);
      setStatus(LinkStatus.Error);
      return;
    }
    const linkStatus = result === OAuthLinkConfirmStatus.Linked ? 'ok' : 'expired';
    void navigate(`/oauth/${encodeURIComponent(provider)}/linked?status=${linkStatus}`, { replace: true });
  };

  const handleCancel = async () => {
    setWorking(true);
    setErrorMsg('');
    const ok = await revokeOAuthLink(provider, username, token, setErrorMsg);
    setWorking(false);
    setStatus(ok ? LinkStatus.Revoked : LinkStatus.Error);
  };

  const renderBody = () => {
    switch (status) {
      case LinkStatus.Invalid:
        return <AlertBanner>This account-link is missing required information. Please use the link from your email.</AlertBanner>;
      case LinkStatus.Revoked:
        return <AlertBanner severity={Severity.Info}>The account-link request was cancelled. No changes were made.</AlertBanner>;
      case LinkStatus.Error:
        return (
          <>
            <AlertBanner>{errorMsg || 'This account-link has expired or was already used.'}</AlertBanner>
            <Button variant={ButtonVariant.Primary} size={ButtonSize.Large} onClick={() => void navigate('/auth', { replace: true })}>
              Go to sign in
            </Button>
          </>
        );
      case LinkStatus.Idle:
      default:
        return (
          <>
            <Message>
              Link <strong>{label}</strong> to your Thorium account <strong>{username}</strong>?
            </Message>
            {working ? (
              <LoadingSpinner loading={true} />
            ) : (
              <Actions>
                <Button variant={ButtonVariant.Ok} size={ButtonSize.Large} onClick={() => void handleConfirm()}>
                  Confirm
                </Button>
                <Button variant={ButtonVariant.Secondary} size={ButtonSize.Large} onClick={() => void handleCancel()}>
                  Cancel
                </Button>
              </Actions>
            )}
          </>
        );
    }
  };

  return (
    <Page title="Link account · Thorium">
      <Centered>
        <CardBox>
          <SimpleTitle>Link a sign-in provider</SimpleTitle>
          {renderBody()}
        </CardBox>
      </Centered>
    </Page>
  );
};

export default OAuthLink;
