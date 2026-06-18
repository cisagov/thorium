import { useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

// project imports
import { CardBox, Centered, Message } from '../oauth/shared.styled';
import Page from '@components/pages/Page';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import SimpleTitle from '@components/shared/titles/SimpleTitle';
import { verifyEmail } from '@thorpi/users';
import { EmailVerifyStatus } from '@models/users';

/**
 * Catch page for the account-creation verification email link
 * (`{base_url}/users/verify/{username}/email/{token}`). It is inert until the user clicks so email
 * security scanners that pre-fetch the link don't consume the single-use token. Confirming calls the
 * `/api` verify endpoint via XHR (which verifies the email and returns 204, or a uniform 401 for an
 * expired/used token) and then navigates to the themed verified landing page.
 */
const EmailVerify = () => {
  const { username = '', token = '' } = useParams();
  const navigate = useNavigate();
  const valid = username !== '' && token !== '';

  const [working, setWorking] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Call the API verify endpoint via XHR, then land the user on the themed verified page. A 204
  // (Verified) or a uniform 401 (Expired) both navigate to the terminal page with the matching status;
  // only an unexpected failure (e.g. the network is down) keeps the user here with an error so a
  // transient outage isn't misreported as a consumed token. Only fires on an explicit user click.
  const handleConfirm = async () => {
    setWorking(true);
    setErrorMsg('');
    const result = await verifyEmail(username, token, setErrorMsg);
    if (result === EmailVerifyStatus.Error) {
      setWorking(false);
      return;
    }
    const status = result === EmailVerifyStatus.Verified ? 'ok' : 'expired';
    void navigate(`/users/verify/${encodeURIComponent(username)}/verified?status=${status}`, { replace: true });
  };

  return (
    <Page title="Verify email · Thorium">
      <Centered>
        <CardBox>
          <SimpleTitle>Verify your email</SimpleTitle>
          {valid ? (
            <>
              <Message>
                Confirm the email address for your Thorium account <strong>{username}</strong>.
              </Message>
              {errorMsg && <AlertBanner>{errorMsg}</AlertBanner>}
              {working ? (
                <LoadingSpinner loading={true} />
              ) : (
                <Button variant={ButtonVariant.Ok} size={ButtonSize.Large} onClick={() => void handleConfirm()}>
                  Verify email
                </Button>
              )}
            </>
          ) : (
            <Message>This verification link is missing required information. Please use the link from your email.</Message>
          )}
        </CardBox>
      </Centered>
    </Page>
  );
};

export default EmailVerify;
