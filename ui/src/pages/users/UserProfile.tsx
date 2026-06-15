import { useRef, useState } from 'react';
import { Col, Container, Form, Row } from 'react-bootstrap';
import { MdAdd, MdDelete } from 'react-icons/md';
import styled from 'styled-components';

// project imports
import Page from '@components/pages/Page';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import UserAvatar from '@components/shared/UserAvatar';
import Subtitle from '@components/shared/titles/Subtitle';
import { useAuth } from '@utilities/auth';
import { fileToResizedBlob } from '@utilities/image';
import { PROFILE_GRAPHIC_ACCEPT, validateProfileGraphic } from '@utilities/profileGraphic';
import { getThoriumRoleBadge } from '@utilities/role';
import { deleteUserImage, updateUser, uploadUserImage } from '@thorpi/users';
import { BUTTON_BADGE_GAP } from '@styles';
import { ThoriumRole } from '@models/users';
import TokenView from '@components/pages/users/TokenView';

// spec: ./UserProfile.spec.md

const ProfileCard = styled.div`
  border: none;
  background-color: var(--thorium-body-bg);
  flex: column;
  justify-content-center;
  align-items: center;
  padding: 1rem;

  // hidden token
  .hidden {
    color: var(--thorium-secondary-text);
  }

  .wrap-token {
    overflow-wrap: anywhere;
  }
  max-width: 1250px;
  min-width: 480px;
`;

const Themes = ['Dark', 'Light', 'Ocean', 'Crab', 'Automatic'];

// Rounded pill for the role/group badges. The background comes from a passed `bg-*` color class;
// the flex layout plus collapsed line-height keeps the label centered within the pill (the same
// treatment as the sign-in MethodPill), which the react-bootstrap badge did not do.
const InfoPill = styled.span`
  display: inline-block;
  text-align: center;
  line-height: 1;
  /* trim the font's half-leading to cap-height/baseline so the label is optically centered by the
     symmetric padding rather than sitting high (see MethodPill for the full rationale) */
  text-box-trim: trim-both;
  text-box-edge: cap alphabetic;
  border-radius: 999px;
  padding: 0.45rem 0.75rem;
  font-size: 0.85rem;
  color: var(--thorium-button-text);
`;

// Wrapping container for a set of pills. `gap` spaces both axes, so pills stay separated even when
// they wrap onto multiple rows (inline-block pills would otherwise stack with no vertical margin).
const PillGroup = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${BUTTON_BADGE_GAP};
`;

type RoleProps = {
  role: ThoriumRole;
};

const Role: React.FC<RoleProps> = ({ role }) => {
  const badge = getThoriumRoleBadge(role);
  return (
    <Container>
      <Row>
        <Col xs={2}>
          <Subtitle>Role</Subtitle>
        </Col>
        <Col>{badge && <InfoPill className={badge.className}>{badge.label}</InfoPill>}</Col>
      </Row>
    </Container>
  );
};

const Groups = ({ groups }: { groups: string[] | undefined }) => {
  return (
    <Container>
      <Row>
        <Col xs={2}>
          <Subtitle className="me-4">Groups</Subtitle>
        </Col>
        <Col>
          <PillGroup>
            {groups &&
              [...groups].sort().map((group: string, idx: number) => (
                <InfoPill key={idx} className="bg-blue">
                  {group}
                </InfoPill>
              ))}
          </PillGroup>
        </Col>
      </Row>
    </Container>
  );
};

const Theme = ({ theme }: { theme: string | undefined }) => {
  const { refreshUserInfo } = useAuth();
  // Send API new user theme settings
  const updateTheme = (theme: string) => {
    const settings = { settings: { theme: theme } };
    void updateUser(settings, console.log).then(() => {
      void refreshUserInfo(true);
    });
  };
  return (
    <Container>
      <Row>
        <Col xs={2}>
          <Subtitle>Theme</Subtitle>
        </Col>
        <Col className="d-flex justify-content-start">
          <Form>
            <Form.Group>
              <Form.Select value={theme ? theme : ''} onChange={(e) => updateTheme(String(e.target.value))}>
                {Themes.map((theme) => (
                  <option key={theme} value={theme}>
                    {theme}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Form>
        </Col>
      </Row>
    </Container>
  );
};

const MethodSection = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(0, 10fr);
  gap: 1rem;
  padding: 0 0.75rem;
  align-items: start;
`;

const MethodBadges = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
`;

const MethodPill = styled.span<{ $tone: 'ok' | 'warn' | 'muted' }>`
  display: inline-block;
  text-align: center;
  line-height: 1;
  /* Trim the font's half-leading down to the cap-height/baseline so the label is optically centered
     by the symmetric padding, instead of sitting high inside the empty reserved descender space that
     align-items:center would otherwise center on. Only takes effect on a non-flex box; ignored by
     browsers without text-box support (degrading to the untrimmed look). */
  text-box-trim: trim-both;
  text-box-edge: cap alphabetic;
  border-radius: 999px;
  padding: 0.45rem 0.75rem;
  font-size: 0.85rem;
  color: var(--thorium-button-text);
  background-color: ${({ $tone }) =>
    $tone === 'ok' ? 'var(--thorium-ok-bg)' : $tone === 'warn' ? 'var(--thorium-warning-bg)' : 'var(--thorium-secondary-panel-bg)'};
`;

const MethodGuidance = styled.p`
  color: var(--thorium-secondary-text);
  font-size: 0.9rem;
  margin: 0;
`;

// Read-only summary of how this account can sign in. Linked-provider management is not shown
// because the current API does not expose a user's linked providers on whoami.
const SignInMethods: React.FC<{ local?: boolean; verified?: boolean }> = ({ local, verified }) => {
  return (
    <MethodSection>
      <Subtitle>Sign-in</Subtitle>
      <div>
        <MethodBadges>
          {/* `local` mirrors the backend ScrubbedUser.local (the account has a password set) */}
          {local && <MethodPill $tone="ok">Local Login</MethodPill>}
          <MethodPill $tone={verified ? 'ok' : 'warn'}>{verified ? 'Email verified' : 'Email not verified'}</MethodPill>
        </MethodBadges>
        <MethodGuidance>
          To add a single sign-on provider, sign in with that provider using this same email address. You&apos;ll receive an email to
          confirm linking it to your account.
        </MethodGuidance>
      </div>
    </MethodSection>
  );
};

// Longest-edge bound (px) the uploaded icon is downscaled to before upload, keeping the
// stored S3 object small.
const PROFILE_ICON_MAX_PX = 256;

// Rendered diameter of the profile avatar on this page.
const PROFILE_AVATAR_PX = 150;

const AvatarColumn = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
`;

// Positioning context for the avatar and the corner delete control.
const AvatarStack = styled.div`
  position: relative;
  width: ${PROFILE_AVATAR_PX}px;
  height: ${PROFILE_AVATAR_PX}px;
`;

// The avatar itself is the upload control: clicking it opens the file picker. Dims slightly on hover
// to signal it's interactive. The explicit text color keeps the default placeholder icon
// (`FaCircleUser`, which draws in currentColor) on the theme text color inside this button.
const AvatarButton = styled.button`
  display: block;
  width: ${PROFILE_AVATAR_PX}px;
  height: ${PROFILE_AVATAR_PX}px;
  padding: 0;
  border: none;
  background: none;
  border-radius: 50%;
  color: var(--thorium-text);
  cursor: pointer;

  &:hover:not(:disabled) {
    filter: brightness(0.9);
  }

  &:disabled {
    cursor: default;
  }
`;

// Anchors the delete control to the bottom-right corner of the avatar's bounding square. The absolute
// positioning lives here (not on the button) so the tooltip's trigger wrapper isn't collapsed to a
// zero-size box at the stack origin, which would otherwise anchor the tooltip to the left.
const DeleteCorner = styled.div`
  position: absolute;
  right: 0;
  bottom: 0;
  line-height: 0;
`;

// Red trash toggle sitting flush in the bottom-right corner of the avatar square. Matches the token
// revoke control's danger styling.
const DeleteButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: none;
  background: none;
  color: var(--thorium-danger-bg);
  cursor: pointer;

  &:hover:not(:disabled) {
    filter: brightness(1.15);
  }

  &:disabled {
    opacity: 0.5;
    cursor: default;
  }
`;

// Neutral add (+) control shown in the same corner when no picture is set; clicking it (like the
// avatar) opens the file picker.
const AddButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: none;
  background: none;
  color: var(--thorium-text);
  cursor: pointer;

  &:hover:not(:disabled) {
    color: var(--thorium-link-text);
  }

  &:disabled {
    opacity: 0.5;
    cursor: default;
  }
`;

const HiddenFileInput = styled.input`
  display: none;
`;

// Profile icon with click-to-upload and a corner delete control. Clicking the avatar opens the file
// picker; static images are resized client-side while animated GIFs and video are uploaded as-is, sent
// as multipart form data (stored in S3, fetched lazily via useUserImage so whoami stays lightweight).
// The trash button, shown only when an icon is set, removes it.
const ProfileImage = () => {
  const { userInfo, refreshUserInfo } = useAuth();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // bumped after an upload/remove to force UserAvatar to refetch the icon at the same path
  const [imageVersion, setImageVersion] = useState(0);

  // refetch the icon (after an upload/remove) and refresh has_image on the cached user
  const refresh = async () => {
    await refreshUserInfo(true);
    setImageVersion((v) => v + 1);
  };

  // validate the selection and upload it as this user's icon; static images are downscaled first while
  // animated GIFs and video are uploaded as-is (a canvas resize would flatten/fail to encode them)
  const handleFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // reset the input so selecting the same file again re-triggers onChange
    e.target.value = '';
    if (!file) {
      return;
    }
    const check = validateProfileGraphic(file);
    if (!check.ok) {
      setError(check.error);
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const graphic = check.resize ? await fileToResizedBlob(file, PROFILE_ICON_MAX_PX) : file;
      if (await uploadUserImage(graphic, setError)) {
        await refresh();
      }
    } catch {
      setError('Failed to process the selected image.');
    } finally {
      setBusy(false);
    }
  };

  // clear this user's icon
  const handleRemove = async () => {
    setError(null);
    setBusy(true);
    if (await deleteUserImage(setError)) {
      await refresh();
    }
    setBusy(false);
  };

  return (
    <AvatarColumn>
      <AvatarStack>
        <OverlayTipRight tip="Click to upload" block>
          <AvatarButton
            type="button"
            disabled={busy}
            onClick={() => fileInputRef.current?.click()}
            aria-label="Click to upload profile picture"
          >
            <UserAvatar username={userInfo?.username} hasImage={userInfo?.has_image} size={PROFILE_AVATAR_PX} version={imageVersion} />
          </AvatarButton>
        </OverlayTipRight>
        <DeleteCorner>
          {userInfo?.has_image ? (
            <OverlayTipRight tip="Delete this profile picture">
              <DeleteButton type="button" disabled={busy} onClick={() => void handleRemove()} aria-label="Delete this profile picture">
                <MdDelete size={22} />
              </DeleteButton>
            </OverlayTipRight>
          ) : (
            // no own tooltip here: the avatar already shows "Click to upload", so a second overlay on the
            // corner add control would just double up on the same empty-state hint
            <AddButton type="button" disabled={busy} onClick={() => fileInputRef.current?.click()} aria-label="Upload a profile picture">
              <MdAdd size={22} />
            </AddButton>
          )}
        </DeleteCorner>
      </AvatarStack>
      <HiddenFileInput ref={fileInputRef} type="file" accept={PROFILE_GRAPHIC_ACCEPT} onChange={(e) => void handleFileSelected(e)} />
      {error && <AlertBanner severity={Severity.Error}>{error}</AlertBanner>}
    </AvatarColumn>
  );
};

const UserProfile = () => {
  const { userInfo } = useAuth();

  return (
    <Page title="Profile · Thorium" className="d-flex justify-content-center">
      <ProfileCard>
        <Row className="d-flex justify-content-center">
          <h2 className="pb-3 d-flex justify-content-center">{userInfo?.username}</h2>
        </Row>
        <Row className="d-flex justify-content-center">
          <ProfileImage />
        </Row>
        <hr />
        {/* Group membership */}
        <Groups groups={userInfo?.groups} />
        <hr />
        {/* Thorium role (not group role) */}
        {userInfo && <Role role={userInfo.role} />}
        <hr />
        {/* User Token + Scoped Tokens */}
        <TokenView />
        <hr />
        {/* Sign-in methods (local password / SSO + verification status) */}
        <SignInMethods local={userInfo?.local} verified={userInfo?.verified} />
        <hr />
        {/* UI Theme */}
        <Theme theme={userInfo?.settings?.theme} />
      </ProfileCard>
    </Page>
  );
};

export default UserProfile;
