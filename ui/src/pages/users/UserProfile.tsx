import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Badge, Button, Col, Container, Form, Modal, Row } from 'react-bootstrap';
import styled from 'styled-components';
import { FaCircleUser } from 'react-icons/fa6';

// project imports
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import { useAuth } from '@utilities/auth';
import { getThoriumRoleBadge } from '@utilities/role';
import { updateUser } from '@thorpi/users';
import { ThoriumRole } from '@models/users';

const ProfileCard = styled.div`
  width: 50rem;
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

  /*$grid-breakpoints: (
    xs: 0,
    sm: 576px,
    md: 768px,
    lg: 992px,
    xl: 1200px,
    xxl: 1400px
  );*/

  // 576px
  @media (max-width: 576px) {
    width: 400px;
  }
`;

const Themes = ['Dark', 'Light', 'Ocean', 'Crab', 'Automatic'];

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
        <Col>
          {badge && (
            <Badge pill bg="" className={`${badge.className} px-3 py-2`}>
              {badge.label}
            </Badge>
          )}
        </Col>
      </Row>
    </Container>
  );
};

const RevokeTokenModal = ({ show, onHide }: { show: boolean; onHide: () => void }) => {
  const { revoke } = useAuth();
  const navigate = useNavigate();
  // call thorium logout route and then
  const handleRevoke = () => {
    void revoke().then(() => {
      void navigate('/');
    });
  };
  return (
    <Modal show={show} onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>Revoke Your Token?</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        Revoking your token will automatically log you out of this page and any currently running or queued analysis jobs (reactions) may
        fail. Are you sure?
      </Modal.Body>
      <Modal.Footer className="d-flex justify-content-center">
        <Button className="danger-btn" onClick={() => handleRevoke()}>
          Confirm
        </Button>
      </Modal.Footer>
    </Modal>
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
          {groups &&
            [...groups].sort().map((group: string, idx: number) => (
              <Badge key={idx} pill bg="" className="bg-blue px-3 py-2 me-1">
                {group}
              </Badge>
            ))}
        </Col>
      </Row>
    </Container>
  );
};

const Token = () => {
  const [showRevokeTokenModal, setShowRevokeTokenModal] = useState(false);
  const [tokenShowing, setTokenShowing] = useState(false);
  const { userInfo } = useAuth();

  // toggle display of revoke token model from previous value
  const handleToggleRevokeTokenModalDisplay = () => {
    setShowRevokeTokenModal((prev) => !prev);
  };
  return (
    <Container>
      <Row>
        <Col xs={2}>
          <Subtitle>Token</Subtitle>
        </Col>
        <Col xs={10}>
          <Row>
            <Col>
              <div className="wrap-token">
                {tokenShowing ? (
                  <p>{userInfo?.token}</p>
                ) : (
                  <p className="hidden">****************************************************************</p>
                )}
              </div>
            </Col>
          </Row>
        </Col>
      </Row>
      <Row>
        <Col className="d-flex justify-content-center pt-2">
          <Button className="primary-btn" onClick={() => setTokenShowing(!tokenShowing)}>
            {tokenShowing ? 'Hide' : 'Show'}
          </Button>
          <Button className="danger-btn" onClick={() => handleToggleRevokeTokenModalDisplay()}>
            Revoke
          </Button>
        </Col>
      </Row>
      <Row className="pt-3">
        <Col xs={2}>
          <Subtitle>Expiry</Subtitle>
        </Col>
        <Col>
          <p>{userInfo?.token_expiration}</p>
        </Col>
      </Row>
      <RevokeTokenModal show={showRevokeTokenModal} onHide={handleToggleRevokeTokenModalDisplay} />
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
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 0.25rem 0.75rem;
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

const UserProfile = () => {
  const { userInfo } = useAuth();

  return (
    <Page title="Profile · Thorium" className="d-flex justify-content-center">
      <ProfileCard>
        <Row className="d-flex justify-content-center">
          <FaCircleUser size={150} />
        </Row>
        <Row className="d-flex justify-content-center">
          <h2 className="pt-3 d-flex justify-content-center">{userInfo?.username}</h2>
        </Row>
        <hr />
        {/* Group membership */}
        <Groups groups={userInfo?.groups} />
        <hr />
        {/* Thorium role (not group role) */}
        {userInfo && <Role role={userInfo.role} />}
        <hr />
        {/* User Token */}
        <Token />
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
