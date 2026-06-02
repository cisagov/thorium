import { useEffect, useState } from 'react';
import { Badge, ButtonGroup, Button, Card, Col, Form, Modal, Row } from 'react-bootstrap';
import AlertBanner from '@components/shared/alerts/AlertBanner';

// project imports
import Page from '@components/pages/Page';
import { OmnibarUsers } from '@components/pages/search/omnibar/Bars';
import { Clause } from '@components/pages/search/omnibar/ClauseTypes';
import {
  getGroupsFromClauses,
  getStringFieldFromClauses,
  getStringFieldListFromClauses,
  matchesStringClauses,
} from '@components/pages/search/omnibar/utils';
import NoResultsBanner from '@components/shared/alerts/NoResultsBanner';
import Title from '@components/shared/titles/Title';
import { OverlayTipLeft } from '@components/shared/overlay/tips';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { useAuth } from '@utilities/auth';
import { getThoriumRole } from '@utilities/role';
import { hasOverlap } from '@utilities/groups';
import { deleteUser, listUsers, updateSingleUser } from '@thorpi/users';
import { RoleKey, UserInfo } from '@models/users';

/**
 * Filter users client-side by the omnibar clauses: username/email (substring for `includes`, exact
 * for `is`), group membership overlap, Thorium role (the resolved string role from
 * {@link getThoriumRole}), and the boolean `verified`/`local` flags (clause value `'true'`/`'false'`;
 * empty means no filter).
 */
export const filterUsers = (users: UserInfo[], clauses: Clause[]): UserInfo[] => {
  const groups = getGroupsFromClauses(clauses);
  const roles = getStringFieldListFromClauses(clauses, 'role');
  const verified = getStringFieldFromClauses(clauses, 'verified');
  const local = getStringFieldFromClauses(clauses, 'local');

  return users.filter((user) => {
    const usernameFilter = matchesStringClauses(clauses, 'username', user.username);
    const emailFilter = matchesStringClauses(clauses, 'email', user.email);
    const groupFilter = groups.length > 0 ? hasOverlap(user.groups, groups) : true;
    const roleFilter = roles.length > 0 ? roles.includes(getThoriumRole(user.role)) : true;
    const verifiedFilter = verified === '' ? true : user.verified === (verified === 'true');
    const localFilter = local === '' ? true : user.local === (local === 'true');
    return usernameFilter && emailFilter && groupFilter && roleFilter && verifiedFilter && localFilter;
  });
};

type SingleUserInfoProps = {
  user: UserInfo;
  impersonate: (userToken: string, tokenExpires: string) => void;
  // reload the full user list (shows the loading spinner) after an update or delete
  reloadUsers: () => void;
};

// component to represent each user's info
const SingleUserInfo: React.FC<SingleUserInfoProps> = ({ user, impersonate, reloadUsers }) => {
  const [singleUserRole, setSingleUserRole] = useState(getThoriumRole(user.role));
  return (
    <Card key={user.username} className="panel mt-1">
      <Row className="align-items-center m-2">
        <Col className="username-col">
          <h5 className="text mt-2">{user.username}</h5>
        </Col>
        <Col className="user-role-col">
          <small>
            <i className="secondary-text">{singleUserRole}</i>
          </small>
        </Col>
        <Col className="user-group-col">
          {user.groups.sort().map((group) => (
            <Badge bg="" key={group} className="m-1 bg-cadet">
              {group}
            </Badge>
          ))}
        </Col>
        <Col>
          <ManipulateUserButtons
            impersonate={impersonate}
            username={user.username}
            token={user.token}
            role={singleUserRole}
            user={user}
            setSingleUserRole={setSingleUserRole}
            reloadUsers={reloadUsers}
          />
        </Col>
      </Row>
    </Card>
  );
};

type ManipulateUserButtonsProps = {
  impersonate: (userToken: string, tokenExpires: string) => void;
  username: string;
  token: string;
  role: RoleKey;
  user: UserInfo;
  setSingleUserRole: (role: RoleKey) => void;
  reloadUsers: () => void;
};

// component for buttons related to each user
const ManipulateUserButtons: React.FC<ManipulateUserButtonsProps> = ({
  impersonate,
  username,
  token,
  role,
  user,
  setSingleUserRole,
  reloadUsers,
}) => {
  const [deleteError, setDeleteError] = useState('');
  // Delete user modal state manipulation
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const handleCloseDeleteModal = () => setShowDeleteModal(false);
  const handleShowDeleteModal = () => setShowDeleteModal(true);
  // Impersonate modal state manipulation
  const [showImpersonateModal, setShowImpersonateModal] = useState(false);
  const handleCloseImpersonateModal = () => setShowImpersonateModal(false);
  const handleShowImpersonateModal = () => setShowImpersonateModal(true);
  return (
    <ButtonGroup>
      <OverlayTipLeft
        tip={`Admins have the ability to change a user's role
        to Admin, Analyst, Developer, or User.`}
      >
        <EditRoles role={role} username={username} user={user} setRole={setSingleUserRole} reloadUsers={reloadUsers} />
      </OverlayTipLeft>
      <OverlayTipLeft
        tip={`Masquerade as ${username} after logging out of
        your current Thorium Session. This is used to troubleshoot Thorium UI
        issues that are specific to a individual user.`}
      >
        <Button className="primary-btn" size="sm" onClick={handleShowImpersonateModal}>
          Masquerade
        </Button>
      </OverlayTipLeft>
      <Modal show={showImpersonateModal} onHide={handleCloseImpersonateModal} backdrop="static" keyboard={false}>
        <Modal.Header closeButton>
          <Modal.Title>Masquerade as {username}?</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Do you really want to logout of your current session and login as <b>{username}</b>?
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          <Button
            className="warning-btn"
            onClick={() => {
              handleCloseImpersonateModal();
              impersonate(token, user.token_expiration);
            }}
          >
            Confirm
          </Button>
        </Modal.Footer>
      </Modal>
      <OverlayTipLeft tip={`Delete this user.`}>
        <Button className="warning-btn" size="sm" onClick={handleShowDeleteModal}>
          Delete
        </Button>
      </OverlayTipLeft>
      <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} backdrop="static" keyboard={false}>
        <Modal.Header closeButton>
          <Modal.Title>Confirm deletion?</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          Do you really want to delete <b>{username}</b>
          {"'s"} user account?
          {deleteError != '' && <AlertBanner>{deleteError}</AlertBanner>}
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          <Button
            className="danger-btn"
            onClick={() =>
              void deleteUser(username, setDeleteError).then((success) => {
                if (success) {
                  handleCloseDeleteModal();
                  // refresh the list so the deleted user is removed (shows the loading spinner)
                  reloadUsers();
                }
              })
            }
          >
            Confirm
          </Button>
        </Modal.Footer>
      </Modal>
    </ButtonGroup>
  );
};

type EditRolesProps = {
  role: RoleKey;
  username: string;
  user: UserInfo;
  setRole: (role: RoleKey) => void;
  reloadUsers: () => void;
};

// component to edit role
const EditRoles: React.FC<EditRolesProps> = ({ role, username, user, setRole, reloadUsers }) => {
  const [showEditRoleModal, setShowEditRoleModal] = useState(false);
  const [updateRoleError, setUpdateRoleError] = useState('');
  const [editRole, setEditRole] = useState(role);
  const [newK8s, setNewK8s] = useState(user.role.Developer ? user.role.Developer.k8s : true);
  const [newBareMetal, setNewBareMetal] = useState(user.role.Developer ? user.role.Developer.bare_metal : false);
  const [newWindows, setNewWindows] = useState(user.role.Developer ? user.role.Developer.windows : false);
  const [newExternal, setNewExternal] = useState(user.role.Developer ? user.role.Developer.external : false);

  // close edit role modal
  const handleCloseEditRoleModal = (response) => {
    if (!response || (response && editRole != RoleKey.Developer)) {
      // reset developer values back to default if leaving modal with no update
      setNewK8s(user.role.Developer ? user.role.Developer.k8s : true);
      setNewBareMetal(user.role.Developer ? user.role.Developer.bare_metal : false);
      setNewWindows(user.role.Developer ? user.role.Developer.windows : false);
      setNewExternal(user.role.Developer ? user.role.Developer.external : false);
    }
    if (response) {
      setRole(editRole);
    } else {
      setEditRole(role);
    }
    setShowEditRoleModal(false);
  };

  // open edit role modal
  const handleShowEditRoleModal = () => setShowEditRoleModal(true);

  const updateRole = async () => {
    let roleInfo = {};
    // if role is developer send configuration changes
    if (editRole == RoleKey.Developer) {
      roleInfo = {
        role: {
          Developer: {
            k8s: newK8s,
            bare_metal: newBareMetal,
            windows: newWindows,
            external: newExternal,
          },
        },
      };
    } else {
      roleInfo = { role: editRole };
    }
    if (Object.keys(roleInfo).length) {
      const response = await updateSingleUser(roleInfo, username, setUpdateRoleError);
      if (response) {
        // close the modal
        handleCloseEditRoleModal(response);
        // refresh the list so the updated role is reflected (shows the loading spinner)
        reloadUsers();
      }
    }
  };

  return (
    <div>
      <Button className="secondary-btn" size="sm" onClick={handleShowEditRoleModal}>
        Role
      </Button>
      <Modal show={showEditRoleModal} onHide={() => handleCloseEditRoleModal(false)} keyboard={false}>
        <Modal.Header
          closeButton
          onHide={() => {
            setUpdateRoleError('');
          }}
        >
          <Modal.Title>Edit Role</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Form.Group className="mb-2">
            <Form.Select value={editRole} onChange={(e) => setEditRole(e.target.value as RoleKey)}>
              {Object.keys(RoleKey).map((selectedRole) => (
                <option key={selectedRole} value={selectedRole}>
                  {selectedRole}
                </option>
              ))}
            </Form.Select>
          </Form.Group>
          {editRole == RoleKey.Developer && (
            <Row>
              <Col>
                <Form.Group>
                  <Form.Label>
                    <b>K8s</b>
                  </Form.Label>
                  <h6>
                    <Form.Check type="switch" id="collect-logs" label="" checked={newK8s} onChange={() => setNewK8s(!newK8s)} />
                  </h6>
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>
                    <b>Bare Metal</b>
                  </Form.Label>
                  <h6>
                    <Form.Check
                      type="switch"
                      id="collect-logs"
                      label=""
                      checked={newBareMetal}
                      onChange={() => setNewBareMetal(!newBareMetal)}
                    />
                  </h6>
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>
                    <b>Windows</b>
                  </Form.Label>
                  <h6>
                    <Form.Check type="switch" id="collect-logs" label="" checked={newWindows} onChange={() => setNewWindows(!newWindows)} />
                  </h6>
                </Form.Group>
              </Col>
              <Col>
                <Form.Group>
                  <Form.Label>
                    <b>External</b>
                  </Form.Label>
                  <h6>
                    <Form.Check
                      type="switch"
                      id="collect-logs"
                      label=""
                      checked={newExternal}
                      onChange={() => setNewExternal(!newExternal)}
                    />
                  </h6>
                </Form.Group>
              </Col>
            </Row>
          )}
          {updateRoleError != '' && updateRoleError != 'Successful' && <AlertBanner>{updateRoleError}</AlertBanner>}
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          <Button className="ok-btn" disabled={role == editRole && role != RoleKey.Developer} onClick={() => void updateRole()}>
            Update
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

// component to view a list of users
const UserBrowsing = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [clauses, setClauses] = useState<Clause[]>([]);
  const { checkCookie, impersonate } = useAuth();

  const filteredUsers = filterUsers(users, clauses);

  // get user details
  const getUserInfo = async () => {
    setLoading(true);
    const reqUsers = (await listUsers(() => void checkCookie(), true)) as UserInfo[] | null;
    if (reqUsers) {
      setUsers(reqUsers);
    }
    setLoading(false);
  };

  // need user info to validate creator permissions
  useEffect(() => {
    void getUserInfo();
  }, []);

  return (
    <Page title="Users · Thorium">
      <Row className="d-flex justify-content-md-center">
        <Col xs={1} sm={1} md={1}>
          <Title>Users</Title>
        </Col>
      </Row>
      <Row className="d-flex justify-content-center">
        <OmnibarUsers clauses={clauses} setClauses={setClauses} users={users} />
      </Row>
      <LoadingSpinner loading={loading}></LoadingSpinner>
      {!loading && filteredUsers.length === 0 && <NoResultsBanner type="Users" />}
      <Row>
        {!loading &&
          filteredUsers.length > 0 &&
          filteredUsers
            .sort((a, b) => a.username.localeCompare(b.username))
            .map((user) => (
              <SingleUserInfo
                key={user.username}
                user={user}
                impersonate={(token, expires) => void impersonate(token, expires)}
                reloadUsers={() => void getUserInfo()}
              />
            ))}
      </Row>
    </Page>
  );
};

export default UserBrowsing;
