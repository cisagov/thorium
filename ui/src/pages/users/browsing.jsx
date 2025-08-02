import React, { useEffect, useState } from 'react';
import { Alert, Badge, ButtonGroup, Button, Card, Col, Form, Modal, Row } from 'react-bootstrap';

// project imports
import { LoadingSpinner, OverlayTipLeft, Title, Page } from '@components';
import { getThoriumRole, useAuth } from '@utilities';
import { deleteUser, listUsers, updateSingleUser } from '@thorpi';

// component to represent each user's info
const SingleUserInfo = ({ user, impersonate }) => {
  const [singleUserRole, setSingleUserRole] = useState(user.role);

  return (
    <Card key={user.username} className="panel mt-1">
      <Row className="align-items-center m-2">
        <Col className="username-col">
          <h5 className="text mt-2">{user.username}</h5>
        </Col>
        <Col className="user-role-col">
          <small>
            <i className="secondary-text">{getThoriumRole(singleUserRole)}</i>
          </small>
        </Col>
        <Col className="user-group-col">
          {user.groups.sort().map((group, idx) => (
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
            role={getThoriumRole(singleUserRole)}
            user={user}
            singleUserRole={singleUserRole}
            setSingleUserRole={setSingleUserRole}
          />
        </Col>
      </Row>
    </Card>
  );
};

// component for buttons related to each user
const ManipulateUserButtons = ({ impersonate, username, token, role, user, singleUserRole, setSingleUserRole }) => {
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
        tip={`Admins have the ability to change user's role
        to Admin, User, or Developer.`}
      >
        <EditRoles role={role} username={username} user={user} setRole={setSingleUserRole} />
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
              impersonate(token, user.expires);
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
          {deleteError != '' && <Alert variant="danger">{deleteError}</Alert>}
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          <Button
            className="danger-btn"
            onClick={async () => {
              if (await deleteUser(username, setDeleteError)) {
                handleCloseDeleteModal();
                getUserInfo();
              }
            }}
          >
            Confirm
          </Button>
        </Modal.Footer>
      </Modal>
    </ButtonGroup>
  );
};

// component to edit role
const EditRoles = ({ role, username, user, setRole }) => {
  const [showEditRoleModal, setShowEditRoleModal] = useState(false);
  const [updateRoleError, setUpdateRoleError] = useState('');
  const [editRole, setEditRole] = useState(role);
  const [newK8s, setNewK8s] = useState(user.role.Developer ? user.role.Developer.k8s : true);
  const [newBareMetal, setNewBareMetal] = useState(user.role.Developer ? user.role.Developer.bare_metal : false);
  const [newWindows, setNewWindows] = useState(user.role.Developer ? user.role.Developer.windows : false);
  const [newExternal, setNewExternal] = useState(user.role.Developer ? user.role.Developer.external : false);
  const handleCloseEditRoleModal = (response) => {
    if (!response || (response && editRole != 'Developer')) {
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

  const handleShowEditRoleModal = () => setShowEditRoleModal(true);

  const updateRole = async () => {
    let roleInfo = {};
    // if role is developer send configuration changes
    if (editRole == 'Developer') {
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
            <Form.Select value={editRole} onChange={(e) => setEditRole(e.target.value)}>
              {['Admin', 'Analyst', 'Developer', 'User'].map((selectedRole) => (
                <option key={selectedRole} value={selectedRole}>
                  {selectedRole}
                </option>
              ))}
            </Form.Select>
          </Form.Group>
          {editRole == 'Developer' && (
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
                      onChange={(e) => setNewBareMetal(!newBareMetal)}
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
                    <Form.Check
                      type="switch"
                      id="collect-logs"
                      label=""
                      checked={newWindows}
                      onChange={(e) => setNewWindows(!newWindows)}
                    />
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
                      onChange={(e) => setNewExternal(!newExternal)}
                    />
                  </h6>
                </Form.Group>
              </Col>
            </Row>
          )}
          {updateRoleError != '' && updateRoleError != 'Successful' && (
            <center>
              <Alert variant="danger">{updateRoleError}</Alert>
            </center>
          )}
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          <Button className="ok-btn" disabled={role == editRole && role != 'Developer'} onClick={() => updateRole()}>
            Update
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

// component to view a list of users
const Users = () => {
  const [loading, setLoading] = useState(false);
  const [users, setUsers] = useState([]);
  const { checkCookie, impersonate } = useAuth();

  // get user details
  const getUserInfo = async () => {
    setLoading(true);
    const reqUsers = await listUsers(checkCookie, true);
    if (reqUsers) {
      setUsers(reqUsers);
    }
    setLoading(false);
  };

  // need user info to validate creator permissions
  useEffect(() => {
    getUserInfo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Page title="Users · Thorium">
      <Row className="d-flex justify-content-md-center">
        <Col xs={1} sm={1} md={1}>
          <Title>Users</Title>
        </Col>
      </Row>
      <LoadingSpinner loading={loading}></LoadingSpinner>
      <Row>
        {users.length > 0 &&
          users
            .sort((a, b) => a.username.localeCompare(b.username))
            .map((user) => <SingleUserInfo key={user.username} user={user} impersonate={impersonate} />)}
      </Row>
    </Page>
  );
};

export default Users;
