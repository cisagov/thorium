import { Button, Col, Modal, Row } from 'react-bootstrap';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { Group, GroupRoleKey, GroupRoleUpdate, GroupUpdate } from '@models/groups';
import { useState } from 'react';
import { fetchGroups } from '@utilities/fetch';
import { deleteGroup, updateGroup } from '@thorpi/groups';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import { useAuth } from '@utilities/auth';
import { isGroupOwner } from '@utilities/permissions';

interface UpdateGroupButtonProps {
  group: Group;
  changes: GroupUpdate;
  disableUpdate: boolean;
  refreshSingleGroup: (name: string) => Promise<void>;
}

interface LeaveGroupButtonProps {
  group: Group;
  username: string;
  setLoading: (next: boolean) => void;
  setGroups: (next: Record<string, Group>) => void;
}

interface DeleteGroupButtonProps {
  group: Group;
  setLoading: (next: boolean) => void;
  setGroups: (next: Record<string, Group>) => void;
}

export const UpdateGroupButton = ({ group, changes, disableUpdate, refreshSingleGroup }: UpdateGroupButtonProps) => {
  const [updateError, setUpdateError] = useState('');
  const [showUpdateModal, setShowUpdateModal] = useState(false);
  const handleCloseUpdateModal = () => {
    setShowUpdateModal(false);
    setUpdateError('');
  };
  const handleShowUpdateModal = () => setShowUpdateModal(true);
  const replaceExp = /,/g;
  return (
    <>
      <OverlayTipTop
        tip={`Submit pending changes. Button will be dark green
                            when there are pending changes.`}
      >
        <Button className="primary-btn" disabled={disableUpdate} onClick={handleShowUpdateModal}>
          Update
        </Button>
      </OverlayTipTop>
      <Modal show={showUpdateModal} onHide={handleCloseUpdateModal} backdrop="static" keyboard={false}>
        <Modal.Header closeButton>
          <Modal.Title>Confirm update?</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <center>
            {changes.description && (
              <Row>
                <Col>
                  <b>Description:</b> {changes.description ? changes.description : 'REMOVED'}
                </Col>
              </Row>
            )}
            {changes.clear_description && (
              <Row>
                <Col>
                  <b>Description Removed</b>
                </Col>
              </Row>
            )}
            {changes.owners && changes.owners.direct_add && changes.owners.direct_add.length > 0 && (
              <Row>
                <Col>
                  <b>Add Owner(s): </b>
                  {changes.owners.direct_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.owners && changes.owners.metagroups_add && changes.owners.metagroups_add.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Add Metagroup(s) to Owners: </b>
                  {changes.owners.metagroups_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.owners && changes.owners.direct_remove && changes.owners.direct_remove.length > 0 && (
              <Row>
                <Col>
                  <b>Remove Owners(s): </b>
                  {changes.owners.direct_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.owners && changes.owners.metagroups_remove && changes.owners.metagroups_remove.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Remove Metagroup(s) From Owners: </b>
                  {changes.owners.metagroups_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.managers && changes.managers.direct_add && changes.managers.direct_add.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Add Manager(s): </b>
                  {changes.managers.direct_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.managers && changes.managers.metagroups_add && changes.managers.metagroups_add.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Add Metagroup(s) to Manager(s): </b>
                  {changes.managers.metagroups_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.managers && changes.managers.direct_remove && changes.managers.direct_remove.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Remove Manager(s): </b>
                  {changes.managers.direct_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.managers && changes.managers.metagroups_remove && changes.managers.metagroups_remove.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Remove Metagroup(s) from Managers: </b>
                  {changes.managers.metagroups_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.users && changes.users && changes.users.direct_add && changes.users.direct_add.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Add User(s): </b>
                  {changes.users.direct_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.users && changes.users.metagroups_add && changes.users.metagroups_add.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Add Metagroup(s) to Users: </b>
                  {changes.users.metagroups_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.users && changes.users.direct_remove && changes.users.direct_remove.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Remove Users(s): </b>
                  {changes.users.direct_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.users && changes.users.metagroups_remove && changes.users.metagroups_remove.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Remove Metagroup(s) from Users: </b>
                  {changes.users.metagroups_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.monitors && changes.monitors.direct_add && changes.monitors.direct_add.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Add Monitor(s): </b>
                  {changes.monitors.direct_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.monitors && changes.monitors.metagroups_add && changes.monitors.metagroups_add.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Add Metagroup(s) to Monitors: </b>
                  {changes.monitors.metagroups_add.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.monitors && changes.monitors.direct_remove && changes.monitors.direct_remove.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Remove Monitor(s): </b>
                  {changes.monitors.direct_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {changes.monitors && changes.monitors.metagroups_remove && changes.monitors.metagroups_remove.length > 0 && (
              <Row>
                <Col className="word-break-all">
                  <b>Remove Metagroup(s) from Monitors: </b>
                  {changes.monitors.metagroups_remove.toString().replace(replaceExp, ', ')}
                </Col>
              </Row>
            )}
            {updateError && <AlertBanner className="word-break-all">{updateError.replace(replaceExp, ', ')}</AlertBanner>}
          </center>
        </Modal.Body>
        <Modal.Footer className="d-flex justify-content-center">
          <Button
            className="ok-btn m-1"
            onClick={() => {
              void (async () => {
                if (await updateGroup(group.name, changes, setUpdateError)) {
                  // Refresh only the edited group so its accordion stays open and just its content rerenders.
                  handleCloseUpdateModal();
                  await refreshSingleGroup(group.name);
                }
              })();
            }}
          >
            Confirm
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export const LeaveGroupButton = ({ group, username, setLoading, setGroups }: LeaveGroupButtonProps) => {
  const [showLeaveModal, setShowLeaveModal] = useState(false);
  const [leaveError, setLeaveError] = useState('');
  const handleCloseLeaveModal = () => setShowLeaveModal(false);
  const roleAction: GroupUpdate = {};

  // Determine the removable role this user directly holds. Owners cannot leave
  // (they must be removed by another owner), and analyst access is a global
  // Thorium role rather than a per-group membership, so neither is leavable.
  let leaveRole: GroupRoleKey | '' = '';
  if (group.managers.combined.includes(username)) {
    leaveRole = GroupRoleKey.Manager;
  } else if (group.users.combined.includes(username)) {
    leaveRole = GroupRoleKey.User;
  } else if (group.monitors.combined.includes(username)) {
    leaveRole = GroupRoleKey.Monitor;
  }

  if (leaveRole) {
    (roleAction as Record<string, GroupRoleUpdate>)[String(leaveRole.toLowerCase() + 's')] = {
      direct_remove: [username],
    };
    return (
      <>
        <OverlayTipTop
          tip={
            'Leave this group. Owners cannot leave their\
            group and must be removed by another owner.'
          }
        >
          <Button className="primary-btn" onClick={() => setShowLeaveModal(true)}>
            Leave
          </Button>
        </OverlayTipTop>
        <Modal show={showLeaveModal} onHide={handleCloseLeaveModal}>
          <Modal.Header closeButton>
            <Modal.Title>{`Confirm Leave ${group.name}?`}</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            Do you really want to leave the <b>{group.name}</b> group?
            {leaveError != '' && <AlertBanner className="mt-3 mb-2">{leaveError}</AlertBanner>}
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              className="warning-btn"
              onClick={() => {
                void (async () => {
                  if (await updateGroup(group.name, roleAction, setLeaveError)) {
                    void fetchGroups(setGroups as (groups: Record<string, Group> | Group[] | string[]) => void, setLoading, true);
                  }
                })();
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
      </>
    );
  } else {
    return null;
  }
};

export const DeleteGroupButton = ({ group, setLoading, setGroups }: DeleteGroupButtonProps) => {
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const { userInfo } = useAuth();
  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
    setDeleteError('');
  };
  const handleShowDeleteModal = () => setShowDeleteModal(true);

  // user must be a system admin or a group owner to delete a group (Group::is_owner)
  if (isGroupOwner(group, userInfo!)) {
    return (
      <>
        <OverlayTipTop
          tip={`Delete this group. Only system admins and
              group owners can delete a group.`}
        >
          <Button className="warning-btn" onClick={handleShowDeleteModal}>
            Delete
          </Button>
        </OverlayTipTop>
        <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} backdrop="static" keyboard={false}>
          <Modal.Header closeButton>
            <Modal.Title>Confirm deletion?</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            Do you really want to delete the <b>{group.name}</b> group?
            {deleteError != '' && <AlertBanner>{deleteError}</AlertBanner>}
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              className="danger-btn"
              onClick={() => {
                void (async () => {
                  if (await deleteGroup(group.name, setDeleteError)) {
                    void fetchGroups(setGroups as (groups: Record<string, Group> | Group[] | string[]) => void, setLoading, true);
                  }
                })();
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
      </>
    );
  } else {
    return null;
  }
};
