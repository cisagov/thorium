import React, { useEffect, useState } from 'react';
import { Accordion, Alert, Button, Badge, ButtonGroup, ButtonToolbar, Col, Container, Form, Modal, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';
import Select from 'react-select';
import CreatableSelect from 'react-select/creatable';

// project imports
import {
  GroupRoleBadge,
  LoadingSpinner,
  GroupMemberCount,
  OverlayTipRight,
  OverlayTipTop,
  OverlayTipLeft,
  Subtitle,
  Title,
  Page,
} from '@components';
import { createReactSelectStyles, fetchGroups, getGroupRole, isGroupAdmin, useAuth } from '@utilities';
import { createGroup, deleteGroup, listUsers, updateGroup } from '@thorpi';

// styles for react select badges
const ownerStyles = createReactSelectStyles('White', 'DarkSlateBlue');
const managerStyles = createReactSelectStyles('White', 'CornFlowerBlue');
const userStyles = createReactSelectStyles('White', 'CadetBlue');
const monitorStyles = createReactSelectStyles('White', 'DimGray');

const Groups = () => {
  const [loading, setLoading] = useState(false);
  const [groups, setGroups] = useState({});
  const [allUsers, setAllUSers] = useState([]);
  const { userInfo, checkCookie } = useAuth();

  // get a list of all Thorium users
  const fetchAllUsers = async () => {
    const reqUsers = await listUsers(console.log, false);
    if (reqUsers) {
      setAllUSers(reqUsers.sort());
    }
  };

  // get list of groups and users on initial page load
  useEffect(() => {
    // Get group details on page load
    fetchGroups(setGroups, setLoading, true);
    fetchAllUsers();
    // if user info changes, we want to get the group details again
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userInfo]);

  const LeaveGroupButton = ({ group }) => {
    const [showLeaveModal, setShowLeaveModal] = useState(false);
    const [leaveError, setLeaveError] = useState('');
    const handleCloseLeaveModal = () => setShowLeaveModal(false);
    const role = getGroupRole(group, userInfo.username);
    const roleAction = {};

    // Owners cannot leave a group, neither can admins which don't have a group role
    if (role && role != 'Owner') {
      // && role != ''
      roleAction[String(role.toLowerCase() + 's')] = {
        direct_remove: [userInfo.username],
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
              <Modal.Title>{`Confirm Leave ${group}?`}</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              Do you really want to leave the <b>{group.name}</b> group?
              {leaveError != '' && (
                <center>
                  <Alert variant="danger" className="mt-3 mb-2">
                    {leaveError}
                  </Alert>
                </center>
              )}
            </Modal.Body>
            <Modal.Footer className="d-flex justify-content-center">
              <Button
                className="warning-btn"
                onClick={async () => {
                  if (await updateGroup(group.name, roleAction, setLeaveError)) {
                    fetchGroups(setGroups, setLoading, true);
                  }
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

  // display delete button and confirmation modal
  const DeleteGroupButton = ({ group }) => {
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [deleteError, setDeleteError] = useState('');
    const handleCloseDeleteModal = () => {
      setShowDeleteModal(false);
      setDeleteError('');
    };
    const handleShowDeleteModal = () => setShowDeleteModal(true);

    // user must be a system admin or a group owner to delete a group
    const groupRole = getGroupRole(group, userInfo.username);
    if (userInfo.role == 'Admin' || groupRole == 'Owner') {
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
              {deleteError != '' && (
                <center>
                  <Alert variant="danger">{deleteError}</Alert>
                </center>
              )}
            </Modal.Body>
            <Modal.Footer className="d-flex justify-content-center">
              <Button
                className="danger-btn"
                onClick={async () => {
                  if (await deleteGroup(group.name, setDeleteError)) {
                    fetchGroups(setGroups, setLoading, true);
                  }
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

  // display update group button and confirmation modals
  const UpdateGroupButton = ({ group, changes, disableUpdate }) => {
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
              {updateError && (
                <center>
                  <Alert variant="danger" className="word-break-all">
                    {updateError.replace(replaceExp, ', ')}
                  </Alert>
                </center>
              )}
            </center>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              className="ok-btn m-1"
              onClick={async () => {
                if (await updateGroup(group.name, changes, setUpdateError)) {
                  fetchGroups(setGroups, setLoading, true);
                }
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
      </>
    );
  };

  const GroupInfo = ({ group, allUsers }) => {
    // Owners
    const combinedOwners = 'owners' in group && 'combined' in group.owners ? group.owners.combined.sort() : [];
    const metagroupOwners = 'owners' in group && 'metagroups' in group.owners ? group.owners.metagroups.sort() : [];
    const directOwners = 'owners' in group && 'direct' in group.owners ? group.owners.direct.sort() : [];
    // Managers
    const combinedManagers = 'managers' in group && 'combined' in group.managers ? group.managers.combined.sort() : [];
    const metagroupManagers = 'managers' in group && 'metagroups' in group.managers ? group.managers.metagroups.sort() : [];
    const directManagers = 'managers' in group && 'direct' in group.managers ? group.managers.direct.sort() : [];
    // Analysts
    const analysts = 'analysts' in group ? group.analysts.sort() : [];
    // Users
    const combinedUsers = 'users' in group && 'combined' in group.users ? group.users.combined.sort() : [];
    const metagroupUsers = 'users' in group && 'metagroups' in group.users ? group.users.metagroups.sort() : [];
    const directUsers = 'users' in group && 'direct' in group.users ? group.users.direct.sort() : [];
    // Monitors
    const combinedMonitors = 'monitors' in group && 'combined' in group.monitors ? group.monitors.combined.sort() : [];
    const metagroupMonitors = 'monitors' in group && 'metagroups' in group.monitors ? group.monitors.metagroups.sort() : [];
    const directMonitors = 'monitors' in group && 'metagroups' in group.monitors ? group.monitors.direct.sort() : [];
    const [description, setDescription] = useState('description' in group ? group.description : '');
    const [disableUpdate, setDisableUpdate] = useState(true);
    const [groupChanges, setGroupChanges] = useState({});

    // Create options for drop downs for use by react/select field
    const usersWithRoles = [...directOwners, ...directManagers, ...directUsers, ...directMonitors];

    const startingUserOptions =
      allUsers.length == 0
        ? []
        : allUsers.reduce((options, name) => {
            if (!usersWithRoles.includes(name)) {
              options.push({ value: name, label: name });
            }
            return options;
          }, []);
    const [ownerOptions, setOwnerOptions] = useState(startingUserOptions);
    const [managerOptions, setManagerOptions] = useState(startingUserOptions);
    const [userOptions, setUserOptions] = useState(startingUserOptions);
    const [monitorOptions, setMonitorOptions] = useState(startingUserOptions);

    // generate selected owners from starting group roles for use by react/select
    const [selectedDirectOwners, setSelectedDirectOwners] = useState(
      directOwners.map((owner) => {
        return { value: owner, label: owner };
      }),
    );
    // generate selected ldap group owners from starting group roles for use by react/select
    const [selectedMetagroupOwners, setSelectedMetagroupOwners] = useState(
      metagroupOwners.map((owner) => {
        return { value: owner, label: owner };
      }),
    );
    // generate selected managers from starting group roles for use by react/select
    const [selectedDirectManagers, setSelectedDirectManagers] = useState(
      directManagers.map((manager) => {
        return { value: manager, label: manager };
      }),
    );
    // generate selected ldap group managers from starting group roles for use by react/select
    const [selectedMetagroupManagers, setSelectedMetagroupManagers] = useState(
      metagroupManagers.map((manager) => {
        return { value: manager, label: manager };
      }),
    );
    // generate selected users from starting group roles for use by react/select
    const [selectedDirectUsers, setSelectedDirectUsers] = useState(
      directUsers.map((user) => {
        return { value: user, label: user };
      }),
    );
    // generate selected ldap group users from starting group roles for use by react/select
    const [selectedMetagroupUsers, setSelectedMetagroupUsers] = useState(
      metagroupUsers.map((user) => {
        return { value: user, label: user };
      }),
    );
    // generate selected monitors for use by react/select
    const [selectedDirectMonitors, setSelectedDirectMonitors] = useState(
      directMonitors.map((monitor) => {
        return { value: monitor, label: monitor };
      }),
    );
    // generate selected ldap group monitors from starting group roles for use by react/select
    const [selectedMetagroupMonitors, setSelectedMetagroupMonitors] = useState(
      metagroupMonitors.map((monitor) => {
        return { value: monitor, label: monitor };
      }),
    );

    // see if user should be able to edit group
    // thorium admins and group owners/managers can edit group membership
    // a user cannot remove themselves from the group if they are an owner
    const groupAdmin = isGroupAdmin(group, userInfo);

    // update the changes to groups structure for eventual
    // submission to the Thorium API
    const updateGroupChanges = (role, type, newValue) => {
      const changes = structuredClone(groupChanges);
      let updatedUsersWithRoles = [];
      // user or ldap group to add/remove to/from role
      let entity = '';
      // action + role that is being modified
      let roleAction = '';
      // opposite action + role for the role being taken (remove vs add)
      let reverseAction = '';
      // remove user/group from a given role
      if (newValue.action == 'remove-value') {
        roleAction = type + '_remove';
        reverseAction = type + '_add';
        entity = newValue.removedValue.value;

        updatedUsersWithRoles = usersWithRoles.filter((user) => {
          // user/group does not have a role
          // we don't add ldap groups to usersWithRoles, they are groups not users
          if (user == entity) {
            return false;
          }
          return true;
        });
        // add a user/group from a given role
      } else if (newValue.action == 'select-option' || newValue.action == 'create-option') {
        roleAction = type + '_add';
        reverseAction = type + '_remove';
        entity = newValue.option.value;

        // only add entity to role list if its a new non-group user
        updatedUsersWithRoles = [...usersWithRoles];
        if (!usersWithRoles.includes(entity)) {
          updatedUsersWithRoles.push(entity);
        }
        // non-supported action, skip all state changes
      } else {
        return;
      }

      // check to see if user was added already
      // remove/add actions cancel out
      if (role in changes && reverseAction in changes[role] && changes[role][reverseAction].includes(entity)) {
        if (changes[role][reverseAction].length == 1) {
          delete changes[role][reverseAction];
        } else if (changes[role][reverseAction].length > 1) {
          // remove item from array by filtering entity out and creating new array
          changes[role][reverseAction] = changes[role][reverseAction].filter((name) => {
            return name != entity;
          });
        }
      } else if (role in changes) {
        // add user to group change structure for the role
        if (roleAction in changes[role]) {
          changes[role][roleAction].push(entity);
        } else {
          changes[role][`${roleAction}`] = [entity];
        }
      } else {
        const newRoleUpdate = {};
        newRoleUpdate[roleAction] = [entity];
        changes[role] = newRoleUpdate;
      }

      // update options for role drop downs for use by react/select field
      // users can only have one role in a given group and are removed
      // from the drop down once added to a group
      const updatedUserOptions =
        allUsers.length == 0
          ? []
          : allUsers.reduce((options, name) => {
              if (!updatedUsersWithRoles.includes(name)) {
                options.push({ value: name, label: name });
              }
              return options;
            }, []);
      setOwnerOptions(updatedUserOptions);
      setManagerOptions(updatedUserOptions);
      setUserOptions(updatedUserOptions);
      setMonitorOptions(updatedUserOptions);

      // get number of group changes to enable/disable update button
      let numChanges = 0;
      Object.entries(changes).map((role) => {
        Object.entries(role[1]).map((entities) => {
          numChanges += entities[1].length;
        });
      });

      // disable update modal/buttons when there are no pending changes
      if (numChanges > 0) {
        setDisableUpdate(false);
      } else {
        setDisableUpdate(true);
      }
      setGroupChanges(changes);
    };

    const updateDescription = (description) => {
      if (description == group.description) return;
      const changes = structuredClone(groupChanges);
      setDescription(description);
      changes['description'] = description;
      setDisableUpdate(false);
      if (description == '') {
        changes['clear_description'] = true;
      } else if ('clear_description' in changes) {
        delete changes['clear_description'];
      }
      setGroupChanges(changes);
    };

    const ModifyGroupButtons = ({ group, admin }) => {
      // only owners, managers and Thorium admins can modify a group
      return (
        <Container>
          <Row>
            <ButtonToolbar className="d-flex justify-content-center">
              <ButtonGroup>
                {admin && <UpdateGroupButton group={group} changes={groupChanges} disableUpdate={disableUpdate} />}
                <LeaveGroupButton group={group} />
                {admin && <DeleteGroupButton group={group} />}
              </ButtonGroup>
            </ButtonToolbar>
          </Row>
        </Container>
      );
    };

    if (!groupAdmin) {
      // return non-editable group info component
      return (
        <Container>
          <Row>
            <Col className="header-col">
              <OverlayTipRight
                tip={`A description of this group, its membership,
                                    and its owned resources.`}
              >
                <b>Description</b> <FaQuestionCircle className="group-tooltip" />
              </OverlayTipRight>
            </Col>
            <Col className="edit-col descr-height">
              <p>{description}</p>
            </Col>
          </Row>
          <Row className="mt-4">
            <Col className="header-col">
              <OverlayTipRight tip={`Analysts have global view into data in Thorium.`}>
                <b>Analysts</b> <FaQuestionCircle className="group-tooltip" />
              </OverlayTipRight>
            </Col>
            <Col className="edit-col mt-2">
              {analysts.map((analyst) => (
                <Badge bg="" className="bg-goldenrod group-edit-badge" key={'analyst_' + analyst}>
                  <b>{analyst}</b>
                </Badge>
              ))}
            </Col>
          </Row>
          <Row className="mt-4">
            <Col className="header-col">
              <OverlayTipRight
                tip={`Owners can access and edit all group resources.
                                    They can also delete the group or remove other
                                    owners from the group.`}
              >
                <b>Owners</b> <FaQuestionCircle className="group-tooltip" />
              </OverlayTipRight>
            </Col>
            <Col className="edit-col mt-2">
              {combinedOwners.length > 0 && <Subtitle>Combined</Subtitle>}
              {combinedOwners.map((owner) => (
                <Badge bg="" className="bg-dark-slate group-edit-badge" key={'combined_owner_' + owner}>
                  <b>{owner}</b>
                </Badge>
              ))}
              {directOwners.length > 0 && <Subtitle>Individuals</Subtitle>}
              {directOwners.map((owner) => (
                <Badge bg="" className="bg-dark-slate group-edit-badge" key={'owner_' + owner}>
                  <b>{owner}</b>
                </Badge>
              ))}
              {metagroupOwners.length > 0 && <Subtitle>Metagroup(s)</Subtitle>}
              {metagroupOwners.map((owner) => (
                <Badge bg="" className="bg-dark-slate group-edit-badge" key={'meta_owner_' + owner}>
                  <b>{owner}</b>
                </Badge>
              ))}
            </Col>
          </Row>
          <Row className="mt-4">
            <Col className="header-col">
              <OverlayTipRight
                tip={`Managers can access and edit all group resources
                                    but cannot delete the group or remove owners.`}
              >
                <b>Managers</b> <FaQuestionCircle className="group-tooltip" />
              </OverlayTipRight>
            </Col>
            <Col className="edit-col mt-2">
              {combinedManagers.length > 0 && <Subtitle>Combined</Subtitle>}
              {combinedManagers.map((manager) => (
                <Badge bg="" className="bg-corn-flower group-edit-badge" key={'combined_manager_' + manager}>
                  <b>{manager}</b>
                </Badge>
              ))}
              {directManagers.length > 0 && <Subtitle>Individuals</Subtitle>}
              {directManagers.map((manager) => (
                <Badge bg="" className="bg-corn-flower group-edit-badge" key={'manager_' + manager}>
                  <b>{manager}</b>
                </Badge>
              ))}
              {metagroupManagers.length > 0 && <Subtitle>Metagroup(s)</Subtitle>}
              {metagroupManagers.map((manager) => (
                <Badge bg="" className="bg-corn-flower group-edit-badge" key={'meta_manager_' + manager}>
                  <b>{manager}</b>
                </Badge>
              ))}
            </Col>
          </Row>
          <Row className="mt-4">
            <Col className="header-col">
              <OverlayTipRight
                tip={`Users can run pipelines and access files
                                    owned by this group.`}
              >
                <b>Users</b> <FaQuestionCircle className="group-tooltip" />
              </OverlayTipRight>
            </Col>
            <Col className="edit-col mt-2">
              {combinedUsers.length > 0 && <Subtitle>Combined</Subtitle>}
              {combinedUsers.map((user) => (
                <Badge bg="" className="bg-cadet group-edit-badge" key={'combined_user_' + user}>
                  <b>{user}</b>
                </Badge>
              ))}
              {directUsers.length > 0 && <Subtitle>Individuals</Subtitle>}
              {directUsers.map((user) => (
                <Badge bg="" className="bg-cadet group-edit-badge" key={'user_' + user}>
                  <b>{user}</b>
                </Badge>
              ))}
              {metagroupUsers.length > 0 && <Subtitle>Metagroup(s)</Subtitle>}
              {metagroupUsers.map((user) => (
                <Badge bg="" className="bg-cadet group-edit-badge" key={'meta_user_' + user}>
                  <b>{user}</b>
                </Badge>
              ))}
            </Col>
          </Row>
          <Row className="mt-4">
            <Col className="header-col">
              <OverlayTipRight
                tip={`Monitors can view the status of reactions and access files
                                    owned by a group but cannot run pipelines or modify files.`}
              >
                <b>Monitors</b> <FaQuestionCircle className="group-tooltip" />
              </OverlayTipRight>
            </Col>
            <Col className="edit-col mt-2">
              {combinedMonitors.length > 0 && <Subtitle>Combined</Subtitle>}
              {combinedMonitors.map((monitor) => (
                <Badge bg="" className="bg-grey group-edit-badge" key={'combined_monitor_' + monitor}>
                  <b>{monitor}</b>
                </Badge>
              ))}
              {directMonitors.length > 0 && <Subtitle>Individuals</Subtitle>}
              {directMonitors.map((monitor) => (
                <Badge bg="" className="bg-grey group-edit-badge" key={'monitor_' + monitor}>
                  <b>{monitor}</b>
                </Badge>
              ))}
              {metagroupMonitors.length > 0 && <Subtitle>Metagroup(s)</Subtitle>}
              {metagroupMonitors.map((monitor) => (
                <Badge bg="" className="bg-grey group-edit-badge" key={'meta_monitor_' + monitor}>
                  <b>{monitor}</b>
                </Badge>
              ))}
            </Col>
          </Row>
        </Container>
      );
    } else {
      // return an editable admin group info component
      return (
        <Container>
          <Row>
            <Form>
              <Row>
                <Col className="header-col">
                  <OverlayTipRight
                    tip={`A description of this group, its membership,
                                        and its owned resources.`}
                  >
                    <b>Description</b> <FaQuestionCircle className="group-tooltip" />
                  </OverlayTipRight>
                </Col>
                <Col className="edit-col descr-height">
                  <Form.Control
                    as="textarea"
                    value={description ? description : ''}
                    placeholder="describe this group"
                    onChange={(e) => updateDescription(String(e.target.value))}
                  />
                </Col>
              </Row>
              <Row className="mt-4">
                <Col className="header-col">
                  <OverlayTipRight tip={`Analysts have global view into data in Thorium.`}>
                    <b>Analysts</b> <FaQuestionCircle className="group-tooltip" />
                  </OverlayTipRight>
                </Col>
                <Col className="edit-col mt-2">
                  {analysts.map((analyst) => (
                    <Badge bg="" className="bg-goldenrod group-edit-badge" key={'analyst_' + analyst}>
                      <b>{analyst}</b>
                    </Badge>
                  ))}
                </Col>
              </Row>
              <Row className="mt-4">
                <Col className="header-col">
                  <OverlayTipRight
                    tip={`Owners can access and edit all group resources.
                                        They can also delete the group or remove other
                                        owners from the group.`}
                  >
                    <b>Owners</b> <FaQuestionCircle className="group-tooltip" />
                  </OverlayTipRight>
                </Col>
                <Col className="edit-col">
                  <div className="mt-2">
                    {combinedOwners.length > 0 && <Subtitle>Combined</Subtitle>}
                    {combinedOwners.map((owner) => (
                      <Badge bg="" className="bg-dark-slate group-edit-badge" key={'combined_owner_' + owner}>
                        <b>{owner}</b>
                      </Badge>
                    ))}
                  </div>
                  <div className="mt-3">
                    <Subtitle>Individuals</Subtitle>
                    <Select
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedDirectOwners}
                      onChange={(selected, newValue) => {
                        setSelectedDirectOwners(selected);
                        updateGroupChanges('owners', 'direct', newValue);
                      }}
                      options={ownerOptions}
                      styles={ownerStyles}
                    />
                  </div>
                  <div className="mt-3">
                    <Subtitle>Metagroup(s)</Subtitle>
                    <CreatableSelect
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedMetagroupOwners}
                      onChange={(selected, newValue) => {
                        setSelectedMetagroupOwners(selected);
                        updateGroupChanges('owners', 'metagroups', newValue);
                      }}
                      options={[]}
                      styles={ownerStyles}
                    />
                  </div>
                </Col>
              </Row>
              <Row className="mt-4">
                <Col className="header-col">
                  <OverlayTipRight
                    tip={`Managers can access and edit all group resources
                                        but cannot delete the group or remove owners.`}
                  >
                    <b>Managers</b> <FaQuestionCircle className="group-tooltip" />
                  </OverlayTipRight>
                </Col>
                <Col className="edit-col">
                  <div className="mt-2">
                    {combinedManagers.length > 0 && <Subtitle>Combined</Subtitle>}
                    {combinedManagers.map((manager) => (
                      <Badge bg="" className="bg-corn-flower  group-edit-badge" key={'combined_manager_' + manager}>
                        <b>{manager}</b>
                      </Badge>
                    ))}
                  </div>
                  <div className="mt-3">
                    <Subtitle>Individuals</Subtitle>
                    <Select
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedDirectManagers}
                      onChange={(selected, newValue) => {
                        setSelectedDirectManagers(selected);
                        updateGroupChanges('managers', 'direct', newValue);
                      }}
                      options={managerOptions}
                      styles={managerStyles}
                    />
                  </div>
                  <div className="mt-3">
                    <Subtitle>Metagroup(s)</Subtitle>
                    <CreatableSelect
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedMetagroupManagers}
                      onChange={(selected, newValue) => {
                        setSelectedMetagroupManagers(selected);
                        updateGroupChanges('managers', 'metagroups', newValue);
                      }}
                      options={[]}
                      styles={managerStyles}
                    />
                  </div>
                </Col>
              </Row>
              <Row className="mt-4">
                <Col className="header-col">
                  <OverlayTipRight
                    tip={`Users can run pipelines and access files
                                        owned by this group.`}
                  >
                    <b>Users</b> <FaQuestionCircle className="group-tooltip" />
                  </OverlayTipRight>
                </Col>
                <Col className="edit-col">
                  <div className="mt-2">
                    {combinedUsers.length > 0 && <Subtitle>Combined</Subtitle>}
                    {combinedUsers.map((user) => (
                      <Badge bg="" className="bg-cadet group-edit-badge" key={'combined_user_' + user}>
                        <b>{user}</b>
                      </Badge>
                    ))}
                  </div>
                  <div className="mt-3">
                    <Subtitle>Individuals</Subtitle>
                    <Select
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedDirectUsers}
                      onChange={(selected, newValue) => {
                        setSelectedDirectUsers(selected);
                        updateGroupChanges('users', 'direct', newValue);
                      }}
                      options={userOptions}
                      styles={userStyles}
                    />
                  </div>
                  <div className="mt-3">
                    <Subtitle>Metagroup(s)</Subtitle>
                    <CreatableSelect
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedMetagroupUsers}
                      onChange={(selected, newValue) => {
                        setSelectedMetagroupUsers(selected);
                        updateGroupChanges('users', 'metagroups', newValue);
                      }}
                      options={[]}
                      styles={userStyles}
                    />
                  </div>
                </Col>
              </Row>
              <Row className="mt-4">
                <Col className="header-col">
                  <OverlayTipRight
                    tip={`Monitors can view the status of reactions and access files
                                        owned by a group but cannot run pipelines or modify files.`}
                  >
                    <b>Monitors</b> <FaQuestionCircle className="group-tooltip" />
                  </OverlayTipRight>
                </Col>
                <Col className="edit-col">
                  <div className="mt-2">
                    {combinedMonitors.length > 0 && <Subtitle>Combined</Subtitle>}
                    {combinedMonitors.map((monitor) => (
                      <Badge bg="" className="bg-grey group-edit-badge" key={'combined_monitor_' + monitor}>
                        <b>{monitor}</b>
                      </Badge>
                    ))}
                  </div>
                  <div className="mt-3">
                    <Subtitle>Individuals</Subtitle>
                    <Select
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedDirectMonitors}
                      onChange={(selected, newValue) => {
                        setSelectedDirectMonitors(selected);
                        updateGroupChanges('monitors', 'direct', newValue);
                      }}
                      options={monitorOptions}
                      styles={monitorStyles}
                    />
                  </div>
                  <div className="mt-3">
                    <Subtitle>Metagroup(s)</Subtitle>
                    <CreatableSelect
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedMetagroupMonitors}
                      onChange={(selected, newValue) => {
                        setSelectedMetagroupMonitors(selected);
                        updateGroupChanges('monitors', 'metagroups', newValue);
                      }}
                      options={[]}
                      styles={monitorStyles}
                    />
                  </div>
                </Col>
              </Row>
              <hr />
            </Form>
          </Row>
          <Row>
            <ModifyGroupButtons group={group} admin={groupAdmin} />
          </Row>
        </Container>
      );
    }
  };

  const CreateGroup = () => {
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [createError, setCreateError] = useState('');
    const [newGroupName, setNewGroupName] = useState('');
    const [newGroupDescription, setNewGroupDescription] = useState('');
    const handleCloseCreateModal = () => {
      setShowCreateModal(false);
      setCreateError('');
    };
    const handleShowCreateModal = () => setShowCreateModal(true);

    return (
      <div>
        <OverlayTipLeft tip={'Create a new Group.'}>
          <Button className="ok-btn" onClick={handleShowCreateModal}>
            +
          </Button>
        </OverlayTipLeft>
        <Modal show={showCreateModal} onHide={handleCloseCreateModal} keyboard={false}>
          <Modal.Header closeButton>
            <Modal.Title>Create New Group</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form.Group>
              <Form.Label>
                <b>Name</b>
              </Form.Label>
              <Form.Control type="text" value={newGroupName} placeholder="name" onChange={(e) => setNewGroupName(String(e.target.value))} />
              <Form.Text className="text-muted">Group names can contain lower case letters, numbers, and dashes.</Form.Text>
            </Form.Group>
            <Form.Group>
              <Form.Label>
                <b>Description</b>
              </Form.Label>
              <Form.Control
                as="textarea"
                value={newGroupDescription}
                placeholder="describe this new group"
                onChange={(e) => setNewGroupDescription(String(e.target.value))}
              />
              {createError != '' && (
                <center>
                  <Alert variant="danger">{createError}</Alert>
                </center>
              )}
              <Form.Text className="text-muted">
                {`Group descriptions should explain a group's intended membership and owned
                resources.`}
              </Form.Text>
            </Form.Group>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              className="ok-btn m-1"
              onClick={async () => {
                if (newGroupName != '') {
                  const groupInfo = {
                    name: newGroupName,
                    description: newGroupDescription,
                  };
                  if (await createGroup(groupInfo, setCreateError)) {
                    fetchGroups(setGroups, setLoading, true);
                    checkCookie();
                  }
                } else {
                  setCreateError('you must specify a group name');
                }
              }}
            >
              Create
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    );
  };

  return (
    <Page title="Groups · Thorium">
      <div className="accordion-list">
        <div>
          <OverlayTipRight tip={`You have access to view ${Object.keys(groups).length} group(s).`}>
            <h2>
              <Badge bg="" className="count-badge">
                {Object.keys(groups).length}
              </Badge>
            </h2>
          </OverlayTipRight>
        </div>
        <Title>Groups</Title>
        <div>
          <h2>
            <CreateGroup />
          </h2>
        </div>
      </div>
      <LoadingSpinner loading={loading}></LoadingSpinner>
      <Accordion alwaysOpen>
        {groups &&
          Object.keys(groups).map((group) => (
            <Accordion.Item key={group} eventKey={group}>
              <Accordion.Header>
                <Container className="accordion-list">
                  <Col className="accordion-item-name mt-2">
                    <div className="text">{group}</div>
                  </Col>
                  <Col className="accordion-item-relation sm-members d-flex justify-content-start mt-2">
                    <small>
                      <i>
                        <GroupMemberCount group={groups[group]} />
                      </i>
                    </small>
                  </Col>
                  <Col className="accordion-item-ownership d-flex justify-content-center">
                    <GroupRoleBadge group={groups[group]} user={userInfo} />
                  </Col>
                </Container>
              </Accordion.Header>
              <Accordion.Body>
                <GroupInfo group={groups[group]} allUsers={allUsers} />
              </Accordion.Body>
            </Accordion.Item>
          ))}
      </Accordion>
    </Page>
  );
};

export default Groups;
