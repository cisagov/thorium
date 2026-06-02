import { useEffect, useState } from 'react';
import { Accordion, Button, Badge, ButtonGroup, ButtonToolbar, Col, Form, Modal, Row } from 'react-bootstrap';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import { FaQuestionCircle } from 'react-icons/fa';
import Select from 'react-select';
import CreatableSelect from 'react-select/creatable';
import type { ActionMeta, MultiValue } from 'react-select';

// project imports
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import Title from '@components/shared/titles/Title';
import GroupMemberCount from '@components/pages/groups/GroupMemberCount';
import GroupRoleBadge from '@components/pages/groups/GroupRoleBadge';
import NoResultsBanner from '@components/shared/alerts/NoResultsBanner';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { OverlayTipRight, OverlayTipTop, OverlayTipLeft } from '@components/shared/overlay/tips';
import { OmnibarGroups } from '@components/pages/search/omnibar/Bars';
import { Clause } from '@components/pages/search/omnibar/ClauseTypes';
import { getGroupsFromClauses, getStringFieldListFromClauses } from '@components/pages/search/omnibar/utils';
import { getAllGroupUsers, hasOverlap } from '@utilities/groups';
import { useAuth } from '@utilities/auth';
import { createReactSelectStyles } from '@utilities/select';
import { canModifyGroup, isGroupOwner } from '@utilities/permissions';
import { fetchGroups } from '@utilities/fetch';
import { listUsers } from '@thorpi/users';
import { createGroup, deleteGroup, getGroup, updateGroup } from '@thorpi/groups';
import { GroupRoleKey, type Group, type GroupUpdate, type GroupRoleUpdate } from '@models/groups';

interface SelectOption {
  value: string;
  label: string;
}

interface LeaveGroupButtonProps {
  group: Group;
}

interface DeleteGroupButtonProps {
  group: Group;
}

interface UpdateGroupButtonProps {
  group: Group;
  changes: GroupUpdate;
  disableUpdate: boolean;
}

interface GroupInfoProps {
  group: Group;
  allUsers: string[];
}

interface ModifyGroupButtonsProps {
  group: Group;
  admin: boolean;
}

// styles for react select badges
const ownerStyles = createReactSelectStyles('White', 'DarkSlateBlue');
const managerStyles = createReactSelectStyles('White', 'CornFlowerBlue');
const userStyles = createReactSelectStyles('White', 'CadetBlue');
const monitorStyles = createReactSelectStyles('White', 'DimGray');

const filterGroups = (groups: Record<string, Group>, clauses: Clause[]): Record<string, Group> => {
  const clauseGroups = getGroupsFromClauses(clauses);
  const clauseUsers = getStringFieldListFromClauses(clauses, 'Users');
  const clauseOwners = getStringFieldListFromClauses(clauses, 'Owners');
  const clauseManagers = getStringFieldListFromClauses(clauses, 'Managers');

  const e = Object.entries(groups).filter(([name, obj]) => {
    const users = getAllGroupUsers(obj.users);
    const owners = getAllGroupUsers(obj.owners);
    const managers = getAllGroupUsers(obj.managers);

    const groupTest = clauseGroups.length > 0 ? clauseGroups.includes(name) : true;
    const userTest = clauseUsers.length > 0 ? hasOverlap(users, clauseUsers) : true;
    const ownerTest = clauseOwners.length > 0 ? hasOverlap(owners, clauseOwners) : true;
    const managerTest = clauseManagers.length > 0 ? hasOverlap(managers, clauseManagers) : true;

    return groupTest && userTest && ownerTest && managerTest;
  });
  return Object.fromEntries(e);
};

const Groups = () => {
  const [loading, setLoading] = useState(false);
  const [groups, setGroups] = useState<Record<string, Group>>({});
  const [allUsers, setAllUSers] = useState<string[]>([]);
  const [clauses, setClauses] = useState<Clause[]>([]);
  const { userInfo, checkCookie } = useAuth();

  const filteredGroups = filterGroups(groups, clauses);
  // get a list of all Thorium users
  const fetchAllUsers = async () => {
    const reqUsers = await listUsers(console.log, false);
    if (reqUsers) {
      setAllUSers((reqUsers as string[]).sort());
    }
  };

  // Refetch a single group and replace just that entry in place. Used after an edit so only the
  // changed group's content rerenders, without reloading the whole list or toggling the page
  // spinner (which keeps every other open accordion untouched).
  const refreshSingleGroup = async (name: string) => {
    const fresh = await getGroup(name, console.log);
    if (fresh) {
      setGroups((prev) => ({ ...prev, [name]: fresh }));
    }
  };

  // get list of groups and users on initial page load
  useEffect(() => {
    // Get group details on page load
    void fetchGroups(setGroups as (groups: Record<string, Group> | Group[] | string[]) => void, setLoading, true);
    void fetchAllUsers();
    // if user info changes, we want to get the group details again
  }, [userInfo]);

  const LeaveGroupButton = ({ group }: LeaveGroupButtonProps) => {
    const [showLeaveModal, setShowLeaveModal] = useState(false);
    const [leaveError, setLeaveError] = useState('');
    const handleCloseLeaveModal = () => setShowLeaveModal(false);
    const username = userInfo!.username;
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

  // display delete button and confirmation modal
  const DeleteGroupButton = ({ group }: DeleteGroupButtonProps) => {
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [deleteError, setDeleteError] = useState('');
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

  // display update group button and confirmation modals
  const UpdateGroupButton = ({ group, changes, disableUpdate }: UpdateGroupButtonProps) => {
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

  const GroupInfo = ({ group, allUsers }: GroupInfoProps) => {
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
    const directMonitors = 'monitors' in group && 'direct' in group.monitors ? group.monitors.direct.sort() : [];
    const [description, setDescription] = useState<string | undefined>('description' in group ? group.description : '');
    const [disableUpdate, setDisableUpdate] = useState(true);
    const [groupChanges, setGroupChanges] = useState<GroupUpdate>({});

    // Create options for drop downs for use by react/select field
    const usersWithRoles = [...directOwners, ...directManagers, ...directUsers, ...directMonitors];

    const startingUserOptions: SelectOption[] =
      allUsers.length == 0
        ? []
        : allUsers.reduce<SelectOption[]>((options, name) => {
            if (!usersWithRoles.includes(name)) {
              options.push({ value: name, label: name });
            }
            return options;
          }, []);
    const [ownerOptions, setOwnerOptions] = useState<SelectOption[]>(startingUserOptions);
    const [managerOptions, setManagerOptions] = useState<SelectOption[]>(startingUserOptions);
    const [userOptions, setUserOptions] = useState<SelectOption[]>(startingUserOptions);
    const [monitorOptions, setMonitorOptions] = useState<SelectOption[]>(startingUserOptions);

    // generate selected owners from starting group roles for use by react/select
    const [selectedDirectOwners, setSelectedDirectOwners] = useState<readonly SelectOption[]>(
      directOwners.map((owner) => {
        return { value: owner, label: owner };
      }),
    );
    // generate selected ldap group owners from starting group roles for use by react/select
    const [selectedMetagroupOwners, setSelectedMetagroupOwners] = useState<readonly SelectOption[]>(
      metagroupOwners.map((owner) => {
        return { value: owner, label: owner };
      }),
    );
    // generate selected managers from starting group roles for use by react/select
    const [selectedDirectManagers, setSelectedDirectManagers] = useState<readonly SelectOption[]>(
      directManagers.map((manager) => {
        return { value: manager, label: manager };
      }),
    );
    // generate selected ldap group managers from starting group roles for use by react/select
    const [selectedMetagroupManagers, setSelectedMetagroupManagers] = useState<readonly SelectOption[]>(
      metagroupManagers.map((manager) => {
        return { value: manager, label: manager };
      }),
    );
    // generate selected users from starting group roles for use by react/select
    const [selectedDirectUsers, setSelectedDirectUsers] = useState<readonly SelectOption[]>(
      directUsers.map((user) => {
        return { value: user, label: user };
      }),
    );
    // generate selected ldap group users from starting group roles for use by react/select
    const [selectedMetagroupUsers, setSelectedMetagroupUsers] = useState<readonly SelectOption[]>(
      metagroupUsers.map((user) => {
        return { value: user, label: user };
      }),
    );
    // generate selected monitors for use by react/select
    const [selectedDirectMonitors, setSelectedDirectMonitors] = useState<readonly SelectOption[]>(
      directMonitors.map((monitor) => {
        return { value: monitor, label: monitor };
      }),
    );
    // generate selected ldap group monitors from starting group roles for use by react/select
    const [selectedMetagroupMonitors, setSelectedMetagroupMonitors] = useState<readonly SelectOption[]>(
      metagroupMonitors.map((monitor) => {
        return { value: monitor, label: monitor };
      }),
    );

    // see if user should be able to edit group
    // thorium admins and group owners/managers can edit non-owner membership (Group::modifiable)
    const groupAdmin = canModifyGroup(group, userInfo!);
    // only thorium admins and group owners can edit the owners role (Group::is_owner)
    const groupOwner = isGroupOwner(group, userInfo!);

    // update the changes to groups structure for eventual
    // submission to the Thorium API
    const updateGroupChanges = (role: string, type: string, newValue: ActionMeta<SelectOption>) => {
      const changes = structuredClone(groupChanges) as Record<string, Record<string, string[]>>;
      let updatedUsersWithRoles: string[] = [];
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
        entity = newValue.option!.value;

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
          changes[role][reverseAction] = changes[role][reverseAction].filter((name: string) => {
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
        const newRoleUpdate: Record<string, string[]> = {};
        newRoleUpdate[roleAction] = [entity];
        changes[role] = newRoleUpdate;
      }

      // update options for role drop downs for use by react/select field
      // users can only have one role in a given group and are removed
      // from the drop down once added to a group
      const updatedUserOptions: SelectOption[] =
        allUsers.length == 0
          ? []
          : allUsers.reduce<SelectOption[]>((options, name) => {
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

    const updateDescription = (description: string) => {
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

    const ModifyGroupButtons = ({ group, admin }: ModifyGroupButtonsProps) => {
      // only owners, managers and Thorium admins can modify a group
      return (
        <>
          <Row>
            <ButtonToolbar className="d-flex justify-content-center">
              <ButtonGroup>
                {admin && <UpdateGroupButton group={group} changes={groupChanges} disableUpdate={disableUpdate} />}
                <LeaveGroupButton group={group} />
                {admin && <DeleteGroupButton group={group} />}
              </ButtonGroup>
            </ButtonToolbar>
          </Row>
        </>
      );
    };

    if (!groupAdmin) {
      // return non-editable group info component
      return (
        <>
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
        </>
      );
    } else {
      // return an editable admin group info component
      return (
        <>
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
                  {/* only owners/admins can edit the owners role; managers see it read-only */}
                  {groupOwner ? (
                    <>
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
                        <Select<SelectOption, true>
                          isMulti
                          isSearchable
                          isClearable={false}
                          defaultValue={selectedDirectOwners}
                          onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
                            setSelectedDirectOwners(selected);
                            updateGroupChanges('owners', 'direct', newValue);
                          }}
                          options={ownerOptions}
                          styles={ownerStyles}
                        />
                      </div>
                      <div className="mt-3">
                        <Subtitle>Metagroup(s)</Subtitle>
                        <CreatableSelect<SelectOption, true>
                          isMulti
                          isSearchable
                          isClearable={false}
                          defaultValue={selectedMetagroupOwners}
                          onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
                            setSelectedMetagroupOwners(selected);
                            updateGroupChanges('owners', 'metagroups', newValue);
                          }}
                          options={[]}
                          styles={ownerStyles}
                        />
                      </div>
                    </>
                  ) : (
                    <div className="mt-2">
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
                    </div>
                  )}
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
                    <Select<SelectOption, true>
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedDirectManagers}
                      onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
                        setSelectedDirectManagers(selected);
                        updateGroupChanges('managers', 'direct', newValue);
                      }}
                      options={managerOptions}
                      styles={managerStyles}
                    />
                  </div>
                  <div className="mt-3">
                    <Subtitle>Metagroup(s)</Subtitle>
                    <CreatableSelect<SelectOption, true>
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedMetagroupManagers}
                      onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
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
                    <Select<SelectOption, true>
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedDirectUsers}
                      onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
                        setSelectedDirectUsers(selected);
                        updateGroupChanges('users', 'direct', newValue);
                      }}
                      options={userOptions}
                      styles={userStyles}
                    />
                  </div>
                  <div className="mt-3">
                    <Subtitle>Metagroup(s)</Subtitle>
                    <CreatableSelect<SelectOption, true>
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedMetagroupUsers}
                      onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
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
                    <Select<SelectOption, true>
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedDirectMonitors}
                      onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
                        setSelectedDirectMonitors(selected);
                        updateGroupChanges('monitors', 'direct', newValue);
                      }}
                      options={monitorOptions}
                      styles={monitorStyles}
                    />
                  </div>
                  <div className="mt-3">
                    <Subtitle>Metagroup(s)</Subtitle>
                    <CreatableSelect<SelectOption, true>
                      isMulti
                      isSearchable
                      isClearable={false}
                      defaultValue={selectedMetagroupMonitors}
                      onChange={(selected: MultiValue<SelectOption>, newValue: ActionMeta<SelectOption>) => {
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
        </>
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
              {createError != '' && <AlertBanner>{createError}</AlertBanner>}
              <Form.Text className="text-muted">
                {`Group descriptions should explain a group's indended membership and owned
                resources.`}
              </Form.Text>
            </Form.Group>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              className="ok-btn m-1"
              onClick={() => {
                void (async () => {
                  if (newGroupName != '') {
                    const groupInfo = {
                      name: newGroupName,
                      description: newGroupDescription,
                    };
                    if (await createGroup(groupInfo, setCreateError)) {
                      void fetchGroups(setGroups as (groups: Record<string, Group> | Group[] | string[]) => void, setLoading, true);
                      void checkCookie();
                    }
                  } else {
                    setCreateError('you must specify a group name');
                  }
                })();
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
      <div className="d-flex justify-content-between">
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
      <div className="d-flex justify-content-center">
        <OmnibarGroups clauses={clauses} setClauses={setClauses} groups={groups} />
      </div>
      <LoadingSpinner loading={loading}></LoadingSpinner>
      {!loading && Object.keys(filteredGroups).length === 0 && <NoResultsBanner type="Groups" />}
      <Accordion alwaysOpen>
        {filteredGroups &&
          Object.keys(filteredGroups)
            .sort()
            .map((group) => (
              <Accordion.Item key={group} eventKey={group}>
                <Accordion.Header>
                  <Col className="accordion-item-name mt-2">
                    <div className="text">{group}</div>
                  </Col>
                  <Col className="accordion-item-relation mt-2">
                    <small>
                      <i>
                        <GroupMemberCount group={filteredGroups[group]} />
                      </i>
                    </small>
                  </Col>
                  <Col className="accordion-item-ownership d-flex justify-content-center">
                    {userInfo && <GroupRoleBadge group={filteredGroups[group]} user={userInfo} />}
                  </Col>
                </Accordion.Header>
                <Accordion.Body>
                  <GroupInfo group={filteredGroups[group]} allUsers={allUsers} />
                </Accordion.Body>
              </Accordion.Item>
            ))}
      </Accordion>
    </Page>
  );
};

export default Groups;
