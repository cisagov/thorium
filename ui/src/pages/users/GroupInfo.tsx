import { useState } from 'react';
import { Badge, ButtonGroup, ButtonToolbar, Col, Form, Row } from 'react-bootstrap';
import { FaQuestionCircle } from 'react-icons/fa';
import Select from 'react-select';
import CreatableSelect from 'react-select/creatable';
import type { ActionMeta, MultiValue } from 'react-select';

// project imports
import Subtitle from '@components/shared/titles/Subtitle';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { useAuth } from '@utilities/auth';
import { canModifyGroup, isGroupOwner } from '@utilities/permissions';
import { type Group, type GroupUpdate } from '@models/groups';
import { createReactSelectStyles } from '@utilities/select';
import { DeleteGroupButton, LeaveGroupButton, UpdateGroupButton } from './GroupButtons';

interface SelectOption {
  value: string;
  label: string;
}

interface GroupInfoProps {
  group: Group;
  allUsers: string[];
  setLoading: (next: boolean) => void;
  setGroups: (next: Record<string, Group>) => void;
  refreshSingleGroup: (name: string) => Promise<void>;
}

interface ModifyGroupButtonsProps {
  group: Group;
  admin: boolean;
  groupChanges: GroupUpdate;
  disableUpdate: boolean;
  setLoading: (next: boolean) => void;
  setGroups: (next: Record<string, Group>) => void;
  refreshSingleGroup: (name: string) => Promise<void>;
}

const ModifyGroupButtons = ({
  group,
  admin,
  groupChanges,
  disableUpdate,
  setLoading,
  setGroups,
  refreshSingleGroup,
}: ModifyGroupButtonsProps) => {
  const { userInfo } = useAuth();
  // only owners, managers and Thorium admins can modify a group
  return (
    <>
      <Row>
        <ButtonToolbar className="d-flex justify-content-center">
          <ButtonGroup>
            {admin && (
              <UpdateGroupButton
                group={group}
                changes={groupChanges}
                disableUpdate={disableUpdate}
                refreshSingleGroup={refreshSingleGroup}
              />
            )}
            <LeaveGroupButton group={group} username={userInfo?.username || ''} setLoading={setLoading} setGroups={setGroups} />
            {admin && <DeleteGroupButton group={group} setLoading={setLoading} setGroups={setGroups} />}
          </ButtonGroup>
        </ButtonToolbar>
      </Row>
    </>
  );
};

// styles for react select badges
const ownerStyles = createReactSelectStyles('White', 'DarkSlateBlue');
const managerStyles = createReactSelectStyles('White', 'CornFlowerBlue');
const userStyles = createReactSelectStyles('White', 'CadetBlue');
const monitorStyles = createReactSelectStyles('White', 'DimGray');

const GroupInfo = ({ group, allUsers, setGroups, setLoading, refreshSingleGroup }: GroupInfoProps) => {
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
  const { userInfo } = useAuth();

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
          <ModifyGroupButtons
            group={group}
            admin={groupAdmin}
            groupChanges={groupChanges}
            disableUpdate={disableUpdate}
            setGroups={setGroups}
            setLoading={setLoading}
            refreshSingleGroup={refreshSingleGroup}
          />
        </Row>
      </>
    );
  }
};

export default GroupInfo;
