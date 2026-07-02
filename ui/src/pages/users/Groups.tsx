import { useEffect, useState } from 'react';
import { Accordion, Button, Badge, Col, Form, Modal } from 'react-bootstrap';
import AlertBanner from '@components/shared/alerts/AlertBanner';

// project imports
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import GroupMemberCount from '@components/pages/groups/GroupMemberCount';
import GroupRoleBadge from '@components/pages/groups/GroupRoleBadge';
import NoResultsBanner from '@components/shared/alerts/NoResultsBanner';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { OverlayTipRight, OverlayTipLeft } from '@components/shared/overlay/tips';
import { OmnibarGroups } from '@components/shared/inputs/omnibar/Bars';
import { Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { defaultTimeSelection } from '@components/shared/inputs/omnibar/timepicker/utils';
import { useOmnibarUrlState } from '@components/shared/inputs/omnibar/useOmnibarUrlState';
import { getGroupsFromClauses, getStringFieldListFromClauses } from '@components/shared/inputs/omnibar/utils';
import { getAllGroupUsers, hasOverlap } from '@utilities/groups';
import { useAuth } from '@utilities/auth';
import { fetchGroups } from '@utilities/fetch';
import { listUsers } from '@thorpi/users';
import { createGroup, getGroup } from '@thorpi/groups';
import { type Group } from '@models/groups';
import GroupInfo from './GroupInfo';

interface CreateGroupProps {
  setLoading: (next: boolean) => void;
  setGroups: (next: Record<string, Group>) => void;
}

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

const CreateGroup = ({ setGroups, setLoading }: CreateGroupProps) => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createError, setCreateError] = useState('');
  const [newGroupName, setNewGroupName] = useState('');
  const [newGroupDescription, setNewGroupDescription] = useState('');
  const { checkCookie } = useAuth();

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

const Groups = () => {
  const [loading, setLoading] = useState(false);
  const [groups, setGroups] = useState<Record<string, Group>>({});
  const [allUsers, setAllUSers] = useState<string[]>([]);
  // omnibar filters live in the URL so a filtered group list is shareable
  const { clauses, setClauses } = useOmnibarUrlState({ clauses: [], time: defaultTimeSelection() });
  const { userInfo } = useAuth();

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
            <CreateGroup setGroups={setGroups} setLoading={setLoading} />
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
                  <GroupInfo
                    group={filteredGroups[group]}
                    allUsers={allUsers}
                    setGroups={setGroups}
                    setLoading={setLoading}
                    refreshSingleGroup={refreshSingleGroup}
                  />
                </Accordion.Body>
              </Accordion.Item>
            ))}
      </Accordion>
    </Page>
  );
};

export default Groups;
