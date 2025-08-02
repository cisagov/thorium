import React from 'react';
import { Badge, Button, Tooltip, OverlayTrigger } from 'react-bootstrap';

// Get the badge for a given role
const GroupRoleBadge = ({ group, user }) => {
  if (group.owners.combined.includes(user.username)) {
    return (
      <OverlayTrigger
        placement="bottom"
        overlay={
          <Tooltip>
            You are an Owner of this group. Owners can add/remove any member within the group. An owner can also access and edit all group
            resources.
          </Tooltip>
        }
      >
        <Badge bg="" className="bg-dark-slate group-badge">
          <div className="mb-2">Owner</div>
        </Badge>
      </OverlayTrigger>
    );
  } else if (group.managers.combined.includes(user.username)) {
    return (
      <OverlayTrigger
        placement="bottom"
        overlay={
          <Tooltip>
            You are a Manager of this group. Managers can edit non-Owner membership within the group as well as all group resources.
          </Tooltip>
        }
      >
        <Badge bg="" className="bg-corn-flower group-badge">
          <div className="mb-2">Manager</div>
        </Badge>
      </OverlayTrigger>
    );
  } else if (group.users.combined.includes(user.username)) {
    return (
      <OverlayTrigger
        placement="bottom"
        overlay={
          <Tooltip>
            You are a User in this group. A user can view group membership and resources. Users can also upload samples/repos and conduct
            analysis on them.
          </Tooltip>
        }
      >
        <Badge bg="" className="bg-cadet group-badge">
          <div className="mb-2">User</div>
        </Badge>
      </OverlayTrigger>
    );
  } else if (group.monitors.combined.includes(user.username)) {
    return (
      <OverlayTrigger
        placement="bottom"
        overlay={
          <Tooltip>
            You are a Monitor of this group. Monitors can view group membership, track running jobs and analyze tool results. Monitors
            cannot run jobs or modify any group resources.
          </Tooltip>
        }
      >
        <Badge bg="" className="bg-grey group-badge">
          <div className="mb-2">Monitor</div>
        </Badge>
      </OverlayTrigger>
    );
  } else if (user.role == 'Admin') {
    return (
      <OverlayTrigger placement="bottom" overlay={<Tooltip>You are a Thorium admin. You have all the permissions.</Tooltip>}>
        <Badge bg="" className="bg-maroon group-badge">
          <div className="mb-2">Admin</div>
        </Badge>
      </OverlayTrigger>
    );
  }
};

// Returns the total member count including all users,
//   monitors, managers and owners
const GroupMemberCount = ({ group }) => {
  const count =
    group.owners.combined.length + group.users.combined.length + group.managers.combined.length + group.monitors.combined.length;
  // use plural string if multiple members
  if (count > 1) {
    return <>{count} Members</>;
  } else {
    return <>{count} Member</>;
  }
};

const SelectGroups = ({ groups, setGroups, disabled, clearState = null }) => {
  if (groups) {
    return (
      <>
        {Object.keys(groups)
          .sort()
          .map((group) => (
            <Button
              key={group}
              className={`m-1 primary-btn ${groups[group] ? 'selected' : 'unselected'}`}
              variant=""
              disabled={disabled}
              onClick={() => {
                setGroups({ ...groups, [group]: !groups[group] });
                if (clearState) {
                  clearState();
                }
              }}
            >
              <font size="3">
                <b>{group}</b>
              </font>
            </Button>
          ))}
      </>
    );
  }
  return null;
};

export { GroupRoleBadge, GroupMemberCount, SelectGroups };
