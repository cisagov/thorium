import { Badge, Tooltip, OverlayTrigger } from 'react-bootstrap';

// project imports
import { getGroupRoleBadge } from '@utilities/role';
import { Group } from '@models/groups';
import { UserInfo } from '@models/users';

interface GroupRoleBadgeProps {
  group: Group;
  user: UserInfo;
}

// Get the badge for a given role
const GroupRoleBadge: React.FC<GroupRoleBadgeProps> = ({ group, user }) => {
  const badge = getGroupRoleBadge(group, user);
  // user has no role in this group and is not an admin
  if (!badge) {
    return null;
  }
  return (
    <OverlayTrigger placement="bottom" overlay={<Tooltip>{badge.tooltip}</Tooltip>}>
      <Badge bg="" className={`${badge.className} group-badge`}>
        <div className="mb-2">{badge.label}</div>
      </Badge>
    </OverlayTrigger>
  );
};

export default GroupRoleBadge;
