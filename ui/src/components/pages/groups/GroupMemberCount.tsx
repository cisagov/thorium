// project imports
import { getGroupMemberCount } from '@utilities/groups';
import { Group } from '@models/groups';

interface GroupMemberCountProps {
  group: Group;
}

// Returns the total member count including all users,
//   monitors, managers, analysts and owners
const GroupMemberCount: React.FC<GroupMemberCountProps> = ({ group }) => {
  const count = getGroupMemberCount(group);
  // use plural string if multiple members
  if (count > 1) {
    return <>{count} Members</>;
  } else {
    return <>{count} Member</>;
  }
};

export default GroupMemberCount;
